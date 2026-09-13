"""AIChannel mixin for streaming response generation with tool loops."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._ai_coalescers import _ThinkingCoalescer, _ToolCallDeltaCoalescer
from roomkit.channels._ai_loop_rules import (
    AIToolLoopRulesMixin,
    _accumulate_usage,
    final_round_reason,
)
from roomkit.channels._ai_resilience import _StreamRetryBoundary
from roomkit.channels._ai_stream_external_tools import _ExternalStreamTools
from roomkit.channels._ai_stream_round import _StreamRound, _StreamRoundState
from roomkit.models.channel import ChannelOutput
from roomkit.models.event import RoomEvent
from roomkit.models.streaming import (
    LoopEndMarker,
    LoopEndReason,
    StreamDelta,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.models.tool_call import AIResponseEvent, response_transcript
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
)
from roomkit.providers.utils import _aclose_stream
from roomkit.realtime.base import EphemeralEventType
from roomkit.telemetry.base import Attr, SpanKind, TelemetryProvider
from roomkit.telemetry.context import get_current_span

if TYPE_CHECKING:
    from roomkit.channels.ai import _ToolLoopContext
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.providers.ai.base import StreamEvent
    from roomkit.telemetry.noop import NoopTelemetryProvider


logger = logging.getLogger("roomkit.channels.ai")


@dataclass
class _StreamTurnState:
    """Invocation-owned state; each round contributes its delivered fragments."""

    loop_ctx: _ToolLoopContext
    telemetry: TelemetryProvider
    span_id: str
    room_id: str | None
    usage: dict[str, int] = field(default_factory=dict)
    segments: list[list[str]] = field(default_factory=list)
    tool_calls_count: int = 0
    tool_rounds_count: int = 0
    reason: LoopEndReason = "completed"
    started_at: float = field(default_factory=time.monotonic)
    dedup_prefix: str = ""
    saw_tool_call: bool = False


@runtime_checkable
class AIStreamingHost(Protocol):
    """Contract: capabilities a host class must provide for AIStreamingMixin.

    Attributes provided by the host's ``__init__``:
        _provider: AI provider for generation.
        _max_tool_rounds: Maximum tool-loop iterations.
        _tool_loop_timeout_seconds: Optional wall-clock timeout for the loop.
        _tool_loop_warn_after: Log a warning after this many rounds.
        _tool_handler: Tool call handler (or ``None`` if tools disabled).
        _active_loops: Registry of currently running tool loops.
        _text_streams: Count of text-only streams currently being produced.
        _after_response_hook: Optional callback fired after response generation.
        channel_id: Unique identifier for this channel.

    Properties / methods provided by other mixins:
        _build_context: ``AIContextMixin`` — builds AI context from room state.
        _drain_steering_queue: ``AISteeringMixin`` — drains pending directives.
        _generate_stream_with_retry: ``AIResilienceMixin`` — stream with retry.
        _publish_thinking_event: ``AIEventsMixin`` — publish thinking events.
        _publish_tool_event: ``AIEventsMixin`` — publish tool call events.
        _telemetry_provider: ``AIGenerationMixin`` property — telemetry provider.

    The shared per-round loop rules (force-stop, empty-retry, budget, parts
    assembly, tool execution) come from :class:`AIToolLoopRulesMixin`, the
    mixin's own base — see :class:`AIToolLoopRulesHost` for that contract.
    """

    _provider: Any
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _max_empty_retries: int
    _thinking_coalesce_ms: float
    _thinking_coalesce_chars: int
    _tool_handler: Any
    _active_loops: dict[str, _ToolLoopContext]
    _text_streams: int
    _after_response_hook: Any
    _before_generation_hook: Any
    _before_tool_call_hook: Any
    _tool_call_hook: Any
    _external_tool_handler: Any
    channel_id: str

    async def _build_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext: ...
    def _drain_steering_queue(
        self, context: AIContext, loop_ctx: _ToolLoopContext
    ) -> tuple[AIContext, bool]: ...
    async def _generate_stream_with_retry(
        self, context: AIContext
    ) -> AsyncIterator[StreamEvent | _StreamRetryBoundary]: ...
    async def _publish_thinking_event(
        self,
        event_type: EphemeralEventType,
        room_id: str,
        thinking: str,
        round_idx: int,
    ) -> None: ...
    async def _publish_tool_event(
        self,
        event_type: EphemeralEventType,
        room_id: str,
        tool_calls: list[Any],
        round_idx: int,
        *,
        duration_ms: int | None = ...,
    ) -> None: ...
    @property
    def _telemetry_provider(self) -> NoopTelemetryProvider: ...


class AIStreamingMixin(AIToolLoopRulesMixin):
    """Streaming AI response generation with tool loop and deduplication.

    Host contract: :class:`AIStreamingHost`.
    """

    _provider: Any
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _max_empty_retries: int
    _thinking_coalesce_ms: float
    _thinking_coalesce_chars: int
    _tool_handler: Any
    _active_loops: dict[str, Any]
    _text_streams: int
    _after_response_hook: Any
    _before_generation_hook: Any
    _before_tool_call_hook: Any
    _tool_call_hook: Any
    _external_tool_handler: Any
    channel_id: str

    # Cross-mixin methods — Any annotations avoid MRO shadowing.
    # _build_context is NOT annotated here: it's a real typed method on
    # AIContextMixin whose return type must be preserved for subclasses
    # (Agent.super()._build_context()). Call sites use type: ignore instead.
    _drain_steering_queue: Any  # see AIStreamingHost
    _generate_stream_with_retry: Any  # see AIStreamingHost
    _publish_thinking_event: Any  # see AIStreamingHost
    _publish_tool_event: Any  # see AIStreamingHost
    _telemetry_provider: Any  # see AIStreamingHost

    def _new_thinking_coalescer(self, room_id: str | None, round_idx: int) -> _ThinkingCoalescer:
        """Coalescer bound to this channel's publish hook and window config."""
        return _ThinkingCoalescer(
            self._publish_thinking_event,
            room_id,
            round_idx,
            flush_ms=self._thinking_coalesce_ms,
            flush_chars=self._thinking_coalesce_chars,
        )

    async def _close_thinking_window(
        self,
        coalescer: _ThinkingCoalescer,
        room_id: str,
        thinking_parts: list[str],
        round_idx: int,
        *,
        published: int,
    ) -> int:
        """Flush the buffered reasoning, publish ``THINKING_END``, return the offset.

        The window closes whenever the model stops reasoning and starts
        producing — a text delta, a tool call's first fragment, or the end of
        the stream — and on every abnormal exit of a round: a cancelled turn,
        a provider that died mid-reasoning, a consumer that stopped reading.
        One place for every call site, so a new exit cannot close a window
        differently from the others.

        Whatever closed it, ``THINKING_END`` carries the block reasoned so
        far. The subscriber has been reading that block in deltas, so an
        empty payload on an abnormal exit would be a second contract to
        learn for the one case where the block is already at hand; and the
        deltas the coalescer still holds go out ahead of the close instead
        of dying with the round.

        A round can open several windows (reason, answer, reason again), and
        each ``THINKING_END`` must carry its own block. ``published`` is how
        many of ``thinking_parts`` earlier windows already sent; the caller
        keeps the returned value and hands it back at the next close. The list
        itself is never truncated — the tool loop replays it whole into the
        assistant message it sends back to the model.
        """
        await coalescer.flush()
        await self._publish_thinking_event(
            EphemeralEventType.THINKING_END,
            room_id,
            "".join(thinking_parts[published:]),
            round_idx,
        )
        return len(thinking_parts)

    def _new_tool_call_coalescer(
        self, room_id: str | None, round_idx: int
    ) -> _ToolCallDeltaCoalescer:
        """Coalescer bound to this channel's publish hook and window config.

        It shares the thinking windows on purpose: both bound the rate at which
        one round's in-progress work reaches the bus, and a second pair of knobs
        would be public surface with no demonstrated need behind it.
        """
        return _ToolCallDeltaCoalescer(
            self._publish_tool_event,
            room_id,
            round_idx,
            flush_ms=self._thinking_coalesce_ms,
            flush_chars=self._thinking_coalesce_chars,
        )

    async def _start_streaming_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response handle (generator starts on consumption)."""
        ai_context = await self._build_context(event, binding, context)  # ty: ignore[unresolved-attribute]
        ai_context, blocked = await self._fire_before_generation_hook(ai_context, event)  # ty: ignore[unresolved-attribute]
        if blocked:
            return ChannelOutput.empty()
        return ChannelOutput(
            responded=True,
            response_stream=self._stream_text_with_thinking(ai_context),
            response_metadata=ai_context.response_metadata,
        )

    async def _stream_text_with_thinking(
        self, ai_context: AIContext
    ) -> AsyncIterator[StreamDelta]:
        """Yield text deltas + thinking markers, publish realtime events.

        Two parallel mechanisms by design:

        * **Inline (channel stream)** — every ``StreamThinkingDelta`` becomes
          a :class:`ThinkingDeltaMarker` yielded in arrival order alongside
          text deltas. Channels that want to render reasoning in line with
          the answer (CLI, web) consume them; text-only channels filter
          them out via ``isinstance(chunk, str)``.

        * **Out-of-band (realtime bus)** — one ``THINKING_END`` event per
          reasoning window, carrying that window's block and nothing else,
          is published for observers (dashboards, audit logs). A model that
          reasons, answers and reasons again opens several windows in one
          round, and a subscriber appends what it receives. This matches the
          tool-loop and non-streaming paths so payloads stay consistent.

        Falls back to ``generate_stream`` for providers that don't expose
        a structured stream.
        """
        # Counted from the first consumption to the close of the generator, the
        # way a tool loop registers itself in ``_active_loops`` for the same
        # span: a caller retiring this object (``active_turns``) must know a
        # text-only stream is still being produced, and this path has no loop
        # context to register.
        self._text_streams += 1
        # Declared ahead of the try so the finally can reach them: a provider
        # that dies mid-reasoning, or a consumer that stops reading, leaves
        # this stream at a point where the window is still open.
        # ``thinking_started`` is True exactly while a window is open on the
        # bus.
        room_id = ai_context.room.room.id if ai_context.room else None
        started_at = time.monotonic()
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        completed = False
        thinking_parts: list[str] = []
        thinking_published = 0
        thinking_started = False
        coalescer = self._new_thinking_coalescer(room_id, round_idx=0)
        stream: Any = None
        try:
            if not self._provider.supports_structured_streaming:
                stream = self._provider.generate_stream(ai_context)
                async for chunk in stream:
                    text_parts.append(chunk)
                    yield chunk
                completed = True
                return

            # Through the resilience wrapper, like every structured generation:
            # retry, fallback and overflow compaction are the wrapper's to give,
            # never a per-path courtesy.
            stream = self._generate_stream_with_retry(ai_context)
            async for ev in stream:
                if isinstance(ev, _StreamRetryBoundary):
                    continue
                if isinstance(ev, StreamThinkingDelta):
                    if not thinking_started and room_id:
                        thinking_started = True
                        await self._publish_thinking_event(
                            EphemeralEventType.THINKING_START, room_id, "", 0
                        )
                    thinking_parts.append(ev.thinking)
                    # Buffer each delta and publish in windows on the realtime bus so
                    # remote subscribers (browser WS clients, etc.) stream the
                    # reasoning as it arrives, not only the buffered text at
                    # THINKING_END. The ``thinking`` field carries the delta, not the
                    # accumulator — clients append to their own buffer.
                    await coalescer.add(ev.thinking)
                    yield ThinkingDeltaMarker(thinking=ev.thinking)
                elif isinstance(ev, StreamTextDelta):
                    if thinking_started and thinking_parts and room_id:
                        thinking_started = False
                        thinking_published = await self._close_thinking_window(
                            coalescer, room_id, thinking_parts, 0, published=thinking_published
                        )
                    text_parts.append(ev.text)
                    yield ev.text
                elif isinstance(ev, StreamDone):
                    usage.update(ev.usage)

            # Thinking with no following text — close the boundary anyway so
            # subscribers see the reasoning even if the model emitted nothing else.
            if thinking_started and thinking_parts and room_id:
                thinking_started = False
                await self._close_thinking_window(
                    coalescer, room_id, thinking_parts, 0, published=thinking_published
                )
            completed = True
        finally:
            try:
                await _aclose_stream(stream)
                # A window still open here was left by an abnormal exit — a
                # provider error, a consumer that closed the stream — and
                # closes with the block reasoned so far, so THINKING_START
                # never stays unpaired. Publishing is best-effort: the error
                # that ended the stream is the one that propagates.
                if thinking_started and thinking_parts and room_id:
                    await self._close_thinking_window(
                        coalescer, room_id, thinking_parts, 0, published=thinking_published
                    )
            finally:
                # The close publishes, and a publish that suspends can be
                # cancelled under a consumer already being torn down. The
                # count comes down whatever happens to it, or a caller
                # retiring the channel waits for zero forever.
                self._text_streams -= 1
                # Exhaustion, not merely entering finally, marks a completed
                # response. Provider errors and consumers closing early must
                # not report a successful turn to evaluation/accounting hooks.
                if completed and self._after_response_hook:
                    try:
                        segments, transcript = response_transcript(["".join(text_parts)])
                        await self._after_response_hook(
                            AIResponseEvent(
                                channel_id=self.channel_id,
                                response_content=transcript,
                                segments=segments,
                                room_id=room_id,
                                usage=usage,
                                thinking="".join(thinking_parts),
                                latency_ms=int((time.monotonic() - started_at) * 1000),
                                streaming=True,
                                # No tool loop ran, and the hook fires only on
                                # exhaustion: the turn ended on its own terms.
                                loop_end_reason="completed",
                            )
                        )
                    except Exception:
                        logger.debug("After-response hook failed (streaming)", exc_info=True)

    async def _start_streaming_tool_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response that handles tool calls between rounds."""
        from roomkit.channels.ai import _current_loop_ctx

        ai_context = await self._build_context(event, binding, context)  # ty: ignore[unresolved-attribute]
        ai_context, blocked = await self._fire_before_generation_hook(ai_context, event)  # ty: ignore[unresolved-attribute]
        if blocked:
            return ChannelOutput.empty()
        # The generator below executes when the CONSUMER iterates the
        # stream — by then handle_event has reset the loop contextvar, so
        # the parent ctx (participant role, room, the toolset stamped by
        # _build_context) must be captured NOW and passed explicitly.
        return ChannelOutput(
            responded=True,
            response_stream=self._run_streaming_tool_loop(
                ai_context, parent_loop_ctx=_current_loop_ctx.get()
            ),
            response_metadata=ai_context.response_metadata,
        )

    def _record_stream_usage(self, total: dict[str, int], usage: dict[str, Any]) -> None:
        """Accumulate every counter and project the input/output metrics."""
        _accumulate_usage(total, usage)
        telemetry = self._telemetry_provider
        for counter in ("input_tokens", "output_tokens"):
            telemetry.record_metric(
                f"roomkit.llm.{counter}",
                float(usage.get(counter, 0)),
                unit="tokens",
                attributes={"channel_id": self.channel_id},
            )

    @asynccontextmanager
    async def _streaming_tool_turn(
        self, context: AIContext, parent_loop_ctx: _ToolLoopContext | None
    ) -> AsyncIterator[_StreamTurnState]:
        """Own the invocation context, activity registration and telemetry span."""
        from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext

        parent = parent_loop_ctx if parent_loop_ctx is not None else _current_loop_ctx.get()
        room_id = context.room.room.id if context.room else None
        loop_ctx = _ToolLoopContext.for_loop(parent, room_id)
        _current_loop_ctx.set(loop_ctx)
        self._active_loops[loop_ctx.loop_id] = loop_ctx
        try:
            telemetry = self._telemetry_provider
            span_id = telemetry.start_span(
                SpanKind.LLM_GENERATE,
                "llm.generate",
                parent_id=get_current_span(),
                room_id=room_id,
                channel_id=self.channel_id,
                attributes={
                    Attr.PROVIDER: type(self._provider).__name__,
                    Attr.LLM_STREAMING: True,
                },
            )
            turn = _StreamTurnState(loop_ctx, telemetry, span_id, room_id)
            failed = False
            try:
                yield turn
            except Exception as exc:
                failed = True
                telemetry.end_span(span_id, status="error", error_message=str(exc))
                raise
            finally:
                if not failed:
                    await self._finish_streaming_tool_turn(turn)
        finally:
            # Finalization may itself be cancelled while publishing a hook.
            self._active_loops.pop(loop_ctx.loop_id, None)
            _current_loop_ctx.set(None)

    async def _finish_streaming_tool_turn(self, turn: _StreamTurnState) -> None:
        """Report the delivered transcript and the counters accumulated by this turn."""
        attributes: dict[str, Any] = {Attr.LLM_TOOL_COUNT: turn.tool_calls_count}
        if turn.usage.get("input_tokens") or turn.usage.get("output_tokens"):
            attributes[Attr.LLM_INPUT_TOKENS] = turn.usage.get("input_tokens", 0)
            attributes[Attr.LLM_OUTPUT_TOKENS] = turn.usage.get("output_tokens", 0)
        turn.telemetry.end_span(turn.span_id, attributes=attributes)
        if self._after_response_hook:
            try:
                segments, transcript = response_transcript("".join(text) for text in turn.segments)
                await self._after_response_hook(
                    AIResponseEvent(
                        channel_id=self.channel_id,
                        response_content=transcript,
                        segments=segments,
                        room_id=turn.room_id,
                        tool_calls_count=turn.tool_calls_count,
                        round_count=turn.tool_rounds_count,
                        loop_end_reason=turn.reason,
                        usage={"input_tokens": 0, "output_tokens": 0, **turn.usage},
                        latency_ms=int((time.monotonic() - turn.started_at) * 1000),
                        streaming=True,
                    )
                )
            except Exception:
                logger.debug("After-response hook failed (streaming)", exc_info=True)

    async def _stream_local_tool_round(
        self,
        context: AIContext,
        state: _StreamRoundState,
        turn: _StreamTurnState,
        index: int,
    ) -> AsyncGenerator[StreamDelta, None]:
        """Persist an assistant's calls and surround execution with lifecycle markers."""
        calls = self._cap_round_tool_calls(state.tool_calls, "Streaming tool loop")
        logger.info("Streaming tool round %d: %d call(s)", index + 1, len(calls))
        if state.text:
            turn.dedup_prefix = state.text
        context.messages.append(
            AIMessage(
                role="assistant",
                content=self._build_assistant_parts(
                    state.thinking, state.thinking_signature, state.text, calls
                ),
            )
        )
        for call in calls:
            yield ToolCallStartMarker(
                tool_name=call.name, tool_id=call.id, arguments=call.arguments
            )
        results, duration_ms, executed_arguments = await self._execute_round_tools(
            context, calls, turn.telemetry, turn.room_id, index, parent_span_id=turn.span_id
        )
        turn.tool_calls_count += len(calls)
        turn.tool_rounds_count += 1
        for call, result in zip(calls, results, strict=False):
            value = result.result
            is_error = isinstance(value, str) and value.startswith("Error executing tool")
            yield ToolCallEndMarker(
                tool_name=call.name,
                tool_id=call.id,
                arguments=executed_arguments.get(call.id, call.arguments),
                result=value,
                status="failed" if is_error else "completed",
                duration_ms=duration_ms,
                error=value if is_error else None,
                structured_content=result.structured_content,
            )
        if turn.room_id:
            await self._publish_tool_event(
                EphemeralEventType.TOOL_CALL_END,
                turn.room_id,
                results,
                index,
                duration_ms=duration_ms,
            )

    async def _run_streaming_tool_loop(
        self, context: AIContext, *, parent_loop_ctx: _ToolLoopContext | None = None
    ) -> AsyncIterator[StreamDelta]:
        """Orchestrate generation, termination decisions and local tool rounds."""
        async with self._streaming_tool_turn(context, parent_loop_ctx) as turn:
            loop_ctx = turn.loop_ctx
            external = _ExternalStreamTools(
                channel_id=self.channel_id,
                room_id=turn.room_id,
                publish=self._publish_tool_event,
                handler=self._external_tool_handler,
                before=self._before_tool_call_hook,
                after=self._tool_call_hook,
            )
            context, cancelled = self._drain_steering_queue(context, loop_ctx)
            if cancelled:
                turn.reason = "cancelled"
                yield LoopEndMarker(reason=turn.reason, rounds=0)
                return
            rules = self._new_loop_state("Streaming tool loop")

            for index in range(self._max_tool_rounds + 1):
                if loop_ctx.cancel_event.is_set():
                    turn.reason = "cancelled"
                    yield LoopEndMarker(reason=turn.reason, rounds=index)
                    return
                context = self._prepare_round_context(context, loop_ctx, rules, index)
                round_ = _StreamRound(
                    index=index,
                    room_id=turn.room_id,
                    cancel_event=loop_ctx.cancel_event,
                    thinking_coalescer=self._new_thinking_coalescer(turn.room_id, index),
                    new_composition=partial(self._new_tool_call_coalescer, turn.room_id, index),
                    publish_thinking=self._publish_thinking_event,
                    close_thinking=self._close_thinking_window,
                    record_usage=partial(self._record_stream_usage, turn.usage),
                    prefix=turn.dedup_prefix,
                    external_tools=external if self._tool_handler is None else None,
                )
                turn.segments.append(round_.state.reported)
                async with aclosing(
                    round_.stream(self._generate_stream_with_retry(context))
                ) as deltas:
                    async for delta in deltas:
                        yield delta
                state = round_.state

                if state.cancelled or loop_ctx.force_stop:
                    turn.reason = "cancelled" if state.cancelled else "force_stopped"
                    yield LoopEndMarker(reason=turn.reason, rounds=index)
                    return
                if not state.tool_calls:
                    if self._try_empty_retry(
                        context,
                        loop_ctx,
                        rules,
                        had_tool_round=turn.saw_tool_call,
                        final_text=state.text,
                        finish_reason=state.finish_reason,
                    ):
                        continue
                    turn.reason = final_round_reason(
                        had_tool_round=turn.saw_tool_call,
                        final_text=state.text,
                        finish_reason=state.finish_reason,
                        deadline_exceeded=rules.deadline_exceeded(),
                        force_stopped=loop_ctx.force_stop,
                    )
                    yield LoopEndMarker(reason=turn.reason, rounds=index)
                    return

                turn.saw_tool_call = True
                if self._tool_handler is None:
                    await external.observe_calls(state.tool_calls)
                    turn.reason = "completed"
                    yield LoopEndMarker(reason=turn.reason, rounds=index)
                    return
                if index >= self._max_tool_rounds:
                    logger.warning(
                        "Streaming tool loop reached max_tool_rounds=%d", self._max_tool_rounds
                    )
                    turn.reason = "max_rounds"
                    yield LoopEndMarker(reason=turn.reason, rounds=index)
                    return
                if rules.deadline_exceeded():
                    logger.warning(
                        "Streaming tool loop timeout after %d rounds (%.0fs)",
                        index,
                        self._tool_loop_timeout_seconds,
                    )
                    turn.reason = "timeout"
                    yield LoopEndMarker(reason=turn.reason, rounds=index)
                    return

                rules.warn_if_needed(index)
                async with aclosing(
                    self._stream_local_tool_round(context, state, turn, index)
                ) as deltas:
                    async for delta in deltas:
                        yield delta
                context, cancelled = self._drain_steering_queue(context, loop_ctx)
                if cancelled:
                    turn.reason = "cancelled"
                    yield LoopEndMarker(reason=turn.reason, rounds=index)
                    return

            # An empty-response retry can consume the final generation slot.
            turn.reason = "max_rounds"
            yield LoopEndMarker(reason=turn.reason, rounds=self._max_tool_rounds)
