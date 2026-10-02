"""AIChannel mixin for the tool loop every turn runs, whatever its provider
streams and whether it carries tools (RFC §6.4)."""

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
    _ToolLoopState,
    final_round_reason,
    interrupts_turn,
    require_schema_answer,
    turn_span_status,
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
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.models.tool_call import AIResponseEvent, response_transcript
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    ProviderError,
)
from roomkit.providers.utils import _aclose_stream
from roomkit.realtime.base import EphemeralEventType
from roomkit.telemetry.base import Attr, SpanKind, TelemetryProvider
from roomkit.telemetry.context import get_current_span
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext

if TYPE_CHECKING:
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
    # Each round's reasoning, for the turn's ON_AI_RESPONSE.
    thinking: list[str] = field(default_factory=list)
    # The provider error that interrupted the turn after a round (reason
    # ``error``): the turn reaches its end on it, then it is raised (RFC §6.4).
    error: Exception | None = None

    def end(self, reason: LoopEndReason, rounds: int) -> LoopEndMarker:
        """End the loop on *reason*: the marker the consumer reads it from."""
        self.reason = reason
        return LoopEndMarker(reason=reason, rounds=rounds, usage=dict(self.usage))


def _turn_span_attributes(turn: _StreamTurnState) -> dict[str, Any]:
    """What the turn's rounds used, whether or not the turn reached its end."""
    attributes: dict[str, Any] = {Attr.LLM_TOOL_COUNT: turn.tool_calls_count}
    if turn.usage.get("input_tokens") or turn.usage.get("output_tokens"):
        attributes[Attr.LLM_INPUT_TOKENS] = turn.usage.get("input_tokens", 0)
        attributes[Attr.LLM_OUTPUT_TOKENS] = turn.usage.get("output_tokens", 0)
    return attributes


def _end_unfinished_turn(turn: _StreamTurnState, exc: BaseException) -> None:
    """End the span of a turn whose loop did not reach its end (RFC §6.4).

    A raise is an error. A close at a yield (a barge-in, a transport that
    stopped reading, a consumer that refused the answer) or a cancelled task
    is a cancellation. Such a turn reports nothing, so its span is where what
    its rounds used stays on record.
    """
    attributes = _turn_span_attributes(turn)
    if isinstance(exc, Exception):
        turn.telemetry.end_span(
            turn.span_id, status="error", error_message=str(exc), attributes=attributes
        )
    else:
        turn.telemetry.end_span(turn.span_id, status="cancelled", attributes=attributes)


def _round_stop_reason(
    state: _StreamRoundState, loop_ctx: _ToolLoopContext
) -> LoopEndReason | None:
    """Why the loop stops after a round, if it does; a stop that came after the
    model's last event counts as one that came during it."""
    if state.cancelled or loop_ctx.cancel_event.is_set():
        return "cancelled"
    return "force_stopped" if loop_ctx.force_stop else None


async def _unrun_call_ends(calls: list[Any]) -> AsyncGenerator[StreamDelta, None]:
    """A failed end for each announced call a stop kept from running (RFC §21.3)."""
    for call in calls:
        yield ToolCallEndMarker(
            tool_name=call.name,
            tool_id=call.id,
            arguments=call.arguments,
            status="failed",
            error="cancelled",
        )


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
    _after_response_hook: Any
    _before_generation_hook: Any
    _before_tool_call_hook: Any
    _tool_report_hook: Any
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
    def _record_declared_tools(
        self, loop_ctx: _ToolLoopContext, tools: list[Any] | None
    ) -> None: ...
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


async def _answered_or_raise(
    context: AIContext, deltas: AsyncGenerator[StreamDelta, None]
) -> AsyncIterator[StreamDelta]:
    """The streaming tool loop, failing a constrained turn that ends without
    its answer (see :func:`require_schema_answer`)."""
    try:
        async for delta in deltas:
            if isinstance(delta, LoopEndMarker):
                try:
                    require_schema_answer(context, delta.reason)
                except Exception as refused:
                    # Raised inside the loop, so its turn ends as the error it is.
                    await deltas.athrow(refused)
                    raise
            yield delta
    finally:
        await _aclose_stream(deltas)


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
    _after_response_hook: Any
    _before_generation_hook: Any
    _before_tool_call_hook: Any
    _tool_report_hook: Any
    _external_tool_handler: Any
    channel_id: str

    # Cross-mixin methods — Any annotations avoid MRO shadowing.
    # _build_context is NOT annotated here: it's a real typed method on
    # AIContextMixin whose return type must be preserved for subclasses
    # (Agent.super()._build_context()). Call sites use type: ignore instead.
    _drain_steering_queue: Any  # see AIStreamingHost
    _generate_stream_with_retry: Any  # see AIStreamingHost
    _record_declared_tools: Any  # see AIStreamingHost
    _publish_thinking_event: Any  # see AIStreamingHost
    _publish_tool_event: Any  # see AIStreamingHost
    _telemetry_provider: Any  # see AIStreamingHost
    _log_provider_error: Any  # AIGenerationMixin: one log line for a failed turn

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

    async def _start_streaming_tool_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response that handles tool calls between rounds."""
        ai_context = await self._build_context(event, binding, context)  # ty: ignore[unresolved-attribute]
        ai_context, blocked = await self._fire_before_generation_hook(ai_context, event)  # ty: ignore[unresolved-attribute]
        if blocked:
            return ChannelOutput.empty()
        # The generator below executes when the CONSUMER iterates the
        # stream — by then handle_event has reset the loop contextvar, so
        # the parent ctx (participant role, room, the toolset stamped by
        # _build_context) must be captured NOW and passed explicitly. So is
        # the span the turn answers under (the broadcast's), which the
        # consumer no longer runs in.
        return ChannelOutput(
            responded=True,
            response_stream=_answered_or_raise(
                ai_context,
                self._run_streaming_tool_loop(
                    ai_context,
                    parent_loop_ctx=_current_loop_ctx.get(),
                    parent_span_id=get_current_span(),
                ),
            ),
            response_metadata=ai_context.response_metadata,
        )

    def _record_stream_usage(
        self, total: dict[str, int], rules: _ToolLoopState, usage: dict[str, Any]
    ) -> None:
        """Record a generation's usage: into the turn's total, against its
        budget, and as the input/output metrics."""
        rules.count(total, usage)
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
        self,
        context: AIContext,
        parent_loop_ctx: _ToolLoopContext | None,
        parent_span_id: str | None = None,
    ) -> AsyncIterator[_StreamTurnState]:
        """Own the invocation context, activity registration and telemetry span."""
        # This body runs in the CONSUMER's context, which may hold a loop
        # context of its own (a handler draining a child channel's stream):
        # that is the value to put back when the turn ends, by value rather
        # than by token, since the turn may end in yet another context.
        enclosing_ctx = _current_loop_ctx.get()
        parent = parent_loop_ctx if parent_loop_ctx is not None else enclosing_ctx
        room = context.room.room if context.room else None
        room_id = room.id if room is not None else None
        loop_ctx = _ToolLoopContext.for_loop(parent, room_id, room=room)
        _current_loop_ctx.set(loop_ctx)
        self._active_loops[loop_ctx.loop_id] = loop_ctx
        try:
            telemetry = self._telemetry_provider
            span_id = telemetry.start_span(
                SpanKind.LLM_GENERATE,
                "llm.generate",
                parent_id=parent_span_id or get_current_span(),
                room_id=room_id,
                channel_id=self.channel_id,
                attributes={
                    Attr.PROVIDER: type(self._provider).__name__,
                    Attr.LLM_STREAMING: True,
                },
            )
            turn = _StreamTurnState(loop_ctx, telemetry, span_id, room_id)
            try:
                yield turn
            except BaseException as exc:
                await self._end_raised_turn(turn, exc)
                raise
            await self._finish_streaming_tool_turn(turn)
        finally:
            # Finalization may itself be cancelled while publishing a hook.
            self._active_loops.pop(loop_ctx.loop_id, None)
            _current_loop_ctx.set(enclosing_ctx)

    async def _end_raised_turn(self, turn: _StreamTurnState, exc: BaseException) -> None:
        """End a turn left by an exception: reported when the provider interrupted
        it after a round, since it reached its end on that error (RFC §6.4)."""
        if exc is turn.error:
            await self._finish_streaming_tool_turn(turn)
        else:
            _end_unfinished_turn(turn, exc)

    async def _finish_streaming_tool_turn(self, turn: _StreamTurnState) -> None:
        """Report the delivered transcript and the counters accumulated by this turn."""
        turn.telemetry.end_span(
            turn.span_id,
            status=turn_span_status(turn.reason),
            error_message=None if turn.error is None else str(turn.error),
            attributes=_turn_span_attributes(turn),
        )
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
                        declared_tools=list(turn.loop_ctx.declared_tools.values()),
                        thinking="\n\n".join(turn.thinking),
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
        # A stop that came while the calls were announced: none of them runs,
        # and the loop ends cancelled at its next check (RFC §21.3).
        ends = (
            _unrun_call_ends(calls)
            if turn.loop_ctx.cancel_event.is_set()
            else self._run_announced_calls(context, calls, turn, index)
        )
        async with aclosing(ends) as deltas:
            async for delta in deltas:
                yield delta

    async def _run_announced_calls(
        self, context: AIContext, calls: list[Any], turn: _StreamTurnState, index: int
    ) -> AsyncGenerator[StreamDelta, None]:
        """Execute a round's announced calls and yield one end marker per call."""
        results, duration_ms, executed_arguments = await self._execute_round_tools(
            context, calls, turn.telemetry, turn.room_id, index, parent_span_id=turn.span_id
        )
        turn.tool_calls_count += len(calls)
        turn.tool_rounds_count += 1
        for call, result in zip(calls, results, strict=False):
            value = result.result
            is_error = result.is_error
            yield ToolCallEndMarker(
                tool_name=call.name,
                tool_id=call.id,
                arguments=executed_arguments.get(call.id, call.arguments),
                result=value,
                status="failed" if is_error else "completed",
                duration_ms=duration_ms,
                # ``error`` is text; a failure that answered with content parts
                # flattens the way any text consumer of that result would.
                error=result.as_text() if is_error else None,
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

    async def _stream_generation(
        self, round_: _StreamRound, context: AIContext, turn: _StreamTurnState, index: int
    ) -> AsyncGenerator[StreamDelta, None]:
        """One round's generation; a provider error once a round ran ends the turn.

        The rounds already reached the room, so the turn reaches its end on
        the error, its ON_AI_RESPONSE fired, and the error then reaches the
        consumer (RFC §6.4). A turn that fails before any round is one log
        line here and the error itself.
        """
        try:
            async with aclosing(
                round_.stream(self._generate_stream_with_retry(context))
            ) as deltas:
                async for delta in deltas:
                    yield delta
        except ProviderError as exc:
            if not interrupts_turn(exc, after_round=turn.saw_tool_call):
                self._log_provider_error(exc)
                raise
            logger.exception("Streaming tool loop interrupted by a provider error after a round")
            turn.error = exc
            yield turn.end("error", index)
            raise
        if round_.state.thinking:
            turn.thinking.append(round_.state.thinking)

    async def _run_streaming_tool_loop(
        self,
        context: AIContext,
        *,
        parent_loop_ctx: _ToolLoopContext | None = None,
        parent_span_id: str | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """Orchestrate generation, termination decisions and local tool rounds."""
        async with self._streaming_tool_turn(context, parent_loop_ctx, parent_span_id) as turn:
            loop_ctx = turn.loop_ctx
            external = _ExternalStreamTools(
                channel_id=self.channel_id,
                room_id=turn.room_id,
                publish=self._publish_tool_event,
                handler=self._external_tool_handler,
                before=self._before_tool_call_hook,
                report=self._tool_report_hook,
            )
            context, cancelled = self._drain_steering_queue(context, loop_ctx)
            if cancelled:
                yield turn.end("cancelled", 0)
                return
            rules = self._new_loop_state("Streaming tool loop", loop_ctx.turn_budget)

            for index in range(self._max_tool_rounds + 1):
                if loop_ctx.cancel_event.is_set():
                    yield turn.end("cancelled", index)
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
                    record_usage=partial(self._record_stream_usage, turn.usage, rules),
                    prefix=turn.dedup_prefix,
                    external_tools=external if self._tool_handler is None else None,
                )
                turn.segments.append(round_.state.reported)
                # What this round declares, as the provider receives it.
                self._record_declared_tools(loop_ctx, context.tools)
                async with aclosing(
                    self._stream_generation(round_, context, turn, index)
                ) as deltas:
                    async for delta in deltas:
                        yield delta
                state = round_.state

                stop = _round_stop_reason(state, loop_ctx)
                if stop is not None:
                    yield turn.end(stop, index)
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
                    reason = final_round_reason(
                        had_tool_round=turn.saw_tool_call,
                        final_text=state.text,
                        finish_reason=state.finish_reason,
                        limit=rules.limit_passed(),
                        force_stopped=loop_ctx.force_stop,
                    )
                    yield turn.end(reason, index)
                    return

                turn.saw_tool_call = True
                if self._tool_handler is None:
                    await external.observe_calls(state.tool_calls)
                    yield turn.end("completed", index)
                    return
                if index >= self._max_tool_rounds:
                    logger.warning(
                        "Streaming tool loop reached max_tool_rounds=%d", self._max_tool_rounds
                    )
                    yield turn.end("max_rounds", index)
                    return
                limit = rules.limit_reached(index)
                if limit is not None:
                    yield turn.end(limit, index)
                    return

                rules.warn_if_needed(index)
                async with aclosing(
                    self._stream_local_tool_round(context, state, turn, index)
                ) as deltas:
                    async for delta in deltas:
                        yield delta
                context, cancelled = self._drain_steering_queue(context, loop_ctx)
                if cancelled:
                    yield turn.end("cancelled", index)
                    return

            # An empty-response retry can consume the final generation slot.
            yield turn.end("max_rounds", self._max_tool_rounds)
