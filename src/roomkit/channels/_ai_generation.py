"""AIChannel mixin for non-streaming response generation with tool loop."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Container, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.channels._ai_loop_rules import (
    AIToolLoopRulesMixin,
    final_round_reason,
    interrupts_turn,
    require_schema_answer,
    turn_span_status,
)
from roomkit.channels._served_tools import CollisionLog
from roomkit.channels._tool_event_result import tool_event_payload
from roomkit.channels._turn_notes import turn_input
from roomkit.models.channel import ChannelOutput
from roomkit.models.enums import EventType
from roomkit.models.event import (
    INTERRUPTION_MARKER_KEY,
    EventSource,
    RoomEvent,
    TextContent,
    ToolCallContent,
)
from roomkit.models.streaming import LoopEndReason
from roomkit.models.tool_call import (
    AIGenerationEvent,
    AIResponseEvent,
    DeclaredTool,
    response_transcript,
)
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    ProviderError,
)
from roomkit.realtime.base import EphemeralEventType
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.context import get_current_span
from roomkit.telemetry.noop import NoopTelemetryProvider

if TYPE_CHECKING:
    from roomkit.channels.ai import _ToolLoopContext
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.enums import ChannelType
    from roomkit.providers.ai.base import AIProvider, AITool, AIToolCall, AIToolResultPart


@dataclass
class ToolRound:
    """Record of one tool execution round in the non-streaming tool loop.

    ``executed_arguments`` maps a call id to the arguments its handler ran
    with, once folds and a ``BEFORE_TOOL_USE`` rewrite applied; the calls keep
    what the model asked for.
    """

    text_before: str
    tool_calls: list[AIToolCall]
    results: list[AIToolResultPart]
    duration_ms: int
    executed_arguments: dict[str, dict[str, Any]] = field(default_factory=dict)

    def arguments_ran(self, call: AIToolCall) -> dict[str, Any]:
        """What *call*'s handler ran with; the request when it never ran."""
        return self.executed_arguments.get(call.id, call.arguments)


@dataclass
class ToolLoopResult:
    """Result of the non-streaming tool loop with full round history.

    ``reason`` is this loop's counterpart of the streaming
    :class:`~roomkit.models.streaming.LoopEndMarker`: the loop knows which of
    its rules ended the turn, and without a name a force-stopped or
    round-capped turn is indistinguishable from a completed one — the exact
    lie the marker was introduced to stop. It reaches consumers as the
    ``loop_end_reason`` key on the response MESSAGE event's metadata.
    """

    response: AIResponse
    rounds: list[ToolRound] = field(default_factory=list)
    reason: LoopEndReason = "completed"
    # The provider error that interrupted the turn after a round (reason
    # ``error``): the turn is delivered, and it is an error too (RFC §6.4).
    error: Exception | None = None
    # The union of what every round declared to the provider, for the turn's
    # ``AIResponseEvent`` (see ``AIResponseEvent.declared_tools``).
    declared_tools: list[DeclaredTool] = field(default_factory=list)


logger = logging.getLogger("roomkit.channels.ai")


class _TurnInterruptedError(Exception):
    """A generation failed once a tool round had run: the turn is interrupted."""

    def __init__(self, error: ProviderError) -> None:
        super().__init__(str(error))
        self.error = error


#: The terminal message of a turn the provider interrupted after a round: the
#: rounds already reached the room, so this is all it has not read (RFC §6.4).
INTERRUPTED_MARKER = "[Response interrupted]"


@runtime_checkable
class AIGenerationHost(Protocol):
    """Contract: capabilities a host class must provide for AIGenerationMixin.

    Attributes provided by the host's ``__init__``:
        _provider: AI provider for generation.
        _max_tool_rounds: Maximum tool-loop iterations.
        _tool_loop_timeout_seconds: Optional wall-clock timeout for the loop.
        _tool_loop_warn_after: Log a warning after this many rounds.
        _tool_handler: Tool call handler (or ``None`` if tools disabled).
        _active_loops: Registry of currently running tool loops.
        _after_response_hook: Optional callback fired after response generation.
        channel_id: Unique identifier for this channel.
        provider_name: Human-readable provider name.
        channel_type: Channel type enum value.

    Methods provided by other mixins:
        _build_context: ``AIContextMixin`` — builds AI context from room state.
        _drain_steering_queue: ``AISteeringMixin`` — drains pending directives.
        _generate_with_retry: ``AIResilienceMixin`` — generate with retry/fallback.
        _publish_thinking_event: ``AIEventsMixin`` — publish thinking events.
        _publish_tool_event: ``AIEventsMixin`` — publish tool call events.

    The shared per-round loop rules (force-stop, empty-retry, budget, parts
    assembly, tool execution) come from :class:`AIToolLoopRulesMixin`, the
    mixin's own base — see :class:`AIToolLoopRulesHost` for that contract.
    """

    _provider: AIProvider
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _max_empty_retries: int
    _tool_handler: Any
    _active_loops: dict[str, _ToolLoopContext]
    _after_response_hook: Any
    _before_generation_hook: Any
    channel_id: str
    provider_name: str
    channel_type: ChannelType

    async def _build_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext: ...
    def _drain_steering_queue(
        self, context: AIContext, loop_ctx: _ToolLoopContext
    ) -> tuple[AIContext, bool]: ...
    async def _generate_with_retry(self, context: AIContext) -> AIResponse: ...
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


class AIGenerationMixin(AIToolLoopRulesMixin):
    """Non-streaming AI response generation with tool loop.

    Host contract: :class:`AIGenerationHost`.
    """

    _provider: Any
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _max_empty_retries: int
    _tool_handler: Any
    _active_loops: dict[str, _ToolLoopContext]
    _after_response_hook: Any
    _before_generation_hook: Any
    channel_id: str
    provider_name: str
    channel_type: Any

    # Cross-mixin methods — Any annotations avoid MRO shadowing.
    # _build_context is NOT annotated here: it's a real typed method on
    # AIContextMixin whose return type must be preserved for subclasses
    # (Agent.super()._build_context()). Call sites use type: ignore instead.
    _drain_steering_queue: Any  # see AIGenerationHost
    _served_tool_names: Any  # AIToolsMixin: what the channel and orchestration serve
    _collisions: CollisionLog
    _generate_with_retry: Any  # see AIGenerationHost
    _record_declared_tools: Any  # see AIGenerationHost
    _publish_thinking_event: Any  # see AIGenerationHost
    _publish_tool_event: Any  # see AIGenerationHost

    @property
    def _telemetry_provider(self) -> NoopTelemetryProvider:
        """Access telemetry provider (set by register_channel)."""
        return getattr(self, "_telemetry", None) or NoopTelemetryProvider()

    async def _fire_before_generation_hook(
        self, ai_context: AIContext, event: RoomEvent
    ) -> tuple[AIContext, bool]:
        """Fire BEFORE_AI_GENERATION hook. Returns ``(context, blocked)``."""
        if not self._before_generation_hook:
            return ai_context, False
        gen_event = AIGenerationEvent(
            ai_context=ai_context,
            channel_id=self.channel_id,
            room_id=event.room_id,
            provider_name=self.provider_name,
        )
        # Read before the hook runs: it may edit the list in place.
        declared = {tool.name for tool in ai_context.tools or []}
        sync_result = await self._before_generation_hook(gen_event)
        if not sync_result.allowed:
            logger.info(
                "AI generation blocked by hook (reason=%s, blocked_by=%s)",
                sync_result.reason,
                sync_result.blocked_by,
            )
            return ai_context, True
        # A hook that REPLACES the context (rather than mutating it) brings a
        # record of its own. The turn has one record — the one the events are
        # built from — so the loop context adopts the hook's: a tool handler's
        # writes then land where the reply reads, whichever object the hook
        # returned.
        from roomkit.channels.ai import _current_loop_ctx

        loop_ctx = _current_loop_ctx.get()
        if loop_ctx is not None:
            loop_ctx.response_metadata = gen_event.ai_context.response_metadata
            # The input the hook left is the one a compaction keeps whole.
            loop_ctx.turn_input = turn_input(gen_event.ai_context.messages)
            _adopt_hook_toolset(
                loop_ctx,
                declared,
                gen_event.ai_context.tools,
                served=self._served_tool_names(loop_ctx.room_id),
                collisions=self._collisions,
            )
        return gen_event.ai_context, False

    def _log_provider_error(self, exc: ProviderError) -> None:
        """One log line for a failed turn, its level by what the status says."""
        if exc.status_code == 404:
            logger.error(
                "AI model not found (channel=%s, provider=%s): %s",
                self.channel_id,
                exc.provider,
                exc,
            )
        elif exc.status_code and exc.status_code >= 500:
            logger.error(
                "AI provider server error (channel=%s, provider=%s, status=%s): %s",
                self.channel_id,
                exc.provider,
                exc.status_code,
                exc,
            )
        else:
            # Connect-refused/timeout (status None), rate-limit (429), other
            # 4xx: expected transients — one WARNING line, no traceback (the
            # error is re-raised and surfaced to the caller regardless).
            logger.warning(
                "AI provider error (channel=%s, provider=%s, status=%s): %s",
                self.channel_id,
                exc.provider,
                exc.status_code,
                exc,
            )

    async def _generate_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Generate an AI response, executing tool calls if needed."""
        ai_context = await self._build_context(event, binding, context)  # ty: ignore[unresolved-attribute]
        ai_context, blocked = await self._fire_before_generation_hook(ai_context, event)
        if blocked:
            return ChannelOutput.empty()
        telemetry = self._telemetry_provider
        _t0 = time.monotonic()
        span_id = telemetry.start_span(
            SpanKind.LLM_GENERATE,
            "llm.generate",
            parent_id=get_current_span(),
            room_id=event.room_id,
            channel_id=self.channel_id,
            attributes={
                Attr.PROVIDER: type(self._provider).__name__,
                Attr.LLM_STREAMING: False,
            },
        )
        try:
            loop_result = await self._run_tool_loop(ai_context, parent_span_id=span_id)
            require_schema_answer(ai_context, loop_result.reason)
        except ProviderError as exc:
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            self._log_provider_error(exc)
            # Propagate so the broadcast path fires ON_ERROR — mirrors the
            # streaming path (which raises out of stream consumption). Swallowing
            # into an empty output would leave the turn with no error surfaced.
            raise
        except asyncio.CancelledError:
            # Cancelled from outside: the span ends, never left open (RFC §6.4).
            telemetry.end_span(span_id, status="cancelled")
            raise
        except Exception:
            telemetry.end_span(span_id, status="error", error_message="AI provider failed")
            logger.exception("AI provider failed for channel %s", self.channel_id)
            raise

        response = loop_result.response
        tool_calls_count = sum(len(rnd.tool_calls) for rnd in loop_result.rounds)
        usage = response.usage or {}
        interrupted = loop_result.error
        telemetry.end_span(
            span_id,
            status=turn_span_status(loop_result.reason),
            error_message=None if interrupted is None else str(interrupted),
            attributes={
                Attr.LLM_INPUT_TOKENS: usage.get("input_tokens", 0),
                Attr.LLM_OUTPUT_TOKENS: usage.get("output_tokens", 0),
                Attr.LLM_TOOL_COUNT: tool_calls_count,
            },
        )

        # Build events: interleaved text segments and tool calls
        response_events = self._build_response_events(
            loop_result,
            event.room_id,
            event.chain_depth + 1,
            response.usage,
            response_metadata=ai_context.response_metadata,
            parent_event_id=event.parent_event_id,
        )

        if self._after_response_hook:
            # What each tool round said before its calls, then the answer —
            # the same segments the response events above persist.
            segments, transcript = response_transcript(
                [rnd.text_before for rnd in loop_result.rounds] + [response.content or ""]
            )
            try:
                await self._after_response_hook(
                    AIResponseEvent(
                        channel_id=self.channel_id,
                        response_content=transcript,
                        segments=segments,
                        room_id=event.room_id,
                        tool_calls_count=tool_calls_count,
                        round_count=len(loop_result.rounds),
                        loop_end_reason=loop_result.reason,
                        declared_tools=loop_result.declared_tools,
                        usage=response.usage or {},
                        thinking=response.thinking or "",
                        latency_ms=int((time.monotonic() - _t0) * 1000),
                    )
                )
            except Exception:
                logger.debug("After-response hook failed", exc_info=True)

        return ChannelOutput(
            responded=True,
            response_events=response_events,
            response_metadata=ai_context.response_metadata,
            error=loop_result.error,
        )

    def _build_response_events(
        self,
        loop_result: ToolLoopResult,
        room_id: str,
        chain_depth: int,
        usage: dict[str, Any] | None,
        response_metadata: Mapping[str, Any] | None = None,
        parent_event_id: str | None = None,
    ) -> list[RoomEvent]:
        """Build interleaved text + tool call events from tool loop result.

        When there are no tool rounds, returns a single MESSAGE event
        (backward compatible). When rounds exist, returns text segments
        and tool call events in order, sharing a correlation_id.

        ``response_metadata`` (``AIContext.response_metadata``, the turn's
        live record) is merged into every MESSAGE event's metadata in its
        final state — the events are built once the loop has run, so
        turn-level attribution set by the host at any point of the turn
        travels with the reply from creation, persisted and broadcast
        without any post-hoc rewrite.

        ``parent_event_id`` is the trigger's thread root (already normalised
        by the locked pipeline); every response event inherits it so the
        reply lands in the same thread. ``None`` keeps the reply top-level.
        """
        source = EventSource(
            channel_id=self.channel_id,
            channel_type=self.channel_type,
            provider=self.provider_name,
        )
        response = loop_result.response
        # ``loop_end_reason`` rides every MESSAGE with the usage: a consumer
        # that reads the reply's metadata can tell a completed turn from one
        # the loop cut (force-stop, round cap, timeout) without a stream.
        message_metadata = {
            **(response_metadata or {}),
            "ai_usage": usage,
            "loop_end_reason": loop_result.reason,
        }
        final_metadata = _final_message_metadata(message_metadata, loop_result)

        if not loop_result.rounds:
            # No tool calls — single text event (existing behavior)
            return [
                RoomEvent(
                    room_id=room_id,
                    source=source,
                    content=TextContent(body=response.content),
                    chain_depth=chain_depth,
                    parent_event_id=parent_event_id,
                    metadata=message_metadata,
                )
            ]

        # Build interleaved segment events
        correlation_id = uuid4().hex
        events: list[RoomEvent] = []
        event_fields: dict[str, Any] = {
            "room_id": room_id,
            "source": source,
            "chain_depth": chain_depth,
            "correlation_id": correlation_id,
            "parent_event_id": parent_event_id,
        }

        for rnd in loop_result.rounds:
            # Text segment before this tool round
            if rnd.text_before:
                events.append(
                    RoomEvent(
                        room_id=room_id,
                        source=source,
                        type=EventType.MESSAGE,
                        content=TextContent(body=rnd.text_before),
                        chain_depth=chain_depth,
                        correlation_id=correlation_id,
                        parent_event_id=parent_event_id,
                        metadata=dict(response_metadata or {}),
                    )
                )

            # Tool call start + end events
            for tc, rp in zip(rnd.tool_calls, rnd.results, strict=False):
                events.extend(
                    _tool_call_events(tc, rp, rnd.arguments_ran(tc), rnd.duration_ms, event_fields)
                )

        # Final text segment (the last response after all tool rounds)
        if response.content:
            events.append(
                RoomEvent(
                    room_id=room_id,
                    source=source,
                    type=EventType.MESSAGE,
                    content=TextContent(body=response.content),
                    chain_depth=chain_depth,
                    correlation_id=correlation_id,
                    parent_event_id=parent_event_id,
                    metadata=final_metadata,
                )
            )

        # With no final text (a cancelled turn, an empty answer), the turn's
        # own record rides its last message instead: a consumer reads the end
        # reason and the usage off the reply whichever way the turn ended.
        last_message = next(
            (i for i in reversed(range(len(events))) if events[i].type == EventType.MESSAGE),
            None,
        )
        if not response.content and last_message is not None:
            last = events[last_message]
            events[last_message] = last.model_copy(
                update={"metadata": {**last.metadata, **message_metadata}}
            )

        # Ensure at least one MESSAGE event exists so the response is not
        # silently dropped (some models return tool calls with no text).
        has_message = last_message is not None
        if not has_message:
            events.append(
                RoomEvent(
                    room_id=room_id,
                    source=source,
                    type=EventType.MESSAGE,
                    content=TextContent(body=""),
                    chain_depth=chain_depth,
                    correlation_id=correlation_id,
                    parent_event_id=parent_event_id,
                    metadata=message_metadata,
                )
            )

        return events

    async def _run_tool_loop(
        self, context: AIContext, *, parent_span_id: str | None = None
    ) -> ToolLoopResult:
        """Generate -> execute tools -> re-generate until a text response."""
        from roomkit.channels.ai import (
            _current_loop_ctx,
            _ToolLoopContext,
        )

        room = context.room.room if context.room else None
        # The value to put back when the loop ends: a caller running inside
        # its own tool loop (a handler that runs a child channel's turn) must
        # find its context intact afterwards. Restored by value, not by token:
        # the loop may end in a context other than the one it began in.
        enclosing_ctx = _current_loop_ctx.get()
        loop_ctx = _ToolLoopContext.for_loop(
            enclosing_ctx, room.id if room is not None else None, room=room
        )
        _current_loop_ctx.set(loop_ctx)
        self._active_loops[loop_ctx.loop_id] = loop_ctx
        rounds: list[ToolRound] = []
        # The turn's token totals, summed over EVERY generation round — a
        # multi-round loop reported through its final response alone would
        # under-count by every round but the last (the streaming loop already
        # sums per round; this is the same rule on this path).
        total_usage: dict[str, int] = {}
        state = self._new_loop_state("Tool loop", loop_ctx.turn_budget)

        async def _generate(ctx: AIContext) -> AIResponse:
            # What this round declares, as the provider receives it: after the
            # hook, the re-filter and the loop's own injections.
            self._record_declared_tools(loop_ctx, ctx.tools)
            resp: AIResponse = await self._generate_with_retry(ctx)
            state.count(total_usage, resp.usage or {})
            return resp

        async def _generate_after_round(ctx: AIContext) -> AIResponse:
            # Once a round ran, a provider failure interrupts the turn rather
            # than losing it: the rounds are kept (RFC §6.4). Overflow
            # recovery (compact and replay once) already ran inside
            # ``_generate_with_retry``: whatever reaches here is spent.
            try:
                return await _generate(ctx)
            except ProviderError as exc:
                if not interrupts_turn(exc, after_round=True):
                    raise
                raise _TurnInterruptedError(exc) from exc

        try:
            context, should_cancel = self._drain_steering_queue(context, loop_ctx)
            if should_cancel:
                return ToolLoopResult(
                    response=AIResponse(content="", tool_calls=[]), reason="cancelled"
                )
            # The first round is prepared as every later one, and as the
            # streaming loop's: Tool Search collapses what the hook left.
            context = self._prepare_round_context(context, loop_ctx, state, 0)
            response: AIResponse = await _generate(context)
            telemetry = self._telemetry_provider
            room_id = context.room.room.id if context.room else None
            reason: LoopEndReason = "completed"

            if response.thinking and room_id:
                await self._publish_thinking_event(
                    EphemeralEventType.THINKING_START, room_id, "", 0
                )
                await self._publish_thinking_event(
                    EphemeralEventType.THINKING_END, room_id, response.thinking, 0
                )

            for round_idx in range(self._max_tool_rounds):
                if not response.tool_calls or self._tool_handler is None:
                    # Final answer reached. If it is empty *after* we ran tools,
                    # the model skipped verbalizing the result — re-prompt once
                    # (bounded) for the final answer instead of returning nothing.
                    if self._try_empty_retry(
                        context,
                        loop_ctx,
                        state,
                        had_tool_round=bool(rounds),
                        final_text=response.content or "",
                        finish_reason=response.finish_reason,
                    ):
                        # Before any round ran, a provider failure is the
                        # turn's own error, as on the streaming loop (RFC §6.4).
                        regenerate = _generate_after_round if rounds else _generate
                        response = await regenerate(context)
                        continue
                    reason = final_round_reason(
                        had_tool_round=bool(rounds),
                        final_text=response.content or "",
                        finish_reason=response.finish_reason,
                        limit=state.limit_passed(),
                        force_stopped=loop_ctx.force_stop,
                    )
                    break

                if loop_ctx.cancel_event.is_set():
                    logger.info("Tool loop cancelled before round %d", round_idx)
                    reason = "cancelled"
                    break

                limit = state.limit_reached(round_idx)
                if limit is not None:
                    reason = limit
                    break

                state.warn_if_needed(round_idx)

                round_calls = self._cap_round_tool_calls(response.tool_calls, state.log_label)

                logger.info(
                    "Tool round %d: %d call(s)",
                    round_idx + 1,
                    len(round_calls),
                )

                parts = self._build_assistant_parts(
                    response.thinking or "",
                    response.thinking_signature,
                    response.content or "",
                    round_calls,
                )
                context.messages.append(AIMessage(role="assistant", content=parts))

                result_parts, duration_ms, executed_arguments = await self._execute_round_tools(
                    context,
                    round_calls,
                    telemetry,
                    room_id,
                    round_idx,
                    parent_span_id=parent_span_id,
                )

                if room_id:
                    await self._publish_tool_event(
                        EphemeralEventType.TOOL_CALL_END,
                        room_id,
                        result_parts,
                        round_idx,
                        duration_ms=duration_ms,
                    )

                # Track the round for persistence
                rounds.append(
                    ToolRound(
                        text_before=response.content or "",
                        tool_calls=round_calls,
                        results=result_parts,
                        duration_ms=duration_ms,
                        executed_arguments=executed_arguments,
                    )
                )

                context, should_cancel = self._drain_steering_queue(context, loop_ctx)
                if should_cancel:
                    logger.info("Tool loop cancelled after round %d", round_idx)
                    reason = "cancelled"
                    # The round's text is already its own segment: the
                    # cancelled turn adds no terminal text (RFC §6.4).
                    response = AIResponse(content="", tool_calls=[])
                    break

                # Anti-loop ripcord (force_stop): one final generation, told
                # to answer in plain text, then stop without running its calls.
                context = self._prepare_round_context(context, loop_ctx, state, round_idx)
                if loop_ctx.force_stop:
                    response = await _generate_after_round(context)
                    reason = "force_stopped"
                    break

                response = await _generate_after_round(context)

                if response.thinking and room_id:
                    await self._publish_thinking_event(
                        EphemeralEventType.THINKING_START, room_id, "", round_idx + 1
                    )
                    await self._publish_thinking_event(
                        EphemeralEventType.THINKING_END,
                        room_id,
                        response.thinking,
                        round_idx + 1,
                    )
            else:
                # Round budget exhausted (no break). A response still asking
                # for tools was cut mid-work — its pending calls are dropped
                # from execution AND from the transcript (their assistant
                # message was never appended), so no provider sees an orphan.
                # One that answered on the last round is classified by the
                # same rule as any final round.
                if response.tool_calls:
                    reason = "max_rounds"
                    logger.warning(
                        "Tool loop reached max_tool_rounds=%d with %d pending tool "
                        "call(s) dropped",
                        self._max_tool_rounds,
                        len(response.tool_calls),
                    )
                else:
                    reason = final_round_reason(
                        had_tool_round=bool(rounds),
                        final_text=response.content or "",
                        finish_reason=response.finish_reason,
                        limit=state.limit_passed(),
                        force_stopped=loop_ctx.force_stop,
                    )

            if total_usage:
                response = response.model_copy(update={"usage": dict(total_usage)})
            return ToolLoopResult(
                response=response,
                rounds=rounds,
                reason=reason,
                declared_tools=list(loop_ctx.declared_tools.values()),
            )
        except _TurnInterruptedError as interrupted:
            # The provider's error is its SDK's own string (status codes,
            # request ids, model and organisation names), not for the room:
            # it goes to the log, where an operator can correlate it, as the
            # delivery pipeline does (`providers/http_errors.py`). The room
            # gets the marker alone: each round's text is already its own
            # message (RFC §6.4).
            logger.exception("Tool loop interrupted by a provider error after a round")
            return ToolLoopResult(
                response=AIResponse(
                    content=INTERRUPTED_MARKER, tool_calls=[], usage=dict(total_usage)
                ),
                rounds=rounds,
                reason="error",
                declared_tools=list(loop_ctx.declared_tools.values()),
                error=interrupted.error,
            )
        finally:
            self._active_loops.pop(loop_ctx.loop_id, None)
            _current_loop_ctx.set(enclosing_ctx)


def _final_message_metadata(
    message_metadata: dict[str, Any], loop_result: ToolLoopResult
) -> dict[str, Any]:
    """The metadata of the turn's final message.

    The interruption marker says the turn was cut; it is no answer (RFC
    §6.4), and readers tell it from one by its key.
    """
    if loop_result.error is None:
        return message_metadata
    return {**message_metadata, INTERRUPTION_MARKER_KEY: True}


def _adopt_hook_toolset(
    loop_ctx: _ToolLoopContext,
    declared: set[str],
    left: list[AITool] | None,
    *,
    served: Container[str],
    collisions: CollisionLog,
) -> None:
    """Make what BEFORE_AI_GENERATION left of the toolset it saw the turn's base.

    Every round re-filters from ``all_context_tools``: without this, a tool
    the hook withdrew would come back (and run), and one it added would
    vanish. The hook saw the toolset the policy and skill gating leave, Tool
    Search's catalogue included, so a tool it removes is gone from every
    round, reveal and call; one it never saw (gated by a skill, denied by the
    policy) stays in the base for those filters to decide. A tool it adds is
    pinned for the turn: Tool Search never defers it (RFC §6.4). A name the
    channel or orchestration serves keeps its definition: the hook may
    withdraw it, never redefine it, nor add a tool under it (RFC §21.1).
    """
    if loop_ctx.all_context_tools is None:
        return
    kept = {tool.name: tool for tool in left or []}
    withdrawn = declared - kept.keys()
    original = {tool.name: tool for tool in loop_ctx.all_context_tools}
    for name, tool in kept.items():
        # A tool added under a served name, or a served tool redefined.
        if name in served and original.get(name) != tool:
            collisions.served(name)
    base = [
        t if t.name in served else kept.get(t.name, t)
        for t in loop_ctx.all_context_tools
        if t.name not in withdrawn
    ]
    known = {tool.name for tool in base}
    added = {name for name in kept if name not in known and name not in served}
    base.extend(kept[name] for name in kept if name in added)
    loop_ctx.all_context_tools = base
    loop_ctx.withdrawn_tools = loop_ctx.withdrawn_tools | withdrawn
    loop_ctx.hook_pinned = loop_ctx.hook_pinned | added


def _tool_call_events(
    tc: AIToolCall,
    rp: AIToolResultPart,
    executed: dict[str, Any],
    duration_ms: int,
    event_fields: dict[str, Any],
) -> list[RoomEvent]:
    """The TOOL_CALL_START and TOOL_CALL_END events of one call of a round.

    The start carries what the model asked for, the end what the handler ran
    with, as the streaming loop's markers do.
    """
    result, structured = tool_event_payload(
        getattr(rp, "result", None), getattr(rp, "structured_content", None)
    )
    # Read the outcome, never the body: the tool loop already knows whether
    # this call failed. Matching the prose sentence it writes for a raised
    # handler missed every other failure — a refused call, a multimodal
    # result — and misread a tool whose own output happened to start that way.
    is_error = bool(getattr(rp, "is_error", False))
    start = ToolCallContent(
        tool_name=tc.name, tool_id=tc.id, arguments=tc.arguments, status="pending"
    )
    end = ToolCallContent(
        tool_name=tc.name,
        tool_id=tc.id,
        arguments=executed,
        result=result,
        status="failed" if is_error else "completed",
        duration_ms=duration_ms,
        error=rp.as_text() if is_error else None,
        structured_content=structured,
    )
    return [
        RoomEvent(type=EventType.TOOL_CALL_START, content=start, **event_fields),
        RoomEvent(type=EventType.TOOL_CALL_END, content=end, **event_fields),
    ]
