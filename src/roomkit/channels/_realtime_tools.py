"""Tool call handling for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import threading
import time
from collections.abc import Container, Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._ai_policy import policy_admits, policy_refusal
from roomkit.channels._realtime_context import (
    _current_voice_session,
    own_call_orphaned,
    serving_call,
    spare_own_orphaned_call,
)
from roomkit.channels._realtime_tool_calls import RealtimeToolCall, ToolCallBook
from roomkit.channels._realtime_tool_executor import (
    ToolCallDoor,
    judge_tool_call,
    report_failed_call,
    run_tool_call,
)
from roomkit.channels._served_tools import CollisionLog, declared_once, dict_tool_name
from roomkit.channels._skill_constants import TOOL_ACTIVATE_SKILL
from roomkit.channels._tool_registry import ChannelRegistry, ToolSource, tool_dict
from roomkit.channels._tool_search_constants import TOOL_CALL_TOOL, TOOL_LIST_TOOLS
from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import (
    ToolCallEvent,
)
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.context import reset_span, set_current_span
from roomkit.tools._outcome import OutcomeKind, ToolOutcome
from roomkit.tools.context import ToolCallContext, _current_tool_call
from roomkit.tools.result import (
    GateRefusal,
    bounded_result,
    declined_answer,
    pre_execution_denial,
    result_text,
)
from roomkit.tools.timeout import ToolTimeouts, answer_within
from roomkit.tools.validation import fold_hoisted_arguments, validate_tool_arguments
from roomkit.voice.base import VoiceSessionState

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.tools.policy import ToolPolicy
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")

_LOOP_SEGMENT_BUDGET_S = 0.050
"""Sync work on the event loop past this delays realtime pacing.

The SIP pacer's jitter headroom is 60ms — one fused stretch beyond it is
an audible drop-out on a concurrent call.  Tool-call segments are timed
individually so the culprit is named in the logs without an asyncio
set_debug hunt."""


@runtime_checkable
class RealtimeToolsHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeToolsMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_rooms: Maps session IDs to room IDs.
        _session_spans: Active telemetry session span per session.
        _turn_spans: Active telemetry turn span per session.
        _session_tools: Per-session tool definitions.
        _tool_handler: User-provided tool handler callback.
        _tools: Default tool definitions.
        _mute_on_tool_call: Whether to mute mic during tool execution.
        _tool_result_max_length: Max characters for tool result.
        _skill_support: Skill infrastructure support.
        _tool_policy: The channel's tool policy, or None.
        _session_roles: The participant role each session's policy resolves for.
        _provider: The realtime voice provider.
        _transport: The voice backend transport.
        _framework: The RoomKit framework instance (or None).
        channel_id: Channel identifier.
        _telemetry_provider: Telemetry provider for spans.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, Any]
    _turn_spans: dict[str, Any]
    _session_tools: dict[str, Any]
    _session_config_locks: dict[str, asyncio.Lock]
    _tool_handler: Any
    _tools: Any
    _system_prompt: str | None
    _mute_on_tool_call: bool
    _tool_result_max_length: int
    _skill_support: Any
    _tool_policy: ToolPolicy | None
    _session_roles: dict[str, str | None]
    _collisions: CollisionLog
    _registry: ChannelRegistry
    _tool_timeouts: ToolTimeouts
    _tool_search_support: Any
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    _transcription_order_locks: dict[str, asyncio.Lock]
    _awaiting_tool_response: set[str]
    _tool_calls: ToolCallBook
    _scheduled_tasks: set[asyncio.Task[Any]]
    channel_id: str
    _telemetry_provider: Any

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    def _update_idle_event(self, session_id: str) -> None: ...

    def _expect_provider_output(self, session_id: str) -> None: ...


def _timed_result_text(name: str, raw: Any) -> str:
    """*raw* as the text a voice provider reads, warning when flattening it
    held the event loop past the realtime segment budget."""
    started = time.perf_counter()
    text = result_text(raw)
    held = time.perf_counter() - started
    if held > _LOOP_SEGMENT_BUDGET_S:
        # Pure sync CPU (wall == loop hold), and it runs on the FULL result
        # before truncation caps it.
        logger.warning(
            "Tool %s result serialization held the event loop for "
            "%.0fms (%d chars, budget ~%.0fms) — concurrent "
            "realtime audio may underrun; return a string or a "
            "compact reference instead of a large object",
            name,
            held * 1000,
            len(text),
            _LOOP_SEGMENT_BUDGET_S * 1000,
        )
    return text


class _ProviderDoor:
    """The provider's function call: its outcome goes back as the call's result."""

    channel_serves = True

    def __init__(self, channel: RealtimeToolsMixin) -> None:
        self._channel = channel

    async def deliver(self, call: RealtimeToolCall, outcome: ToolOutcome) -> bool:
        return await self._channel._deliver_tool_result(call, result_text(outcome.result))


class _ToolCallSpan:
    """The telemetry span of one realtime tool call, closed by its outcome."""

    def __init__(self, telemetry: Any, span_id: Any, name: str) -> None:
        self._telemetry = telemetry
        self._span_id = span_id
        self._name = name
        self._open = True

    def close(self, outcome: ToolOutcome) -> None:
        """End the span as *outcome* ended the call."""
        if outcome.kind is OutcomeKind.REFUSED:
            self.end(attributes={Attr.REALTIME_TOOL_DENIED: True})
        elif outcome.kind is OutcomeKind.CANCELLED:
            self.end(status="cancelled")
        elif outcome.kind is OutcomeKind.FAILED:
            self.end(status="error", error_message=f"tool {self._name} failed")
        else:
            self.end()

    def end(self, **kwargs: Any) -> None:
        if self._open:
            self._open = False
            self._telemetry.end_span(self._span_id, **kwargs)


class RealtimeToolsMixin:
    """Tool call execution for RealtimeVoiceChannel.

    Host contract: :class:`RealtimeToolsHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, Any]
    _turn_spans: dict[str, Any]
    _session_tools: dict[str, Any]
    _session_config_locks: dict[str, asyncio.Lock]
    _tool_handler: Any
    _tools: Any
    _system_prompt: str | None
    _mute_on_tool_call: bool
    _tool_result_max_length: int
    _skill_support: Any
    _tool_policy: ToolPolicy | None
    _session_roles: dict[str, str | None]
    _collisions: CollisionLog
    _registry: ChannelRegistry
    _tool_timeouts: ToolTimeouts
    _tool_search_support: Any
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    _transcription_order_locks: dict[str, asyncio.Lock]
    _awaiting_tool_response: set[str]
    _tool_calls: ToolCallBook
    _scheduled_tasks: set[asyncio.Task[Any]]
    channel_id: str
    _telemetry_provider: Any

    _track_task: Any  # see RealtimeToolsHost — cross-mixin
    _session_answer_depth: Any  # RealtimeTranscriptionMixin — cross-mixin
    _expect_provider_output: Any
    _update_idle_event: Any
    _compose_session_prompt: Any
    _compose_session_tools: Any

    def _on_provider_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Handle tool call from provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if session.state == VoiceSessionState.ENDED:
            return
        call = RealtimeToolCall(session, call_id, name, arguments, mutes=self._mute_on_tool_call)
        if not self._open_tool_call(call):
            self._track_task(
                loop,
                self._refuse_duplicate_call(call),
                name=f"rt_tool_duplicate:{session.id}:{call_id}",
            )
            return
        call.task = self._track_task(
            loop,
            self._handle_tool_call(call),
            name=f"rt_tool_call:{session.id}:{call_id}",
        )
        call.task.add_done_callback(lambda _: self._close_tool_call(call))

    def _on_provider_tool_call_cancelled(self, session: VoiceSession, call_ids: list[str]) -> Any:
        """Provider callback: the model abandoned outstanding calls (RFC §12.4).

        The handler still running for one of them is working for a result
        nobody will read. Its task is cancelled and the call is reported to
        ON_TOOL_CALL's observers as cancelled. A call no longer in the books
        (its result left before the cancellation arrived) has no event to
        build: the provider dropped the stale result and logged it. A call
        whose result went out, or whose outcome the observers already
        received, is left to finish for the same reason: a second event would
        put two outcomes on one ``tool_call_id``, and the result it is
        submitting is the provider's to drop. A reconnect the call's own
        handler caused (a handoff reconfiguring its session) orphans that call
        too, and it is not abandoned: it runs on, its result kept off the wire
        (RFC §9.3).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if session.state == VoiceSessionState.ENDED:
            return
        for call_id in call_ids:
            call = self._tool_calls.get(session.id, call_id)
            if call is None or not call.abandonable:
                logger.debug(
                    "Cancelled tool call %s is not in flight for session %s", call_id, session.id
                )
                continue
            if self._spared_by_own_reconnect(session, call_id, call.name):
                continue
            if call.delivered or call.reported:
                logger.debug(
                    "Cancelled tool call %s already gave its outcome for session %s",
                    call_id,
                    session.id,
                )
                continue
            if call.task is not None:
                call.task.cancel()
            logger.info(
                "Tool call %s(%s) cancelled by the model for session %s",
                call.name,
                call_id,
                session.id,
            )
            self._track_task(
                loop,
                self._report_cancelled_tool_call(call),
                name=f"rt_tool_cancelled:{session.id}:{call_id}",
            )

    @staticmethod
    def _spared_by_own_reconnect(session: VoiceSession, call_id: str, name: str) -> bool:
        """Whether the call's own handler caused the reconnect that orphaned it.

        Such a call is not abandoned: its handler runs on, its result stays
        off the wire, and its outcome is reported as usual (RFC §9.3).
        """
        if not spare_own_orphaned_call(session.id, call_id):
            return False
        logger.info(
            "Tool call %s(%s) lost its id to the reconnect its own handler caused; "
            "the handler runs on and its result stays off the wire (session %s)",
            name,
            call_id,
            session.id,
        )
        return True

    async def _report_cancelled_tool_call(
        self,
        call: RealtimeToolCall,
        hint: str = "The model abandoned this call before its result; nothing was sent.",
    ) -> None:
        """Report an abandoned call to ON_TOOL_CALL's observers (RFC §9.3)."""
        body = json.dumps({"error": "Tool call cancelled", "tool": call.name, "hint": hint})
        await report_failed_call(self, call, ToolOutcome(OutcomeKind.CANCELLED, body))

    async def _report_ended_calls(self, calls: list[RealtimeToolCall]) -> None:
        """Report the calls the session's end interrupted, each once, as
        cancelled; a call whose result went out was not interrupted (RFC §12.4)."""
        for call in calls:
            if not call.delivered:
                await self._report_cancelled_tool_call(
                    call, hint="The session ended before its result; nothing was sent."
                )

    async def _refuse_duplicate_call(self, call: RealtimeToolCall) -> None:
        """Report a call whose id names a call still in flight, sending nothing.

        The id's one result is the first call's, which runs on (RFC §12.4).
        """
        logger.warning(
            "Tool call %s(%s) arrived while a call with its id is in flight (session %s); "
            "refused, the first one runs on",
            call.name,
            call.call_id,
            call.session.id,
        )
        body = json.dumps({"error": f"Tool call '{call.call_id}' is already running"})
        await report_failed_call(self, call, ToolOutcome(OutcomeKind.REFUSED, body))

    def _session_room(self, session: VoiceSession) -> str | None:
        with self._state_lock:
            return self._session_rooms.get(session.id)

    def _open_tool_call(self, call: RealtimeToolCall) -> bool:
        """Put *call* on the books; False when its id names a call in flight.

        A call that holds the input muted mutes it, and the input stays muted
        until the last such call in flight ends (RFC §12.4). The call holds
        idle while it runs, and a result it sends holds it until the
        continuation (``_expect_provider_output``).
        """
        session_id = call.session.id
        if call.room_id is None:
            call.room_id = self._session_room(call.session)
        muted_before = self._tool_calls.muting(session_id)
        if not self._tool_calls.open(call):
            return False
        if call.mutes and not muted_before and self._transport is not None:
            self._transport.set_input_muted(call.session, True)
        self._update_idle_event(session_id)
        return True

    def _close_tool_call(self, call: RealtimeToolCall) -> None:
        """Take *call* off the books, releasing the input once no call holds it."""
        session_id = call.session.id
        self._tool_calls.close(call)
        released = call.mutes and not self._tool_calls.muting(session_id)
        if released and self._transport is not None:
            self._transport.set_input_muted(call.session, False)
        self._update_idle_event(session_id)

    async def _handle_tool_call(self, call: RealtimeToolCall) -> None:
        if call.session.state == VoiceSessionState.ENDED:
            return
        with serving_call(call.session.id, call.call_id):
            await self._execute_tool_call(call)

    async def _execute_tool_call(self, call: RealtimeToolCall) -> None:
        """Serve a provider's function call and submit its outcome (RFC §12.4)."""
        session = call.session
        if session.state == VoiceSessionState.ENDED:
            return
        await self._after_earlier_transcriptions(session)
        if session.state == VoiceSessionState.ENDED:
            return
        call.room_id = self._session_room(session)
        with self._tool_call_span(call, SpanKind.REALTIME_TOOL_CALL, "realtime_tool") as span:
            outcome = await run_tool_call(self, call, _ProviderDoor(self))
            span.close(outcome)
        logger.info(
            "Tool call %s(%s) %s for session %s", call.name, call.call_id, outcome.kind, session.id
        )

    async def _after_earlier_transcriptions(self, session: VoiceSession) -> None:
        """Wait for the transcriptions the provider emitted before this call.

        A tool call must not overtake them: the user final that closes the
        current utterance travels the serialised transcription queue, while
        tool calls run in their own task — unbarriered, the tool reaches the
        application first and the late final reads as new user speech. The
        call passes through the same FIFO lock, then releases it: tool
        execution itself must not hold transcriptions back.
        """
        with self._state_lock:
            order_lock = self._transcription_order_locks.setdefault(session.id, asyncio.Lock())
        async with order_lock:
            pass

    @contextlib.contextmanager
    def _tool_call_span(
        self, call: RealtimeToolCall, kind: SpanKind, prefix: str, **attributes: Any
    ) -> Iterator[_ToolCallSpan]:
        """The call's telemetry span, under the session's, the session's span
        current while the call runs; a call cut short ends it cancelled."""
        session_id = call.session.id
        with self._state_lock:
            session_span = self._session_spans.get(session_id)
            parent = self._turn_spans.get(session_id) or session_span
        token = set_current_span(session_span) if session_span else None
        telemetry = self._telemetry_provider
        span_id = telemetry.start_span(
            kind,
            f"{prefix}:{call.name}",
            parent_id=parent,
            attributes={
                Attr.REALTIME_TOOL_NAME: call.name,
                "tool_call_id": call.call_id,
                **attributes,
            },
            room_id=call.room_id,
            session_id=session_id,
            channel_id=self.channel_id,
        )
        span = _ToolCallSpan(telemetry, span_id, call.name)
        try:
            yield span
        except asyncio.CancelledError:
            span.end(status="cancelled")
            raise
        finally:
            span.end()
            if token is not None:
                reset_span(token)

    def _unwrap_call_tool(self, call: RealtimeToolCall) -> str | None:
        """Unwrap the tool a fixed-declaration ``call_tool`` carries into *call*,
        so its books and its reports name that tool; why the transport is
        unreadable, if it is."""
        support = self._tool_search_support
        session = call.session
        if not (
            call.name == TOOL_CALL_TOOL
            and support
            and support.uses_call_tool
            and support.active(session.id)
        ):
            return None
        call.name, call.arguments, transport_error = support.unwrap_call(
            call.arguments, session.id
        )
        return transport_error

    # -- ToolCallHost: the steps the executor serves a call with -------------

    def _tool_framework(self, call: RealtimeToolCall) -> RoomKit | None:
        return self._framework if call.room_id else None

    def _tool_event(self, call: RealtimeToolCall, result: str | None) -> ToolCallEvent:
        """The ON_TOOL_CALL event of one call on this session."""
        return ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.REALTIME_VOICE,
            tool_call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
            result=result,
            room_id=call.room_id,
            session=call.session,
        )

    async def _authorize_call(
        self, call: RealtimeToolCall, door: ToolCallDoor
    ) -> tuple[GateRefusal | None, RoomContext | None]:
        """The pre-execution gate (RFC §12.4), after a fixed-declaration
        ``call_tool`` is unwrapped into the tool it carries."""
        if door.channel_serves:
            transport_error = self._unwrap_call_tool(call)
            if transport_error is not None:
                return GateRefusal(json.dumps({"error": transport_error})), None
        call.arguments, denial, context = await self._authorize_realtime_tool(
            call.name,
            call.arguments,
            call.call_id,
            call.room_id,
            call.session,
            channel_serves=door.channel_serves,
        )
        return denial, context

    def _call_ended(self, call: RealtimeToolCall) -> bool:
        return call.session.state == VoiceSessionState.ENDED

    async def _serve_channel_tool(
        self, call: RealtimeToolCall, door: ToolCallDoor, carrying: RoomContext | None
    ) -> ToolOutcome | None:
        """Tool Search and skill activation, which reconfigure the session
        around their delivery; ``None`` for any other call."""
        if self._tool_search_support and self._tool_search_support.is_search_tool(call.name):
            return await self._serve_tool_search(call, door)
        if call.name == TOOL_ACTIVATE_SKILL and self._skill_support:
            return await self._serve_skill_activation(call, door, carrying)
        return None

    async def _answer_call(self, call: RealtimeToolCall, carrying: RoomContext | None) -> str:
        """A skill tool's answer, else the handler's, as the text the model reads.

        Raises :class:`~roomkit.core.exceptions.UnservedToolCallError` when
        nothing serves the call, and lets the handler's
        :class:`~roomkit.core.exceptions.ToolRefusedError` through.
        """
        name, session = call.name, call.session
        if self._skill_support and self._skill_support.is_skill_tool(name):
            answer = self._skill_support.handle_tool_call(name, call.arguments, session.id)
            return await answer_within(self._call_timeout(name, call.room_id), name, answer)
        if not self._serves_tool(name, call.room_id or session.room_id):
            raise UnservedToolCallError(name)
        logger.info(
            "Executing tool %s(%s) via handler for session %s", name, call.call_id, session.id
        )
        t_seg = time.perf_counter()
        raw = await self._call_tool_handler(call, carrying)
        logger.debug(
            "tool %s handler segment: %.0fms wall", name, (time.perf_counter() - t_seg) * 1000
        )
        text = _timed_result_text(name, raw)
        # Yield so realtime pacing gets a slot between the handler segment and
        # hook dispatch — sync hooks run inline next and would otherwise fuse
        # with this segment into one loop step.
        await asyncio.sleep(0)
        return text

    def _bound_call_result(self, call: RealtimeToolCall, text: str) -> str:
        """*text* within ``tool_result_max_length`` (RFC §21.5). An activated
        skill's instructions are exempt, and so is the complete schema
        ``list_tools(name=...)`` reads: the model needs it whole to call the
        tool."""
        if call.name == TOOL_ACTIVATE_SKILL:
            return text
        if call.name == TOOL_LIST_TOOLS and call.arguments.get("name"):
            return text
        return bounded_result(text, self._tool_result_max_length, call.name)

    async def _call_tool_handler(
        self, call: RealtimeToolCall, gate_context: RoomContext | None
    ) -> Any:
        """The answer to one call, run inside the tool call context (RFC §21.4),
        whichever door brought the call: the tool orchestration set up for the
        room, else the host's handler. ``current_tool_call()`` names the call,
        its room and this channel, as on every channel."""
        session = call.session
        loop_ctx = await self._realtime_loop_context(session, call.room_id, gate_context)
        call_ctx = ToolCallContext(
            room_id=loop_ctx.room_id or "", tool_call_id=call.call_id, channel_id=self.channel_id
        )
        token = _current_voice_session.set(session)
        loop_token = _current_loop_ctx.set(loop_ctx)
        call_token = _current_tool_call.set(call_ctx)
        try:
            timeout = self._call_timeout(call.name, loop_ctx.room_id)
            answer = self._answer(call.name, call.arguments, loop_ctx.room_id)
            return await answer_within(timeout, call.name, answer)
        finally:
            _current_tool_call.reset(call_token)
            _current_loop_ctx.reset(loop_token)
            _current_voice_session.reset(token)

    def _call_timeout(self, name: str, room_id: str | None) -> float | None:
        """The bound of one call to *name* (RFC §21.6): the channel's, unless
        the tool keeps a bound of its own."""
        return self._registry.bound(name, room_id, self._tool_timeouts)

    async def _answer(self, name: str, arguments: dict[str, Any], room_id: str | None) -> Any:
        """The answer of what orchestration set up for *room_id*, else of the
        host's handler; a handler that declines the call raises
        :class:`~roomkit.core.exceptions.UnservedToolCallError` (RFC §21.4).

        Only the host's answer may be the "not mine" envelope: orchestration's
        tools answer what they ran.
        """
        entry = self._registry.lookup(name, room_id)
        if entry is not None and entry.serve is not None:
            result = entry.serve(arguments)
            return await result if inspect.isawaitable(result) else result
        return declined_answer(await self._tool_handler(name, arguments), name)

    def _serves_tool(self, name: str, room_id: str | None) -> bool:
        """Whether something serves a call to *name* in *room_id*: what
        orchestration set up there, or the host's handler."""
        entry = self._registry.lookup(name, room_id)
        return (entry is not None and entry.serve is not None) or self._tool_handler is not None

    async def _realtime_loop_context(
        self, session: VoiceSession, room_id: str | None, gate_context: RoomContext | None
    ) -> _ToolLoopContext:
        """The per-call context a handler reads through ``roomkit.tools`` (RFC §21.4).

        The realtime path runs no turn, so it builds the context itself around
        the handler call: the session's room and participant as the room and
        actor, no response record to merge (``has_turn`` is off, so
        ``current_response_metadata()`` answers ``None``), and the Room as
        loaded for this call: the gate's when a BEFORE_TOOL_USE hook made it
        build a context, one indexed read otherwise, taken under the
        framework's lease like every store read a channel makes. That read is
        the price of a sync accessor on a path that awaits the handler anyway;
        the two session values cost nothing. A handler shared with an
        ``AIChannel`` then answers the same questions on both paths.
        """
        ctx = _ToolLoopContext()
        ctx.has_turn = False
        ctx.room_id = room_id or session.room_id
        ctx.actor_id = session.participant_id
        # The call belongs to the model's answer, as deep as its transcript.
        ctx.chain_depth = self._session_answer_depth(session.id).answer
        if gate_context is not None:
            ctx.room = gate_context.room
        elif self._framework is not None and ctx.room_id:
            with self._framework._resource_lease():
                ctx.room = await self._framework.store.get_room(ctx.room_id)
        return ctx

    async def _serve_skill_activation(
        self, call: RealtimeToolCall, door: ToolCallDoor, carrying: RoomContext | None
    ) -> ToolOutcome:
        """Serve ``activate_skill``: ON_TOOL_CALL decides, then delivery, then gates.

        The SYNC hooks run on the result before it goes out, as for any other
        tool, so a hook that blocks ``activate_skill`` blocks the activation
        too: the model reads the refusal and no gate opens. They run outside
        the session's configuration lock, which a hook may itself need to
        reconfigure the session.
        """
        support, session = self._skill_support, call.session
        lock = self._session_config_locks.get(session.id)
        if lock is None:
            return _session_ended()
        tools = self._session_base_tools(session.id)
        result, skill = await support.prepare_activation(call.arguments, session.id, tools)
        outcome = await judge_tool_call(
            self, call, ToolOutcome(OutcomeKind.SERVED, result), carrying
        )
        # Provider updates (discovery, handoff, activation) are serialised on
        # this lock; the catalogue may have changed while the hooks ran.
        async with lock:
            if session.state == VoiceSessionState.ENDED:
                return _session_ended()
            if skill is not None and outcome.kind is OutcomeKind.SERVED:
                missing = support.missing_required_tools(
                    skill, self._session_base_tools(session.id)
                )
                if missing:
                    outcome = ToolOutcome(
                        OutcomeKind.REFUSED, support.missing_tools_error(missing)
                    )
                    skill = None
            # The call ID belongs to the current connection. Deliver before
            # native reconfiguration can replace that connection.
            call.delivered = True
            delivered = await door.deliver(call, outcome)
            if delivered and skill is not None and outcome.kind is OutcomeKind.SERVED:
                await self._open_skill_gates(session, skill)
        return outcome

    async def _open_skill_gates(self, session: VoiceSession, skill: Any) -> None:
        """Give the session a delivered activation's rules, then commit it."""
        support = self._skill_support
        if self._provider.supports_mid_session_reconfigure:
            base_tools = self._session_base_tools(session.id)
            visible = self._compose_session_tools(session, base_tools, pending_skill=skill)
            addendum = support.activated_skills_prompt(session.id, skill)
            if addendum or skill.metadata.gated_tool_names:
                prompt = self._compose_session_prompt(
                    session,
                    session.metadata.get("system_prompt", self._system_prompt),
                    pending_skill=skill,
                )
                await self._provider.reconfigure(session, tools=visible, system_prompt=prompt)
        if session.state != VoiceSessionState.ENDED:
            support.commit_activation(session.id, skill)

    def _session_base_tools(self, session_id: str) -> list[dict[str, Any]]:
        """The session's authorized catalogue, read under the state lock."""
        with self._state_lock:
            return self._session_tools.get(session_id, self._tools or [])

    def _session_catalogue(self, session_id: str) -> list[dict[str, Any]]:
        """Every tool the session declares beside the channel's own: its base
        catalogue, then what orchestration set up for its room (RFC §19.7).
        What a call is checked, validated and recovered against."""
        with self._state_lock:
            base = list(self._session_tools.get(session_id, self._tools or []))
            room_id = self._session_rooms.get(session_id)
        return base + self._orchestration_dicts(room_id, {dict_tool_name(t) for t in base})

    def _orchestration_dicts(
        self, room_id: str | None, skip: Container[str | None] = ()
    ) -> list[dict[str, Any]]:
        """The declarations of the tools orchestration declares in *room_id*'s
        sessions, bar the names in *skip*."""
        return [
            tool_dict(entry.definition)
            for entry in self._registry.entries(room_id, source=ToolSource.ORCHESTRATION)
            if entry.traits.always_declared and entry.name not in skip
        ]

    async def _deliver_tool_result(self, call: RealtimeToolCall, result: str) -> bool:
        """Send *call*'s one result (RFC §12.4); whether it reached a live session.

        The call counts as delivered from here on: a cancellation that lands
        while the result goes out, or a step that fails after it, adds no
        second outcome.
        """
        call.delivered = True
        return await self._submit_realtime_tool_result(call.session, call.call_id, result)

    async def _submit_realtime_tool_result(
        self, session: VoiceSession, call_id: str, result: str
    ) -> bool:
        """Send a call's result; whether it reached a live session.

        False when the session ended, or when the call lost its id to a
        reconnect its own handler caused: the new socket never issued it, and
        no provider output is awaited for it (RFC §9.3).
        """
        if session.state == VoiceSessionState.ENDED:
            return False
        if own_call_orphaned(session.id, call_id):
            # The new socket never issued this id (RFC §9.3)
            return False
        self._expect_provider_output(session.id)
        await self._provider.submit_tool_result(session, call_id, result)
        return session.state != VoiceSessionState.ENDED

    def _tool_parameters(self, name: str, session: VoiceSession) -> dict[str, Any] | None:
        """Return the declared ``parameters`` schema for realtime tool *name*.

        ``None`` when the tool's schema is unknown (skips argument validation).
        """
        if self._tool_search_support and self._tool_search_support.is_search_tool(name):
            for tool in self._tool_search_support.search_tool_dicts():
                if tool["name"] == name:
                    params = tool.get("parameters")
                    return params if isinstance(params, dict) else None
        if self._skill_support and self._skill_support.is_skill_tool(name):
            for tool in self._skill_support.skill_tool_dicts():
                if tool["name"] == name:
                    params = tool.get("parameters")
                    return params if isinstance(params, dict) else None
        for t in self._session_catalogue(session.id):
            if isinstance(t, dict) and t.get("name") == name:
                params = t.get("parameters")
                return params if isinstance(params, dict) else None
        return None

    def _is_declared_realtime_tool(
        self, name: str, session: VoiceSession, served: Container[str] | None = None
    ) -> bool:
        """Return whether *name* is in a non-empty session tool catalogue.

        An empty catalogue retains the historical hook-only/dynamic-handler
        mode. Once declarations exist, however, a provider cannot invent an
        undeclared name and reach a generic dispatcher.

        Infrastructure tools (Tool Search, skills) are declared by the channel
        rather than by the caller's catalogue, so they answer ``True`` without
        appearing in it.
        """
        if name in (self._channel_tool_names() if served is None else served):
            return True
        tools = self._session_catalogue(session.id)
        if not tools:
            return True
        return any(isinstance(tool, dict) and tool.get("name") == name for tool in tools)

    def _channel_tool_names(self) -> frozenset[str]:
        """The tools this channel serves itself: Tool Search's and the skills'."""
        return frozenset(e.name for e in self._registry.entries(None, source=ToolSource.CHANNEL))

    def _exempt_tool_names(self) -> frozenset[str]:
        """The channel's own tools that escape the policy and skill gating (RFC §21.1)."""
        return frozenset(self._registry.names(None, lambda traits: traits.exempt))

    def _declared_once(
        self, tools: list[dict[str, Any]], room_id: str | None
    ) -> list[dict[str, Any]]:
        """A session's host tools in *room_id*: none under a name the channel or
        orchestration declares itself there, each name once (RFC §21.1,
        :func:`declared_once`). Those are composed in afterwards."""
        served = self._channel_tool_names() | self._registry.names(
            room_id, lambda traits: traits.always_declared
        )
        return declared_once(tools, dict_tool_name, served, self._collisions)

    def _tool_reachable(self, name: str, session_id: str) -> bool:
        """Whether the session may call *name*: its tool policy and skill gating.

        What Tool Search may name in its results and listings (RFC §21.1); the
        pre-execution gate enforces the same rule on the call itself.
        """
        if not policy_admits(self._session_policy(session_id), name, self._exempt_tool_names()):
            return False
        support = self._skill_support
        return support is None or not support.is_gated(name, session_id)

    def _session_policy(self, session_id: str) -> ToolPolicy | None:
        """The tool policy resolved for the session's participant (RFC §12.4)."""
        if self._tool_policy is None:
            return None
        return self._tool_policy.resolve(self._session_roles.get(session_id))

    def _policy_filter(self, session_id: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The part of *tools* the session's policy admits.

        Tool Search's ``call_tool`` transport stays declared: it is no tool of
        its own, and the policy applies to the tool it names, at the gate.
        """
        policy = self._session_policy(session_id)
        if policy is None:
            return tools
        search = self._tool_search_support
        exempt = self._exempt_tool_names()
        return [
            t
            for t in tools
            if (search is not None and search.is_search_tool(str(t.get("name", ""))))
            or policy_admits(policy, str(t.get("name", "")), exempt)
        ]

    async def _refresh_session_role(self, session: VoiceSession, room_id: str | None) -> None:
        """Read the participant's role again, so a role changed during the
        session holds at the gate from the next call on (RFC §12.4)."""
        policy = self._tool_policy
        if policy is None or not policy.role_overrides or not (self._framework and room_id):
            return
        role = await self._resolve_session_role(room_id, session.participant_id)
        if session.id in self._session_roles:
            self._session_roles[session.id] = role

    async def _resolve_session_role(self, room_id: str | None, participant_id: str) -> str | None:
        """The session participant's role, where a policy has overrides to read."""
        policy = self._tool_policy
        if policy is None or not policy.role_overrides or not (self._framework and room_id):
            return None
        participant = await self._framework.store.get_participant(room_id, participant_id)
        return participant.role if participant is not None else None

    async def _authorize_realtime_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
        room_id: str | None,
        session: VoiceSession,
        *,
        channel_serves: bool = True,
    ) -> tuple[dict[str, Any], GateRefusal | None, RoomContext | None]:
        """Pre-execution gate for realtime tool calls (parity with the classic
        AI path), in RFC §12.4's order.

        *channel_serves* says whether this entry serves the channel's own
        tools (Tool Search, skills): the provider's function calls do; a
        reasoning backend's calls and a recovered spoken call reach the
        handler only, so on them no name is the channel's (RFC §21.1).

        Checks the tool is declared, folds a flattened hub-tool call back into
        ``params`` and validates the arguments against the declared schema,
        applies the tool policy and skill gating, and runs BEFORE_TOOL_USE so
        a block prevents the side effect rather than only hiding the result.
        Hooks may replace the arguments through ``metadata["arguments"]``; the
        replacement is validated before it can reach the handler.

        Returns the effective arguments, an optional denial result, and the
        room context this gate built — ``None`` when it built none. The caller
        hands that context to :meth:`_fire_tool_hook` as ``carrying`` so one
        tool call deserialises the room history once instead of twice.
        """
        served = self._channel_tool_names() if channel_serves else frozenset()
        if not self._is_declared_realtime_tool(name, session, served):
            logger.warning("Realtime provider requested undeclared tool %s", name)
            undeclared = json.dumps({"error": f"Tool '{name}' is not declared"})
            return arguments, GateRefusal(undeclared), None
        params = self._tool_parameters(name, session)
        arguments, invalid = self._validated_realtime_arguments(name, arguments, params)
        if invalid is not None:
            return arguments, GateRefusal(invalid), None
        await self._refresh_session_role(session, room_id)
        exempt = self._exempt_tool_names() if channel_serves else frozenset()
        refusal = self._access_refusal(name, session.id, exempt)
        if refusal is not None:
            return arguments, GateRefusal(refusal), None
        return await self._before_realtime_tool_use(
            name, arguments, params, call_id, room_id, session
        )

    def _validated_realtime_arguments(
        self, name: str, arguments: dict[str, Any], params: dict[str, Any] | None
    ) -> tuple[dict[str, Any], str | None]:
        """The model's arguments checked against the declared schema (fail-closed),
        after repairing a hub tool's flattened ``params``: same gate, same order
        as the classic AI path."""
        if params is None:
            return arguments, None
        folded, fold_error = fold_hoisted_arguments(params, arguments)
        if fold_error is not None:
            logger.warning("Realtime tool %s arguments ambiguous: %s", name, fold_error)
            return arguments, json.dumps(
                {"error": f"Invalid arguments for '{name}': {fold_error}"}
            )
        if folded is not None:
            logger.info(
                "Realtime tool %s: folded hoisted arguments %s into its container "
                "(provider=%s, model=%s)",
                name,
                sorted(set(arguments) - set(folded)),
                self._provider.name,
                self._provider.model_name,
            )
            arguments = folded
        arg_error = validate_tool_arguments(params, arguments)
        if arg_error is not None:
            logger.warning("Realtime tool %s arguments rejected: %s", name, arg_error)
            return arguments, json.dumps({"error": f"Invalid arguments for '{name}': {arg_error}"})
        return arguments, None

    def _access_refusal(self, name: str, session_id: str, exempt: Container[str]) -> str | None:
        """Why the session may not call *name*: its tool policy, resolved for
        its participant, then skill gating, as on the classic path."""
        if not policy_admits(self._session_policy(session_id), name, exempt):
            logger.warning("Realtime tool %s blocked by policy", name)
            return json.dumps({"error": policy_refusal(name)})
        # Hiding a gated tool from the catalogue is not enforcement — the model
        # may still name one it saw before the skill was deactivated.
        if self._skill_support is not None and self._skill_support.is_gated(name, session_id):
            logger.warning("Realtime tool %s blocked by skill gating", name)
            return json.dumps(
                {
                    "error": (
                        f"Tool '{name}' is gated by a skill. "
                        "Activate the skill first using activate_skill."
                    )
                }
            )
        return None

    async def _before_realtime_tool_use(
        self,
        name: str,
        arguments: dict[str, Any],
        params: dict[str, Any] | None,
        call_id: str,
        room_id: str | None,
        session: VoiceSession,
    ) -> tuple[dict[str, Any], GateRefusal | None, RoomContext | None]:
        """BEFORE_TOOL_USE as every channel runs it, which needs a framework and
        a room to run room hooks; the arguments it leaves are validated again."""
        framework = self._framework
        if framework is None or not room_id:
            return arguments, None, None
        pre_event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.REALTIME_VOICE,
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
            result=None,
            room_id=room_id,
            session=session,
        )
        decision, context = await framework._decide_before_tool_use(pre_event, self.channel_id)
        if not decision:
            logger.info("Realtime tool %s denied by BEFORE_TOOL_USE hook", name)
            denial = json.dumps({"error": pre_execution_denial(name, decision.reason)})
            return arguments, GateRefusal(denial, decision.detail), context
        arguments, invalid = _rewritten_arguments(name, arguments, params, decision.arguments)
        return arguments, GateRefusal(invalid) if invalid is not None else None, context

    async def _serve_tool_search(self, call: RealtimeToolCall, door: ToolCallDoor) -> ToolOutcome:
        """Serve a Tool Search call, serialized with activation and handoff.

        The result goes out, bounded, then the session is reconfigured to
        declare what it revealed: the call id belongs to the current
        connection, and a provider update can replace it. The observers hear
        of the call after, outside the configuration lock (they may request a
        handoff), as a report: the model already read the result, so nothing
        a hook returns replaces it (RFC §9.3).
        """
        session = call.session
        lock = self._session_config_locks.get(session.id)
        if lock is None:
            return _session_ended()
        async with lock:
            if session.state == VoiceSessionState.ENDED:
                return _session_ended()
            result, updated = await self._tool_search_support.handle_tool_call(
                call.name, call.arguments, session.id
            )
            outcome = ToolOutcome(OutcomeKind.SERVED, self._bound_call_result(call, result))
            call.delivered = True
            delivered = await door.deliver(call, outcome)
            if (
                delivered
                and updated is not None
                and self._provider.supports_mid_session_reconfigure
            ):
                await self._reveal_tools(session)
        logger.info(
            "Tool-search %s(%s) handled for session %s (%d tools now visible)",
            call.name,
            call.call_id,
            session.id,
            len(updated) if updated is not None else 0,
        )
        await self._report_search_call(call, str(outcome.result))
        return outcome

    async def _reveal_tools(self, session: VoiceSession) -> None:
        """Declare to the session the tools Tool Search revealed."""
        with self._state_lock:
            base_tools = self._session_tools.get(session.id, self._tools or [])
        await self._provider.reconfigure(
            session,
            tools=self._compose_session_tools(session, base_tools),
            system_prompt=self._compose_session_prompt(
                session, session.metadata.get("system_prompt", self._system_prompt)
            ),
        )

    async def _report_search_call(self, call: RealtimeToolCall, result: str) -> None:
        """Report a delivered Tool Search call to every ON_TOOL_CALL hook."""
        framework = self._tool_framework(call)
        if framework is None:
            return
        call.reported = True
        try:
            await framework._report_tool_call(self._tool_event(call, result), self.channel_id)
        except Exception:
            logger.debug(
                "ON_TOOL_CALL report failed for tool-search tool %s", call.name, exc_info=True
            )


def _session_ended() -> ToolOutcome:
    """The outcome of a call its session ended before it was served."""
    return ToolOutcome(OutcomeKind.CANCELLED, json.dumps({"error": "The session has ended."}))


def _rewritten_arguments(
    name: str,
    arguments: dict[str, Any],
    params: dict[str, Any] | None,
    rewritten: dict[str, Any] | None,
) -> tuple[dict[str, Any], str | None]:
    """The arguments BEFORE_TOOL_USE left, returned or edited in place, checked
    against the schema again."""
    effective = rewritten if rewritten is not None else arguments
    # No fold here, deliberately: a hook's rewritten arguments are user code,
    # and repairing them would hide the hook's bug. The model's own call was
    # already folded above.
    arg_error = validate_tool_arguments(params, effective) if params is not None else None
    if arg_error is not None:
        logger.warning("Realtime tool %s post-hook arguments rejected: %s", name, arg_error)
        return effective, json.dumps(
            {"error": f"Invalid rewritten arguments for '{name}': {arg_error}"}
        )
    return effective, None
