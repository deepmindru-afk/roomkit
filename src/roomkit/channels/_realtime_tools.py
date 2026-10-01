"""Tool call handling for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
from collections.abc import Container
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._ai_policy import policy_admits, policy_refusal
from roomkit.channels._realtime_context import (
    _current_voice_session,
    own_call_orphaned,
    serving_call,
    spare_own_orphaned_call,
)
from roomkit.channels._served_tools import CollisionLog, declared_once, dict_tool_name
from roomkit.channels._skill_constants import TOOL_ACTIVATE_SKILL
from roomkit.channels._tool_registry import ChannelRegistry, ToolSource, tool_dict
from roomkit.channels._tool_search_constants import TOOL_CALL_TOOL
from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext
from roomkit.core.exceptions import ToolRefusedError
from roomkit.core.hooks import SyncPipelineResult
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.models.tool_call import (
    ToolCallEvent,
    fold_tool_call_rewrite,
    observed_call_event,
)
from roomkit.providers.ai.base import AITextPart
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.tools.result import (
    GateRefusal,
    as_tool_result,
    before_tool_use_detail,
    failure_detail,
    hook_errors_detail,
    is_unknown_tool_answer,
    pre_execution_denial,
    tool_call_verdict,
    tool_failure,
    unserved_tool_error,
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


def result_text(raw: Any) -> str:
    """Flatten a tool handler result for a voice provider.

    A handler shared with an ``AIChannel`` may answer with a content-part
    list (text + images); a speech provider cannot consume an image, so the
    list flattens the way ``AIToolResultPart.as_text()`` does — text joined,
    ``[image]`` placeholders. ``json.dumps`` on such a list would raise on
    the pydantic parts instead. Anything else is JSON, as on every channel
    (RFC §21.4).
    """
    value = as_tool_result(raw)
    if isinstance(value, str):
        return value
    return "\n".join(p.text if isinstance(p, AITextPart) else "[image]" for p in value)


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
    _pending_tool_calls: dict[str, dict[str, tuple[str, dict[str, Any]]]]
    _reported_tool_calls: dict[str, set[str]]
    _scheduled_tasks: set[asyncio.Task[Any]]
    channel_id: str
    _telemetry_provider: Any

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    def _update_idle_event(self, session_id: str) -> None: ...

    def _expect_provider_output(self, session_id: str) -> None: ...


def _hook_outcome(
    hook_result: Any, tool_event: ToolCallEvent, handler_result: str | None, name: str
) -> tuple[str, bool]:
    """The result a realtime model reads after ON_TOOL_CALL, and whether it failed.

    The chain is read as on every channel (:func:`tool_call_verdict`, RFC
    §9.3): a MODIFY counts like the override, and a result replaced by an
    empty value is replaced, never kept.
    """
    verdict = tool_call_verdict(hook_result, tool_event)
    if verdict.result is not None:
        return result_text(verdict.result), verdict.blocked
    if handler_result is not None:
        return handler_result, False
    # Nothing served this call: no handler, and the hooks that could have
    # answered it did not, a hook that raised included (its message goes to
    # the observers, never to the model, RFC §9.3). Reporting ``{"status":
    # "ok"}`` would be a success for work nobody did, which the model then
    # acts on and an audit trail records as a completed call.
    return unserved_tool_error(name), True


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
    _pending_tool_calls: dict[str, dict[str, tuple[str, dict[str, Any]]]]
    _reported_tool_calls: dict[str, set[str]]
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
        self._begin_tool_call(session.id, call_id, name, arguments)
        task = self._track_task(
            loop,
            self._handle_tool_call(session, call_id, name, arguments),
            name=f"rt_tool_call:{session.id}:{call_id}",
        )
        task.add_done_callback(lambda _: self._finish_tool_call(session.id, call_id))

    def _on_provider_tool_call_cancelled(self, session: VoiceSession, call_ids: list[str]) -> Any:
        """Provider callback: the model abandoned outstanding calls (RFC §12.4).

        The handler still running for one of them is working for a result
        nobody will read. Its task is cancelled — found by name, the way the
        session-end drain finds it — and the call is reported to ON_TOOL_CALL's
        observers as cancelled. A call no longer in the books (its result left
        before the cancellation arrived) has no event to build: the provider
        dropped the stale result and logged it. A call still in the books whose
        outcome the observers already received is left to finish for the same
        reason: a second event would put two outcomes on one ``tool_call_id``,
        and the result it is submitting is the provider's to drop. A reconnect
        the call's own handler caused (a handoff reconfiguring its session)
        orphans that call too, and it is not abandoned: it runs on, its result
        kept off the wire (RFC §9.3).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if session.state == VoiceSessionState.ENDED:
            return
        pending = self._pending_tool_calls.get(session.id) or {}
        for call_id in call_ids:
            recorded = pending.get(call_id)
            if recorded is None:
                logger.debug(
                    "Cancelled tool call %s is not in flight for session %s", call_id, session.id
                )
                continue
            if self._spared_by_own_reconnect(session, call_id, recorded[0]):
                continue
            if self._tool_call_reported(session.id, call_id):
                logger.debug(
                    "Cancelled tool call %s already reported its outcome for session %s",
                    call_id,
                    session.id,
                )
                continue
            name, arguments = recorded
            self._cancel_tool_call_task(session.id, call_id)
            logger.info(
                "Tool call %s(%s) cancelled by the model for session %s", name, call_id, session.id
            )
            with self._state_lock:
                room_id = self._session_rooms.get(session.id)
            self._track_task(
                loop,
                self._report_cancelled_tool_call(session, call_id, name, arguments, room_id),
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

    def _cancel_tool_call_task(self, session_id: str, call_id: str) -> None:
        """Cancel one call's handler task, found by the name it was tracked under."""
        task_name = f"rt_tool_call:{session_id}:{call_id}"
        for task in list(self._scheduled_tasks):
            if task.get_name() == task_name and not task.done():
                task.cancel()

    async def _report_cancelled_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        room_id: str | None,
    ) -> None:
        """Report an abandoned call to ON_TOOL_CALL's observers (RFC §9.3)."""
        body = json.dumps(
            {
                "error": "Tool call cancelled",
                "tool": name,
                "hint": "The model abandoned this call before its result; nothing was sent.",
            }
        )
        await self._fire_tool_refusal(
            session, call_id, name, arguments, body, room_id, cancelled=True
        )

    def _begin_tool_call(
        self, session_id: str, call_id: str, name: str, arguments: dict[str, Any]
    ) -> None:
        # The call holds idle while it runs, and a result it sends holds it
        # until the continuation (``_expect_provider_output``). The provider's
        # response state is not touched: a call that ends owing nothing
        # (cancelled, or spared by its own reconnect) leaves nothing to wait on.
        self._pending_tool_calls.setdefault(session_id, {})[call_id] = (name, arguments)
        self._update_idle_event(session_id)

    def _finish_tool_call(self, session_id: str, call_id: str) -> None:
        pending = self._pending_tool_calls.get(session_id)
        if pending is not None:
            pending.pop(call_id, None)
        reported = self._reported_tool_calls.get(session_id)
        if reported is not None:
            reported.discard(call_id)
        self._update_idle_event(session_id)

    def _mark_tool_call_reported(self, session_id: str, call_id: str) -> None:
        """Record that ON_TOOL_CALL's observers received this call's outcome."""
        self._reported_tool_calls.setdefault(session_id, set()).add(call_id)

    def _tool_call_reported(self, session_id: str, call_id: str) -> bool:
        return call_id in (self._reported_tool_calls.get(session_id) or ())

    async def _handle_tool_call(
        self, session: VoiceSession, call_id: str, name: str, arguments: dict[str, Any]
    ) -> None:
        if session.state == VoiceSessionState.ENDED:
            return
        self._begin_tool_call(session.id, call_id, name, arguments)
        try:
            with serving_call(session.id, call_id):
                await self._execute_tool_call(session, call_id, name, arguments)
        finally:
            self._finish_tool_call(session.id, call_id)

    async def _execute_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Execute a tool call and submit the result to the provider.

        If a ``tool_handler`` was provided, it is called directly.
        The ``ON_TOOL_CALL`` hook is then fired (handler result, if any,
        is passed as ``event.result`` so the hook can observe or override).
        """
        if session.state == VoiceSessionState.ENDED:
            return
        name, arguments, transport_error = self._unwrap_call_tool(
            session, call_id, name, arguments
        )
        # Order barrier: a tool call must not overtake the transcriptions the
        # provider emitted before it. The user final that closes the current
        # utterance travels the serialised transcription queue, while tool
        # calls run in their own task — unbarriered, the tool reaches the
        # application first and the late final reads as new user speech.
        # Pass through the same FIFO lock, then release: tool execution
        # itself must not hold transcriptions back.
        with self._state_lock:
            order_lock = self._transcription_order_locks.setdefault(session.id, asyncio.Lock())
        async with order_lock:
            pass
        if session.state == VoiceSessionState.ENDED:
            return

        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
            _rt_parent = self._session_spans.get(session.id)
            parent = self._turn_spans.get(session.id) or _rt_parent

        from roomkit.telemetry.context import reset_span, set_current_span

        _rt_tok = set_current_span(_rt_parent) if _rt_parent else None

        telemetry = self._telemetry_provider
        tool_span_id = telemetry.start_span(
            SpanKind.REALTIME_TOOL_CALL,
            f"realtime_tool:{name}",
            parent_id=parent,
            attributes={Attr.REALTIME_TOOL_NAME: name, "tool_call_id": call_id},
            room_id=room_id,
            session_id=session.id,
            channel_id=self.channel_id,
        )

        if self._mute_on_tool_call and self._transport is not None:
            self._transport.set_input_muted(session, True)

        try:
            result_str: str
            if transport_error is not None:
                body = json.dumps({"error": transport_error})
                await self._submit_realtime_tool_result(session, call_id, body)
                telemetry.end_span(tool_span_id)
                # After the wire: the provider is holding a turn open on this
                # result, and an audit hook must not stand in front of it. Same
                # order for the gate's own refusal and the fallback below —
                # each of them answers before it reports.
                await self._fire_tool_refusal(session, call_id, name, arguments, body, room_id)
                return

            # Pre-execution gate (parity with the classic AI path): validate
            # arguments and run BEFORE_TOOL_USE BEFORE the call is routed, so a
            # block prevents the side effect instead of only hiding the result.
            # Its scope is every call: hook-only mode serves the tool from
            # ON_TOOL_CALL, and an infrastructure tool serves itself, but a
            # host auditing or denying tool use must see both.
            arguments, denial, gate_context = await self._authorize_realtime_tool(
                name, arguments, call_id, room_id, session
            )
            if session.state == VoiceSessionState.ENDED:
                telemetry.end_span(tool_span_id, status="cancelled")
                return
            if denial is not None:
                await self._submit_realtime_tool_result(session, call_id, denial.body)
                telemetry.end_span(tool_span_id)
                logger.info(
                    "Realtime tool %s(%s) denied before execution for session %s",
                    name,
                    call_id,
                    session.id,
                )
                await self._fire_gate_refusal(session, call_id, name, arguments, denial, room_id)
                return

            # Tool Search infrastructure tools — handle internally
            if self._tool_search_support and self._tool_search_support.is_search_tool(name):
                await self._dispatch_tool_search_call(
                    session, call_id, name, arguments, room_id, tool_span_id
                )
                return

            # Skill infrastructure tools — handle internally
            if self._skill_support and self._skill_support.is_skill_tool(name):
                result_str = await self._deliver_skill_call(
                    session, call_id, name, arguments, room_id, gate_context
                )
                telemetry.end_span(tool_span_id)
                logger.info(
                    "Skill tool %s(%s) handled for session %s",
                    name,
                    call_id,
                    session.id,
                )
                return

            try:
                result_str = await self._serve_gated_tool_call(
                    session, call_id, name, arguments, room_id, gate_context
                )
            except ToolRefusedError as refusal:
                # Ends this call the way the pre-execution denial above does:
                # the span says denied, the model reads the handler's words,
                # and the turn carries on.
                body = refusal.message
                if len(body) > self._tool_result_max_length:
                    body = self._truncate_tool_result(body, name, call_id, session.id)
                await self._submit_realtime_tool_result(session, call_id, body)
                await self._fire_tool_refusal(session, call_id, name, arguments, body, room_id)
                telemetry.end_span(tool_span_id, attributes={Attr.REALTIME_TOOL_DENIED: True})
                logger.info(
                    "Tool call %s(%s) refused by its handler for session %s",
                    name,
                    call_id,
                    session.id,
                )
                return
            await self._submit_realtime_tool_result(session, call_id, result_str)

            telemetry.end_span(tool_span_id)
            logger.info(
                "Tool call %s(%s) handled for session %s",
                name,
                call_id,
                session.id,
            )

        except asyncio.CancelledError:
            telemetry.end_span(tool_span_id, status="cancelled")
            raise
        except Exception as exc:
            telemetry.end_span(tool_span_id, status="error", error_message=f"tool {name} failed")
            logger.exception("Error handling tool call %s for session %s", call_id, session.id)
            body = tool_failure(name, exc)
            try:
                await self._submit_realtime_tool_result(session, call_id, body)
            except Exception:
                logger.exception("Error submitting fallback tool result")
            await self._report_raised_call(session, call_id, name, arguments, room_id, exc)
        finally:
            if self._mute_on_tool_call and self._transport is not None:
                self._transport.set_input_muted(session, False)
            if _rt_tok is not None:
                reset_span(_rt_tok)

    def _unwrap_call_tool(
        self, session: VoiceSession, call_id: str, name: str, arguments: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str | None]:
        """The tool a fixed-declaration ``call_tool`` carries, its arguments, and
        why the transport is unreadable, if it is; any other call as it came."""
        support = self._tool_search_support
        if not (
            name == TOOL_CALL_TOOL
            and support
            and support.uses_call_tool
            and support.active(session.id)
        ):
            return name, arguments, None
        name, arguments, transport_error = support.unwrap_call(arguments, session.id)
        # The books name the call the model issued; a cancellation report
        # should name the tool it wrapped.
        pending = self._pending_tool_calls.get(session.id)
        if pending is not None and call_id in pending:
            pending[call_id] = (name, arguments)
        return name, arguments, transport_error

    async def _serve_gated_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        room_id: str | None,
        gate_context: RoomContext | None,
    ) -> str:
        """Serve a tool call that already passed the pre-execution gate.

        Runs the channel's ``tool_handler`` if there is one, fires
        ``ON_TOOL_CALL`` (which may observe or override the result), and caps
        the result length. Shared by the provider's own tool calls and by a
        reasoning backend's (RFC §12.4.1), which differ only in where the
        result then goes.

        Raises :class:`~roomkit.core.exceptions.ToolRefusedError` when the
        handler declines the call. It is not caught here: a refusal and a
        served result end their caller's span differently and read differently
        in its log, so the caller is where the distinction is spent.
        """
        handler_result: str | None = None
        if self._serves_tool(name, room_id or session.room_id):
            logger.info(
                "Executing tool %s(%s) via handler for session %s",
                name,
                call_id,
                session.id,
            )
            t_seg = time.perf_counter()
            # ``ToolRefusedError`` travels out of here on purpose. Flattening
            # it into the returned string would put the outcome back in the
            # body, which is what this whole mechanism removes, and both
            # callers below own a span and a log line that have to know.
            raw = await self._call_tool_handler(session, name, arguments, room_id, gate_context)
            logger.debug(
                "tool %s handler segment: %.0fms wall",
                name,
                (time.perf_counter() - t_seg) * 1000,
            )

            t_seg = time.perf_counter()
            handler_result = result_text(raw)
            ser_s = time.perf_counter() - t_seg
            if ser_s > _LOOP_SEGMENT_BUDGET_S:
                # Pure sync CPU (wall == loop hold), and it runs on the
                # FULL result before truncation caps it.
                logger.warning(
                    "Tool %s result serialization held the event loop for "
                    "%.0fms (%d chars, budget ~%.0fms) — concurrent "
                    "realtime audio may underrun; return a string or a "
                    "compact reference instead of a large object",
                    name,
                    ser_s * 1000,
                    len(handler_result),
                    _LOOP_SEGMENT_BUDGET_S * 1000,
                )
            if is_unknown_tool_answer(handler_result):
                # Every handler said the tool is not theirs: nothing served
                # the call, which the hooks may still serve (RFC §21.4).
                handler_result = None
            # Yield so realtime pacing gets a slot between the handler
            # segment and hook dispatch — sync hooks run inline next and
            # would otherwise fuse with this segment into one loop step.
            await asyncio.sleep(0)

        # Run ON_TOOL_CALL hook (if framework + room).
        tool_event = self._realtime_tool_event(
            session, call_id, name, arguments, handler_result, room_id
        )

        if self._framework and room_id:
            result_str = await self._fire_tool_hook(
                tool_event, room_id, handler_result, name, call_id, session, gate_context
            )
            # Same reason as the post-handler yield: don't fuse hook
            # dispatch with submission into one loop step.
            await asyncio.sleep(0)
        elif handler_result is not None:
            result_str = handler_result
        else:
            result_str = unserved_tool_error(name)

        if len(result_str) > self._tool_result_max_length:
            result_str = self._truncate_tool_result(result_str, name, call_id, session.id)
        return result_str

    async def _call_tool_handler(
        self,
        session: VoiceSession,
        name: str,
        arguments: dict[str, Any],
        room_id: str | None,
        gate_context: RoomContext | None,
    ) -> Any:
        """The answer to one call, run inside the tool call context (RFC §21.4),
        whichever path brought the call: the tool orchestration set up for the
        room, else the host's handler."""
        loop_ctx = await self._realtime_loop_context(session, room_id, gate_context)
        token = _current_voice_session.set(session)
        loop_token = _current_loop_ctx.set(loop_ctx)
        try:
            waits = self._registry.waits(name, loop_ctx.room_id)
            timeout = self._tool_timeouts.for_call(name, waits=waits)
            answer = self._answer(name, arguments, loop_ctx.room_id)
            return await answer_within(timeout, name, answer)
        finally:
            _current_loop_ctx.reset(loop_token)
            _current_voice_session.reset(token)

    async def _answer(self, name: str, arguments: dict[str, Any], room_id: str | None) -> Any:
        """The answer of what orchestration set up for *room_id*, else of the
        host's handler."""
        entry = self._registry.lookup(name, room_id)
        if entry is not None and entry.serve is not None:
            result = entry.serve(arguments)
            return await result if inspect.isawaitable(result) else result
        return await self._tool_handler(name, arguments)

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

    async def _deliver_skill_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        room_id: str | None,
        carrying: RoomContext | None = None,
    ) -> str:
        """Serve a skill tool: ON_TOOL_CALL decides, then delivery, then gates.

        The SYNC hooks run on the result before it goes out, as for any other
        tool, so a hook that blocks ``activate_skill`` blocks the activation
        too: the model reads the refusal and no gate opens. They run outside
        the session's configuration lock, which a hook may itself need to
        reconfigure the session.
        """
        support = self._skill_support
        if name != TOOL_ACTIVATE_SKILL:
            result = await support.handle_tool_call(name, arguments, session.id)
            result, _ = await self._screen_skill_result(
                session, call_id, name, arguments, result, room_id, carrying
            )
            await self._submit_realtime_tool_result(session, call_id, result)
            return result
        lock = self._session_config_locks.get(session.id)
        if lock is None:
            return json.dumps({"error": "Session ended before skill activation"})
        tools = self._session_base_tools(session.id)
        result, skill = await support.prepare_activation(arguments, session.id, tools)
        result, failed = await self._screen_skill_result(
            session, call_id, name, arguments, result, room_id, carrying
        )
        # Provider updates (discovery, handoff, activation) are serialised on
        # this lock; the catalogue may have changed while the hooks ran.
        async with lock:
            if session.state == VoiceSessionState.ENDED:
                return json.dumps({"error": "Session ended before skill activation"})
            if skill is not None and not failed:
                missing = support.missing_required_tools(
                    skill, self._session_base_tools(session.id)
                )
                if missing:
                    result, skill = support.missing_tools_error(missing), None
            # The call ID belongs to the current connection. Submit before
            # native reconfiguration can replace that connection.
            delivered = await self._submit_realtime_tool_result(session, call_id, result)
            if delivered and skill is not None and not failed:
                await self._open_skill_gates(session, skill)
            return result

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

    def _realtime_tool_event(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        result: str | None,
        room_id: str | None,
    ) -> ToolCallEvent:
        """The ON_TOOL_CALL event of one call on this session."""
        return ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.REALTIME_VOICE,
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
            result=result,
            room_id=room_id,
            session=session,
        )

    async def _screen_skill_result(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        result: str,
        room_id: str | None,
        carrying: RoomContext | None = None,
    ) -> tuple[str, bool]:
        """Run ON_TOOL_CALL on a skill tool's result; return what the model reads
        and whether the call failed (a hook blocked it)."""
        if not (self._framework and room_id):
            return result, False
        event = self._realtime_tool_event(session, call_id, name, arguments, result, room_id)
        return await self._fire_tool_hook_outcome(
            event, room_id, result, name, call_id, session, carrying
        )

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
        """BEFORE_TOOL_USE, which needs a framework and a room to run room
        hooks; the arguments it leaves are validated again."""
        framework = self._framework
        if framework is None or not room_id:
            return arguments, None, None
        # Building a context costs two store reads; skip it when nothing listens.
        # Schema validation above stays unconditional — it needs no context.
        if not framework.hook_engine.has_hooks(HookTrigger.BEFORE_TOOL_USE):
            return arguments, None, None
        context = await framework._build_context(room_id)
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
        hook_result = await framework.hook_engine.run_sync_hooks(
            room_id, HookTrigger.BEFORE_TOOL_USE, pre_event, context, skip_event_filter=True
        )
        await framework._emit_framework_event(
            "before_tool_use",
            room_id=room_id,
            channel_id=self.channel_id,
            data={
                "tool_name": name,
                "tool_call_id": call_id,
                "allowed": hook_result.allowed,
                "reason": hook_result.reason,
            },
        )
        if not hook_result.allowed:
            logger.info("Realtime tool %s denied by BEFORE_TOOL_USE hook", name)
            return arguments, _hook_refusal(name, hook_result), context
        arguments, invalid = _rewritten_arguments(name, arguments, params, hook_result.metadata)
        return arguments, GateRefusal(invalid) if invalid is not None else None, context

    async def _report_raised_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        room_id: str | None,
        exc: Exception,
    ) -> None:
        """Tell the observers a call raised, once, whatever reached the model (RFC §9.3).

        The model reads :func:`tool_failure`, the class alone; the observers
        get the message on ``error_detail``. A call already reported (served
        before a later step failed, the submission itself for instance) adds no
        second outcome: that failure is in the log.
        """
        if self._tool_call_reported(session.id, call_id):
            return
        body = tool_failure(name, exc)
        await self._fire_tool_refusal(
            session, call_id, name, arguments, body, room_id, detail=failure_detail(exc)
        )

    async def _fire_gate_refusal(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        denial: GateRefusal,
        room_id: str | None,
    ) -> None:
        """Report a call the pre-execution gate refused, with its detail."""
        await self._fire_tool_refusal(
            session, call_id, name, arguments, denial.body, room_id, detail=denial.detail
        )

    async def _fire_tool_refusal(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        result: str,
        room_id: str | None,
        *,
        cancelled: bool = False,
        detail: str | None = None,
    ) -> None:
        """Fire ON_TOOL_CALL for a call that failed, was refused, or was abandoned.

        *detail* is a raised call's full failure, or the error of a
        BEFORE_TOOL_USE hook that failed closed, for the observers only
        (``ToolCallEvent.error_detail``).

        The pre-execution gate returns before anything serves the call, and a
        failure inside it lands in the fallback below — neither path reached
        the hook, so a host auditing tool use saw a denied tool as a tool the
        agent never called. ``cancelled`` is the third outcome: the model
        discarded the call before its result (RFC §9.3), and the event says so
        beside ``is_error`` rather than leaving it to be read as a refusal.

        Observational by construction: it reaches the ASYNC observers of
        ON_TOOL_CALL and no further. A SYNC hook is the one that can *serve* a
        call, so a refused call must not reach one — the gate exists to prevent
        the side effect, not to hide it. A hook that raises must not turn the
        refusal into a crash either.
        """
        if not self._framework or not room_id:
            return
        self._mark_tool_call_reported(session.id, call_id)
        event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.REALTIME_VOICE,
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
            result=result,
            room_id=room_id,
            session=session,
            is_error=True,
            cancelled=cancelled,
            error_detail=detail,
        )
        data: dict[str, Any] = {
            "tool_name": name,
            "tool_call_id": call_id,
            "channel_type": str(ChannelType.REALTIME_VOICE),
            "is_error": True,
        }
        if cancelled:
            data["cancelled"] = True
        try:
            context = await self._framework._build_context(room_id)
            await self._framework.hook_engine.run_observers(
                room_id,
                HookTrigger.ON_TOOL_CALL,
                event,
                context,
                skip_event_filter=True,
            )
            await self._framework._emit_framework_event(
                "tool_call",
                room_id=room_id,
                channel_id=self.channel_id,
                data=data,
            )
        except Exception:
            logger.debug(
                "ON_TOOL_CALL observation failed for refused tool %s", name, exc_info=True
            )

    async def _fire_tool_hook(
        self,
        tool_event: Any,
        room_id: str,
        handler_result: str | None,
        name: str,
        call_id: str,
        session: VoiceSession,
        carrying: RoomContext | None = None,
    ) -> str:
        """Fire ON_TOOL_CALL and return the result the model reads."""
        result_str, _ = await self._fire_tool_hook_outcome(
            tool_event, room_id, handler_result, name, call_id, session, carrying
        )
        return result_str

    async def _fire_tool_hook_outcome(
        self,
        tool_event: Any,
        room_id: str,
        handler_result: str | None,
        name: str,
        call_id: str,
        session: VoiceSession,
        carrying: RoomContext | None = None,
    ) -> tuple[str, bool]:
        """Fire ON_TOOL_CALL hook; return the final result and whether it failed.

        ``carrying`` is the context the pre-execution gate already built for
        this same call, when it built one: handing it over spares the room
        history a second deserialisation per tool call.
        """
        assert self._framework is not None  # guarded by caller  # noqa: S101
        t_seg = time.perf_counter()
        # No hook, no chain to run and no context to build for one: the call
        # stands as served. The observers wait for the outcome: they see the
        # result the model reads, and a call nothing served is the hooks'
        # chance to serve it, not a report (RFC §9.3).
        context: RoomContext | None = None
        hook_result = SyncPipelineResult(event=tool_event)
        if self._framework.hook_engine.has_hooks(HookTrigger.ON_TOOL_CALL):
            context = await self._framework._build_context(room_id, carrying=carrying)
            hook_result = await self._framework.hook_engine.run_sync_hooks(
                room_id,
                HookTrigger.ON_TOOL_CALL,
                tool_event,
                context,
                skip_event_filter=True,
                fold=fold_tool_call_rewrite,
                fire_observers=False,
            )
        if handler_result is not None:
            # The firing carried the handler's result: that was the report. A
            # cancellation landing between here and the wire must not add a
            # second outcome for the same call.
            self._mark_tool_call_reported(session.id, call_id)
        # Wall time, not loop hold — sync hooks may legitimately await I/O.
        logger.debug(
            "tool %s ON_TOOL_CALL segment: %.0fms wall",
            name,
            (time.perf_counter() - t_seg) * 1000,
        )
        result_str, failed = _hook_outcome(hook_result, tool_event, handler_result, name)
        await self._report_hook_outcome(
            tool_event, hook_result, handler_result, result_str, failed, context, session
        )
        return result_str, failed

    async def _report_hook_outcome(
        self,
        tool_event: ToolCallEvent,
        hook_result: Any,
        handler_result: str | None,
        result_str: str,
        failed: bool,
        context: RoomContext | None,
        session: VoiceSession,
    ) -> None:
        """Tell ON_TOOL_CALL's observers the call's outcome, once (RFC §9.3).

        A served call's observers see the result the model is about to read,
        on the event the chain left; a blocked one's, the withheld outcome. A
        call nothing served had its firing with no result, a chance to serve
        it rather than a report: a hook that served it gives the outcome,
        marked reported so a cancellation adds none; otherwise it is a
        refusal, stated with the body the model is about to read, and a
        refusal carries its own framework event.
        """
        assert self._framework is not None  # guarded by caller  # noqa: S101
        room_id = str(tool_event.room_id)
        call_id, name = tool_event.tool_call_id, tool_event.name
        engine = self._framework.hook_engine
        if handler_result is None and failed:
            await self._fire_tool_refusal(
                session,
                call_id,
                name,
                tool_event.arguments,
                result_str,
                room_id,
                detail=hook_errors_detail(hook_result),
            )
            return
        if handler_result is None and hook_result.allowed:
            self._mark_tool_call_reported(session.id, call_id)
        if context is not None:  # None when no ON_TOOL_CALL hook is registered
            await engine.run_observers(
                room_id,
                HookTrigger.ON_TOOL_CALL,
                observed_call_event(hook_result, tool_event, result_str),
                context,
                skip_event_filter=True,
            )
        await self._framework._emit_framework_event(
            "tool_call",
            room_id=room_id,
            channel_id=self.channel_id,
            data={
                "tool_name": name,
                "tool_call_id": call_id,
                "channel_type": str(ChannelType.REALTIME_VOICE),
            },
        )

    async def _dispatch_tool_search_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        room_id: str | None,
        tool_span_id: Any,
    ) -> None:
        """Serialize discovery with activation and handoff, then notify observers."""
        lock = self._session_config_locks.get(session.id)
        if lock is None:
            return
        async with lock:
            if session.state == VoiceSessionState.ENDED:
                return
            result_str, updated = await self._tool_search_support.handle_tool_call(
                name, arguments, session.id
            )
            # The pending call belongs to the current connection. Deliver its
            # result before a provider update can replace that connection.
            delivered = await self._submit_realtime_tool_result(session, call_id, result_str)
            if not delivered:
                return
            if updated is not None and self._provider.supports_mid_session_reconfigure:
                with self._state_lock:
                    base_tools = self._session_tools.get(session.id, self._tools or [])
                await self._provider.reconfigure(
                    session,
                    tools=self._compose_session_tools(session, base_tools),
                    system_prompt=self._compose_session_prompt(
                        session, session.metadata.get("system_prompt", self._system_prompt)
                    ),
                )

        # Observers may request a handoff, so never call them under the
        # configuration lock. The model already read the result, so the
        # firing is a report: nothing a hook returns replaces it, and the
        # observers see it whatever a hook decided, a BLOCK included.
        if self._framework and room_id:
            search_event = self._realtime_tool_event(
                session, call_id, name, arguments, result_str, room_id
            )
            try:
                await self._framework._report_tool_call(search_event, self.channel_id)
            except Exception:
                logger.debug(
                    "ON_TOOL_CALL report failed for tool-search tool %s",
                    name,
                    exc_info=True,
                )

        self._telemetry_provider.end_span(tool_span_id)
        logger.info(
            "Tool-search %s(%s) handled for session %s (%d tools now visible)",
            name,
            call_id,
            session.id,
            len(updated) if updated is not None else 0,
        )

    def _truncate_tool_result(
        self,
        result_str: str,
        name: str,
        call_id: str,
        session_id: str,
    ) -> str:
        """Truncate an oversized tool result with a notice."""
        original_len = len(result_str)
        logger.warning(
            "Tool result for %s(%s) truncated from %d to %d chars (session %s)",
            name,
            call_id,
            original_len,
            self._tool_result_max_length,
            session_id,
        )
        notice = (
            f"\n... [truncated — original result was {original_len} chars. "
            "The full content has been delivered to the client.]"
        )
        return result_str[: self._tool_result_max_length - len(notice)] + notice


def _hook_refusal(name: str, hook_result: Any) -> GateRefusal:
    """What the model reads of a call BEFORE_TOOL_USE refused, and the observers' detail.

    A BLOCK's reason is the hook's own words for the model. A hook that failed
    closed gives it the plain denial, and its error goes to the observers only
    (RFC §9.3).
    """
    detail = before_tool_use_detail(hook_result)
    reason = hook_result.reason if not hook_result.failed_closed else None
    return GateRefusal(json.dumps({"error": reason or pre_execution_denial(name)}), detail)


def _rewritten_arguments(
    name: str,
    arguments: dict[str, Any],
    params: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """The arguments BEFORE_TOOL_USE left, returned or edited in place, checked
    against the schema again."""
    rewritten = metadata.get("arguments")
    if "arguments" in metadata and not isinstance(rewritten, dict):
        logger.error(
            "BEFORE_TOOL_USE hook returned non-object arguments for realtime tool %s "
            "— denying tool call",
            name,
        )
        error = f"Invalid rewritten arguments for '{name}': expected an object"
        return arguments, json.dumps({"error": error})
    effective = rewritten if isinstance(rewritten, dict) else arguments
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
