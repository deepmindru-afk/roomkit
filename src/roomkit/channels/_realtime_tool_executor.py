"""One realtime tool call, served the same way whatever door it came through (RFC §12.4).

A speech-to-speech channel has several doors to the host's tools: the
provider's function call, a call recovered from speech, a reasoning backend's
call, and a conference's call. Every call takes the same steps: the
pre-execution gate, the serving inside the tool call context, ON_TOOL_CALL,
the bound on the result, the delivery, the report. Only the delivery differs
between doors, so a door is its delivery; the host is the channel that owns
the call and knows how to gate it and how to answer it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from roomkit.core.exceptions import ToolRefusedError, UnservedToolCallError
from roomkit.models.tool_call import ToolCallVerdict
from roomkit.tools._outcome import OutcomeKind, ToolOutcome, read_outcome
from roomkit.tools.context import (
    ToolCallContext,
    _current_loop_ctx,
    _current_tool_call,
    _ToolLoopContext,
)
from roomkit.tools.result import (
    GateRefusal,
    cancelled_tool_error,
    failure_detail,
    read_tool_call_verdict,
    result_text,
    tool_failure,
    unserved_tool_error,
)

if TYPE_CHECKING:
    from roomkit.channels._realtime_tool_calls import RealtimeToolCall
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.models.room import Room
    from roomkit.models.tool_call import ToolCallEvent
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_tools")

ABANDONED_BY_PROVIDER = "The provider abandoned this call"
"""Why a call the provider reported abandoned was cancelled: the model
discarded it, a reconnect orphaned it, the provider's wait on it timed out,
or the connection ended (RFC §12.4)."""


class ToolCallDoor(Protocol):
    """Where a call's outcome goes: the one thing that differs between doors."""

    channel_serves: bool
    """Whether the door serves the channel's own tools (Tool Search, skills):
    the provider's function calls do; a recovered or a backend call reaches
    the host's tools only (RFC §21.1)."""

    async def deliver(self, call: RealtimeToolCall, outcome: ToolOutcome) -> bool:
        """Hand *outcome* to whoever waits for it; whether it reached them."""
        ...


class ToolCallHost(Protocol):
    """The channel that owns a call: how it gates the call and answers it."""

    @property
    def channel_id(self) -> str: ...

    def _tool_framework(self, call: RealtimeToolCall) -> RoomKit | None:
        """The kit whose hooks judge and observe *call*; ``None`` without one
        or without a room."""
        ...

    def _tool_event(self, call: RealtimeToolCall, result: str | None) -> ToolCallEvent:
        """The ON_TOOL_CALL event of *call*, carrying *result*."""
        ...

    async def _authorize_call(
        self, call: RealtimeToolCall, door: ToolCallDoor
    ) -> tuple[GateRefusal | None, RoomContext | None]:
        """The pre-execution gate: why *call* may not run, and the context the
        gate built. Leaves the arguments to run with on *call*."""
        ...

    def _call_ended(self, call: RealtimeToolCall) -> bool:
        """Whether whoever issued *call* is gone (its session ended)."""
        ...

    async def _serve_channel_tool(
        self, call: RealtimeToolCall, door: ToolCallDoor, carrying: RoomContext | None
    ) -> ToolOutcome | None:
        """Serve one of the channel's own tools, delivery included; ``None``
        when *call* names none."""
        ...

    async def _answer_call(self, call: RealtimeToolCall, carrying: RoomContext | None) -> str:
        """The answer to *call*, as text, inside its tool call context.

        Raises :class:`UnservedToolCallError` when nothing serves the call and
        :class:`ToolRefusedError` when its handler refuses it.
        """
        ...

    def _bound_call_result(self, call: RealtimeToolCall, text: str) -> str:
        """*text* within the bound on what the model reads (RFC §21.5)."""
        ...


async def run_tool_call(
    host: ToolCallHost, call: RealtimeToolCall, door: ToolCallDoor
) -> ToolOutcome:
    """Serve *call* and deliver its outcome through *door*, once, reported once.

    Never raises but a cancellation: a step that fails is the call's failure,
    which the model reads as one and the observers receive.
    """
    try:
        outcome = await _decide(host, call, door)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "Tool call %s(%s) failed on channel %s", call.name, call.call_id, host.channel_id
        )
        outcome = failed_outcome(call, exc)
    try:
        return await finish_tool_call(host, call, door, outcome)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "Could not deliver the outcome of tool call %s(%s) on channel %s",
            call.name,
            call.call_id,
            host.channel_id,
        )
        # A call that already failed keeps its own cause; the delivery's
        # failure is the outcome of a call that had none.
        failed = outcome if outcome.failed else failed_outcome(call, exc)
        await report_failed_call(host, call, failed)
        return failed


async def _decide(host: ToolCallHost, call: RealtimeToolCall, door: ToolCallDoor) -> ToolOutcome:
    """The gate, then the serving: *call*'s outcome before its delivery."""
    if call.unreadable is not None:
        # Nothing runs on arguments that do not read (RFC §6.4, §12.4).
        return ToolOutcome(OutcomeKind.REFUSED, call.unreadable)
    if host._call_ended(call):
        # Whoever issued it is gone: no gate runs for it.
        return _ended_outcome(call)
    denial, carrying = await host._authorize_call(call, door)
    if denial is not None:
        return ToolOutcome(OutcomeKind.REFUSED, denial.body, detail=denial.detail)
    if host._call_ended(call):
        return _ended_outcome(call)
    if door.channel_serves:
        served = await host._serve_channel_tool(call, door, carrying)
        if served is not None:
            return served
    return await serve_tool_call(host, call, carrying)


async def serve_tool_call(
    host: ToolCallHost,
    call: RealtimeToolCall,
    carrying: RoomContext | None,
    answer_with: Callable[[], Awaitable[str]] | None = None,
) -> ToolOutcome:
    """The answer to a gated *call*, as ON_TOOL_CALL leaves it: the host's
    handler's, or *answer_with*'s for one of the channel's own tools."""
    try:
        answer = await (
            answer_with() if answer_with is not None else host._answer_call(call, carrying)
        )
    except UnservedToolCallError:
        # Nothing served it: the hooks may still (RFC §21.4).
        outcome = ToolOutcome(OutcomeKind.UNSERVED, unserved_tool_error(call.name))
    except ToolRefusedError as refusal:
        # A refusal in the handler's words, which the model reads.
        return ToolOutcome(OutcomeKind.REFUSED, refusal.message)
    else:
        outcome = ToolOutcome(OutcomeKind.SERVED, answer)
    return await judge_tool_call(host, call, outcome, carrying)


async def judge_tool_call(
    host: ToolCallHost,
    call: RealtimeToolCall,
    outcome: ToolOutcome,
    carrying: RoomContext | None,
    *,
    admit: Callable[[], bool] | None = None,
) -> ToolOutcome:
    """*outcome* once ON_TOOL_CALL's SYNC chain judged it, read as on every
    channel (RFC §9.3): a block withholds the result, a hook's result replaces
    the handler's or serves a call nothing served, and a call nothing served
    failed. The judgement reports a served or blocked call to the observers,
    claiming the call's one report before it does. *admit*, asked after the
    chain and before the observers, can keep that report back: the caller
    then reports what it delivers instead."""
    served = outcome.result if outcome.kind is OutcomeKind.SERVED else None
    framework = host._tool_framework(call)
    verdict: ToolCallVerdict | None = None
    if framework is not None:
        event = host._tool_event(call, None if served is None else str(served))
        if served is not None:
            # The handler's structured copy, which the SYNC chain may replace
            # or clear and the observers then receive (RFC §9.3).
            event = replace(event, structured_content=call.structured_content)

        def claim() -> bool:
            return (admit is None or admit()) and call.claim_report()

        verdict = await framework._judge_tool_call(
            event, host.channel_id, carrying=carrying, claim=claim
        )
        # Don't fuse hook dispatch with the delivery into one loop step.
        await asyncio.sleep(0)
    reading = read_tool_call_verdict(call.name, verdict, served)
    kind = read_outcome(reading)
    if kind is OutcomeKind.UNSERVED:
        detail = verdict.error_detail if verdict is not None else None
        return ToolOutcome(kind, unserved_tool_error(call.name), detail=detail)
    return ToolOutcome(kind, reading.result)


async def finish_tool_call(
    host: ToolCallHost, call: RealtimeToolCall, door: ToolCallDoor, outcome: ToolOutcome
) -> ToolOutcome:
    """Bound *outcome*, deliver it once and report a failure once (RFC §12.4).

    The delivery precedes the report: the provider holds a turn open on the
    result, and an observer must not stand in front of it.
    """
    bounded = replace(outcome, result=host._bound_call_result(call, result_text(outcome.result)))
    await deliver_once(call, door, bounded)
    if outcome.failed:
        # The bound is the model's copy: a refusal's observers receive its raw
        # message (RFC §21.5).
        await report_failed_call(host, call, outcome)
    return bounded


async def deliver_once(call: RealtimeToolCall, door: ToolCallDoor, outcome: ToolOutcome) -> bool:
    """Deliver *outcome* through *door* unless *call*'s one result already went
    out (RFC §12.4); whether it reached whoever waits for it.

    The call counts as delivered from here on: a cancellation that lands while
    the result goes out, or a step that fails after it, adds no second outcome.
    """
    if call.delivered:
        return False
    call.delivered = True
    return await door.deliver(call, outcome)


async def refuse_duplicate_call(host: ToolCallHost, call: RealtimeToolCall) -> None:
    """Report a call whose id names a call still in flight, sending nothing:
    the id's one result is the first call's, which runs on (RFC §12.4)."""
    logger.warning(
        "Tool call %s(%s) arrived while a call with its id is in flight on channel %s; "
        "refused, the first one runs on",
        call.name,
        call.call_id,
        host.channel_id,
    )
    body = json.dumps({"error": f"Tool call '{call.call_id}' is already running"})
    await report_failed_call(host, call, ToolOutcome(OutcomeKind.REFUSED, body))


async def submit_tool_outcome(
    provider: RealtimeVoiceProvider,
    session: VoiceSession,
    call_id: str,
    result: str,
    *,
    failed: bool,
) -> None:
    """Send a call's result through *provider*, as an error when the call
    failed, for a protocol that can say so (RFC §12.4)."""
    submit = provider.submit_tool_error if failed else provider.submit_tool_result
    await submit(session, call_id, result)


async def report_cancelled_call(host: ToolCallHost, call: RealtimeToolCall, why: str) -> None:
    """Report *call*, interrupted before its result, once, as cancelled (RFC §9.3)."""
    body = cancelled_tool_error(call.name, f"{why} before its result; nothing was sent.")
    await report_failed_call(host, call, ToolOutcome(OutcomeKind.CANCELLED, body))


async def report_interrupted_calls(
    host: ToolCallHost, calls: list[RealtimeToolCall], why: str
) -> None:
    """Report each call an ending interrupted, once: as cancelled before its
    result went out, and with what the model read when the ending cut its
    report short (RFC §9.3, §12.4)."""
    await asyncio.gather(*(_report_interrupted(host, call, why) for call in calls))


async def _report_interrupted(host: ToolCallHost, call: RealtimeToolCall, why: str) -> None:
    if not call.delivered:
        await report_cancelled_call(host, call, why)
    elif call.read is not None:
        await _report_read_call(host, call, call.read)


async def _report_read_call(host: ToolCallHost, call: RealtimeToolCall, read: str) -> None:
    """Tell ON_TOOL_CALL's observers alone what the model read of *call*,
    unless its report was already made (RFC §9.3)."""
    framework = host._tool_framework(call)
    if framework is None or not call.claim_report():
        return
    try:
        await framework._observe_failed_tool_call(host._tool_event(call, read), host.channel_id)
    except Exception:
        logger.warning("ON_TOOL_CALL observation failed for tool %s", call.name, exc_info=True)


async def report_failed_call(
    host: ToolCallHost, call: RealtimeToolCall, outcome: ToolOutcome
) -> None:
    """Tell ON_TOOL_CALL's observers *call* failed, was refused or was
    cancelled, unless its outcome was already reported (RFC §9.3)."""
    framework = host._tool_framework(call)
    if framework is None or not call.claim_report():
        return
    event = replace(
        host._tool_event(call, result_text(outcome.result)),
        is_error=True,
        cancelled=outcome.kind is OutcomeKind.CANCELLED,
        error_detail=outcome.detail,
    )
    try:
        await framework._observe_failed_tool_call(event, host.channel_id)
    except Exception:
        logger.warning("ON_TOOL_CALL observation failed for tool %s", call.name, exc_info=True)


def failed_outcome(call: RealtimeToolCall, exc: BaseException) -> ToolOutcome:
    """The outcome of a call that raised: the model reads the tool's failure
    and its class, the observers the message (RFC §9.3)."""
    return ToolOutcome(
        OutcomeKind.FAILED, tool_failure(call.name, exc), detail=failure_detail(exc)
    )


async def tool_loop_context(
    framework: RoomKit | None,
    room_id: str | None,
    *,
    actor_id: str | None,
    chain_depth: int,
    room: Room | None = None,
) -> _ToolLoopContext:
    """The per-call context a handler reads through ``roomkit.tools`` (RFC §21.4).

    A realtime door runs no turn, so the context is built around the handler
    call: the call's room and actor, the chain depth of the answer that issued
    it, no response record to merge (``has_turn`` is off, so
    ``current_response_metadata()`` answers ``None``), and the Room as loaded
    for this call: *room* when a gate already loaded it, one indexed read
    under the framework's lease otherwise. A handler shared with an
    ``AIChannel`` then answers the same questions on every path.
    """
    ctx = _ToolLoopContext()
    ctx.has_turn = False
    ctx.room_id = room_id
    ctx.actor_id = actor_id
    ctx.chain_depth = chain_depth
    if room is not None:
        ctx.room = room
    elif framework is not None and room_id:
        with framework._resource_lease():
            ctx.room = await framework.store.get_room(room_id)
    return ctx


@contextlib.contextmanager
def serving_tool_call(
    call: RealtimeToolCall, channel_id: str, loop_ctx: _ToolLoopContext
) -> Iterator[None]:
    """Run a handler inside *call*'s tool call context: ``current_tool_call()``
    names the call, its room and the channel, as on every channel (RFC §21.4).
    The structured copy the handler leaves there stays on *call*, for
    ON_TOOL_CALL to see (RFC §9.3)."""
    call_ctx = ToolCallContext(
        room_id=loop_ctx.room_id or "", tool_call_id=call.call_id, channel_id=channel_id
    )
    call_token = _current_tool_call.set(call_ctx)
    loop_token = _current_loop_ctx.set(loop_ctx)
    try:
        yield
        call.structured_content = call_ctx.structured_content
    finally:
        _current_loop_ctx.reset(loop_token)
        _current_tool_call.reset(call_token)


def _ended_outcome(call: RealtimeToolCall) -> ToolOutcome:
    """The outcome of a call whose issuer is gone before it was served."""
    body = cancelled_tool_error(call.name, "The session ended before its result.")
    return ToolOutcome(OutcomeKind.CANCELLED, body)
