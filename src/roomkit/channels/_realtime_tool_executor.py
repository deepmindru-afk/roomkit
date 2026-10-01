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
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from roomkit.core.exceptions import ToolRefusedError, UnservedToolCallError
from roomkit.models.tool_call import ToolCallVerdict
from roomkit.tools._outcome import OutcomeKind, ToolOutcome, read_outcome
from roomkit.tools.result import (
    GateRefusal,
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
    from roomkit.models.tool_call import ToolCallEvent

logger = logging.getLogger("roomkit.channels.realtime_tools")


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

    channel_id: str

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
    denial, carrying = await host._authorize_call(call, door)
    if denial is not None:
        return ToolOutcome(OutcomeKind.REFUSED, denial.body, detail=denial.detail)
    if host._call_ended(call):
        return ToolOutcome(OutcomeKind.CANCELLED, unserved_tool_error(call.name))
    if door.channel_serves:
        served = await host._serve_channel_tool(call, door, carrying)
        if served is not None:
            return served
    return await serve_tool_call(host, call, carrying)


async def serve_tool_call(
    host: ToolCallHost, call: RealtimeToolCall, carrying: RoomContext | None
) -> ToolOutcome:
    """The answer to a gated *call*, as ON_TOOL_CALL leaves it."""
    try:
        answer = await host._answer_call(call, carrying)
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
) -> ToolOutcome:
    """*outcome* once ON_TOOL_CALL's SYNC chain judged it, read as on every
    channel (RFC §9.3): a block withholds the result, a hook's result replaces
    the handler's or serves a call nothing served, and a call nothing served
    failed. The judgement reports a served or blocked call to the observers."""
    served = outcome.result if outcome.kind is OutcomeKind.SERVED else None
    framework = host._tool_framework(call)
    verdict: ToolCallVerdict | None = None
    if framework is not None:
        event = host._tool_event(call, None if served is None else str(served))
        verdict = await framework._judge_tool_call(event, host.channel_id, carrying=carrying)
        # Don't fuse hook dispatch with the delivery into one loop step.
        await asyncio.sleep(0)
    reading = read_tool_call_verdict(call.name, verdict, served)
    kind = read_outcome(reading)
    if kind is OutcomeKind.UNSERVED:
        detail = verdict.error_detail if verdict is not None else None
        return ToolOutcome(kind, unserved_tool_error(call.name), detail=detail)
    if framework is not None:
        call.reported = True
    return ToolOutcome(kind, reading.result)


async def finish_tool_call(
    host: ToolCallHost, call: RealtimeToolCall, door: ToolCallDoor, outcome: ToolOutcome
) -> ToolOutcome:
    """Bound *outcome*, deliver it once and report a failure once (RFC §12.4).

    The delivery precedes the report: the provider holds a turn open on the
    result, and an observer must not stand in front of it.
    """
    outcome = replace(outcome, result=host._bound_call_result(call, result_text(outcome.result)))
    if not call.delivered:
        call.delivered = True
        await door.deliver(call, outcome)
    if outcome.failed:
        await report_failed_call(host, call, outcome)
    return outcome


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
        logger.debug("ON_TOOL_CALL observation failed for tool %s", call.name, exc_info=True)


def failed_outcome(call: RealtimeToolCall, exc: BaseException) -> ToolOutcome:
    """The outcome of a call that raised: the model reads the tool's failure
    and its class, the observers the message (RFC §9.3)."""
    return ToolOutcome(
        OutcomeKind.FAILED, tool_failure(call.name, exc), detail=failure_detail(exc)
    )
