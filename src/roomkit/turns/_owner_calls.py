"""The calls a turn makes to its room owner, on the owner (RFC §10.6.5).

A ``tool_call`` runs the handler the owner keeps for the tool, and a
``host_call`` a verb of the turn's ``TurnHost``, each as a task of its own,
in a copy of the execution context of the owner task reading the turn when
the call arrives (the delivery target awaiting the output, or the reader
awaiting an item), under the turn's id: the call holds the reentrancy that
task holds, and a close it calls spares the turn it serves, as when the call
ran in the owner's own code. Its answer goes back once, as a ``tool_result``
or a ``host_result``. A call the executor withdraws is cancelled, and so is
every call of a turn that ends for the owner; the client that submitted the
turn waits for them when it closes.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import TYPE_CHECKING

from roomkit.core.exceptions import ToolRefusedError, UnservedToolCallError
from roomkit.tools.result import as_tool_result
from roomkit.turns._codec import encode_result, error_record
from roomkit.turns.base import enter_turn
from roomkit.turns.wire import HostCall, HostResult, ToolCall, ToolResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.turns.base import OwnerHost, OwnerTools
    from roomkit.turns.wire import Control, TurnRequest

logger = logging.getLogger("roomkit.turns")


class OwnerCalls:
    """The owner-served calls of one turn, each a task the turn can cancel.

    *held* is the set of the client's running calls, every turn's, which the
    client waits for when it closes.
    """

    def __init__(
        self,
        request: TurnRequest,
        tools: OwnerTools | None,
        host: OwnerHost | None,
        send: Callable[[Control], Awaitable[None]],
        held: set[asyncio.Task[None]],
    ) -> None:
        self._request = request
        self._tools = tools
        self._host = host
        self._send = send
        self._held = held
        self._running: dict[str, asyncio.Task[None]] = {}

    def start(self, call: ToolCall | HostCall) -> None:
        """Run *call* in a copy of the current task's context."""
        context = contextvars.copy_context()
        context.run(enter_turn, self._request.turn_id)
        task = asyncio.get_running_loop().create_task(
            self._answer(call), context=context, name=f"roomkit-owner-call-{call.call_id}"
        )
        self._running[call.call_id] = task
        self._held.add(task)
        task.add_done_callback(lambda _task: self._forget(call.call_id, task))

    def withdraw(self, call_id: str) -> None:
        """The executor withdrew the call: its handler is cancelled."""
        task = self._running.get(call_id)
        if task is not None and not task.cancelling():
            task.cancel()

    def cancel_all(self) -> None:
        """The turn is over for the owner: every call still running is
        cancelled, but the one running this code (a close it called)."""
        current = asyncio.current_task()
        for task in list(self._running.values()):
            if task is not current and not task.cancelling():
                task.cancel()

    def _forget(self, call_id: str, task: asyncio.Task[None]) -> None:
        self._running.pop(call_id, None)
        self._held.discard(task)

    async def _answer(self, call: ToolCall | HostCall) -> None:
        try:
            if isinstance(call, HostCall):
                result: Control = await self._host_result(call)
            else:
                result = await self._result(call)
        except asyncio.CancelledError as cancelled:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            # The call ended on a cancellation of something it awaited, not
            # of the call: the call failed, and the turn goes on.
            result = _failed(call, cancelled)
        try:
            await self._send(result)
        except (TypeError, ValueError) as exc:
            what = f"verb {call.verb!r}" if isinstance(call, HostCall) else f"tool {call.name!r}"
            refusal = TypeError(f"The result of {what} cannot cross the turn backend")
            refusal.__cause__ = exc
            await self._send(_failed(call, refusal))

    async def _host_result(self, call: HostCall) -> HostResult:
        """What the verb of *call* made, as it crosses back."""
        try:
            if self._host is None:
                raise ValueError(f"No verb {call.verb!r} is served by the room owner")
            value = await self._host.call(self._request, call.verb, call.arguments)
        except Exception as exc:
            logger.debug("Host call %s failed", call.verb, exc_info=True)
            return HostResult(call_id=call.call_id, error=error_record(exc))
        return HostResult(call_id=call.call_id, result=value)

    async def _result(self, call: ToolCall) -> ToolResult:
        """What the handler made of *call*, as it crosses back."""
        try:
            if self._tools is None:
                raise UnservedToolCallError(f"No tool {call.name!r} is served by the room owner")
            answer, structured = await self._tools.serve(self._request, call)
        except UnservedToolCallError as exc:
            return ToolResult(call_id=call.call_id, outcome="unserved", error=error_record(exc))
        except ToolRefusedError as exc:
            return ToolResult(call_id=call.call_id, outcome="refused", error=error_record(exc))
        except Exception as exc:
            logger.debug("Owner-served call %s failed", call.name, exc_info=True)
            return ToolResult(call_id=call.call_id, outcome="failed", error=error_record(exc))
        return ToolResult(
            call_id=call.call_id,
            outcome="served",
            result=encode_result(as_tool_result(answer)),
            structured_content=structured,
        )


def _failed(call: ToolCall | HostCall, exc: BaseException) -> Control:
    """The answer of a call that failed with *exc*."""
    if isinstance(call, HostCall):
        return HostResult(call_id=call.call_id, error=error_record(exc))
    return ToolResult(call_id=call.call_id, outcome="failed", error=error_record(exc))
