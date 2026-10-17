"""The calls a turn makes to its room owner, on the owner (RFC §10.6.5).

A ``tool_call`` runs the handler the owner keeps for the tool, as a task of
its own, in a copy of the execution context of the owner task reading the turn
when the call arrives (the delivery target awaiting the output, or the reader
awaiting an item): the handler holds the reentrancy that task holds, as the
call did when it ran in the owner's own code. Its answer goes back once, as a
``tool_result``. A call the executor withdraws is cancelled, and so is every
call of a turn that ends for the owner.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import TYPE_CHECKING

from roomkit.core.exceptions import UnservedToolCallError
from roomkit.tools.result import as_tool_result
from roomkit.turns._codec import encode_result, error_record
from roomkit.turns.wire import ToolResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.turns.base import OwnerTools
    from roomkit.turns.wire import Control, ToolCall, TurnRequest

logger = logging.getLogger("roomkit.turns")


class OwnerCalls:
    """The owner-served calls of one turn, each a task the turn can cancel."""

    def __init__(
        self,
        request: TurnRequest,
        tools: OwnerTools | None,
        send: Callable[[Control], Awaitable[None]],
    ) -> None:
        self._request = request
        self._tools = tools
        self._send = send
        self._running: dict[str, asyncio.Task[None]] = {}

    def start(self, call: ToolCall) -> None:
        """Run *call*'s handler in a copy of the current task's context."""
        task = asyncio.get_running_loop().create_task(
            self._answer(call),
            context=contextvars.copy_context(),
            name=f"roomkit-owner-call-{call.call_id}",
        )
        self._running[call.call_id] = task
        task.add_done_callback(lambda _task: self._running.pop(call.call_id, None))

    def withdraw(self, call_id: str) -> None:
        """The executor withdrew the call: its handler is cancelled."""
        task = self._running.get(call_id)
        if task is not None:
            task.cancel()

    def cancel_all(self) -> None:
        """The turn is over for the owner: every call still running is cancelled."""
        for task in list(self._running.values()):
            task.cancel()

    async def _answer(self, call: ToolCall) -> None:
        result = await self._result(call)
        try:
            await self._send(result)
        except (TypeError, ValueError) as exc:
            refusal = TypeError(f"The result of tool {call.name!r} cannot cross the turn backend")
            refusal.__cause__ = exc
            await self._send(
                ToolResult(call_id=call.call_id, outcome="failed", error=error_record(refusal))
            )

    async def _result(self, call: ToolCall) -> ToolResult:
        """What the handler made of *call*, as it crosses back."""
        try:
            if self._tools is None:
                raise UnservedToolCallError(f"No tool {call.name!r} is served by the room owner")
            answer, structured = await self._tools.serve(self._request, call)
        except UnservedToolCallError as exc:
            return ToolResult(call_id=call.call_id, outcome="unserved", error=error_record(exc))
        except Exception as exc:
            logger.debug("Owner-served call %s failed", call.name, exc_info=True)
            return ToolResult(call_id=call.call_id, outcome="failed", error=error_record(exc))
        return ToolResult(
            call_id=call.call_id,
            outcome="served",
            result=encode_result(as_tool_result(answer)),
            structured_content=structured,
        )
