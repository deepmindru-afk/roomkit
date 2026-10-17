"""One turn's human-input requests on its executor (RFC §10.6.5).

A request the channel's handler opens is sent to the owner as an
``input_request`` before ``ON_USER_INPUT_REQUIRED`` fires; the owner's
``input_answer`` settles it here. A request that settles here without the
owner's answer (a hook's block, its bound, a release) is sent as an
``input_closed``, and an answer or a rejection the channel's handler receives
is forwarded to the owner, which records it and answers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from roomkit.turns.wire import InputClosed, InputRequest

if TYPE_CHECKING:
    from roomkit.models.pending_input import PendingInput
    from roomkit.tools._turn_inputs import InputSettler
    from roomkit.turns.wire import Frame, InputAnswer

logger = logging.getLogger("roomkit.turns")


class ExecutorInputs:
    """The open human-input requests of one turn, and their way to its owner."""

    def __init__(
        self,
        send: Callable[[Frame], Awaitable[int]],
        host_call: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> None:
        self._send = send
        self._host_call = host_call
        self._open: dict[str, InputSettler] = {}
        self._sending: set[asyncio.Task[Any]] = set()

    async def opened(self, pending: PendingInput, settle: InputSettler) -> None:
        self._open[pending.pending_id] = settle
        await self._send(
            InputRequest(
                request_id=pending.pending_id,
                tool_name=pending.tool_name,
                arguments=pending.arguments,
            )
        )

    def closed(self, pending_id: str, reason: str) -> None:
        if self._open.pop(pending_id, None) is not None:
            self._spawn(self._send(InputClosed(request_id=pending_id, reason=reason)))

    def forward(self, pending_id: str, *, value: str | None, reason: str | None) -> None:
        if value is not None:
            self._spawn(
                self._host_call("answer_input", {"request_id": pending_id, "value": value})
            )
        else:
            arguments = {"request_id": pending_id, "reason": reason or ""}
            self._spawn(self._host_call("reject_input", arguments))

    def answer(self, message: InputAnswer) -> None:
        """The owner's outcome for a request: it settles here."""
        settle = self._open.pop(message.request_id, None)
        if settle is None:
            return
        if message.outcome == "answered":
            settle.settle(message.request_id, value=message.value or "", reason=None)
        else:
            settle.settle(message.request_id, value=None, reason=message.reason or "")

    def _spawn(self, work: Awaitable[Any]) -> None:
        task = asyncio.get_running_loop().create_task(_logged(work))
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)


async def _logged(work: Awaitable[Any]) -> None:
    try:
        await work
    except Exception:
        logger.warning("A human-input message could not reach the room owner", exc_info=True)
