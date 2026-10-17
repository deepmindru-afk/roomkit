"""One turn's human-input requests on its executor (RFC §10.6.5).

A request the channel's handler opens is sent to the owner as an
``input_request`` before ``ON_USER_INPUT_REQUIRED`` fires; the owner's
``input_answer`` settles it here while it is still open. A request settled
here (an answer the channel's handler received, a hook's block, its bound, a
release) is sent as an ``input_closed``, and the owner answers it no more.
Once the turn ends, nothing more is sent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns.wire import InputClosed, InputRequest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.models.pending_input import PendingInput
    from roomkit.tools._turn_inputs import InputSettler
    from roomkit.turns.wire import Frame, InputAnswer

logger = logging.getLogger("roomkit.turns")


class ExecutorInputs:
    """The open human-input requests of one turn, and their way to its owner."""

    def __init__(self, send: Callable[[Frame], Awaitable[int]]) -> None:
        self._send = send
        self._loop = asyncio.get_running_loop()
        self._open: dict[str, InputSettler] = {}
        self._running = True
        self._sending: set[asyncio.Task[None]] = set()

    @property
    def open(self) -> bool:
        return self._running

    def close(self) -> None:
        """The turn ended: its requests are the handler's alone."""
        self._running = False
        self._open.clear()

    async def opened(self, pending: PendingInput, settle: InputSettler) -> None:
        if not self._running:
            return
        self._open[pending.pending_id] = settle
        await self._send(
            InputRequest(
                request_id=pending.pending_id,
                tool_name=pending.tool_name,
                arguments=pending.arguments,
            )
        )

    def closed(self, pending_id: str, reason: str) -> None:
        # The handler may be answered from another thread: the frame is
        # sent from the turn's loop.
        self._loop.call_soon_threadsafe(self._send_closed, pending_id, reason)

    def answer(self, message: InputAnswer) -> None:
        """The owner's outcome for a request: it settles here, if still open."""
        settle = self._open.pop(message.request_id, None)
        if settle is None:
            return
        if message.outcome == "answered":
            settle.settle(message.request_id, value=message.value or "", reason=None)
        else:
            settle.settle(message.request_id, value=None, reason=message.reason or "")

    def _send_closed(self, pending_id: str, reason: str) -> None:
        if not self._running or self._open.pop(pending_id, None) is None:
            return
        task = self._loop.create_task(
            self._send_quietly(InputClosed(request_id=pending_id, reason=reason))
        )
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)

    async def _send_quietly(self, frame: Frame) -> None:
        try:
            await self._send(frame)
        except TurnLinkError:
            pass
        except Exception:
            logger.warning("A human-input close could not reach the room owner", exc_info=True)
