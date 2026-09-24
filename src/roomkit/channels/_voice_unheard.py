"""The routed voice turn whose response the user has not heard yet (RFC §12.3.12).

A turn is routed, and until the first audio of its response reaches the
transport, nobody has heard it. A user who resumes speaking in that window is
still on their sentence: the response is held while they speak, then either
superseded by what they added or released to play as it would have.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

from roomkit.models.delivery import DeliveryHandle


@dataclass
class _UnheardTurn:
    handle: DeliveryHandle
    held_since: float | None = None
    """Monotonic onset of the speech holding the response, None when not held."""
    resumed_ms: float = 0.0
    """How long the speech that held the response lasted, once it ended."""
    heard: bool = False
    released: asyncio.Event = field(default_factory=asyncio.Event)


class UnheardTurns:
    """Per session, the routed turn not yet heard, held while the user resumes.

    ``hold`` and ``note_speech_end`` are called from the VAD, which may run on
    the audio thread; everything else runs on the event loop, which is the only
    place an ``asyncio.Event`` is set.
    """

    def __init__(self) -> None:
        self._turns: dict[str, _UnheardTurn] = {}
        self._lock = threading.Lock()

    def register(self, session_id: str, handle: DeliveryHandle, *, speaking: bool) -> None:
        """Track a turn just routed; *speaking* holds it at once (speech already started)."""
        turn = _UnheardTurn(handle=handle)
        if speaking:
            turn.held_since = time.monotonic()
        else:
            turn.released.set()
        with self._lock:
            self._turns[session_id] = turn

    def discard(self, session_id: str, handle: DeliveryHandle) -> None:
        """Forget the session's turn once its delivery ended, if it is still *handle*'s."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is not None and turn.handle is handle:
                del self._turns[session_id]
        if turn is not None and turn.handle is handle:
            turn.released.set()

    def hold(self, session_id: str) -> bool:
        """Speech started: hold the session's unheard response. False when none is unheard."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is None or turn.heard:
                return False
            if turn.held_since is None:
                turn.held_since = time.monotonic()
                turn.resumed_ms = 0.0
                # A fresh event: clearing one from this thread is not safe.
                turn.released = asyncio.Event()
            return True

    def note_speech_end(self, session_id: str) -> None:
        """Speech ended: record how long the speech holding the response lasted."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is not None and turn.held_since is not None:
                turn.resumed_ms = (time.monotonic() - turn.held_since) * 1000

    async def wait_to_play(self, session_id: str) -> None:
        """Before the response's first audio: wait while it is held, then mark it heard."""
        while True:
            with self._lock:
                turn = self._turns.get(session_id)
                if turn is None:
                    return
                if turn.held_since is None:
                    turn.heard = True
                    return
                released = turn.released
            await released.wait()

    def take_superseded(self, session_id: str, min_speech_ms: float) -> DeliveryHandle | None:
        """The held turn the ended speech supersedes, removed from tracking; else None."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is None or turn.held_since is None or turn.resumed_ms < min_speech_ms:
                return None
            del self._turns[session_id]
        turn.released.set()
        return turn.handle

    def release(self, session_id: str) -> None:
        """The speech did not supersede the turn: its held response plays."""
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is None or turn.held_since is None:
                return
            turn.held_since = None
            released = turn.released
        released.set()
