"""Bench backends shared by the runnable realtime examples."""

from __future__ import annotations

from typing import Any

from roomkit import ScenarioVoiceBackend
from roomkit.voice.base import VoiceSession

__all__ = ["IncomingScenarioBackend"]


class IncomingScenarioBackend(ScenarioVoiceBackend):
    """Accept an in-process connection while retaining the bench's capture.

    An example that drives a real provider needs no transport: the session is
    opened in process and the audio the model produces is captured to a WAV
    for inspection. The base backend expects a client to connect, so the
    accept is reduced to registering the session.
    """

    async def accept(self, session: VoiceSession, connection: Any) -> None:
        self._sessions[session.id] = session
