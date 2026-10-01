"""What a LiveKit bot session hands the backend: its callback fanout, as functions.

Its own module because every part of a session reads it — the room view emits
through it, the pumps deliver frames to it, the departure reports the end
through it — and none of them may import the session that composes them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from roomkit.conference._livekit_media import AudioSink, VideoSink
from roomkit.conference.models import BotSession, ConferenceParticipant, ConferenceTrack


@dataclass(frozen=True)
class ConferenceEmissions:
    """The backend's callback fanout, handed to a session as plain functions.

    A session emits without holding the backend, so what it needs from the
    backend is stated here rather than discovered by reaching into it.
    """

    participant_joined: Callable[[str, ConferenceParticipant], Awaitable[None]]
    participant_left: Callable[[str, ConferenceParticipant], Awaitable[None]]
    track_published: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_unpublished: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_muted: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_unmuted: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_audio: AudioSink
    track_video: VideoSink
    active_speaker_changed: Callable[[str, str], Awaitable[None]]
    connection_quality: Callable[[str, str, str], Awaitable[None]]
    bot_session_ended: Callable[[BotSession, str], Awaitable[None]]
