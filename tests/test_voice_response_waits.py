"""A response waits for the playback under way instead of cutting it (RFC §12.2).

The agent's words are cut by the person, never by its own next response: a
background task's result relayed while it finishes a sentence waited for no
one and cut it off. ``say()`` still takes the floor at once.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession
from roomkit.voice.tts.base import TTSProvider

_RATE = 16000


class _TimedTTS(TTSProvider):
    """Each text plays for ``seconds``; records when each one starts and ends."""

    def __init__(self, seconds: float = 0.3) -> None:
        self._seconds = seconds
        self.played: list[tuple[str, float, float]] = []

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        start = time.monotonic()
        yield AudioChunk(data=b"\x00\x00" * int(_RATE * self._seconds), sample_rate=_RATE)
        await asyncio.sleep(self._seconds)
        self.played.append((text, start, time.monotonic()))


async def _room(tts: TTSProvider) -> tuple[RoomKit, VoiceChannel, VoiceSession, list[Any]]:
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel("voice-1", tts=tts, backend=backend)
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    channel.bind_session(
        session,
        "r1",
        ChannelBinding(room_id="r1", channel_id="voice-1", channel_type=ChannelType.VOICE),
    )
    cancelled: list[Any] = []

    @kit.hook(HookTrigger.ON_TTS_CANCELLED, HookExecution.ASYNC)
    async def on_cancelled(event: Any, context: Any) -> None:
        cancelled.append(event)

    return kit, channel, session, cancelled


async def test_responses_play_one_after_the_other_whole() -> None:
    tts = _TimedTTS()
    kit, channel, session, cancelled = await _room(tts)
    texts = ["Je lance ça en arrière-plan.", "Voici les campagnes.", "Autre chose ?"]

    tasks = []
    for text in texts:
        tasks.append(asyncio.create_task(channel._send_tts(session, text, response=True)))  # noqa: SLF001
        await asyncio.sleep(0.05)
    await asyncio.gather(*tasks)
    await kit.close()

    assert cancelled == []
    assert [text for text, _start, _end in tts.played] == texts
    for (_, _, ended), (_, started, _) in zip(tts.played, tts.played[1:], strict=False):
        assert started >= ended  # never two at once


async def test_say_still_takes_the_floor_at_once() -> None:
    tts = _TimedTTS(seconds=1.0)
    kit, channel, session, cancelled = await _room(tts)

    answer = asyncio.create_task(
        channel._send_tts(session, "A long answer.", response=True)  # noqa: SLF001
    )
    await asyncio.sleep(0.2)
    await channel.say(session, "An announcement.")
    await answer
    await kit.close()

    assert [event.reason for event in cancelled] == ["new_tts"]


async def test_the_last_playbacks_drain_leaves_the_next_one_playing() -> None:
    """Without an AEC a playback sent whole is kept for its echo to drain; the
    response that took over is no longer its to end when that delay is up."""
    kit, channel, session, cancelled = await _room(_TimedTTS(seconds=0.1))
    await channel._send_tts(session, "Je lance ça.", response=True)  # noqa: SLF001
    channel._tts = _TimedTTS(seconds=2.5)  # noqa: SLF001
    result = asyncio.create_task(
        channel._send_tts(session, "Voici les campagnes.", response=True)  # noqa: SLF001
    )
    await asyncio.sleep(2.2)  # past the first one's drain

    playing = channel._playing_sessions.get(session.id)  # noqa: SLF001
    assert playing is not None and playing.text == "Voici les campagnes."
    await result
    await kit.close()
    assert cancelled == []
