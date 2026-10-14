"""A response waits for the playback under way instead of cutting it (RFC §12.2).

The agent's words are cut by the person, never by its own next response: a
background task's result relayed while it finishes a sentence waited for no
one and cut it off. ``say()`` still takes the floor at once.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
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


class _PacedBackend(MockVoiceBackend):
    """Sends audio in real time and stops it on ``cancel_audio``; logs each
    playback as (text, start, end, cut). Its transcription send yields, as a
    websocket's does."""

    def __init__(self) -> None:
        super().__init__(capabilities=VoiceCapability.INTERRUPTION)
        self.channel: VoiceChannel | None = None
        self.log: list[tuple[str, float, float, bool]] = []
        self._cuts: dict[str, int] = {}

    async def send_transcription(
        self, session: VoiceSession, text: str, role: str = "user"
    ) -> None:
        await asyncio.sleep(0)
        await super().send_transcription(session, text, role)

    async def cancel_audio(self, session: VoiceSession) -> bool:
        self._cuts[session.id] = self._cuts.get(session.id, 0) + 1
        return True

    async def send_audio(self, session: VoiceSession, audio: Any) -> None:
        cuts = self._cuts.get(session.id, 0)
        assert self.channel is not None
        playing = self.channel._playing_sessions.get(session.id)  # noqa: SLF001
        text = playing.text if playing is not None else "?"
        start, cut = time.monotonic(), False
        async for chunk in audio:
            ends = time.monotonic() + len(chunk.data) / 2 / chunk.sample_rate
            while time.monotonic() < ends and not cut:
                cut = self._cuts.get(session.id, 0) != cuts
                await asyncio.sleep(0.01)
            if cut:
                break
        self.log.append((text, start, time.monotonic(), cut))

    def heard(self) -> list[str]:
        """What played whole, in order; asserts no two of them overlapped."""
        whole = sorted((entry for entry in self.log if not entry[3]), key=lambda e: e[1])
        for before, after in zip(whole, whole[1:], strict=False):
            assert after[1] >= before[2] - 0.02, f"{after[0]!r} played over {before[0]!r}"
        return [entry[0] for entry in whole]


def _wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_RATE)
        wf.writeframes(b"\x00\x00" * int(_RATE * seconds))
    return buf.getvalue()


async def _room(
    tts: TTSProvider, backend: MockVoiceBackend | None = None
) -> tuple[RoomKit, VoiceChannel, VoiceSession, list[Any]]:
    backend = backend or MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel("voice-1", tts=tts, backend=backend)
    if isinstance(backend, _PacedBackend):
        backend.channel = channel
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
    texts = ["I'm starting that in the background.", "Here are the campaigns.", "Anything else?"]

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
    await channel._send_tts(session, "Starting it.", response=True)  # noqa: SLF001
    channel._tts = _TimedTTS(seconds=2.5)  # noqa: SLF001
    result = asyncio.create_task(
        channel._send_tts(session, "Here are the campaigns.", response=True)  # noqa: SLF001
    )
    await asyncio.sleep(2.2)  # past the first one's drain

    playing = channel._playing_sessions.get(session.id)  # noqa: SLF001
    assert playing is not None and playing.text == "Here are the campaigns."
    await result
    await kit.close()
    assert cancelled == []


class _ChunkTTS(TTSProvider):
    """Each text is one chunk of ``seconds`` of audio; streams one chunk per sentence."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=b"\x00\x00" * int(_RATE * self._seconds), sample_rate=_RATE)

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        async for _sentence in text_stream:
            yield AudioChunk(data=b"\x00\x00" * int(_RATE * self._seconds), sample_rate=_RATE)


def _respond(channel: VoiceChannel, session: VoiceSession, text: str) -> asyncio.Task[None]:
    return asyncio.create_task(channel._send_tts(session, text, response=True))  # noqa: SLF001


async def test_waiting_responses_take_the_session_one_at_a_time() -> None:
    """Woken together when the playback under way is sent, only one of them
    plays: the others wait on it, even past a transcription send that yields."""
    backend = _PacedBackend()
    kit, channel, session, cancelled = await _room(_ChunkTTS(0.3), backend)
    tasks = [_respond(channel, session, "First.")]
    await asyncio.sleep(0.05)
    tasks += [_respond(channel, session, "Second."), _respond(channel, session, "Third.")]
    await asyncio.gather(*tasks)
    await kit.close()

    assert backend.heard() == ["First.", "Second.", "Third."]
    assert cancelled == []


async def test_say_while_a_response_waits_plays_before_it_never_with_it() -> None:
    backend = _PacedBackend()
    kit, channel, session, cancelled = await _room(_ChunkTTS(0.5), backend)
    answer = _respond(channel, session, "A long answer.")
    await asyncio.sleep(0.1)
    result = _respond(channel, session, "The task's result.")
    await asyncio.sleep(0.1)
    await channel.say(session, "An announcement.")
    await asyncio.gather(answer, result)
    await kit.close()

    assert backend.heard() == ["An announcement.", "The task's result."]
    assert [(event.reason, event.text) for event in cancelled] == [("new_tts", "A long answer.")]


async def test_a_response_cuts_a_play_under_way() -> None:
    """``play()`` is the application's audio, not the agent's words: a response
    cuts it, as before, even after a spoken playback left its event set.

    The play takes over the response draining its echo as ``say()`` does,
    reporting it cut."""
    backend = _PacedBackend()
    kit, channel, session, cancelled = await _room(_ChunkTTS(0.1), backend)
    await _respond(channel, session, "Hello.")
    hold = asyncio.create_task(channel.play(session, _wav(3.0), text="Hold music"))
    await asyncio.sleep(0.2)
    await _respond(channel, session, "Your answer.")
    await hold
    await kit.close()

    assert backend.heard() == ["Hello.", "Your answer."]
    assert [(event.reason, event.text) for event in cancelled] == [
        ("new_tts", "Hello."),
        ("new_tts", "Hold music"),
    ]


async def test_a_cut_playbacks_drain_leaves_the_next_one_playing() -> None:
    """Without an AEC, the drain of a playback a barge-in cut does not end the
    response that started in its place."""
    backend = _PacedBackend()
    kit, channel, session, _cancelled = await _room(_ChunkTTS(3.0), backend)
    answer = _respond(channel, session, "A long answer.")
    await asyncio.sleep(0.1)
    result = _respond(channel, session, "The task's result.")
    await asyncio.sleep(0.1)
    await channel.interrupt(session, reason="barge_in")
    await asyncio.sleep(2.4)  # past the cut playback's drain

    playing = channel._playing_sessions.get(session.id)  # noqa: SLF001
    assert playing is not None and playing.text == "The task's result."
    await asyncio.gather(answer, result)
    await kit.close()


async def test_a_streamed_response_leaves_the_session_after_its_drain() -> None:
    """Without an AEC, a streamed response ends after its drain: its whole
    text is set on the playback itself, which the drain ends by identity."""
    kit, channel, session, _cancelled = await _room(_ChunkTTS(0.05))
    binding = ChannelBinding(room_id="r1", channel_id="voice-1", channel_type=ChannelType.VOICE)
    event = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
        content=TextContent(body=""),
    )

    async def sentences() -> AsyncIterator[str]:
        yield "Hello there, this is a test. "
        yield "A second sentence here."

    context = RoomContext(room=Room(id="r1"), bindings=[binding])
    await channel.deliver_stream(sentences(), event, binding, context)
    playing = channel._playing_sessions.get(session.id)  # noqa: SLF001
    assert playing is not None and playing.text.endswith("A second sentence here.")
    await asyncio.sleep(2.3)  # past its drain

    assert channel._playing_sessions.get(session.id) is None  # noqa: SLF001
    await kit.close()


async def test_a_response_that_fails_before_it_plays_holds_up_no_later_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A playback that fails once the session is its, before any audio, is
    let go all the same: the next response plays instead of waiting for good."""
    kit, channel, session, _cancelled = await _room(_ChunkTTS(0.1))

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("context store unavailable")

    monkeypatch.setattr(channel, "_begin_tts_turn", unavailable)
    with pytest.raises(RuntimeError, match="context store unavailable"):
        await _respond(channel, session, "First.")
    monkeypatch.undo()
    await asyncio.wait_for(_respond(channel, session, "Second."), timeout=2.0)
    await kit.close()


async def test_a_play_reads_as_playing_until_its_audio_is_sent() -> None:
    """``play()`` takes the session as a response does: after a spoken
    playback left the session's done event set, a play under way still reads
    as playing, so waiting for playback waits for it, and cutting it is a
    cut, not the end of a drain."""
    backend = _PacedBackend()
    kit, channel, session, _cancelled = await _room(_ChunkTTS(0.1), backend)
    await _respond(channel, session, "Hello.")
    hold = asyncio.create_task(channel.play(session, _wav(1.0), text="Hold music"))
    await asyncio.sleep(0.2)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(channel.wait_playback_done("r1"), timeout=0.2)
    playing = channel._playing_sessions[session.id]  # noqa: SLF001
    await channel.interrupt(session)
    await hold
    await kit.close()

    assert playing.stopped_at is not None


async def test_a_play_cuts_the_playback_under_way() -> None:
    """``play()`` does not wait, as ``say()`` does not: the playback under way
    is cut and reported, never left to play under it."""
    backend = _PacedBackend()
    kit, channel, session, cancelled = await _room(_ChunkTTS(3.0), backend)
    answer = _respond(channel, session, "A long answer.")
    await asyncio.sleep(0.2)
    await channel.play(session, _wav(0.3), text="Hold music")
    await answer
    await kit.close()

    assert backend.heard() == ["Hold music"]
    assert [(event.reason, event.text) for event in cancelled] == [("new_tts", "A long answer.")]


async def test_a_cut_play_reports_what_was_heard_of_it() -> None:
    """A ``play()`` cut short reports the audio heard of it, which joins the
    session's played total the echo guard reads."""
    backend = _PacedBackend()
    kit, channel, session, cancelled = await _room(_ChunkTTS(0.1), backend)
    hold = asyncio.create_task(channel.play(session, _wav(3.0), text="Hold music"))
    await asyncio.sleep(0.5)
    await channel.interrupt(session, reason="barge_in")
    await hold
    await kit.close()

    assert [event.text for event in cancelled] == ["Hold music"]
    assert 300 <= cancelled[0].audio_position_ms <= 1000
    played = channel._session_played_ms[session.id]  # noqa: SLF001
    assert 300 <= played <= 1000


class _StuckBackend(MockVoiceBackend):
    """``send_audio()`` never returns, as into a session already gone."""

    def __init__(self) -> None:
        super().__init__(capabilities=VoiceCapability.INTERRUPTION)
        self.sends = 0

    async def send_audio(self, session: VoiceSession, audio: Any) -> None:
        self.sends += 1
        await asyncio.Event().wait()


async def test_responses_waiting_on_a_session_are_let_go_when_it_is_unbound() -> None:
    backend = _StuckBackend()
    kit, channel, session, _cancelled = await _room(_ChunkTTS(0.1), backend)
    first = _respond(channel, session, "First.")
    await asyncio.sleep(0.05)
    second = _respond(channel, session, "Second.")
    await asyncio.sleep(0.05)
    assert backend.sends == 1

    channel.unbind_session(session)
    await asyncio.sleep(0.05)

    assert backend.sends == 2  # no longer waiting on a send that never returns
    for task in (first, second):
        task.cancel()
    await asyncio.gather(first, second, return_exceptions=True)
    await kit.close()


async def test_held_speech_waits_while_a_response_took_the_session_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an AEC, the drain of a playback sent whole that a response took
    over does not hand the speech held during playback on: the bot is still
    speaking, and that response's own end hands it on."""
    backend = _PacedBackend()
    kit, channel, session, _cancelled = await _room(_ChunkTTS(0.5), backend)
    handed_on_while_speaking: list[bool] = []

    async def hand_on(session_id: str) -> None:
        handed_on_while_speaking.append(session_id in channel._playing_sessions)  # noqa: SLF001

    monkeypatch.setattr(channel, "_flush_queued_speech", hand_on)
    first = _respond(channel, session, "First.")
    await asyncio.sleep(0.1)
    second = _respond(channel, session, "Second.")
    await asyncio.gather(first, second)
    await asyncio.sleep(2.4)  # past the second's drain
    await kit.close()

    assert handed_on_while_speaking == [False]
