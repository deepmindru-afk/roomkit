"""A VoiceChannel carries a diarizing STT's speaker labels to the room (RFC §12.2.3)."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels._voice_speakers import (
    UNKNOWN_SPEAKER,
    SpeakerAttribution,
    SpeakerTracker,
    default_sender_name,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, SpeakerSegment, TranscriptionResult
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    MockTurnDetector,
    MockVADProvider,
    TurnDecision,
)
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.tts.mock import MockTTSProvider

_FRAME = AudioFrame(data=b"\x01\x00" * 1600)  # 3200 bytes: one STT buffer flush


class _SlowAI(MockAIProvider):
    """An AI that takes a while to answer, so its reply is still pending."""

    def __init__(self, delay: float) -> None:
        super().__init__(responses=["D'accord."])
        self._delay = delay

    async def generate(self, context: Any) -> Any:
        await asyncio.sleep(self._delay)
        return await super().generate(context)


class _PlainSTT(STTProvider):
    """A continuous STT that does not diarize: one final per chunk."""

    def __init__(self, finals: list[str]) -> None:
        self._finals = list(finals)

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for chunk in audio_stream:
            if any(chunk.data) and self._finals:
                yield TranscriptionResult(text=self._finals.pop(0), is_final=True)


class _DiarizingSTT(STTProvider):
    """Answers one scripted final per chunk it receives, all on one stream.

    ``end_after`` ends each stream after that many finals, as a provider that
    closes its stream would.
    """

    def __init__(self, finals: list[list[SpeakerSegment]], *, end_after: int | None = None):
        self._finals = list(finals)
        self._end_after = end_after
        self.streams = 0
        self.chunks: list[AudioChunk] = []

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_diarization(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        self.streams += 1
        answered = 0
        async for chunk in audio_stream:
            self.chunks.append(chunk)
            if not any(chunk.data) or not self._finals:
                continue  # silence the channel sent through a quiet spell
            segments = self._finals.pop(0)
            yield TranscriptionResult(
                text=" ".join(s.text for s in segments), is_final=True, segments=segments
            )
            answered += 1
            if self._end_after is not None and answered >= self._end_after:
                return


async def _eventually(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.02)


class _Room:
    """A kit with one continuous VoiceChannel, and what its hooks saw."""

    def __init__(
        self,
        stt: STTProvider,
        pipeline: AudioPipelineConfig | None = None,
        *,
        ai_delay: float | None = None,
    ) -> None:
        self.backend = MockVoiceBackend()
        self.channel = VoiceChannel(
            "voice-1",
            stt=stt,
            tts=MockTTSProvider() if ai_delay is not None else None,
            backend=self.backend,
            pipeline=pipeline or AudioPipelineConfig(),
        )
        self.kit = RoomKit(stt=stt, voice=self.backend)
        self.kit.register_channel(self.channel)
        self._ai = ai_delay is not None
        if ai_delay is not None:
            self.kit.register_channel(AIChannel("ai-1", provider=_SlowAI(ai_delay)))
        self.transcriptions: list[Any] = []
        self.transcribed_at: list[float] = []
        self.speaker_changes: list[Any] = []
        self.rename: dict[str, str] = {}
        self.slow_hook: dict[str, float] = {}
        self.started = time.monotonic()

        @self.kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event: Any, ctx: Any) -> HookResult:
            self.transcriptions.append(event)
            self.transcribed_at.append(time.monotonic() - self.started)
            if event.speaker in self.slow_hook:
                await asyncio.sleep(self.slow_hook[event.speaker])
            if event.speaker in self.rename:
                return HookResult.modify(
                    dataclasses.replace(event, sender_name=self.rename[event.speaker])
                )
            return HookResult.allow()

        @self.kit.hook(HookTrigger.ON_SPEAKER_CHANGE, execution=HookExecution.ASYNC)
        async def on_speaker_change(event: Any, ctx: Any) -> None:
            self.speaker_changes.append(event)

    async def start(self) -> None:
        room = await self.kit.create_room()
        self.room_id = room.id
        await self.kit.attach_channel(room.id, "voice-1")
        if self._ai:
            await self.kit.attach_channel(room.id, "ai-1")
        self.session = await self.kit.join(room.id, "voice-1", participant_id="owner")
        assert self.channel._continuous_stt
        self.started = time.monotonic()

    async def speak(self) -> None:
        await self.backend.simulate_audio_received(self.session, _FRAME)

    async def messages(self, count: int, timeout: float = 3.0) -> list[Any]:
        """The channel's messages in the room, once there are ``count`` of them."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            events = await self.kit.store.list_events(self.room_id, offset=0, limit=50)
            found = [
                e
                for e in events
                if e.type == EventType.MESSAGE and e.source.channel_id == "voice-1"
            ]
            if len(found) >= count or loop.time() > deadline:
                return found
            await asyncio.sleep(0.02)


def _said(messages: list[Any]) -> list[tuple[str | None, str]]:
    return [(m.metadata.get("sender_name"), m.content.body) for m in messages]


class TestSpeakerRules:
    def test_default_names(self) -> None:
        assert default_sender_name("A", 0) == "Speaker A"
        assert default_sender_name("A", 2) == "Speaker A#2"
        assert default_sender_name(None, 1) == UNKNOWN_SPEAKER

    def test_metadata_leaves_the_label_out_for_unattributed_words(self) -> None:
        assert SpeakerAttribution.of("B", 1).metadata() == {
            "sender_name": "Speaker B#1",
            "speaker_epoch": 1,
            "speaker_label": "B",
        }
        assert SpeakerAttribution.of(None, 0).metadata() == {
            "sender_name": UNKNOWN_SPEAKER,
            "speaker_epoch": 0,
        }

    def test_tracker_fires_on_change_and_first_label_of_each_epoch(self) -> None:
        tracker = SpeakerTracker()
        observed = [
            tracker.observe(label, epoch)
            for label, epoch in [
                ("A", 0),  # first labelled segment: fires, new
                ("A", 0),  # same speaker: nothing
                (None, 0),  # unattributed: nothing, and no reset
                ("A", 0),  # still A: nothing
                ("B", 0),  # change: fires, new
                ("A", 0),  # back to A: fires, not new
                ("A", 1),  # new epoch: fires, new again
            ]
        ]
        assert observed == [True, None, None, None, True, False, True]


class TestVoiceChannelCarriesLabels:
    async def test_one_stream_one_message_per_speaker(self) -> None:
        stt = _DiarizingSTT(
            [[SpeakerSegment("A", "Bonjour Julie.")], [SpeakerSegment("B", "Oui, je l'ai lu.")]]
        )
        room = _Room(stt)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 1)
        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 2)
        messages = await room.messages(2)
        await room.channel.close()

        # The stream was kept across the turn: the labels compare.
        assert stt.streams == 1
        assert _said(messages) == [
            ("Speaker A", "Bonjour Julie."),
            ("Speaker B", "Oui, je l'ai lu."),
        ]
        assert [m.metadata["speaker_label"] for m in messages] == ["A", "B"]
        assert {m.source.participant_id for m in messages} == {"owner"}
        assert [(t.speaker, t.speaker_epoch, t.sender_name) for t in room.transcriptions] == [
            ("A", 0, "Speaker A"),
            ("B", 0, "Speaker B"),
        ]
        await _eventually(lambda: len(room.speaker_changes) == 2)
        assert [
            (c.speaker_id, c.is_new_speaker, c.source, c.speaker_epoch)
            for c in room.speaker_changes
        ] == [("A", True, "stt", 0), ("B", True, "stt", 0)]

    async def test_a_final_mixing_speakers_becomes_one_message_each_in_order(self) -> None:
        stt = _DiarizingSTT(
            [
                [
                    SpeakerSegment("A", "Tu viens?"),
                    SpeakerSegment("B", "Oui."),
                    SpeakerSegment(None, "euh"),
                ]
            ]
        )
        room = _Room(stt)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 3)
        messages = await room.messages(3)
        await room.channel.close()

        assert _said(messages) == [
            ("Speaker A", "Tu viens?"),
            ("Speaker B", "Oui."),
            (UNKNOWN_SPEAKER, "euh"),
        ]
        assert "speaker_label" not in messages[2].metadata

    async def test_a_hook_names_the_speaker(self) -> None:
        stt = _DiarizingSTT([[SpeakerSegment("A", "Bonjour.")]])
        room = _Room(stt)
        room.rename = {"A": "Alice"}
        await room.start()

        await room.speak()
        messages = await room.messages(1)
        await room.channel.close()

        assert _said(messages) == [("Alice", "Bonjour.")]
        assert messages[0].metadata["speaker_label"] == "A"

    async def test_a_new_stream_starts_a_new_epoch(self) -> None:
        # The provider ends its stream after each final: the second "A" is
        # another stream's label, and must not read as the first speaker.
        stt = _DiarizingSTT(
            [[SpeakerSegment("A", "Premier.")], [SpeakerSegment("A", "Second.")]], end_after=1
        )
        room = _Room(stt)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 1)
        await asyncio.sleep(0.3)  # the channel reconnects
        await room.speak()
        messages = await room.messages(2)
        await room.channel.close()

        assert stt.streams == 2
        assert _said(messages) == [("Speaker A", "Premier."), ("Speaker A#1", "Second.")]
        assert [m.metadata["speaker_epoch"] for m in messages] == [0, 1]
        await _eventually(lambda: len(room.speaker_changes) == 2)
        assert all(c.is_new_speaker for c in room.speaker_changes)

    async def test_silence_keeps_the_stream_alive_through_a_quiet_spell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A muted mic sends nothing; a provider closes a stream that falls
        # behind real time, which would start a new label epoch.
        monkeypatch.setattr("roomkit.channels.voice._KEPT_STREAM_PACE_S", 0.1)
        stt = _DiarizingSTT([[SpeakerSegment("A", "Bonjour.")], [SpeakerSegment("B", "Oui.")]])
        room = _Room(stt)
        await room.start()

        opened = time.monotonic()
        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 1)
        await asyncio.sleep(0.6)  # no audio: the channel sends silence instead
        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 2)
        elapsed = time.monotonic() - opened
        await room.channel.close()

        assert any(not any(c.data) for c in stt.chunks)
        assert stt.streams == 1
        assert [t.speaker_epoch for t in room.transcriptions] == [0, 0]
        audio = sum(len(c.data) for c in stt.chunks) / (2 * 16000)
        assert audio >= elapsed - 0.35  # kept within a pacing period or so of the clock

    async def test_a_lossy_stream_is_caught_up_in_its_pauses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Frames that arrive short of real time, with no long gap: the deficit
        # of a whole call would otherwise add up until the provider hangs up.
        monkeypatch.setattr("roomkit.channels.voice._KEPT_STREAM_PACE_S", 0.1)
        stt = _DiarizingSTT([])
        room = _Room(stt)
        await room.start()

        opened = time.monotonic()
        for _ in range(8):
            await room.speak()  # 0.1 s of audio...
            await asyncio.sleep(0.25)  # ...every 0.25 s
        elapsed = time.monotonic() - opened
        await room.channel.close()

        audio = sum(len(c.data) for c in stt.chunks) / (2 * 16000)
        assert audio >= elapsed - 0.35
        assert stt.streams == 1

    async def test_finals_are_routed_in_order_even_when_a_hook_is_slow(self) -> None:
        stt = _DiarizingSTT([[SpeakerSegment("A", "Premier.")], [SpeakerSegment("B", "Second.")]])
        room = _Room(stt)
        room.slow_hook = {"A": 0.4}  # B's final arrives while A is in its hook
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 1)
        await room.speak()
        messages = await room.messages(2)
        await room.channel.close()

        assert _said(messages) == [("Speaker A", "Premier."), ("Speaker B", "Second.")]

    async def test_a_speaker_change_closes_the_pending_turn(self) -> None:
        # The detector judges every turn incomplete, and would join both
        # segments; a turn has one speaker, so A's is routed when B speaks.
        detector = MockTurnDetector(
            decisions=[TurnDecision(is_complete=False), TurnDecision(is_complete=False)]
        )
        pipeline = AudioPipelineConfig(turn_detector=detector, turn_incomplete_wait_ms=100)
        stt = _DiarizingSTT([[SpeakerSegment("A", "Je voudrais")], [SpeakerSegment("B", "Non.")]])
        room = _Room(stt, pipeline)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 1)
        await room.speak()
        messages = await room.messages(2)
        await room.channel.close()

        assert _said(messages) == [("Speaker A", "Je voudrais"), ("Speaker B", "Non.")]


class TestTheReplyDoesNotHoldBackTheNextFinal:
    # A user who adds words while the AI is still answering is heard at once,
    # not after the reply has played (and then dropped as echo of it).

    async def test_without_diarization(self) -> None:
        room = _Room(_PlainSTT(["Une table", "pour deux"]), ai_delay=0.8)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 1)
        await asyncio.sleep(0.2)
        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 2)
        messages = await room.messages(2)
        await room.kit.close()

        assert room.transcribed_at[1] < 0.7
        assert [m.content.body for m in messages] == ["Une table", "pour deux"]

    async def test_with_diarization(self) -> None:
        stt = _DiarizingSTT([[SpeakerSegment("A", "Une table")], [SpeakerSegment("B", "Non.")]])
        room = _Room(stt, ai_delay=0.8)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 1)
        await asyncio.sleep(0.2)
        await room.speak()
        await _eventually(lambda: len(room.transcriptions) == 2)
        messages = await room.messages(2)
        await room.kit.close()

        assert room.transcribed_at[1] < 0.7
        assert _said(messages) == [("Speaker A", "Une table"), ("Speaker B", "Non.")]


class TestWhereLabelsCannotBeCarried:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pipeline": AudioPipelineConfig(vad=MockVADProvider(events=[]))},
            {"pipeline": AudioPipelineConfig(), "batch_mode": True},
        ],
        ids=["vad", "batch"],
    )
    def test_refused(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="continuous mode"):
            VoiceChannel("voice-1", stt=_DiarizingSTT([]), backend=MockVoiceBackend(), **kwargs)

    def test_continuous_mode_accepts_it(self) -> None:
        VoiceChannel(
            "voice-1",
            stt=_DiarizingSTT([]),
            backend=MockVoiceBackend(),
            pipeline=AudioPipelineConfig(),
        )
