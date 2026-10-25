"""LocalAudioBackend AEC tests.

Covers transport-level AEC reference feeding, rate mismatch resampling,
and the feeds_aec_reference property.

Tests are skipped when sounddevice/numpy are not installed.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sd = pytest.importorskip("sounddevice")
np = pytest.importorskip("numpy")

from roomkit.voice._device_clock import (  # noqa: E402
    CAPTURED_AT,
    capture_metadata,
    playout_delay,
)
from roomkit.voice.audio_frame import AudioFrame  # noqa: E402
from roomkit.voice.backends._local_speaker import SpeakerBlock  # noqa: E402
from roomkit.voice.backends.base import VoiceBackend  # noqa: E402
from roomkit.voice.backends.local import LocalAudioBackend  # noqa: E402
from roomkit.voice.pipeline.aec.mock import MockAECProvider  # noqa: E402


class TestFeedsAECReferenceProperty:
    """Tests for VoiceBackend.feeds_aec_reference default and LocalAudioBackend override."""

    def test_abc_defaults_false(self):
        """VoiceBackend subclass without override returns False."""

        class MinimalBackend(VoiceBackend):
            @property
            def name(self):
                return "minimal"

            async def connect(self, room_id, participant_id, channel_id, *, metadata=None):
                pass

            async def disconnect(self, session):
                pass

            async def send_audio(self, session, audio):
                pass

        backend = MinimalBackend()
        assert backend.feeds_aec_reference is False

    def test_local_backend_false_without_aec(self):
        """LocalAudioBackend without AEC returns False."""
        backend = LocalAudioBackend(aec=None)
        assert backend.feeds_aec_reference is False

    def test_local_backend_true_with_aec(self):
        """LocalAudioBackend with AEC returns True."""
        aec = MockAECProvider()
        backend = LocalAudioBackend(aec=aec)
        assert backend.feeds_aec_reference is True


class TestAECFeedPlayed:
    """Tests for LocalAudioBackend._aec_feed_played transport-level reference feeding."""

    def test_feed_played_with_rate_mismatch(self):
        """AEC reference resampled when output_rate != input_rate."""
        aec = MockAECProvider()
        backend = LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=24000,
            channels=1,
            block_duration_ms=20,
            aec=aec,
        )

        # Output block size: 24000 * 20/1000 * 1 * 2 = 960 bytes
        output_block = bytearray(b"\x01\x00" * (24000 * 20 // 1000))
        backend._aec_feed_played(output_block, "s1")

        assert len(aec.reference_frames) == 1
        assert aec.reference_frames[0].sample_rate == 16000

    def test_feed_played_without_rate_mismatch(self):
        """AEC reference fed directly when output_rate == input_rate."""
        aec = MockAECProvider()
        backend = LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=16000,
            channels=1,
            block_duration_ms=20,
            aec=aec,
        )

        # Block size: 16000 * 20/1000 * 1 * 2 = 640 bytes
        block = bytearray(b"\x01\x00" * (16000 * 20 // 1000))
        backend._aec_feed_played(block, "s1")

        assert len(aec.reference_frames) == 1
        assert aec.reference_frames[0].sample_rate == 16000

    def test_feed_played_accumulates_partial_blocks(self):
        """_aec_feed_played accumulates data until a full block is available."""
        aec = MockAECProvider()
        backend = LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=16000,
            channels=1,
            block_duration_ms=20,
            aec=aec,
        )

        # Block size = 640 bytes; send less than one block
        backend._aec_feed_played(bytearray(b"\x01\x00" * 100), "s1")  # 200 bytes
        assert len(aec.reference_frames) == 0

        # Send more to complete the block
        backend._aec_feed_played(bytearray(b"\x01\x00" * 220), "s1")  # 440 bytes, total 640
        assert len(aec.reference_frames) == 1

    def test_partial_reference_buffers_are_isolated_per_stream(self):
        """A partial Alice block must never be completed with Bob's audio."""
        aec = MockAECProvider()
        backend = LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=16000,
            channels=1,
            block_duration_ms=20,
            aec=aec,
        )

        backend._aec_feed_played(bytearray(b"\x01\x00" * 100), "alice")
        backend._aec_feed_played(bytearray(b"\x02\x00" * 220), "bob")

        assert aec.reference_frames == []

        backend._aec_feed_played(bytearray(b"\x03\x00" * 220), "alice")
        assert aec.reference_streams == ["alice"]


class TestAECDelayAutoConfiguration:
    def test_uses_combined_portaudio_input_and_output_latency(self):
        aec = MagicMock()
        aec.stream_delay_ms = 0
        backend = LocalAudioBackend(aec=aec)
        backend._speaker._stream = SimpleNamespace(latency=0.0181)

        backend._note_input_latency(37.8)

        aec.set_stream_delay_ms.assert_called_once_with(56)

    def test_preserves_explicit_stream_delay(self):
        aec = MagicMock()
        aec.stream_delay_ms = 80
        backend = LocalAudioBackend(aec=aec)
        backend._speaker._stream = SimpleNamespace(latency=0.02)

        backend._note_input_latency(40)

        aec.set_stream_delay_ms.assert_not_called()


class TestAECDelayFromDeviceClocks:
    """As browsers do: the render-to-capture delay from the devices' clocks, every
    captured frame, rather than PortAudio's nominal latencies (296 ms announced
    on a MacBook where the clocks give 42 ms)."""

    @staticmethod
    def _backend(stream_delay_ms: int = 0) -> tuple[LocalAudioBackend, MagicMock]:
        aec = MagicMock()
        aec.stream_delay_ms = stream_delay_ms
        return LocalAudioBackend(aec=aec), aec

    @staticmethod
    def _captured(ago: float) -> AudioFrame:
        return AudioFrame(data=b"\x00\x00" * 320, metadata={CAPTURED_AT: time.monotonic() - ago})

    @staticmethod
    def _played(backend: LocalAudioBackend, playout_delay: float | None) -> None:
        backend._on_speaker_block(
            SpeakerBlock(b"", 0, False, False, False, playout_delay=playout_delay)
        )

    def test_playout_plus_capture_delay_every_frame(self):
        backend, aec = self._backend()
        self._played(backend, 0.0142)
        backend._follow_aec_delay(self._captured(0.0274))
        (delay,), _ = aec.set_stream_delay_ms.call_args
        assert 41 <= delay <= 43

    def test_jitter_is_not_told_again_a_change_is(self):
        backend, aec = self._backend()
        self._played(backend, 0.014)
        backend._follow_aec_delay(self._captured(0.027))
        backend._follow_aec_delay(self._captured(0.030))  # 3 ms more: jitter
        assert aec.set_stream_delay_ms.call_count == 1
        self._played(backend, 0.180)  # Bluetooth headphones now
        backend._follow_aec_delay(self._captured(0.027))
        assert aec.set_stream_delay_ms.call_count == 2
        (delay,), _ = aec.set_stream_delay_ms.call_args
        assert 205 <= delay <= 209

    def test_an_explicit_delay_is_left_alone(self):
        backend, aec = self._backend(stream_delay_ms=80)
        self._played(backend, 0.014)
        backend._follow_aec_delay(self._captured(0.027))
        aec.set_stream_delay_ms.assert_not_called()

    def test_nothing_is_told_without_both_clocks(self):
        backend, aec = self._backend()
        backend._follow_aec_delay(self._captured(0.027))  # no block played yet
        self._played(backend, None)  # a host API that reports no clock
        backend._follow_aec_delay(self._captured(0.027))
        self._played(backend, 0.014)
        backend._follow_aec_delay(AudioFrame(data=b"\x00\x00" * 320))  # an unstamped frame
        aec.set_stream_delay_ms.assert_not_called()

    def test_a_frame_replayed_from_a_backlog_is_no_measure(self):
        backend, aec = self._backend()
        self._played(backend, 0.014)
        backend._follow_aec_delay(self._captured(2.0))
        aec.set_stream_delay_ms.assert_not_called()

    def test_the_portaudio_seed_stands_down_once_the_clocks_spoke(self):
        backend, aec = self._backend()
        backend._speaker._stream = SimpleNamespace(latency=0.0532)
        self._played(backend, 0.014)
        backend._follow_aec_delay(self._captured(0.027))
        aec.stream_delay_ms = 41
        aec.set_stream_delay_ms.reset_mock()
        backend._note_input_latency(242.6)  # a shared source subscribed late
        aec.set_stream_delay_ms.assert_not_called()

    def test_the_speaker_reports_each_blocks_playout_delay(self):
        backend, _ = self._backend()
        speaker = backend._speaker
        out = bytearray(speaker._blocksize * speaker._frame_width)
        clock = SimpleNamespace(currentTime=12.0, outputBufferDacTime=12.0142)
        speaker._callback(out, speaker._blocksize, clock, None)
        assert backend._playout_delay == pytest.approx(0.0142)

    def test_the_captured_frames_say_when_they_were_captured(self):
        clock = SimpleNamespace(currentTime=5.0274, inputBufferAdcTime=5.0)
        stamp = capture_metadata(clock)[CAPTURED_AT]
        assert time.monotonic() - stamp == pytest.approx(0.0274, abs=0.005)
        assert capture_metadata(SimpleNamespace(currentTime=0, inputBufferAdcTime=0)) == {}
        assert capture_metadata(object()) == {}
        assert playout_delay(SimpleNamespace(currentTime=3.0, outputBufferDacTime=0)) is None


class TestAECDebugTaps:
    """RMK-552: the backend shows its own AEC to the debug taps, every captured
    frame with the reference fed meanwhile, on the event loop."""

    async def test_every_captured_frame_brings_its_reference(self):
        aec = MockAECProvider()
        backend = LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=16000,
            channels=1,
            block_duration_ms=20,
            aec=aec,
            mute_mic_during_playback=False,
        )
        backend._loop = asyncio.get_running_loop()
        taps: list[tuple[str, bytes]] = []
        unsubscribe = backend.on_aec_tap(lambda session, stage, f: taps.append((stage, f.data)))
        backend.on_audio_received(lambda session, frame: None)
        handle = backend._make_frame_handler(SimpleNamespace(id="s1"))
        frame = AudioFrame(data=b"\x01\x00" * 320, sample_rate=16000, channels=1, sample_width=2)

        handle(frame)  # nothing played yet
        backend._aec_feed_played(bytearray(b"\x02\x00" * 320), "s1")  # 20 ms played
        handle(frame)
        await asyncio.sleep(0)

        assert taps == [
            ("transport_raw", frame.data),
            ("aec_reference", bytes(640)),
            ("transport_raw", frame.data),
            ("aec_reference", b"\x02\x00" * 320),
        ]
        unsubscribe()
        handle(frame)
        await asyncio.sleep(0)
        assert len(taps) == 4

    def test_no_aec_no_taps(self):
        assert LocalAudioBackend(aec=None).on_aec_tap(lambda *a: None) is None
