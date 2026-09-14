"""RTP time follows transmitted samples, including audio sent ahead by the pacer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from roomkit.voice.backends._sip_types import SIPSessionState
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.base import VoiceSession
from tests.test_voice_session_lifecycle import sip  # noqa: F401


@pytest.fixture
def sender(sip: tuple[SIPVoiceBackend, MagicMock], monkeypatch: pytest.MonkeyPatch):  # noqa: F811, ANN201
    backend, media = sip
    session = VoiceSession(id="sip", room_id="r", participant_id="p", channel_id="sip")
    state = SIPSessionState(session=session, call_session=media)
    backend._session_states[session.id] = state
    clock = SimpleNamespace(now=100.0)
    # Replace only this module's clock; asyncio and the fixtures keep real time.
    monkeypatch.setattr(
        "roomkit.voice.backends.sip_audio.time", SimpleNamespace(monotonic=lambda: clock.now)
    )

    def send(at: float, duration_ms: int) -> None:
        clock.now = 100.0 + at
        pcm = b"\x01\x00" * (state.codec_rate * duration_ms // 1000)
        backend._send_pcm_bytes(session, media, pcm)

    return state, media, send


@pytest.mark.parametrize("rate", [8000, 16000])
def test_continuous_bursts_have_contiguous_rtp_timestamps(sender, rate: int) -> None:  # noqa: ANN001
    state, media, send = sender
    state.codec_rate = rate
    for index in range(5):
        send(index * 0.2, 200)

    timestamps = [call.args[1] for call in media.send_audio_pcm.call_args_list]
    # G.722 samples at 16 kHz but retains the 8 kHz RTP clock.
    assert timestamps == list(range(0, 50 * 160, 160))


def test_pacer_headroom_covers_a_pause_between_individual_packets(sender) -> None:  # noqa: ANN001
    _, media, send = sender
    send(0, 80)
    for _ in range(8):
        send(0, 20)
    send(0.18, 20)

    assert [call.args[1] for call in media.send_audio_pcm.call_args_list] == list(
        range(0, 13 * 160, 160)
    )


def test_real_idle_gap_advances_only_beyond_audio_already_sent(sender) -> None:  # noqa: ANN001
    _, media, send = sender
    send(0, 200)
    send(0.5, 20)

    assert media.send_audio_pcm.call_args.args[1] == pytest.approx(4000, abs=1)


def test_partial_packets_do_not_advance_the_clock_before_transmission(sender) -> None:  # noqa: ANN001
    state, media, send = sender
    send(0, 20)
    send(0.25, 10)
    assert state.send_timestamp == 160
    assert media.send_audio_pcm.call_count == 1
    send(0.5, 10)
    assert media.send_audio_pcm.call_args.args[1] == pytest.approx(4000, abs=1)


def test_packet_timing_jitter_does_not_insert_silence(sender) -> None:  # noqa: ANN001
    _, media, send = sender
    for index in range(10):
        send(index * 0.021, 20)
    assert [call.args[1] for call in media.send_audio_pcm.call_args_list] == list(
        range(0, 10 * 160, 160)
    )
