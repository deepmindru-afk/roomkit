"""µ-law on the vendored WebRTC websocket (Twilio / phone media streams).

The reference hashes were generated with ``audioop`` (CPython 3.12 and its
``audioop-lts`` port agree), so they hold whichever codec the websocket uses.
"""

from __future__ import annotations

import base64
import hashlib
import types

import pytest

pytest.importorskip("roomkit.webrtc", reason="roomkit[fastrtc] transport deps not installed")

import numpy as np  # noqa: E402

from roomkit.voice._limits import MAX_INBOUND_AUDIO_FRAME_BYTES  # noqa: E402
from roomkit.webrtc.websocket import WebSocketHandler, convert_to_mulaw  # noqa: E402

# Every int16 value, ascending, through convert_to_mulaw at 8 kHz in and out.
_ENCODE_SHA256 = "01bbcf0dd4d106f64bfb74e6bef1e4da00a6148c7099f1e0ba026b855d882f28"
# Every µ-law byte 0..255 decoded to little-endian int16.
_DECODE_SHA256 = "3dab54339e520bb2c924826e3b72a917a2b612e9fd12fc867500f1d983a75827"


def _handler(*, phone_mode: bool, input_sample_rate: int) -> WebSocketHandler:
    stream_handler = types.SimpleNamespace(
        phone_mode=phone_mode, input_sample_rate=input_sample_rate
    )
    return WebSocketHandler(stream_handler, None, None, None)  # type: ignore[arg-type]


def _media(payload: bytes) -> dict[str, object]:
    return {"event": "media", "media": {"payload": base64.b64encode(payload).decode()}}


def test_convert_to_mulaw_matches_audioop() -> None:
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    encoded = convert_to_mulaw(pcm, 8000, 8000)
    assert hashlib.sha256(encoded).hexdigest() == _ENCODE_SHA256


def test_convert_to_mulaw_silence_and_full_scale() -> None:
    pcm = np.array([0, 32767, -32768], dtype=np.int16)
    assert convert_to_mulaw(pcm, 8000, 8000) == b"\xff\x80\x00"


@pytest.mark.parametrize("phone_mode", [False, True])
def test_media_audio_matches_audioop(phone_mode: bool) -> None:
    handler = _handler(phone_mode=phone_mode, input_sample_rate=8000)
    audio = handler._media_audio(_media(bytes(range(256))))
    assert audio.dtype == np.int16
    assert hashlib.sha256(audio.astype("<i2").tobytes()).hexdigest() == _DECODE_SHA256


def test_media_audio_drops_an_oversized_frame(caplog: pytest.LogCaptureFixture) -> None:
    handler = _handler(phone_mode=False, input_sample_rate=8000)
    oversized = b"\xff" * (MAX_INBOUND_AUDIO_FRAME_BYTES + 3)
    with caplog.at_level("WARNING"):
        assert handler._media_audio(_media(oversized)) is None
    assert "oversized media frame" in caplog.text


def test_media_audio_resamples_phone_audio_to_the_input_rate() -> None:
    handler = _handler(phone_mode=True, input_sample_rate=16000)
    audio = handler._media_audio(_media(b"\xff" * 160))
    assert audio.dtype == np.int16
    assert len(audio) == 320
    assert not audio.any()
