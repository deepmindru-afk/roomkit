"""The 16-bit PCM guard a voice or conference channel puts before playing a chunk."""

from __future__ import annotations

import pytest

from roomkit.voice.base import AudioChunk, require_pcm16


class TestRequirePcm16:
    @pytest.mark.parametrize("fmt", ["pcm_s16le", "pcm"])
    def test_decoded_16_bit_pcm_is_accepted(self, fmt: str) -> None:
        require_pcm16(AudioChunk(data=b"\x00\x00", format=fmt), "publish_audio")

    @pytest.mark.parametrize("fmt", ["mp3", "opus", "ulaw", "mulaw", "alaw", "wav"])
    def test_an_encoded_chunk_is_refused(self, fmt: str) -> None:
        """Encoding belongs to the backend: a caller choosing the wire format
        defeats the boundary (RFC sections 12.2 and 12.10.3).
        """
        with pytest.raises(
            ValueError, match=f"publish_audio expects decoded PCM, got format '{fmt}'"
        ):
            require_pcm16(AudioChunk(data=b"\x00", format=fmt), "publish_audio")

    def test_another_pcm_width_is_refused_rather_than_reinterpreted(self) -> None:
        """Read as 16-bit signed, float samples would not fail: they would play noise."""
        with pytest.raises(ValueError, match="16-bit signed"):
            require_pcm16(AudioChunk(data=b"\x00" * 4, format="pcm_f32le"), "publish_audio")
