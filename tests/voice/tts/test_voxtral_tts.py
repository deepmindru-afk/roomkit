"""VoxtralTTSProvider against a fake Mistral speech API (httpx.MockTransport).

The streamed answers are shaped as the live service sent them on 2026-10-21:
server-sent events, base64 audio of 32-bit float samples.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import struct
import wave
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.voxtral import (
    DEFAULT_VOICE,
    SAMPLE_RATE,
    VoxtralTTSConfig,
    VoxtralTTSError,
    VoxtralTTSProvider,
    pcm16,
)


def _f32(*samples: float) -> bytes:
    return struct.pack(f"<{len(samples)}f", *samples)


def _s16(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events
    )


def _delta(audio: bytes) -> dict[str, Any]:
    return {"type": "speech.audio.delta", "audio_data": base64.b64encode(audio).decode()}


_DONE = {"type": "speech.audio.done", "usage": {"prompt_tokens": 15}}


class _Pieces(httpx.AsyncByteStream):
    """A body served in the pieces given, as the network cuts it."""

    def __init__(self, body: bytes, size: int) -> None:
        self._pieces = [body[i : i + size] for i in range(0, len(body), size)]

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for piece in self._pieces:
            yield piece


@contextlib.contextmanager
def _api(respond: Callable[[httpx.Request], httpx.Response]) -> Iterator[list[httpx.Request]]:
    real = httpx.AsyncClient
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond(request)

    with patch(
        "httpx.AsyncClient",
        side_effect=lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs),
    ):
        yield seen


def _streamed(body: bytes, piece: int = 7) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=_Pieces(body, piece)
    )


async def _render(provider: VoxtralTTSProvider, text: str, **kwargs: Any) -> list[AudioChunk]:
    chunks = [chunk async for chunk in provider.synthesize_stream(text, **kwargs)]
    await provider.close()
    return chunks


def _provider(**config: Any) -> VoxtralTTSProvider:
    return VoxtralTTSProvider(VoxtralTTSConfig(api_key="test-key", **config))


class TestConfig:
    def test_defaults(self) -> None:
        provider = _provider()
        assert provider.name == "VoxtralTTS"
        assert provider.default_voice == DEFAULT_VOICE == "voice_478"
        assert provider.supports_streaming_input is False
        assert "api_key" not in repr(VoxtralTTSConfig(api_key="sk-secret"))

    def test_api_key_is_required(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            VoxtralTTSConfig(api_key="")

    def test_a_key_no_header_can_carry_is_refused_without_echoing_it(self) -> None:
        with pytest.raises(ValueError, match="non-ASCII") as info:
            VoxtralTTSConfig(api_key="sk-…secret")
        assert "secret" not in str(info.value)

    def test_expressiveness_below_one_is_refused(self) -> None:
        # The service answers 422 below 1.0 (measured 2026-10-21).
        with pytest.raises(ValueError, match="expressiveness"):
            VoxtralTTSConfig(api_key="k", expressiveness=0.5)


class TestPcm16:
    def test_float_samples_become_16_bit_and_are_clipped(self) -> None:
        assert pcm16(_f32(0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -2.0)) == _s16(
            0, 16383, -16383, 32767, -32767, 32767, -32767
        )


class TestStream:
    async def test_the_request_asks_for_streamed_pcm(self) -> None:
        body = _sse(_delta(_f32(0.5)), _DONE)
        with _api(_streamed(body)) as seen:
            await _render(_provider(expressiveness=2.5), "Hello [emotion=happy] there.")

        request = seen[0]
        assert str(request.url) == "https://api.mistral.ai/v1/audio/speech"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "voxtral-mini-tts-3",
            "input": "Hello [emotion=happy] there.",
            "voice_id": "voice_478",
            "response_format": "pcm",
            "stream": True,
            "expressiveness": 2.5,
        }

    async def test_a_call_voice_overrides_the_configured_one(self) -> None:
        with _api(_streamed(_sse(_DONE))) as seen:
            await _render(_provider(), "Bonjour.", voice="voice_3137")

        body = json.loads(seen[0].content)
        assert body["voice_id"] == "voice_3137"
        assert "expressiveness" not in body

    async def test_audio_is_16_bit_24khz_and_a_split_sample_is_carried_over(self) -> None:
        samples = _f32(0.5, -0.5, 1.0)
        # The first delta ends two bytes into the second sample.
        body = _sse(_delta(samples[:6]), _delta(samples[6:]), _DONE)
        with _api(_streamed(body, piece=5)):
            chunks = await _render(_provider(), "Hi.")

        assert [c.sample_rate for c in chunks] == [SAMPLE_RATE] * 3
        assert b"".join(c.data for c in chunks) == _s16(16383, -16383, 32767)
        assert [len(c.data) for c in chunks] == [2, 4, 0]
        assert chunks[-1].is_final

    async def test_nothing_after_done_is_read(self) -> None:
        body = _sse(_delta(_f32(0.5)), _DONE, _delta(_f32(-0.5)))
        with _api(_streamed(body)):
            chunks = await _render(_provider(), "Hi.")

        assert b"".join(c.data for c in chunks) == _s16(16383)

    async def test_synthesize_returns_a_16_bit_wav(self) -> None:
        body = _sse(_delta(_f32(*([0.25] * 2400))), _DONE)
        with _api(_streamed(body, piece=4096)):
            provider = _provider()
            content = await provider.synthesize("Short one.")
            await provider.close()

        assert content.mime_type == "audio/wav"
        assert content.transcript == "Short one."
        assert content.duration_seconds == pytest.approx(0.1)
        wav = base64.b64decode(content.url.split(",", 1)[1])
        with wave.open(io.BytesIO(wav)) as reader:
            assert (reader.getframerate(), reader.getsampwidth(), reader.getnchannels()) == (
                24000,
                2,
                1,
            )
            assert reader.getnframes() == 2400

    @pytest.mark.parametrize(
        ("status", "body", "code", "error_type", "retryable"),
        [
            (
                404,
                {
                    "object": "error",
                    "message": "Voice 'nope' not found.",
                    "type": "invalid_voice",
                    "code": "1902",
                },
                "1902",
                "invalid_voice",
                False,
            ),
            (
                400,
                {"object": "error", "message": "Either ref_audio or voice must be provided."},
                None,
                None,
                False,
            ),
            (
                422,
                {"detail": [{"msg": "Input should be greater than or equal to 1"}]},
                None,
                None,
                False,
            ),
            (429, {"object": "error", "message": "Rate limit exceeded"}, None, None, True),
            (502, {}, None, None, True),
        ],
    )
    async def test_a_refusal_raises_before_any_audio(
        self,
        status: int,
        body: dict[str, Any],
        code: str | None,
        error_type: str | None,
        retryable: bool,
    ) -> None:
        refusal = httpx.Response(status, json=body)
        with _api(lambda _: refusal), pytest.raises(VoxtralTTSError) as info:
            await _render(_provider(), "Hi.", voice="nope")

        assert info.value.status_code == status
        assert info.value.code == code
        assert info.value.error_type == error_type
        assert info.value.retryable is retryable
        assert str(info.value).startswith(f"VoxtralTTS request failed ({status})")

    async def test_a_render_cut_short_raises_rather_than_ending(self) -> None:
        # The body ends without speech.audio.done: the audio is not all there.
        with (
            _api(_streamed(_sse(_delta(_f32(0.5))))),
            pytest.raises(VoxtralTTSError, match="cut short") as info,
        ):
            await _render(_provider(), "Hi.")

        assert info.value.retryable is True

    async def test_an_error_event_mid_stream_raises(self) -> None:
        error = {"object": "error", "message": "Internal error", "code": 1000}
        body = _sse(_delta(_f32(0.5))) + f"data: {json.dumps(error)}\n\n".encode()
        with _api(_streamed(body)), pytest.raises(VoxtralTTSError, match="Internal error"):
            await _render(_provider(), "Hi.")


def _voice(slug: str | None, name: str, languages: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "id": extra.pop("id", f"uuid-{name.lower()}"),
        "slug": slug,
        "name": name,
        "languages": languages,
        "gender": extra.pop("gender", "female"),
        "type": extra.pop("type", "preset"),
        "kind": extra.pop("kind", "pvc"),
        "description": extra.pop("description", ""),
        **extra,
    }


class TestVoices:
    async def test_every_page_is_read_once_and_mapped(self) -> None:
        pages = {
            None: {
                "data": [
                    _voice("voice_478", "Sheila", ["en_us"]),
                    _voice("gb_jane_neutral", "Jane - Neutral", ["en_gb"], kind="ivc"),
                ],
                "next_page_token": "p2",
            },
            "p2": {
                "data": [
                    _voice("voice_3137", "Manon", ["fr"]),
                    _voice(
                        None,
                        "My voice",
                        ["fr", "en_us"],
                        id="0b5c-saved",
                        type="custom",
                        description="Cloned from the onboarding call",
                    ),
                ],
                "next_page_token": None,
            },
        }

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=pages[request.url.params.get("page_token")])

        with _api(respond) as seen:
            provider = _provider()
            voices = await provider.list_voices()
            french = await provider.list_voices(language="fr")
            await provider.close()

        assert [str(r.url.path) for r in seen] == ["/v2/audio/voices"] * 2
        assert [(v.id, v.language) for v in voices] == [
            ("voice_478", "en-US"),
            ("gb_jane_neutral", "en-GB"),
            ("voice_3137", "fr"),
            ("0b5c-saved", "multilingual"),
        ]
        assert voices[3].attributes == {"type": "custom", "kind": "pvc"}
        assert voices[3].description == "Cloned from the onboarding call"
        assert voices[0].description is None
        assert [v.id for v in french] == ["voice_3137"]

    async def test_a_refused_listing_raises(self) -> None:
        with _api(lambda _: httpx.Response(401, json={"detail": "Invalid API Key"})):
            provider = _provider()
            with pytest.raises(VoxtralTTSError, match="Invalid API Key") as info:
                await provider.list_voices()
            await provider.close()

        assert info.value.status_code == 401

    def test_the_offline_catalog_holds_the_default_and_names_every_voice(self) -> None:
        catalog = VoxtralTTSProvider.available_voices()
        ids = [v.id for v in catalog]
        assert DEFAULT_VOICE in ids
        assert len(ids) == len(set(ids)) == 36
        assert all(v.name and v.language and v.gender for v in catalog)
        assert {v.id for v in catalog if v.language == "fr"} == {
            "voice_592",
            "voice_3137",
            "voice_589",
            "voice_3141",
        }
