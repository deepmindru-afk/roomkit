"""FluxionsTTSProvider against a fake Fluxions API (httpx.MockTransport)."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from roomkit.voice.tts.context import TTSContextLevel
from roomkit.voice.tts.fluxions import (
    SAMPLE_RATE,
    FluxionsTTSConfig,
    FluxionsTTSProvider,
)


def _voice(short: str, full: str, **extra: object) -> dict[str, object]:
    return {"id": short, "voice_id": full, "name": short.title(), **extra}


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class FakeFluxions:
    """Serves the voice lists (``catalogs``, one per listing, then ``mine``) and renders."""

    def __init__(
        self,
        catalogs: list[list[dict[str, object]]],
        audio: list[bytes] | None = None,
        render_status: Callable[[str], int] = lambda voice: 200,
        mine: list[dict[str, object]] | None = None,
    ) -> None:
        self.catalogs = catalogs
        self.mine = mine or []
        self.streams: list[_Chunks] = []
        self.audio = audio if audio is not None else [b"\x01\x00\x02\x00"]
        self.render_status = render_status
        self.listings = 0
        self.renders: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/vui/voices":
            catalog = self.catalogs[min(self.listings, len(self.catalogs) - 1)]
            self.listings += 1
            return httpx.Response(200, json={"voices": catalog})
        if request.url.path == "/vui/v1/voices/mine":
            return httpx.Response(200, json={"voices": self.mine})
        self.renders.append(request)
        voice = json.loads(request.content)["voice"]
        status = self.render_status(voice)
        if status != 200:
            return httpx.Response(status, json={"detail": f"render failed: {voice}"})
        self.streams.append(_Chunks(self.audio))
        return httpx.Response(200, stream=self.streams[-1])


def _provider(fake: FakeFluxions, **config: object) -> FluxionsTTSProvider:
    provider = FluxionsTTSProvider(FluxionsTTSConfig(api_key="fx-key", **config))
    provider._client = httpx.AsyncClient(
        base_url="https://api.fluxions.ai",
        headers={"Authorization": "fx-key"},
        transport=httpx.MockTransport(fake.handle),
    )
    return provider


async def _pcm(provider: FluxionsTTSProvider, text: str = "Hi.", **kwargs: object) -> bytes:
    return b"".join([c.data async for c in provider.synthesize_stream(text, **kwargs)])


class TestRender:
    async def test_a_short_id_renders_with_the_current_models_id(self) -> None:
        fake = FakeFluxions([[_voice("maeve", "maeve.h1")]])

        await _pcm(_provider(fake))

        [render] = fake.renders
        assert render.url.path == "/vui/v1/tts"
        assert render.url.params["stream"] == "1"
        assert json.loads(render.content) == {
            "voice": "maeve.h1",
            "input": "Hi.",
            "response_format": "pcm",
        }

    async def test_every_chunk_holds_whole_samples(self) -> None:
        fake = FakeFluxions(
            [[_voice("maeve", "maeve.h1")]], audio=[b"\x01", b"\x02\x03", b"\x04\x05\x06"]
        )

        chunks = [c async for c in _provider(fake).synthesize_stream("Hi.")]

        assert b"".join(c.data for c in chunks) == b"\x01\x02\x03\x04\x05\x06"
        assert all(len(c.data) % 2 == 0 for c in chunks)
        assert all(c.sample_rate == SAMPLE_RATE for c in chunks)
        assert chunks[-1].is_final and chunks[-1].data == b""

    async def test_closing_the_stream_closes_the_render(self) -> None:
        """A barge-in closes the stream: the HTTP response goes with it."""
        fake = FakeFluxions([[_voice("maeve", "maeve.h1")]], audio=[b"\x00\x00"] * 50)
        stream = _provider(fake).synthesize_stream("Hi.")

        await anext(stream)
        await stream.aclose()

        assert fake.streams[0].closed

    async def test_a_voice_the_list_does_not_carry_is_passed_as_given(self) -> None:
        fake = FakeFluxions([[_voice("maeve", "maeve.h1")]])

        await _pcm(_provider(fake), voice="u-1234")

        assert json.loads(fake.renders[0].content)["voice"] == "u-1234"

    async def test_options_reach_the_render_only_when_set(self) -> None:
        fake = FakeFluxions([[_voice("maeve", "maeve.h1")]])

        await _pcm(_provider(fake, temperature=0.6, max_secs=12.0, verify_chunks=True))

        body = json.loads(fake.renders[0].content)
        assert (body["temperature"], body["max_secs"], body["verify_chunks"]) == (0.6, 12.0, True)

    async def test_synthesize_returns_a_wav(self) -> None:
        fake = FakeFluxions([[_voice("maeve", "maeve.h1")]], audio=[b"\x00\x00" * 2400])

        content = await _provider(fake).synthesize("Hi.")

        wav = base64.b64decode(content.url.split(",", 1)[1])
        assert wav[:4] == b"RIFF" and len(wav) == 44 + 4800
        assert content.duration_seconds == pytest.approx(0.1)


class TestVoiceChanges:
    async def test_a_404_lists_the_voices_again_once(self) -> None:
        """A new model release renames every voice's full id."""
        fake = FakeFluxions(
            [[_voice("maeve", "maeve.hOld")], [_voice("maeve", "maeve.hNew")]],
            render_status=lambda voice: 404 if voice == "maeve.hOld" else 200,
        )
        provider = _provider(fake)
        await provider.warmup()

        await _pcm(provider)

        assert fake.listings == 2
        assert [json.loads(r.content)["voice"] for r in fake.renders] == [
            "maeve.hOld",
            "maeve.hNew",
        ]

    async def test_an_unknown_voice_fails_after_one_new_listing(self) -> None:
        fake = FakeFluxions([[_voice("maeve", "maeve.h1")]], render_status=lambda voice: 404)

        with pytest.raises(httpx.HTTPStatusError):
            await _pcm(_provider(fake), voice="nobody")

        assert (fake.listings, len(fake.renders)) == (2, 2)

    @pytest.mark.parametrize("status", [402, 503])
    async def test_other_errors_are_raised_at_once(self, status: int) -> None:
        fake = FakeFluxions([[_voice("maeve", "maeve.h1")]], render_status=lambda voice: status)

        with pytest.raises(httpx.HTTPStatusError) as raised:
            await _pcm(_provider(fake))

        assert raised.value.response.status_code == status
        assert (fake.listings, len(fake.renders)) == (1, 1)


class TestVoices:
    async def test_list_voices_gives_short_ids_and_hides_hidden_ones(self) -> None:
        fake = FakeFluxions(
            [
                [
                    _voice("maeve", "maeve.h1", gender="female", accent="Irish"),
                    _voice("harry", "harry.h1", style="calm"),
                    _voice("ghost", "ghost.h1", hidden=True),
                ]
            ]
        )

        voices = await _provider(fake).list_voices()

        assert [(v.id, v.gender, v.accent, v.attributes) for v in voices] == [
            ("maeve", "female", "Irish", {}),
            ("harry", None, None, {"style": "calm"}),
        ]
        assert [v.id for v in await _provider(fake).list_voices(gender="female")] == ["maeve"]

    async def test_the_accounts_cloned_voices_are_listed_and_rendered(self) -> None:
        fake = FakeFluxions(
            [[_voice("maeve", "maeve.h1")]], mine=[{"voice_id": "u-42", "name": "Me"}]
        )
        provider = _provider(fake)

        assert [(v.id, v.name) for v in await provider.list_voices()] == [
            ("maeve", "Maeve"),
            ("u-42", "Me"),
        ]
        await _pcm(provider, voice="u-42")
        assert json.loads(fake.renders[0].content)["voice"] == "u-42"

    async def test_a_list_without_short_ids_still_lists_and_renders(self) -> None:
        """The documented shape carries ``voice_id`` alone."""
        fake = FakeFluxions([[{"voice_id": "maeve.h1", "preview_text": "Hello."}]])
        provider = _provider(fake, voice="maeve.h1")

        assert [v.id for v in await provider.list_voices()] == ["maeve.h1"]
        await _pcm(provider)
        assert json.loads(fake.renders[0].content)["voice"] == "maeve.h1"

    async def test_warmup_warns_about_a_voice_fluxions_does_not_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = FakeFluxions([[_voice("maeve", "maeve.h1")]])

        with caplog.at_level("WARNING", logger="roomkit.voice.tts.fluxions"):
            await _provider(fake, voice="nobody").warmup()

        assert "nobody" in caplog.text


class TestProvider:
    def test_no_conversation_context(self) -> None:
        provider = FluxionsTTSProvider(FluxionsTTSConfig(api_key="fx-key"))

        assert provider.context_level == TTSContextLevel.NONE
        assert provider.default_voice == "maeve"
        assert not provider.supports_streaming_input

    def test_the_key_stays_out_of_the_config_repr(self) -> None:
        assert "fx-key" not in repr(FluxionsTTSConfig(api_key="fx-key"))

    async def test_the_client_carries_the_key_and_the_api_root(self) -> None:
        provider = FluxionsTTSProvider(FluxionsTTSConfig(api_key="fx-key", timeout=42.0))

        client = provider._get_client()

        assert client.headers["Authorization"] == "fx-key"
        assert str(client.base_url) == "https://api.fluxions.ai"
        assert (client.timeout.read, client.timeout.connect) == (42.0, 5.0)
        await provider.close()

    async def test_close_releases_the_client(self) -> None:
        fake = FakeFluxions([[_voice("maeve", "maeve.h1")]])
        provider = _provider(fake)

        await provider.close()

        assert provider._client is None

    def test_lazy_getters(self) -> None:
        from roomkit.voice import get_fluxions_tts_config, get_fluxions_tts_provider

        assert get_fluxions_tts_provider() is FluxionsTTSProvider
        assert get_fluxions_tts_config() is FluxionsTTSConfig
