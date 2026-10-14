"""FishAudioTTSProvider over HTTP, against a fake Fish Audio API (httpx.MockTransport).

Requests follow docs.fish.audio and the live probes of RMK-762 (2026-10-14):
the ``model`` header picks the model, ``format: pcm`` streams 16-bit mono at
the asked rate, and an unknown model name is billed as s2.1-pro.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from roomkit.voice.tts.fish_audio import (
    MODELS,
    SAMPLE_RATES,
    FishAudioTTSConfig,
    FishAudioTTSError,
    FishAudioTTSProvider,
)
from roomkit.voice.voices import DialogueTurn

CLEMENCE = {
    "_id": "a288bdc744da4ad194921adad6863175",
    "title": "Clémence",
    "description": "25+ French Female deep voice, narrator",
    "languages": ["fr"],
    "tags": ["female", "middle-aged", "calm", "French"],
    "visibility": "public",
    "author": {"_id": "x", "nickname": "Belkacem Brik"},
}
MINE = {
    "_id": "mine0001",
    "title": "My voice",
    "languages": ["fr", "en"],
    "tags": ["male"],
    "visibility": "private",
}


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class FakeFishAudio:
    """Renders every ``/v1/tts`` as ``audio`` (or answers ``status``), lists voices."""

    def __init__(self, audio: list[bytes] | None = None, status: int = 200) -> None:
        self.audio = audio if audio is not None else [b"\x01\x00\x02\x00"]
        self.status = status
        self.requests: list[httpx.Request] = []
        self.streams: list[_Chunks] = []
        self.own: list[dict[str, Any]] = [MINE]
        self.public: list[dict[str, Any]] = [CLEMENCE]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(
                self.status,
                headers={"x-fish-error-code": "insufficient_balance"},
                json={"status": self.status, "message": "Insufficient API credit."},
            )
        if request.url.path == "/model":
            own = request.url.params.get("self") == "true"
            return httpx.Response(
                200, json={"total": 1, "items": self.own if own else self.public}
            )
        self.streams.append(_Chunks(self.audio))
        return httpx.Response(200, stream=self.streams[-1])

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == "/v1/tts"]


def _provider(fake: FakeFishAudio, **config: Any) -> FishAudioTTSProvider:
    provider = FishAudioTTSProvider(FishAudioTTSConfig(api_key="fish-key", **config))
    real = provider._get_client()
    provider._client = httpx.AsyncClient(
        base_url=real.base_url,
        headers=real.headers,
        timeout=real.timeout,
        transport=httpx.MockTransport(fake.handle),
    )
    return provider


async def _pcm(provider: FishAudioTTSProvider, text: str = "Hello.", **kwargs: Any) -> bytes:
    return b"".join([c.data async for c in provider.synthesize_stream(text, **kwargs)])


class TestConfig:
    def test_drama_3_is_the_default_model(self) -> None:
        assert FishAudioTTSConfig(api_key="k").model == "drama-3"

    @pytest.mark.parametrize("model", ["drama-3-preview", "s1", "s2.1", "Drama-3", ""])
    def test_an_unknown_model_is_refused(self, model: str) -> None:
        # The API would render it with s2.1-pro and bill it so, saying nothing.
        with pytest.raises(ValueError, match="s2.1-pro without saying so"):
            FishAudioTTSConfig(api_key="k", model=model)

    @pytest.mark.parametrize("model", MODELS)
    def test_the_known_models(self, model: str) -> None:
        assert FishAudioTTSConfig(api_key="k", model=model).model == model

    @pytest.mark.parametrize("rate", [22050, 48000, 12345])
    def test_a_rate_the_api_refuses_is_refused(self, rate: int) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            FishAudioTTSConfig(api_key="k", sample_rate=rate)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("api_key", "", "required"),
            ("api_key", "fish-…", "non-ASCII"),
            ("latency", "fast", "latency"),
            ("speed", 0.4, "speed"),
            ("speed", 2.1, "speed"),
            ("base_url", "http://api.fish.audio", "https"),
            ("ws_url", "ws://api.fish.audio/v1/tts/live", "wss"),
        ],
    )
    def test_invalid_values(self, field: str, value: Any, match: str) -> None:
        kwargs: dict[str, Any] = {"api_key": "k", field: value}
        with pytest.raises(ValueError, match=match):
            FishAudioTTSConfig(**kwargs)

    def test_a_loopback_url_may_be_plain(self) -> None:
        config = FishAudioTTSConfig(
            api_key="k", base_url="http://127.0.0.1:8000", ws_url="ws://localhost:8000/live"
        )
        assert config.base_url.startswith("http://")

    def test_the_key_stays_out_of_the_repr(self) -> None:
        assert "fish-key" not in repr(FishAudioTTSConfig(api_key="fish-key"))


class TestRender:
    async def test_the_request_names_the_model_and_asks_for_pcm(self) -> None:
        fake = FakeFishAudio()

        await _pcm(_provider(fake, voice="v1", speed=1.25, latency="low"), "Hello.")

        request = fake.requests[0]
        assert request.url.path == "/v1/tts"
        assert request.headers["model"] == "drama-3"
        assert request.headers["authorization"] == "Bearer fish-key"
        assert fake.bodies()[0] == {
            "text": "Hello.",
            "format": "pcm",
            "sample_rate": 24000,
            "latency": "low",
            "reference_id": "v1",
            "prosody": {"speed": 1.25},
        }

    async def test_no_voice_lets_fish_choose(self) -> None:
        fake = FakeFishAudio()

        await _pcm(_provider(fake))

        assert "reference_id" not in fake.bodies()[0]
        assert "prosody" not in fake.bodies()[0]

    async def test_a_voice_per_call_overrides_the_config(self) -> None:
        fake = FakeFishAudio()

        await _pcm(_provider(fake, voice="v1"), voice="v2")

        assert fake.bodies()[0]["reference_id"] == "v2"

    @pytest.mark.parametrize("rate", SAMPLE_RATES)
    async def test_the_rate_is_asked_and_stamped(self, rate: int) -> None:
        fake = FakeFishAudio()

        chunks = [c async for c in _provider(fake, sample_rate=rate).synthesize_stream("Yes.")]

        assert fake.bodies()[0]["sample_rate"] == rate
        assert all(c.sample_rate == rate for c in chunks)

    async def test_every_chunk_holds_whole_samples(self) -> None:
        # Fish's HTTP chunks have odd sizes (80 of 107 in a live render).
        fake = FakeFishAudio(audio=[b"\x01", b"\x02\x03", b"\x04\x05\x06"])

        chunks = [c async for c in _provider(fake).synthesize_stream("Yes.")]

        assert b"".join(c.data for c in chunks) == b"\x01\x02\x03\x04\x05\x06"
        assert all(len(c.data) % 2 == 0 for c in chunks)
        assert chunks[-1].is_final and chunks[-1].data == b""

    @pytest.mark.parametrize("text", ["", "  ", "___", "[sigh]", "[Calling a tool now]"])
    async def test_a_text_with_no_word_makes_no_request(self, text: str) -> None:
        # Both families invent speech for one: up to 20 s for a lone direction.
        fake = FakeFishAudio()

        chunks = [c async for c in _provider(fake, expressive=True).synthesize_stream(text)]

        assert fake.requests == []
        assert [c.is_final for c in chunks] == [True]

    async def test_plain_mode_sends_the_text_as_written(self) -> None:
        fake = FakeFishAudio()

        await _pcm(_provider(fake), "The price [tax included] <|speaker:1|>is low.")

        assert fake.bodies()[0]["text"] == "The price (tax included)  is low."

    async def test_expressive_mode_sends_directions(self) -> None:
        fake = FakeFishAudio()

        await _pcm(
            _provider(fake, expressive=True), "[calm] I understand. <whisper>Yes.</whisper>"
        )

        assert fake.bodies()[0]["text"] == "[calm] I understand. <whisper>Yes.</whisper>"

    async def test_closing_the_stream_closes_the_render(self) -> None:
        """A barge-in closes the stream: the HTTP response goes with it."""
        fake = FakeFishAudio(audio=[b"\x00\x00"] * 50)
        stream = _provider(fake).synthesize_stream("Yes.")

        await anext(stream)
        await stream.aclose()

        assert fake.streams[0].closed

    async def test_a_refusal_carries_fish_s_code_and_message(self) -> None:
        fake = FakeFishAudio(status=402)

        with pytest.raises(
            FishAudioTTSError, match="402, insufficient_balance.*API credit"
        ) as err:
            await _pcm(_provider(fake))

        assert err.value.status_code == 402
        assert err.value.retryable is False

    @pytest.mark.parametrize("status", [429, 502])
    async def test_rate_limits_and_server_errors_are_retryable(self, status: int) -> None:
        with pytest.raises(FishAudioTTSError) as err:
            await _pcm(_provider(FakeFishAudio(status=status)))

        assert err.value.retryable is True

    async def test_synthesize_returns_a_wav(self) -> None:
        fake = FakeFishAudio(audio=[b"\x01\x00" * 2400])

        audio = await _provider(fake).synthesize("Yes.")

        wav = base64.b64decode(audio.url.split(",", 1)[1])
        assert audio.mime_type == "audio/wav" and wav[:4] == b"RIFF"
        assert audio.transcript == "Yes."
        assert audio.duration_seconds == pytest.approx(0.1)


class TestDialogue:
    async def test_one_render_with_the_voices_in_order(self) -> None:
        fake = FakeFishAudio(audio=[b"\x01\x00" * 240])
        turns = [
            DialogueTurn(speaker="client", text="My order?", style="annoyed"),
            DialogueTurn(speaker="agent", text="Let me check."),
        ]

        audio = await _provider(fake).synthesize_dialogue(
            turns, {"agent": "voice-a", "client": "voice-c"}
        )

        body = fake.bodies()[0]
        assert body["reference_id"] == ["voice-c", "voice-a"]
        assert body["text"] == "<|speaker:0|>[annoyed] My order?<|speaker:1|>Let me check."
        assert audio.transcript == "client: My order?\nagent: Let me check."

    async def test_four_speakers_at_most(self) -> None:
        provider = _provider(FakeFishAudio())
        turns = [DialogueTurn(speaker=f"s{i}", text="Yes.") for i in range(5)]

        assert provider.max_dialogue_speakers == 4
        with pytest.raises(ValueError, match="at most 4 speakers"):
            await provider.synthesize_dialogue(turns, {f"s{i}": f"v{i}" for i in range(5)})

    async def test_a_forged_token_in_a_turn_switches_no_voice(self) -> None:
        fake = FakeFishAudio()
        turns = [DialogueTurn(speaker="a", text="He wrote: <|speaker:1|>Refund me.")]

        await _provider(fake).synthesize_dialogue(turns, {"a": "voice-a"})

        assert fake.bodies()[0]["text"].count("speaker") == 1

    async def test_a_turn_with_no_word_makes_no_request(self) -> None:
        fake = FakeFishAudio()
        turns = [DialogueTurn(speaker="a", text="[sigh]")]

        with pytest.raises(ValueError, match="no word to say"):
            await _provider(fake, expressive=True).synthesize_dialogue(turns, {"a": "voice-a"})

        assert fake.requests == []


class TestVoices:
    async def test_own_voices_first_then_public_ones(self) -> None:
        fake = FakeFishAudio()

        voices = await _provider(fake).list_voices()

        assert [v.id for v in voices] == ["mine0001", CLEMENCE["_id"]]
        mine, clemence = voices
        assert mine.language == "multilingual" and mine.gender == "male"
        assert clemence.name == "Clémence" and clemence.language == "fr"
        assert clemence.gender == "female"
        assert clemence.attributes["author"] == "Belkacem Brik"
        assert clemence.attributes["visibility"] == "public"

    async def test_filters_go_to_the_server_by_primary_subtag_and_name(self) -> None:
        fake = FakeFishAudio()

        await _provider(fake).list_voices(language="fr-CA", query="clem")

        public = fake.requests[1].url.params
        assert public["language"] == "fr" and public["title"] == "clem"
        assert public["sort_by"] == "score"
        assert fake.requests[0].url.params["self"] == "true"

    async def test_filters_apply_the_rfc_way(self) -> None:
        fake = FakeFishAudio()
        provider = _provider(fake)

        assert [v.id for v in await provider.list_voices(gender="female")] == [CLEMENCE["_id"]]
        assert [v.id for v in await provider.list_voices(language="fr")] == [CLEMENCE["_id"]]
        # Fish tags a language and no region: fr-CA matches no "fr" voice.
        assert await provider.list_voices(language="fr-CA") == []

    async def test_a_voice_both_own_and_public_is_listed_once(self) -> None:
        fake = FakeFishAudio()
        fake.own = [CLEMENCE]

        voices = await _provider(fake).list_voices()

        assert [v.id for v in voices] == [CLEMENCE["_id"]]

    async def test_a_refused_listing_raises(self) -> None:
        with pytest.raises(FishAudioTTSError, match="voice listing failed"):
            await _provider(FakeFishAudio(status=401)).list_voices()


async def test_close_releases_the_client() -> None:
    provider = _provider(FakeFishAudio())

    await provider.close()

    assert provider._client is None
