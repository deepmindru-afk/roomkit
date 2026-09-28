"""Gemini TTS: the live voice catalog and dialogue synthesis (RFC §12.2).

The fakes mirror the live API (verified 2026-09-27): ``voices.list`` pages the
catalog with a ``next_page_token``, entries carry Google's own fields (persona,
region, type…), and a two-speaker dialogue binds each speaker to a voice in
``speech_config`` and each text item to its speaker in ``speech_metadata``.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest

from roomkit.voice.tts.audio_utils import wrap_wav
from roomkit.voice.tts.gemini import GeminiTTSConfig, GeminiTTSProvider
from roomkit.voice.voices import DialogueTurn


def _voice(voice_id: str, language: str, gender: str, **fields: Any) -> SimpleNamespace:
    base = {
        "id": voice_id,
        "display_name": fields.pop("display_name", voice_id.title()),
        "language_code": language,
        "gender": gender,
        "accent": None,
        "description": None,
        "persona": None,
        "context": None,
        "region_code": None,
        "model": None,
        "key": None,
        "type": "prebuilt",
        "pitch": None,
        "expire_time": None,
    }
    base.update(fields)
    return SimpleNamespace(**base)


PAGES = [
    SimpleNamespace(
        voices=[
            _voice("voice_abc", "fr-CA", "female", type="prompted", display_name="Ma voix"),
            _voice(
                "fr-ca-advisor-1",
                "fr-CA",
                "female",
                accent="Montreal French",
                persona="High-Trust Advisor",
                region_code="CA",
                description="31-year-old Lawyer from Montreal.",
            ),
        ],
        next_page_token="page-2",
    ),
    SimpleNamespace(
        voices=[
            _voice("fr-fr-advisor-1", "fr-FR", "male"),
            _voice("achernar", "en-US", "female", description="Soft, calm"),
        ],
        next_page_token=None,
    ),
]


class _FakeVoices:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def list(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return PAGES[0] if kwargs.get("page_token") is None else PAGES[1]


class _FakeInteractions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        wav = wrap_wav(b"\x01\x02" * 12000, 24000, 1)  # 0.5 s
        return SimpleNamespace(
            status="completed",
            output_audio=SimpleNamespace(
                data=base64.b64encode(wav).decode(),
                mime_type="audio/wav",
                sample_rate=None,
                channels=None,
            ),
        )


def _provider(**overrides: Any) -> tuple[GeminiTTSProvider, SimpleNamespace]:
    provider = GeminiTTSProvider(GeminiTTSConfig(api_key="test-key", **overrides))
    aio = SimpleNamespace(voices=_FakeVoices(), interactions=_FakeInteractions())
    provider._client = SimpleNamespace(aio=aio)
    return provider, aio


class TestListVoices:
    async def test_the_whole_catalog_is_read_page_by_page(self) -> None:
        provider, aio = _provider()

        voices = await provider.list_voices()

        assert [v.id for v in voices] == [
            "voice_abc",
            "fr-ca-advisor-1",
            "fr-fr-advisor-1",
            "achernar",
        ]
        assert [c.get("page_token") for c in aio.voices.calls] == [None, "page-2"]
        # The service's filters mean something else; none are sent.
        assert all(set(c) == {"page_size", "page_token"} for c in aio.voices.calls)

    async def test_googles_fields_map_to_voice_info(self) -> None:
        provider, _ = _provider()

        advisor = next(v for v in await provider.list_voices() if v.id == "fr-ca-advisor-1")

        assert advisor.language == "fr-CA"
        assert advisor.gender == "female"
        assert advisor.accent == "Montreal French"
        assert advisor.description == "31-year-old Lawyer from Montreal."
        assert advisor.attributes == {
            "persona": "High-Trust Advisor",
            "region_code": "CA",
            "type": "prebuilt",
        }

    async def test_a_language_prefix_finds_every_region(self) -> None:
        """The service answers nothing for ``fr``; RFC §12.2 wants fr-CA and fr-FR."""
        provider, _ = _provider()

        french = await provider.list_voices(language="fr")

        assert [v.id for v in french] == ["voice_abc", "fr-ca-advisor-1", "fr-fr-advisor-1"]

    async def test_filters_combine(self) -> None:
        provider, _ = _provider()

        found = await provider.list_voices(language="fr-CA", gender="female", query="lawyer")

        assert [v.id for v in found] == ["fr-ca-advisor-1"]

    async def test_a_custom_voice_says_its_type(self) -> None:
        provider, _ = _provider()

        mine = (await provider.list_voices(query="ma voix"))[0]

        assert mine.attributes["type"] == "prompted"


TURNS = [
    DialogueTurn(speaker="Client", text="Bonjour, ma facture est trop élevée."),
    DialogueTurn(speaker="Conseillère", text="Je vérifie ça.", style="calme"),
    DialogueTurn(speaker="Client", text="Merci."),
]
VOICES = {"Client": "fr-ca-advisor-2", "Conseillère": "fr-ca-advisor-1"}


class TestDialogue:
    def test_two_speakers_from_3_8_none_before(self) -> None:
        assert _provider()[0].max_dialogue_speakers == 2
        assert _provider(model="gemini-3.1-flash-tts-preview")[0].max_dialogue_speakers == 0

    async def test_each_turn_carries_its_speaker_and_each_speaker_its_voice(self) -> None:
        provider, aio = _provider(language="fr-CA", style_prompt="ignored in a dialogue")

        await provider.synthesize_dialogue(TURNS, VOICES)

        call = aio.interactions.calls[0]
        assert call["model"] == "gemini-3.8-flash-tts"
        assert call["response_format"] == {"type": "audio"}
        assert call["generation_config"] == {
            "speech_config": [
                {"speaker": "Client", "voice": "fr-ca-advisor-2", "language": "fr-CA"},
                {"speaker": "Conseillère", "voice": "fr-ca-advisor-1", "language": "fr-CA"},
            ]
        }
        items = call["input"][0]["content"]
        assert [item["text"] for item in items] == [t.text for t in TURNS]
        assert [item["annotations"] for item in items] == [
            [{"type": "speech_metadata", "speaker": "Client"}],
            [{"type": "speech_metadata", "speaker": "Conseillère", "style": "calme"}],
            [{"type": "speech_metadata", "speaker": "Client"}],
        ]

    async def test_the_clip_is_the_wav_the_service_answered(self) -> None:
        provider, _ = _provider()

        audio = await provider.synthesize_dialogue(TURNS, VOICES)

        assert audio.mime_type == "audio/wav"
        assert audio.duration_seconds == pytest.approx(0.5)
        assert audio.transcript.splitlines() == [
            "Client: Bonjour, ma facture est trop élevée.",
            "Conseillère: Je vérifie ça.",
            "Client: Merci.",
        ]

    async def test_a_third_speaker_is_refused_before_any_call(self) -> None:
        provider, aio = _provider()
        turns = [*TURNS, DialogueTurn(speaker="Gérant", text="Je m'en occupe.")]

        with pytest.raises(ValueError, match="at most 2 speakers"):
            await provider.synthesize_dialogue(turns, {**VOICES, "Gérant": "achernar"})

        assert aio.interactions.calls == []

    async def test_a_model_without_dialogue_refuses_it(self) -> None:
        provider, aio = _provider(model="gemini-3.1-flash-tts-preview")

        with pytest.raises(NotImplementedError):
            await provider.synthesize_dialogue(TURNS, VOICES)

        assert aio.interactions.calls == []
