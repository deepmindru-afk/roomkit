"""Tests for the dedicated Gemini recogniser path of GeminiSTTProvider.

The fake answer mirrors ``gemini-3.5-transcribe`` on the live API (verified
2026-09-27): the transcript in ``output_text``, and, when word timestamps are
asked for, one ``word_info`` annotation per word on the step's text content,
with ``spk:N`` speakers when diarization is on and durations such as
``"0.300s"`` for offsets. No language is ever reported back.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.stt.gemini import (
    RECOGNISER_MODELS,
    GeminiSTTConfig,
    GeminiSTTProvider,
    TranscriptSegment,
    TranscriptWord,
)
from roomkit.voice.stt.gemini_recogniser import is_recogniser

MODEL = "gemini-3.5-transcribe"

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _word(text: str, start: str, end: str, speaker: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        type="word_info", text=text, start_offset=start, end_offset=end, speaker=speaker
    )


def _answer(
    text: str, words: list[SimpleNamespace] | None = None, status: str = "completed"
) -> SimpleNamespace:
    content = SimpleNamespace(type="text", text=text, annotations=words)
    return SimpleNamespace(
        status=status,
        output_text=text,
        steps=[SimpleNamespace(type="model_output", content=[content])],
    )


DIALOGUE = _answer(
    "Salut. Ça va?Oui.",
    [
        _word("Salut.", "0.300s", "0.800s", "spk:3"),
        _word("Ça", "1.300s", "1.400s", "spk:3"),
        _word("va?", "1.400s", "1.700s", "spk:3"),
        _word("Oui.", "63.800s", "64.100s", "spk:0"),
    ],
)


class _FakeInteractions:
    def __init__(self, answer: SimpleNamespace) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.answer


def _provider(
    answer: SimpleNamespace = DIALOGUE, **overrides: Any
) -> tuple[GeminiSTTProvider, _FakeInteractions]:
    config: dict[str, Any] = {"api_key": "test-key", "model": MODEL, **overrides}
    provider = GeminiSTTProvider(GeminiSTTConfig(**config))
    interactions = _FakeInteractions(answer)
    provider._client = SimpleNamespace(aio=SimpleNamespace(interactions=interactions))
    return provider, interactions


def _frame() -> AudioFrame:
    return AudioFrame(data=b"\x01\x02" * 160, sample_rate=16000, channels=1)


# ---------------------------------------------------------------------------
# Model resolution and configuration
# ---------------------------------------------------------------------------


class TestModelResolution:
    @pytest.mark.parametrize("model", [*RECOGNISER_MODELS, "gemini-4.0-transcribe"])
    def test_transcribe_models_are_recognisers(self, model: str) -> None:
        assert is_recogniser(model)

    @pytest.mark.parametrize(
        "model", ["gemini-3.8-flash", "gemini-3.5-transcribe-live", "gemini-3.8-live"]
    )
    def test_other_models_are_prompted(self, model: str) -> None:
        assert not is_recogniser(model)


class TestConfiguration:
    def test_defaults_ask_for_speakers_and_word_timing(self) -> None:
        config = GeminiSTTConfig(api_key="k", model=MODEL)
        assert (config.mode, config.diarize, config.word_timestamps) == ("verbatim", True, True)

    def test_mode_is_case_insensitive(self) -> None:
        config = GeminiSTTConfig(
            api_key="k", model=MODEL, mode="SMART", diarize=False, word_timestamps=False
        )
        assert config.mode == "smart"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"mode": "fast"}, "mode must be one of"),
            ({"custom_vocabulary": ["x"] * 1001}, "at most 1000"),
            ({"model": "gemini-3.8-flash", "mode": "smart"}, "needs a dedicated recogniser"),
            ({"prompt": "Spell it RoomKit"}, "takes no prompt"),
            ({"word_timestamps": False}, "diarize needs word_timestamps"),
            ({"mode": "smart"}, "mode='smart' cannot be combined"),
            ({"mode": "smart", "diarize": False}, "mode='smart' cannot be combined"),
            ({"custom_vocabulary": ["RoomKit"]}, "custom_vocabulary cannot be combined"),
            (
                {"custom_vocabulary": ["RoomKit"], "diarize": False},
                "custom_vocabulary cannot be combined",
            ),
            (
                {"custom_vocabulary": ["RoomKit"], "diarize": False, "word_timestamps": False},
                "custom_vocabulary needs language",
            ),
        ],
    )
    def test_what_the_recogniser_refuses_is_refused_at_construction(
        self, overrides: dict[str, Any], message: str
    ) -> None:
        """Each answers a 400 on the live API, or a transcript cut after one
        sentence for a vocabulary with no language; here it fails before the call."""
        values: dict[str, Any] = {"api_key": "k", "model": MODEL, **overrides}
        with pytest.raises(ValueError, match=message):
            GeminiSTTConfig(**values)

    def test_a_multimodal_model_keeps_its_prompt_and_timing_options(self) -> None:
        config = GeminiSTTConfig(
            api_key="k", prompt="Formal register.", word_timestamps=False, diarize=True
        )
        assert config.prompt == "Formal register."


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequestShape:
    async def test_default_asks_verbatim_with_speakers_and_words(self) -> None:
        provider, interactions = _provider()

        await provider.transcribe_recording(_frame())

        call = interactions.calls[0]
        assert call["model"] == MODEL
        # No prompt and no response schema: the recogniser answers 400 to both.
        assert len(call["input"]) == 1
        assert call["input"][0]["type"] == "audio"
        assert "response_format" not in call
        assert call["generation_config"] == {
            "transcription_config": {
                "mode": {
                    "type": "verbatim",
                    "diarization_mode": "speaker",
                    "timestamp_granularities": ["word"],
                }
            }
        }

    async def test_language_becomes_the_language_code(self) -> None:
        provider, interactions = _provider(language="fr-CA")

        await provider.transcribe_recording(_frame())

        config = interactions.calls[0]["generation_config"]["transcription_config"]
        assert config["language_codes"] == ["fr-CA"]

    async def test_smart_mode_is_sent_as_the_bare_enum(self) -> None:
        provider, interactions = _provider(
            _answer("Salut."), mode="smart", diarize=False, word_timestamps=False
        )

        await provider.transcribe_recording(_frame())

        config = interactions.calls[0]["generation_config"]["transcription_config"]
        assert config == {"mode": "smart"}

    async def test_vocabulary_rides_on_plain_verbatim_with_its_language(self) -> None:
        provider, interactions = _provider(
            _answer("RoomKit."),
            custom_vocabulary=["RoomKit"],
            language="fr-CA",
            diarize=False,
            word_timestamps=False,
        )

        await provider.transcribe_recording(_frame())

        config = interactions.calls[0]["generation_config"]["transcription_config"]
        assert config == {
            "language_codes": ["fr-CA"],
            "custom_vocabulary": ["RoomKit"],
            "mode": {"type": "verbatim"},
        }

    async def test_a_multimodal_model_reads_the_vocabulary_in_its_prompt(self) -> None:
        provider = GeminiSTTProvider(
            GeminiSTTConfig(api_key="k", custom_vocabulary=["RoomKit", "Luge"])
        )

        prompt = provider._build_prompt()

        assert "Spell these terms exactly as written when they are spoken: RoomKit, Luge." in (
            prompt
        )


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


class TestTranscript:
    async def test_turns_are_runs_of_one_speaker_labelled_in_order(self) -> None:
        provider, _ = _provider(language="fr-CA")

        transcript = await provider.transcribe_recording(_frame())

        # spk:3 speaks first, so it is Speaker 1 whatever its service label.
        assert transcript.segments == [
            TranscriptSegment("Speaker 1", "00:00", "00:01", "Salut. Ça va?"),
            TranscriptSegment("Speaker 2", "01:03", "01:04", "Oui."),
        ]
        assert transcript.words[0] == TranscriptWord("Salut.", 0.3, 0.8, "Speaker 1")
        assert transcript.words[-1] == TranscriptWord("Oui.", 63.8, 64.1, "Speaker 2")
        assert transcript.language == "fr-CA"

    async def test_a_start_sent_back_in_time_is_repaired(self) -> None:
        # The live service's answer (2026-09-27): on a change of speaker the
        # first word's start came ten seconds early, dragging its turn back.
        answer = _answer(
            "le mois de novembre. D'accord. Je",
            [
                _word("novembre.", "13.600s", "14.100s", "spk:0"),
                _word("D'accord.", "4.300s", "14.900s", "spk:1"),
                _word("Je", "15.200s", "15.300s", "spk:1"),
            ],
        )
        provider, _ = _provider(answer)

        transcript = await provider.transcribe_recording(_frame())

        assert transcript.words[1] == TranscriptWord("D'accord.", 14.1, 14.9, "Speaker 2")
        assert transcript.segments[1] == TranscriptSegment(
            "Speaker 2", "00:14", "00:15", "D'accord. Je"
        )

    async def test_words_without_speakers_make_one_turn(self) -> None:
        answer = _answer(
            "Bonjour à tous.",
            [
                _word("Bonjour", "0.100s", "0.500s"),
                _word("à", "0.500s", "0.600s"),
                _word("tous.", "0.600s", "1.000s"),
            ],
        )
        provider, _ = _provider(answer, diarize=False)

        transcript = await provider.transcribe_recording(_frame())

        assert transcript.segments == [
            TranscriptSegment("Speaker 1", "00:00", "00:01", "Bonjour à tous.")
        ]
        assert all(word.speaker is None for word in transcript.words)

    async def test_without_words_the_text_is_one_untimed_segment(self) -> None:
        provider, _ = _provider(_answer("Bonjour à tous."), diarize=False, word_timestamps=False)

        transcript = await provider.transcribe_recording(_frame())

        assert transcript.segments == [TranscriptSegment("Speaker 1", "", "", "Bonjour à tous.")]
        assert transcript.words == []

    async def test_silence_is_an_empty_transcript_not_an_error(self) -> None:
        provider, _ = _provider(_answer(""), diarize=False, word_timestamps=False)

        transcript = await provider.transcribe_recording(_frame())

        assert transcript.segments == []
        assert transcript.plain_text == ""

    async def test_an_unfinished_interaction_raises_with_its_status(self) -> None:
        provider, _ = _provider(_answer("", status="failed"))

        with pytest.raises(RuntimeError, match="status=failed"):
            await provider.transcribe_recording(_frame())

    async def test_an_unreadable_offset_raises(self) -> None:
        provider, _ = _provider(_answer("Salut.", [_word("Salut.", "soon", "0.800s", "spk:0")]))

        with pytest.raises(RuntimeError, match="unreadable word offset"):
            await provider.transcribe_recording(_frame())

    async def test_undetected_language_flattens_to_none(self) -> None:
        """The recogniser never reports the language it detected."""
        provider, _ = _provider()

        result = await provider.transcribe(_frame())

        assert result.text == "Salut. Ça va? Oui."
        assert result.language is None
        assert result.is_final is True
