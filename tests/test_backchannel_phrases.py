"""PhraseBackchannelDetector: an utterance made only of acknowledgements is a backchannel."""

from __future__ import annotations

import pytest

from roomkit.voice.pipeline.backchannel import (
    ENGLISH_BACKCHANNELS,
    BackchannelContext,
    PhraseBackchannelDetector,
)


def _is_backchannel(text: str | None, detector: PhraseBackchannelDetector | None = None) -> bool:
    detector = detector or PhraseBackchannelDetector()
    context = BackchannelContext(transcript=text, speech_duration_ms=800)
    return detector.classify(context).is_backchannel


@pytest.mark.parametrize(
    "text",
    [
        "Okay",
        "Okay.",
        "Yeah, right",
        "Okay, interesting",
        "Mm-hmm",
        "mhm",
        "Hmmm",
        "Uh huh",
        "I see",
        "oh okay",
        "Oui",
        "Ouiii",
        "D'accord",
        "d’accord",
        "C'est ça",
        "Ah bon",
        "Tout à fait",
        "Merci.",
        "Merci beaucoup",
        "Ok, merci",
        "Thank you",
        "Thanks a lot",
        "Beurk",
    ],
)
def test_acknowledgements_are_backchannels(text: str) -> None:
    assert _is_backchannel(text)


@pytest.mark.parametrize(
    "text",
    [
        "No",
        "Wait",
        "Stop",
        "Okay, and do you remember the number",
        "Attends, stop",
        "Yeah but what about Calgary",
        "Is it dangerous",
        "Merci, mais attends",
        "Non merci",
        "Thanks, but stop",
    ],
)
def test_anything_else_is_an_interruption(text: str) -> None:
    assert not _is_backchannel(text)


@pytest.mark.parametrize(
    "text",
    [
        # RMK-696: an « Ouh. » cut a story
        "Ouh.",
        "Ouuuh !",
        "Ouah!",
        "Waouh",
        "Wow.",
        "Génial.",
        "Ah ouin.",
        "Ayoye!",
        "Oh là là",
        "Oh la la !",
        "Oulala",
        "Ah ben dis donc",
        "C'est vrai ?",
        "Ben oui.",
        "Ouais ouais ouais",
        "C'est sûr",
        "Bien sûr",
        "Mets-en!",
        "Pas pire",
        "Trop bien!",
        "Très intéressant",
        "C'est malade!",
        "T'as raison",
        "Vas-y",
        "Envoye!",
        "Hahaha!",
        "Ha ha ha",
        "(Rires)",
        "Ouache",
        "Oh mon dieu",
        "uhhuh",
        "um-hum",
        "Oh really?",
        "Oh my god",
        "That's awesome",
        "Very cool",
        "[laughter]",
        "(laughs)",
        "Thanks so much!",
        "Got it, thanks",
        "Sounds good",
        "Fair enough",
        "Ew, gross",
        "Aww",
        "Go on",
        "Makes sense",
    ],
)
def test_reactions_and_regional_acknowledgements_are_backchannels(text: str) -> None:
    assert _is_backchannel(text)


@pytest.mark.parametrize(
    "text",
    [
        "Ouh, attends",
        "Euh, je voulais dire",
        "Ah non",
        "Oh non",
        "Hein?",
        "Pardon?",
        "Pantoute",
        "Ben là!",
        "Voyons donc",
        "Sérieux?",
        "C'est pas vrai",
        "C'est bon",
        "C'est correct",
        "Laisse faire",
        "Je sais",
        "Mouais",
        "Trop fort",
        "C'est trop",
        "Oups",
        "hold on",
        "huh?",
        "excuse me",
        "well",
        "um",
        "really?",
        "no way",
        "oh no",
        "whoa whoa whoa",
        "I know",
        "that's enough",
        "that's okay",
        "yeah but",
    ],
)
def test_what_stops_doubts_or_opens_a_turn_still_cuts(text: str) -> None:
    assert not _is_backchannel(text)


@pytest.mark.parametrize("text", ["mm-mm", "Mm-mm.", "Mmm-mmm", "uh-uh", "huh-uh", "nuh-uh"])
def test_a_hummed_no_cuts(text: str) -> None:
    """Deepgram writes "no" as "mm-mm": split on its hyphen, it read as two "mm"."""
    assert not _is_backchannel(text)
    assert _is_backchannel("Mm-hmm") and _is_backchannel("uh-huh")


@pytest.mark.parametrize("text", [None, "", "  ", "..."])
def test_without_words_the_duration_decides_by_default(text: str | None) -> None:
    assert not _is_backchannel(text)


@pytest.mark.parametrize("text", [None, "", "  ", "..."])
def test_only_words_interrupt_when_asked(text: str | None) -> None:
    """RMK-559: echo or noise the STT made no words of does not cut the bot."""
    words_only = PhraseBackchannelDetector(cut_without_words=False)
    assert _is_backchannel(text, words_only)
    assert not _is_backchannel("Attends, stop", words_only)
    assert _is_backchannel("Okay", words_only)


def test_an_acknowledgement_growing_into_a_question_stops_being_one() -> None:
    """The channel classifies each partial: the later words decide."""
    assert _is_backchannel("Okay,")
    assert not _is_backchannel("Okay, and do you")


def test_too_many_words_is_someone_talking() -> None:
    assert _is_backchannel("yeah yeah okay sure")
    assert not _is_backchannel("yeah yeah yeah okay sure")
    assert _is_backchannel("yeah yeah yeah okay sure", PhraseBackchannelDetector(max_words=6))


@pytest.mark.parametrize("text", [None, "", "  ", "..."])
def test_without_words_the_strategy_judges_on_duration(text: str | None) -> None:
    """No words, no verdict of backchannel: SEMANTIC falls back on duration."""
    decision = PhraseBackchannelDetector().classify(
        BackchannelContext(transcript=text, speech_duration_ms=300)
    )
    assert not decision.is_backchannel
    assert decision.label == "no words"


def test_phrases_can_be_replaced() -> None:
    english_only = PhraseBackchannelDetector(ENGLISH_BACKCHANNELS)
    assert _is_backchannel("okay", english_only)
    assert not _is_backchannel("ouais", english_only)
    assert _is_backchannel("tiens donc", PhraseBackchannelDetector(["tiens donc"]))
