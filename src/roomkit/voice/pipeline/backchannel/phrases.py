"""A backchannel detector that knows the acknowledgements people say."""

from __future__ import annotations

import re
from collections.abc import Iterable

from roomkit.voice.pipeline.backchannel.base import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)

ENGLISH_BACKCHANNELS: tuple[str, ...] = (
    # Continuers and acknowledgements
    "ok",
    "okay",
    "alright",
    "all right",
    "yeah",
    "yes",
    "yep",
    "yup",
    "sure",
    "sure thing",
    "right",
    "mm",
    "mhm",
    "mm-hmm",
    "um-hum",
    "uh-huh",
    "uhhuh",
    "hmm",
    "hum",
    "aha",
    "ah",
    "oh",
    "i see",
    "got it",
    "gotcha",
    "understood",
    "i understand",
    "noted",
    "go on",
    "go ahead",
    "keep going",
    "continue",
    # Agreement
    "exactly",
    "true",
    "indeed",
    "of course",
    "absolutely",
    "definitely",
    "totally",
    "certainly",
    "agreed",
    "i agree",
    "fair enough",
    "makes sense",
    "that makes sense",
    "good point",
    "that's right",
    "you're right",
    # Assessments; "very" is no phrase of its own, nor is "that's", or "that's okay" (a
    # polite no) would pass
    "cool",
    "nice",
    "great",
    "good",
    "perfect",
    "excellent",
    "awesome",
    "amazing",
    "wonderful",
    "fantastic",
    "brilliant",
    "lovely",
    "sweet",
    "neat",
    "incredible",
    "interesting",
    "funny",
    "crazy",
    "very good",
    "very nice",
    "very cool",
    "very interesting",
    "very funny",
    "sounds good",
    "sounds great",
    "love it",
    "that's great",
    "that's cool",
    "that's interesting",
    "that's amazing",
    "that's awesome",
    "that's funny",
    "that's nice",
    # Surprise, sympathy, relief, disgust
    "wow",
    "oh my",
    "my god",
    "oh god",
    "my gosh",
    "gosh",
    "my goodness",
    "jeez",
    "geez",
    "oh boy",
    "oh man",
    "oh dear",
    "yay",
    "aw",
    "ouch",
    "oof",
    "yikes",
    "phew",
    "ew",
    "gross",
    "yuck",
    "ugh",
    # Laughter, spelled out or as the words of a tag ("[laughter]", "(laughs)")
    "ha",
    "haha",
    "hahaha",
    "hahahaha",
    "heh",
    "hehe",
    "laugh",
    "laughs",
    "laughter",
    "laughing",
    "chuckles",
    # Thanks
    "thanks",
    "thank you",
    "thanks a lot",
    "thank you so much",
    "thanks so much",
    "thank you very much",
    "much appreciated",
    "appreciate it",
    "cheers",
)
"""Acknowledgements, continuers, agreement, assessments, reactions, laughter and thanks
in English, said to what the bot just said or did, not to stop it (RMK-696). Left out
on purpose: what asks it to stop or repeat ("wait", "huh"), disagrees ("no", "oh no",
"but"), doubts ("really?", "for sure?", "that's true?") or opens a turn ("um", "well",
"so")."""

FRENCH_BACKCHANNELS: tuple[str, ...] = (
    # Acknowledgements and continuers
    "oui",
    "ouais",
    "ouin",
    "ouain",
    "ok",
    "okay",
    "d'accord",
    "mm",
    "mmh",
    "mhm",
    "hm",
    "hum",
    "ah",
    "oh",
    "ah bon",
    "bon",
    "bien",
    "voilà",
    "je vois",
    "je comprends",
    "compris",
    "entendu",
    "noté",
    "ça marche",
    "pas de problème",
    "pas de souci",
    "pas de soucis",
    "vas-y",
    "allez-y",
    "continue",
    "envoye",
    "envoie",
    # Agreement; « sûr », « vrai » and « certain » stay out, alone or after « c'est »:
    # « C'est vrai ? » doubts
    "exact",
    "exactement",
    "effectivement",
    "en effet",
    "tout à fait",
    "c'est ça",
    "bien sûr",
    "c'est clair",
    "clairement",
    "évidemment",
    "absolument",
    "certainement",
    "carrément",
    "mets-en",
    "en plein ça",
    "t'as raison",
    "tu as raison",
    "vous avez raison",
    # Assessments; « très », « trop » and « c'est » are no phrases of their own: « Trop ! »
    # protests, and « c'est bon » is also "enough!"
    "très bien",
    "très intéressant",
    "très drôle",
    "trop bien",
    "trop cool",
    "trop drôle",
    "super",
    "cool",
    "génial",
    "parfait",
    "excellent",
    "magnifique",
    "incroyable",
    "top",
    "nickel",
    "chouette",
    "sympa",
    "intéressant",
    "drôle",
    "malade",
    "écœurant",
    "écoeurant",
    "tiguidou",
    "pas pire",
    "pas mal",
    "tant mieux",
    "bravo",
    "c'est génial",
    "c'est cool",
    "c'est super",
    "c'est fou",
    "c'est malade",
    "c'est intéressant",
    "nice",
    # Surprise and reactions; "là" is no phrase of its own: « ben là » protests
    "wow",
    "ouah",
    "wouah",
    "waouh",
    "ouh",
    "oh là là",
    "oh la la",
    "oh là là là",
    "ouh là là",
    "ouh là",
    "oulà",
    "houlà",
    "oulala",
    "mon dieu",
    "my god",
    "oh boy",
    "eh boy",
    "ayoye",
    "aïe",
    "ouille",
    "ouch",
    "ouf",
    # Disgust
    "berk",
    "beurk",
    "ouache",
    "yark",
    "dégueu",
    "dégueulasse",
    # Laughter, spelled out or as a subtitle tag ("(Rires)")
    "ha",
    "haha",
    "hahaha",
    "hahahaha",
    "hihi",
    "hihihi",
    "héhé",
    "rires",
    "rire",
    # Thanks
    "merci",
    "merci beaucoup",
    "merci bien",
    "merci mille fois",
    "merci infiniment",
    "c'est gentil",
)
"""The same in French, from France and Québec (« ouin », « ayoye », « mets-en »): thanks
included (live, a « Merci » to « je m'occupe de ça » cut the result that came next,
RMK-555), and reactions (an « Ouh. » cut a story, RMK-696). Left out on purpose:
« non », « attends », « hein », « euh », « ben là », « voyons donc », « c'est bon » (also
"enough!"), « c'est correct » (a polite no in Québec), doubts (« c'est vrai ? »,
« c'est sûr ? ») and what opens a turn (« ben », « dis donc », « justement »)."""

_WORD = re.compile(r"[^\W_]+")  # letters and digits: apostrophes and hyphens split words
_REPEATED = re.compile(r"(\w)\1+")
# A "no" an STT writes as hummed syllables (Deepgram's "mm-mm", "uh-uh", "nuh-uh"):
# split on the hyphen, "mm-mm" read as two "mm", an acknowledgement.
_HUMMED_NO = re.compile(r"\b(?:m+-m+|u+h+-u+h+|h+u+h+-u+h+|n+u+h+-u+h+)\b", re.IGNORECASE)


def _words(text: str) -> tuple[str, ...]:
    """Lowercase words, each repeated letter written once.

    Applied to the phrases and the transcript alike, so a held sound matches
    however the STT spells it: "hmm", "hmmm" and "hm" are one word, as are
    "oui" and "ouiii".
    """
    return tuple(_WORD.findall(_REPEATED.sub(r"\1", text.lower())))


def _heard(transcript: str) -> str:
    """*transcript* with a hummed "no" ("mm-mm") written "no".

    The transcript's only: a phrase list naming "mm-mm" never makes "no" an
    acknowledgement.
    """
    return _HUMMED_NO.sub(" no ", transcript)


class PhraseBackchannelDetector(BackchannelDetector):
    """A backchannel is an utterance made only of known acknowledgements.

    "Okay", "mm-hmm", "yeah, right" or "d'accord" let the bot keep talking;
    "okay, and what about Calgary?" does not, since "and what about Calgary"
    is not an acknowledgement. The channel classifies each partial transcript
    of the speech as it grows, so words added after an acknowledgement still
    cut in.

    The detector reads words. Without any (the transcript is empty or not
    there yet), it judges no utterance a backchannel and the strategy falls
    back on speech duration, as ``CONFIRMED`` does (RFC §12.3.13).

    A partial transcript (``BackchannelContext.is_final`` false) that is
    acknowledgements and then the start of one gives no verdict yet
    (``pending``): its last word may be a piece of the word being said, as a
    streaming STT that writes pieces of words gives it ("Ou" of "Ouais"), or
    a phrase may be half said ("Thank" of "Thank you"). Its next words decide.
    A last word followed by a space or punctuation is whole.

    Args:
        phrases: The acknowledgements, matched case-insensitively and
            regardless of punctuation. Defaults to English and French.
        max_words: Longer utterances are never backchannels, whatever their
            words ("yeah yeah yeah, okay, sure, right" is someone talking).
        cut_without_words: Whether an utterance without words may cut the bot
            (the default): judged no backchannel, the strategy falls back on
            its duration. ``False`` judges it a backchannel, so only words
            interrupt: echo an AEC left, a cough or room noise never cut the
            bot, and neither does a wordless interruption (RFC §12.3.13).
    """

    def __init__(
        self,
        phrases: Iterable[str] = ENGLISH_BACKCHANNELS + FRENCH_BACKCHANNELS,
        *,
        max_words: int = 4,
        cut_without_words: bool = True,
    ) -> None:
        self._phrases = frozenset(words for phrase in phrases if (words := _words(phrase)))
        self._longest = max((len(p) for p in self._phrases), default=0)
        self._max_words = max_words
        self._cut_without_words = cut_without_words

    @property
    def name(self) -> str:
        return "PhraseBackchannelDetector"

    def classify(self, context: BackchannelContext) -> BackchannelDecision:
        transcript = _heard(context.transcript or "")
        words = _words(transcript)
        if not words:
            return BackchannelDecision(
                is_backchannel=not self._cut_without_words, confidence=0.5, label="no words"
            )
        if len(words) > self._max_words:
            return BackchannelDecision(is_backchannel=False, label="interruption")
        reachable = self._reachable(words)
        if reachable[-1]:
            return BackchannelDecision(is_backchannel=True, label="acknowledgement")
        # A last word the transcript ends inside may be a piece of the word.
        open_end = _WORD.fullmatch(transcript[-1:]) is not None
        if not context.is_final and self._may_become_phrases(words, reachable, open_end):
            return BackchannelDecision(
                is_backchannel=False, confidence=0.5, label="unfinished", pending=True
            )
        return BackchannelDecision(is_backchannel=False, label="interruption")

    def _reachable(self, words: tuple[str, ...]) -> list[bool]:
        """For each position of *words*, whether the words before it split
        entirely into known phrases."""
        reachable = [True] + [False] * len(words)
        for start in range(len(words)):
            if not reachable[start]:
                continue
            for size in range(1, min(self._longest, len(words) - start) + 1):
                if words[start : start + size] in self._phrases:
                    reachable[start + size] = True
        return reachable

    def _may_become_phrases(
        self, words: tuple[str, ...], reachable: list[bool], open_end: bool
    ) -> bool:
        """Whether *words* are known phrases, then the start of one: its first
        words, the last of them only begun when *open_end*."""
        return any(
            _begins(phrase, words[start:], open_end)
            for start in range(len(words))
            if reachable[start]
            for phrase in self._phrases
        )


def _begins(phrase: tuple[str, ...], words: tuple[str, ...], open_end: bool) -> bool:
    """Whether *words* begin *phrase* without saying all of it."""
    *whole, last = words
    if len(phrase) < len(words) or phrase[: len(whole)] != tuple(whole):
        return False
    word = phrase[len(whole)]
    if word == last:
        return len(phrase) > len(words)
    return open_end and word.startswith(last)
