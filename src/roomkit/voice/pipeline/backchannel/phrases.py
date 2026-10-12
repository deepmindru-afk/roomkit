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
    "for sure",
    "agreed",
    "i agree",
    "fair enough",
    "makes sense",
    "that makes sense",
    "good point",
    "that's right",
    "that's true",
    "you're right",
    # Assessments; "very" only ever joins listed words ("very cool"), and "that's" is no
    # phrase of its own, or "that's okay" (a polite no) would pass
    "very",
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
    "that's good",
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
    # A news-marker: "really" alone is often a challenge
    "oh really",
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
"but"), or opens a turn ("um", "well", "so")."""

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
    "ben",
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
    # Agreement; "sûr" and "vrai" alone stay out: « Sûr ? », « Vrai ? » doubt
    "exact",
    "exactement",
    "effectivement",
    "en effet",
    "tout à fait",
    "c'est ça",
    "c'est vrai",
    "c'est sûr",
    "bien sûr",
    "c'est clair",
    "clairement",
    "évidemment",
    "absolument",
    "certainement",
    "carrément",
    "justement",
    "certain",
    "mets-en",
    "en plein ça",
    "t'as raison",
    "tu as raison",
    "vous avez raison",
    # Assessments; "très" and "trop" only join listed words, and "c'est" is no phrase of
    # its own, or « c'est bon » (also "enough!") would pass
    "très",
    "très bien",
    "trop",
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
    "dis donc",
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
"enough!") and « c'est correct » (a polite no in Québec)."""

_WORD = re.compile(r"[^\W_]+")  # letters and digits: apostrophes and hyphens split words
_REPEATED = re.compile(r"(\w)\1+")
# A "no" an STT writes as hummed syllables (Deepgram's "mm-mm", "uh-uh", "nuh-uh"):
# split on the hyphen, "mm-mm" read as two "mm", an acknowledgement.
_HUMMED_NO = re.compile(r"\b(?:m+-m+|u+h+-u+h+|h+u+h+-u+h+|n+u+h+-u+h+)\b")


def _words(text: str) -> tuple[str, ...]:
    """Lowercase words, each repeated letter written once.

    Applied to the phrases and the transcript alike, so a held sound matches
    however the STT spells it: "hmm", "hmmm" and "hm" are one word, as are
    "oui" and "ouiii". A hummed "no" ("mm-mm") reads as "no".
    """
    lowered = _HUMMED_NO.sub(" no ", text.lower())
    return tuple(_WORD.findall(_REPEATED.sub(r"\1", lowered)))


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
        words = _words(context.transcript or "")
        if not words:
            return BackchannelDecision(
                is_backchannel=not self._cut_without_words, confidence=0.5, label="no words"
            )
        if len(words) <= self._max_words and self._made_of_phrases(words):
            return BackchannelDecision(is_backchannel=True, label="acknowledgement")
        return BackchannelDecision(is_backchannel=False, label="interruption")

    def _made_of_phrases(self, words: tuple[str, ...]) -> bool:
        """Whether *words* split entirely into known phrases."""
        reachable = [True] + [False] * len(words)
        for start in range(len(words)):
            if not reachable[start]:
                continue
            for size in range(1, min(self._longest, len(words) - start) + 1):
                if words[start : start + size] in self._phrases:
                    reachable[start + size] = True
        return reachable[-1]
