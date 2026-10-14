"""The text a Fish Audio model reads: control tokens neutralised, markup per mode.

Fish Audio models read control tokens inside the text field: ``<|speaker:1|>``
switches to the second voice of a dialogue, ``<|phoneme_start|>…`` rewrites a
pronunciation. RoomKit writes the speaker tokens of a dialogue itself; any other
token in a text is a forgery, so every ``<|…|>`` becomes a space before the text
leaves (RFC §12.2, a turn cannot open another; RFC §6.4, replaced, never
deleted). Measured on s2.1-pro (2026-10-14), where a raw ``<|speaker:1|>`` in a
quoted line switched the voice on every take: a zero-width space inside the
token still switched it (Fish drops invisible characters), ``‹|speaker:1|›``
switched it one take in two, spaces in place of the marks had "speaker" read
aloud, and the whole token replaced by one space neither switched nor spoke.

Directions in square brackets (``[calm]``) and paired tags (``<whisper>``) are
performed, not read. In plain mode the text is spoken as written: brackets
become parentheses (Fish drops bracketed content otherwise) and tag-shaped
markup takes single guillemets, which Fish reads out. Expressive mode lets
both through.

A text with no word to say is never sent: both model families invent speech
for one (measured 2026-10-14: a lone ``[Calling show_join_codes now]`` gave up
to 20 s of unrelated speech over HTTP and 47 s on the socket, ``[pause]``
4 s). On a stream such a piece waits for the next one, whose words it colours.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from roomkit.voice.voices import DialogueTurn

# Characters a reader may drop before reading (Fish drops the zero-width
# space): format characters, variation selectors, the combining grapheme
# joiner, fillers. Between the marks of a token they would hide it.
_INVISIBLE = (
    "\u00ad\u034f\u061c\u115f-\u1160\u17b4-\u17b5\u180b-\u180f"
    "\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\u3164"
    "\ufe00-\ufe0f\ufeff\uffa0\ufff9-\ufffb\U0001bca0-\U0001bca3"
    "\U000e0001\U000e0020-\U000e007f\U000e0100-\U000e01ef"
)
# The marks and their look-alikes (fullwidth, small forms, single guillemets,
# angle brackets, the bars a model may read as "|").
_OPEN = "<\uff1c\ufe64\u2039\u2329\u3008"
_BAR = "|\uff5c\uffe8\u2223\u01c0\ufe31\ufe32\u00a6\u2502"
_CLOSE = ">\uff1e\ufe65\u203a\u232a\u3009"
_OPEN_CHARS = frozenset(_OPEN)
_MAX_NAME = 64

# A whole token, else a lone opening or closing mark. The name is bounded and
# lazy: the only quantifiers next to each other match disjoint classes, so a
# long run of marks costs at most _MAX_NAME steps per opening mark.
_TOKEN = re.compile(
    rf"[{_OPEN}][{_INVISIBLE}]*[{_BAR}][^\n]{{0,{_MAX_NAME}}}?[{_BAR}][{_INVISIBLE}]*[{_CLOSE}]"
    rf"|[{_OPEN}][{_INVISIBLE}]*[{_BAR}]"
    rf"|[{_BAR}][{_INVISIBLE}]*[{_CLOSE}]"
)
# What the end of a streamed piece may hold of a token still to come: an
# opening mark, and the start of a name no closing mark has ended yet.
_PARTIAL_OPENING = re.compile(
    rf"[{_OPEN}][{_INVISIBLE}]*(?:[{_BAR}][^\n{_CLOSE}]{{0,{_MAX_NAME}}})?"
)
_TRAILING_BAR = re.compile(rf"[{_BAR}][{_INVISIBLE}]*\Z")
_MARKUP = re.compile(r"<(/?[A-Za-z][^<>\n]{0,40})>")
# A tag or a direction a piece may end in the middle of.
_PARTIAL_MARKUP = re.compile(r"<(?:/?[A-Za-z][^<>\n]{0,40}|/)?\Z")
_PARTIAL_DIRECTION = re.compile(r"\[[^\[\]\n]{0,100}\Z")
_BRACKETS = str.maketrans({"[": "(", "]": ")"})
# Performed, never read: what a word count leaves out. Neither alternative
# crosses its own opening character, so a run of them stays linear.
_PERFORMED = re.compile(r"\[[^\[\]]*\]|<[^<>]*>")
_WORD = re.compile(r"[^\W_]")  # a letter or a digit: an underscore says nothing


def neutralize_tokens(text: str) -> str:
    """*text* with every control token, and every lone mark of one, made a space."""
    return _TOKEN.sub(" ", text)


def plain(text: str) -> str:
    """*text* to be read as written: brackets as parentheses, markup in guillemets."""
    return _MARKUP.sub("\u2039\\1\u203a", text.translate(_BRACKETS))


def prepare(text: str, *, expressive: bool) -> str:
    """The text Fish reads for *text*: tokens neutralised, markup per mode."""
    text = neutralize_tokens(text)
    return text if expressive else plain(text)


def says_words(prepared: str) -> bool:
    """Whether a prepared text has a word to say, its directions and tags aside."""
    return bool(_WORD.search(_PERFORMED.sub("", prepared)))


def dialogue_text(
    turns: Sequence[DialogueTurn], speakers: Sequence[str], *, expressive: bool
) -> str:
    """One dialogue as Fish reads it: a speaker token, a direction, the turn.

    *speakers* is the order of the voices sent with it. A turn's ``style`` is
    a direction (``[style]``) in either mode: it is an instruction by
    definition. Its brackets become parentheses so that it cannot end early.

    Raises:
        ValueError: A turn has no word to say: Fish would invent speech for it.
    """
    parts: list[str] = []
    for number, turn in enumerate(turns, 1):
        spoken = prepare(turn.text, expressive=expressive)
        if not says_words(spoken):
            raise ValueError(
                f"dialogue turn {number} ({turn.speaker}) has no word to say: "
                "Fish Audio invents speech for a turn without words"
            )
        parts.append(f"<|speaker:{speakers.index(turn.speaker)}|>")
        if turn.style:
            parts.append(f"[{neutralize_tokens(turn.style).translate(_BRACKETS)}] ")
        parts.append(spoken)
    return "".join(parts)


class StreamedText:
    """Prepares a text that arrives in pieces, for the socket.

    A piece's end that could still become a token (an opening mark, an opened
    token not yet closed, a bar a closing mark may follow), a tag, or in
    expressive mode a direction, is held and read with the next piece. A
    piece with no word to say waits for one that has words, and goes with
    it; one left at the end is not sent.
    """

    def __init__(self, *, expressive: bool) -> None:
        self._expressive = expressive
        self._unfinished = ""
        self._wordless = ""

    def feed(self, piece: str) -> str:
        """What can be sent now of what has come so far ("" for nothing)."""
        text = self._unfinished + piece
        cut = _unfinished_from(text, expressive=self._expressive)
        self._unfinished = text[cut:]
        return self._with_words(prepare(text[:cut], expressive=self._expressive))

    def flush(self) -> str:
        """What is left to send at the end: nothing more will come."""
        text, self._unfinished = self._unfinished, ""
        ready = self._with_words(prepare(text, expressive=self._expressive))
        self._wordless = ""
        return ready

    def _with_words(self, prepared: str) -> str:
        text = self._wordless + prepared
        if says_words(text):
            self._wordless = ""
            return text
        self._wordless = text if not text or text.endswith(" ") else f"{text} "
        return ""


def _unfinished_from(text: str, *, expressive: bool) -> int:
    """Where the end of *text* that could still become a token, a tag or a
    direction (in expressive mode) starts."""
    cuts = [_token_tail(text)]
    markup = _PARTIAL_MARKUP.search(text)
    if markup:
        cuts.append(markup.start())
    if expressive and (direction := _PARTIAL_DIRECTION.search(text)):
        cuts.append(direction.start())
    return min(cuts)


def _token_tail(text: str) -> int:
    opening = max(text.rfind(mark) for mark in _OPEN_CHARS)
    if opening >= 0 and _PARTIAL_OPENING.fullmatch(text, opening):
        return opening
    bar = _TRAILING_BAR.search(text)
    return bar.start() if bar else len(text)
