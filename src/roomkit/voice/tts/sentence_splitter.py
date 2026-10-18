"""Sentence boundary splitter for streaming text-to-speech."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from roomkit.models.streaming import SegmentBreakMarker

# Sentence-ending punctuation followed by whitespace, or a line break: a list
# (one item per line, no final punctuation) is otherwise one block for the TTS.
_SENTENCE_END = re.compile(r"[.!?][\s]|\n")


def sentence_pieces(text: str) -> list[str]:
    """*text* cut after each place a sentence may end, the pieces joined
    giving it back: a reader that decides between deltas whether a reply
    goes on decides the same however its deltas were cut (RFC §10.6.4)."""
    pieces: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        pieces.append(text[start : match.end()])
        start = match.end()
    if start < len(text) or not pieces:
        pieces.append(text[start:])
    return pieces


async def split_sentences(
    token_stream: AsyncIterator[str | SegmentBreakMarker],
    min_chunk_chars: int = 20,
) -> AsyncIterator[str]:
    """Buffer streaming tokens and yield complete sentences.

    Accumulates tokens into a buffer.  When the buffer contains a
    sentence-ending punctuation mark (``.``, ``!``, ``?``) followed by
    whitespace, or a line break, **and** the accumulated text is at least
    *min_chunk_chars* long, the sentence is yielded and the buffer is reset.
    Short lines (list items) gather until the chunk reaches that length.

    On stream end any remaining buffered text is yielded as-is (the final
    partial sentence), and so it is at a :class:`SegmentBreakMarker`: the
    text before a tool call ends there, so it is said before the call runs,
    however short, never run on into the next round's text (RFC §12.2 step
    12s.a).

    Args:
        token_stream: Async iterator of text token deltas from an LLM,
            with a :class:`SegmentBreakMarker` where a text segment ends.
        min_chunk_chars: Minimum characters before yielding a sentence.
            Prevents very short fragments like ``"Hi."`` that sound
            unnatural when sent individually to TTS.
    """
    buf = ""

    async for token in token_stream:
        if isinstance(token, SegmentBreakMarker):
            if remaining := buf.strip():
                yield remaining
            buf = ""
            continue
        buf += token

        # Scan for the *last* sentence boundary that satisfies min_chunk_chars
        while True:
            match = _SENTENCE_END.search(buf, min_chunk_chars)
            if match is None:
                break
            # Split right after the punctuation (before the trailing space),
            # or at the line break itself
            split_pos = match.start() if match.group() == "\n" else match.start() + 1
            sentence = buf[:split_pos].strip()
            buf = buf[split_pos:].lstrip()
            if sentence:
                yield sentence

    # Flush remaining text
    remaining = buf.strip()
    if remaining:
        yield remaining
