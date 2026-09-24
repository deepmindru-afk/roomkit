"""Sentence boundary splitter for streaming text-to-speech."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

# Sentence-ending punctuation followed by whitespace, or a line break: a list
# (one item per line, no final punctuation) is otherwise one block for the TTS.
_SENTENCE_END = re.compile(r"[.!?][\s]|\n")


async def split_sentences(
    token_stream: AsyncIterator[str],
    min_chunk_chars: int = 20,
) -> AsyncIterator[str]:
    """Buffer streaming tokens and yield complete sentences.

    Accumulates tokens into a buffer.  When the buffer contains a
    sentence-ending punctuation mark (``.``, ``!``, ``?``) followed by
    whitespace, or a line break, **and** the accumulated text is at least
    *min_chunk_chars* long, the sentence is yielded and the buffer is reset.
    Short lines (list items) gather until the chunk reaches that length.

    On stream end any remaining buffered text is yielded as-is (the final
    partial sentence).

    Args:
        token_stream: Async iterator of text token deltas from an LLM.
        min_chunk_chars: Minimum characters before yielding a sentence.
            Prevents very short fragments like ``"Hi."`` that sound
            unnatural when sent individually to TTS.
    """
    buf = ""

    async for token in token_stream:
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
