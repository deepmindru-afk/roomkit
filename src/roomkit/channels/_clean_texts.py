"""Texts a scan for copies of the runtime's marks found clean (RFC §6.4).

The marks' scan and the notes' header's scan each keep one, so a text a turn
already read is not scanned again by either; the scan for a copy split over
consecutive messages keeps one of the runs it read.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Iterator


class CleanTexts:
    """Digests of texts seen to need no change, most recent last, the oldest
    forgotten past *size*: a turn reads the history the last turn read, and
    scanning it again would cost as much as the first time. Digests only, so
    a long text costs the cache nothing."""

    def __init__(self, size: int) -> None:
        self._digests: OrderedDict[bytes, None] = OrderedDict()
        self._size = size

    def __contains__(self, text: str) -> bool:
        key = _digest(text)
        if key not in self._digests:
            return False
        self._digests.move_to_end(key)
        return True

    def add(self, text: str) -> None:
        self._digests[_digest(text)] = None
        if len(self._digests) > self._size:
            self._digests.popitem(last=False)

    def cleaned(self, text: str, clean: Callable[[str], str]) -> str:
        """*text* as *clean* leaves it; *text* itself, not cleaned again, when
        it was seen to need no change."""
        if text in self:
            return text
        cleaned = clean(text)
        if cleaned == text:
            self.add(text)
        return cleaned

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._digests)


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()
