"""State and text handling for one streamed AI generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from roomkit.providers.ai.base import StreamToolCall


@dataclass
class _StreamRoundState:
    """A generation's raw transcript and the fragments actually delivered."""

    thinking_parts: list[str] = field(default_factory=list)
    thinking_signature: str | None = None
    thinking_started: bool = False
    thinking_published: int = 0
    text_parts: list[str] = field(default_factory=list)
    reported: list[str] = field(default_factory=list)
    tool_calls: list[StreamToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class _PrefixDeduplicator:
    """Withhold a replayed prefix until new text or a mismatch settles it.

    A generation ending inside the prefix still gets its buffered text back:
    it may be the entire final answer. Only a prefix followed by new text is
    suppressed. The consumer decides when returned fragments are delivered.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._active = bool(prefix)
        self._offset = 0
        self._buffer: list[str] = []

    def add(self, text: str) -> list[str]:
        """Return the fragments ready to deliver after this delta."""
        if not self._active:
            return [text]
        end = self._offset + len(text)
        if end <= len(self._prefix):
            if self._prefix[self._offset : end] == text:
                self._offset = end
                self._buffer.append(text)
                return []
            result = [*self._buffer, text]
        else:
            tail = self._prefix[self._offset :]
            if text[: len(tail)] == tail:
                remaining = text[len(tail) :]
                result = [remaining] if remaining else []
            else:
                result = [*self._buffer, text]
        self._active = False
        self._buffer.clear()
        return result

    def finish(self) -> list[str]:
        """Deliver a partial or exact prefix when no new text followed it."""
        result = self._buffer
        self._buffer = []
        return result
