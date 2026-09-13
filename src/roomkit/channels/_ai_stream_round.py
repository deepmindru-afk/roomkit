"""State and text handling for one streamed AI generation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Protocol

from roomkit.channels._ai_coalescers import _ThinkingCoalescer, _ToolCallDeltaCoalescer
from roomkit.channels._ai_resilience import _StreamRetryBoundary
from roomkit.channels._ai_stream_external_tools import _ExternalStreamTools
from roomkit.models.streaming import StreamDelta, ThinkingDeltaMarker
from roomkit.providers.ai.base import (
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.providers.utils import _aclose_stream
from roomkit.realtime.base import EphemeralEventType


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
    cancelled: bool = False

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def thinking(self) -> str:
        return "".join(self.thinking_parts)


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


class _ThinkingWindowCloser(Protocol):
    async def __call__(
        self,
        coalescer: _ThinkingCoalescer,
        room_id: str,
        thinking_parts: list[str],
        round_idx: int,
        *,
        published: int,
    ) -> int: ...


@dataclass
class _StreamRound:
    """Consume one generation and own its stream and observable windows.

    The turn chooses whether another round can run. This component only
    projects provider events and keeps the raw and delivered transcripts.
    A retry boundary starts a new composition attempt within the same round.
    """

    index: int
    room_id: str | None
    cancel_event: asyncio.Event
    thinking_coalescer: _ThinkingCoalescer
    new_composition: Callable[[], _ToolCallDeltaCoalescer]
    publish_thinking: Callable[[EphemeralEventType, str, str, int], Awaitable[None]]
    close_thinking: _ThinkingWindowCloser
    record_usage: Callable[[dict[str, Any]], None]
    prefix: str = ""
    external_tools: _ExternalStreamTools | None = None
    state: _StreamRoundState = field(default_factory=_StreamRoundState, init=False)
    _composition: _ToolCallDeltaCoalescer = field(init=False)
    _dedup: _PrefixDeduplicator = field(init=False)

    def __post_init__(self) -> None:
        self._composition = self.new_composition()
        self._dedup = _PrefixDeduplicator(self.prefix)

    async def _end_thinking(self) -> None:
        state = self.state
        if not (state.thinking_started and state.thinking_parts and self.room_id):
            return
        # Disarm before awaiting a publish: cleanup can itself be cancelled.
        state.thinking_started = False
        state.thinking_published = await self.close_thinking(
            self.thinking_coalescer,
            self.room_id,
            state.thinking_parts,
            self.index,
            published=state.thinking_published,
        )

    async def _add_thinking(self, event: StreamThinkingDelta) -> None:
        state = self.state
        if event.signature:
            state.thinking_signature = event.signature
        if not event.thinking:
            return
        if not state.thinking_started and self.room_id:
            state.thinking_started = True
            await self.publish_thinking(
                EphemeralEventType.THINKING_START, self.room_id, "", self.index
            )
        state.thinking_parts.append(event.thinking)
        await self.thinking_coalescer.add(event.thinking)

    async def close(self) -> None:
        """Close only opened windows, including when another close is cancelled."""
        try:
            await self._end_thinking()
        finally:
            if self.room_id:
                await self._composition.close()

    async def stream(
        self, source: AsyncIterator[StreamEvent | _StreamRetryBoundary]
    ) -> AsyncGenerator[StreamDelta, None]:
        """Yield on demand; the direct consumer must explicitly close this iterator."""
        state = self.state
        try:
            async for event in source:
                if self.cancel_event.is_set():
                    state.cancelled = True
                    await self.close()
                    return
                if isinstance(event, _StreamRetryBoundary):
                    if self.room_id:
                        await self._composition.close()
                        self._composition = self.new_composition()
                elif isinstance(event, StreamThinkingDelta):
                    await self._add_thinking(event)
                    if event.thinking:
                        yield ThinkingDeltaMarker(thinking=event.thinking)
                elif isinstance(event, StreamTextDelta):
                    await self._end_thinking()
                    state.text_parts.append(event.text)
                    for text in self._dedup.add(event.text):
                        state.reported.append(text)
                        yield text
                elif isinstance(event, StreamToolCallDelta):
                    await self._end_thinking()
                    if self.room_id:
                        await self._composition.add(
                            event.index, event.id, event.name, len(event.arguments_delta)
                        )
                elif isinstance(event, StreamToolCall):
                    state.tool_calls.append(event)
                    if self.external_tools is not None:
                        async with aclosing(
                            self.external_tools.stream_call(event, self.index)
                        ) as deltas:
                            async for delta in deltas:
                                yield delta
                elif isinstance(event, StreamDone):
                    state.finish_reason = event.finish_reason
                    if event.usage:
                        self.record_usage(event.usage)

            for text in self._dedup.finish():
                state.reported.append(text)
                yield text
            await self.close()
        finally:
            try:
                await _aclose_stream(source)
            finally:
                await self.close()
