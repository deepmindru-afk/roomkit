"""The parts an assistant round is replayed as: its reasoning, its text and its
calls (RFC §6.4).

One assembly for every reader of a round. The tool loop saw the round stream,
so it knows where each reasoning block came relative to the text and the calls
and replays them there. A caller of ``generate()`` has only the response,
which does not say where its blocks came: they go first, then the text, then
the calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from roomkit.providers.ai.base import (
    AITextPart,
    AIThinkingPart,
    AIToolCall,
    AIToolCallPart,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.providers.ai.thinking_blocks import ThinkingBlocks

type AssistantPart = AITextPart | AIThinkingPart | AIToolCallPart
type RoundCall = StreamToolCall | AIToolCall


@dataclass(frozen=True)
class _Thinking:
    block: int | None


@dataclass(frozen=True)
class _Text:
    segment: int


@dataclass(frozen=True)
class _Call:
    id: str


def call_part(call: RoundCall) -> AIToolCallPart:
    """The history part of a call, its metadata (a thought signature) kept."""
    return AIToolCallPart(
        id=call.id, name=call.name, arguments=call.arguments, metadata=call.metadata
    )


def round_parts(
    thinking: Sequence[AIThinkingPart], text: str, calls: Sequence[RoundCall]
) -> list[AssistantPart]:
    """A round whose reasoning has no known place: the reasoning, then the
    text, then the calls."""
    parts: list[AssistantPart] = list(thinking)
    if text:
        parts.append(AITextPart(text=text))
    return parts + [call_part(call) for call in calls]


class RoundTranscript:
    """What a round said, in the order it came: its reasoning blocks, the
    stretches of text between them and its calls."""

    def __init__(self) -> None:
        self.thinking = ThinkingBlocks()
        self._entries: list[_Thinking | _Text | _Call] = []
        self._segments: list[list[str]] = []

    @property
    def text(self) -> str:
        """Everything the round said, its stretches joined."""
        return "".join("".join(segment) for segment in self._segments)

    def add_thinking(self, delta: StreamThinkingDelta) -> None:
        if self.thinking.add(delta):
            self._entries.append(_Thinking(delta.block))

    def add_text(self, text: str) -> None:
        if not self._entries or not isinstance(self._entries[-1], _Text):
            self._entries.append(_Text(len(self._segments)))
            self._segments.append([])
        self._segments[-1].append(text)

    def add_call(self, call_id: str) -> None:
        self._entries.append(_Call(call_id))

    def parts(self, calls: Sequence[RoundCall]) -> list[AssistantPart]:
        """The round as the next one replays it, keeping *calls* of the ones it
        made. Reasoning a vendor sent in blocks goes back block by block, each
        where it came; a single block goes first, then the text, then the calls.
        """
        if not self.thinking.keyed:
            return round_parts(self.thinking.parts(), self.text, calls)
        pending = {call.id: call for call in calls}
        parts: list[AssistantPart] = []
        for entry in self._entries:
            if isinstance(entry, _Thinking):
                if (part := self.thinking.part(entry.block)) is not None:
                    parts.append(part)
            elif isinstance(entry, _Text):
                if text := "".join(self._segments[entry.segment]):
                    parts.append(AITextPart(text=text))
            elif entry.id in pending:
                parts.append(call_part(pending.pop(entry.id)))
        if pending:
            raise RuntimeError(f"replaying calls the round never made: {sorted(pending)}")
        return parts
