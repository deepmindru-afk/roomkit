"""A response's reasoning blocks, folded from its streamed deltas (RFC §6.4).

A vendor whose reasoning comes in blocks (Anthropic) signs each block, and
refuses a replayed round whose blocks were merged, split or reordered; a
redacted block comes as opaque data, replayed as received. A delta names its
block. One that names none belongs to the response's single block, which keeps
the joined text and the last signature, as a provider without blocks reports
its reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from roomkit.providers.ai.base import AIThinkingPart, StreamThinkingDelta


@dataclass
class _Block:
    text: list[str] = field(default_factory=list)
    signature: str | None = None
    redacted: str | None = None

    def part(self) -> AIThinkingPart | None:
        if self.redacted is not None:
            return AIThinkingPart(thinking="", redacted=self.redacted)
        thinking = "".join(self.text)
        if not (thinking or self.signature):
            return None
        return AIThinkingPart(thinking=thinking, signature=self.signature)


class ThinkingBlocks:
    """The reasoning blocks of one response, in the order they opened."""

    def __init__(self) -> None:
        self._blocks: dict[int | None, _Block] = {}

    def add(self, delta: StreamThinkingDelta) -> bool:
        """Fold *delta* into its block; whether it opened one."""
        held = self._blocks.get(delta.block)
        opened = held is None
        if held is None:
            held = self._blocks[delta.block] = _Block()
        held.text.append(delta.thinking)
        if delta.signature:
            held.signature = delta.signature
        if delta.redacted is not None:
            held.redacted = delta.redacted
        return opened

    @property
    def keyed(self) -> bool:
        """Whether the vendor named its blocks, so their place in the round counts."""
        return any(block is not None for block in self._blocks)

    def part(self, block: int | None) -> AIThinkingPart | None:
        """The part a block replays as, or ``None`` for one with nothing to replay."""
        held = self._blocks.get(block)
        return held.part() if held is not None else None

    def parts(self) -> list[AIThinkingPart]:
        """Every block's part, in the order the blocks opened."""
        return [part for held in self._blocks.values() if (part := held.part()) is not None]
