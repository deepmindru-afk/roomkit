"""A tool call's outcome, typed once where the handler returns (RFC §9.3, §21.4).

Every channel tells a served call from a refused, failed, blocked, unserved or
cancelled one through :class:`ToolOutcome`, instead of reading the failure out
of the result text, and the part the model reads is built from it in one place
(:meth:`ToolOutcome.as_part`), so ``is_error`` cannot be forgotten on one path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from roomkit.providers.ai.base import AIToolResultPart
from roomkit.tools.result import ToolResult, VerdictReading


class OutcomeKind(StrEnum):
    """How a tool call ended."""

    SERVED = "served"
    """A handler or an ON_TOOL_CALL hook answered it."""
    REFUSED = "refused"
    """A gate or the handler refused it, in words the model reads."""
    FAILED = "failed"
    """The handler, or what it handed the work to, raised."""
    BLOCKED = "blocked"
    """An ON_TOOL_CALL hook withheld the result."""
    UNSERVED = "unserved"
    """No handler served it, and no hook did."""
    CANCELLED = "cancelled"
    """It was cancelled or aborted before it answered."""


@dataclass(frozen=True)
class ToolOutcome:
    """One tool call's outcome: how it ended, and what each reader keeps of it."""

    kind: OutcomeKind
    result: ToolResult
    """What the model reads."""
    recorded: Any = None
    """What the room's tool memory keeps, when it differs from :attr:`result`."""
    structured: dict[str, Any] | None = None
    """The call's structured copy (MCP ``structuredContent``); a failure keeps none."""
    remember: bool = True
    """Whether the room's tool memory keeps the call."""
    detail: str | None = None
    """What failed, for the log and the observers only, never the model (RFC §9.3)."""

    @property
    def failed(self) -> bool:
        """Whether the model reads the call as failed (``is_error``)."""
        return self.kind is not OutcomeKind.SERVED

    @property
    def kept(self) -> Any:
        """What the room's tool memory keeps: the recorded answer, else the result."""
        return self.recorded if self.recorded is not None else self.result

    def as_part(
        self, tool_call_id: str, name: str, *, references: Iterable[str] = ()
    ) -> AIToolResultPart:
        """The part the model reads: the one place a call's result part is built."""
        return AIToolResultPart(
            tool_call_id=tool_call_id,
            name=name,
            result=self.result,
            structured_content=self.structured,
            is_error=self.failed,
            references=list(references),
        )


def read_outcome(reading: VerdictReading) -> OutcomeKind:
    """How a call ended, as ON_TOOL_CALL's verdict reads it."""
    if reading.blocked:
        return OutcomeKind.BLOCKED
    return OutcomeKind.SERVED if reading.served else OutcomeKind.UNSERVED
