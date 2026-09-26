"""What every intelligence channel does with an INSTRUCTION (RFC §10.1.1).

Shared by the in-process AI channel and the ACP channel: both must mark the
text as the application's, never a participant's, and both record on the
reply which instruction produced it, as a fingerprint rather than the text.
"""

from __future__ import annotations

import hashlib
from typing import Any

from roomkit.models.delivery import STANDALONE
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent

INSTRUCTION_MARKER = (
    "[Instruction from the application: nobody in this conversation said it. "
    "Act on it now, in your own words, without quoting it.]"
)
"""Prefix of an instruction's text in the model's input (step 6)."""


def instruction_fingerprint(text: str) -> dict[str, Any]:
    """What a reply records of the instruction that produced it (step 6)."""
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "length": len(text)}


def mark_instruction(text: str) -> str:
    """The instruction as the model reads it: marked, never attributed."""
    return f"{INSTRUCTION_MARKER}\n{text}"


def is_standalone(event: RoomEvent) -> bool:
    """Whether *event* is an instruction whose turn reads nothing of the room (step 7)."""
    return event.type == EventType.INSTRUCTION and bool(event.metadata.get(STANDALONE))
