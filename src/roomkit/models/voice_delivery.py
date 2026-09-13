"""Durable identity and ownership of one proactive voice injection."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from roomkit.models.delivery import DeliveryOutcome


class VoiceDeliveryRecord(BaseModel):
    """One reserved injection, retained until its room is deleted.

    An absent outcome means unresolved submission, never permission to retry.
    ``attempt_id`` fences completion from previous owners. Only a recorded
    retryable failure known to precede submission can be claimed again.
    """

    room_id: str
    channel_id: str
    session_id: str
    idempotency_key: str
    content_hash: str
    attempt_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    outcome: DeliveryOutcome | None = None

    @property
    def key_hash(self) -> str:
        """Bounded storage key; room isolation remains an explicit SQL predicate."""
        value = json.dumps([self.channel_id, self.session_id, self.idempotency_key])
        return hashlib.sha256(value.encode()).hexdigest()

    @property
    def retryable(self) -> bool:
        """Whether the completed attempt established a safe retry."""
        result = self.outcome
        return bool(
            result is not None
            and result.status in ("failed", "unavailable")
            and result.error is not None
            and result.error.retryable
            and not result.session_ids
        )
