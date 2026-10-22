"""Timestamps in the serialized form (RFC §1.4): aware, and in UTC."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator


def _in_utc(value: datetime) -> datetime:
    """*value* in UTC. A value without an offset is read as UTC: the
    serialized form never writes one, and a reader MAY read it so."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_in_utc)]
"""A ``datetime`` field of a record: whatever offset it was given, it holds
the same instant in UTC."""
