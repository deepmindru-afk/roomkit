"""Task and observation models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from roomkit.models.enums import TaskStatus


class Task(BaseModel):
    """A task assigned within a room."""

    id: str
    room_id: str
    title: str
    description: str | None = None
    assigned_to: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """An observation produced by an intelligence channel."""

    id: str
    room_id: str
    channel_id: str
    content: str
    category: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DelegatedTaskResult(BaseModel):
    """Result of a completed delegated task."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    child_room_id: str
    parent_room_id: str
    agent_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    output: str | None = None
    error: str | None = None
    exception: SkipJsonSchema[Exception | None] = Field(default=None, exclude=True)
    """The failure itself, its type kept (what the worker's turn raised),
    for a caller that hands it on; held in memory, never serialized (RFC
    §23.3 step 6)."""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
