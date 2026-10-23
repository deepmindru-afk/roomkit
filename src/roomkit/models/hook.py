"""Hook-related models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from roomkit.models.event import RoomEvent
from roomkit.models.task import Observation, Task


class InjectedEvent(BaseModel):
    """An event injected by a hook as a side effect (RFC §9.4).

    It is committed after the event whose hooks injected it, as a pass of
    its own once that event's locked section has released the room lock, so
    another event may commit between them: it runs the ``BEFORE_BROADCAST``
    hooks, which may block it, and counts against the reentry budget. It
    keeps that event's chain depth, carries no idempotency key and takes no
    source check; its visibility names the targets the room binds. It is
    delivered to its transport targets only, never to its own source, and
    asks no agent: an intelligence channel among the targets reads it in its
    history, and a transport hosting a model (a realtime voice channel, a
    conference with a realtime model) takes it into the model's context
    silently. Its delivery set fires ``AFTER_BROADCAST`` like any other.
    """

    event: RoomEvent
    target_channel_ids: list[str] | None = None
    """The channels it is delivered to; ``None`` stores it only."""


VERIFICATION_DECISION_KEY = "roomkit:verification"
"""The hook metadata key :meth:`HookResult.allow_unverified` sets (RFC §11.7.3)."""


class HookResult(BaseModel):
    """Result returned by a sync hook."""

    action: Literal["allow", "block", "modify"]
    event: Any = None
    """The replacement payload when action is "modify".

    Typed loosely because a sync hook receives whatever its trigger passes, and
    only one of them passes a RoomEvent: BEFORE_TTS receives a string, the
    bridge triggers receive frames, the tool and generation triggers receive
    their own event types. Restricting this to RoomEvent made "modify"
    impossible for all of them — the hook raised, the engine logged and carried
    on, and the unmodified payload continued on its way.

    Consumers must check the type they expect before using it.
    """
    reason: str | None = None
    injected_events: list[InjectedEvent] = Field(default_factory=list)
    """Events to inject, whatever the action. Committed on ``BEFORE_BROADCAST``
    only (RFC §9.4): any other trigger drops them with a warning."""
    tasks: list[Task] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_action_fields(self) -> HookResult:
        if self.action == "modify" and self.event is None:
            raise ValueError("action='modify' requires 'event' to be set")
        if self.action == "block" and self.reason is None:
            raise ValueError("action='block' requires 'reason' to be set")
        return self

    @classmethod
    def allow(cls, injected: list[InjectedEvent] | None = None) -> HookResult:
        """Allow the event to proceed."""
        return cls(action="allow", injected_events=injected or [])

    @classmethod
    def allow_unverified(cls) -> HookResult:
        """On ON_VERIFICATION_REQUIRED: let an unverified sender's message through.

        It goes on labelled with the sender's verification; without this, the
        message is held from the room's agents (RFC §11.7.3). A plain
        :meth:`allow` does not pass it: a hook that only watches never lets a
        message through by accident.
        """
        return cls(action="allow", metadata={VERIFICATION_DECISION_KEY: "pass"})

    @classmethod
    def block(
        cls,
        reason: str,
        injected: list[InjectedEvent] | None = None,
        tasks: list[Task] | None = None,
        observations: list[Observation] | None = None,
    ) -> HookResult:
        """Block the event from proceeding."""
        return cls(
            action="block",
            reason=reason,
            injected_events=injected or [],
            tasks=tasks or [],
            observations=observations or [],
        )

    @classmethod
    def modify(
        cls,
        event: Any,
        injected: list[InjectedEvent] | None = None,
    ) -> HookResult:
        """Replace the payload before it proceeds.

        Takes whatever the trigger passed the hook — a string for BEFORE_TTS, a
        RoomEvent for BEFORE_BROADCAST — because restricting it to RoomEvent
        left every other sync trigger unable to use its own constructor.
        """
        return cls(action="modify", event=event, injected_events=injected or [])
