"""Edits and deletes (RFC §10.3): what a change is, what the commit of a
change writes on its target, and the trigger it fires once the room lock is
released."""

from __future__ import annotations

from roomkit.models.enums import EventType, HookTrigger
from roomkit.models.event import (
    CompositeContent,
    DeleteContent,
    EditContent,
    EventContent,
    RoomEvent,
)

type Change = EditContent | DeleteContent

TARGET_NOT_FOUND = "target_event_not_found"
"""Step 8a's refusal of a change that names no target it may change."""

_CHANGE_TYPES = frozenset({EventType.EDIT, EventType.DELETE})


def holds_change(content: EventContent) -> bool:
    """Whether *content* is or holds a change at any depth: an edit, a
    delete, or a composite with one among its parts."""
    if isinstance(content, EditContent | DeleteContent):
        return True
    if isinstance(content, CompositeContent):
        return any(holds_change(part) for part in content.parts)
    return False


def concerns_change(event: RoomEvent) -> bool:
    """Whether step 8a judges *event* (RFC §10.1): an EDIT or a DELETE, or
    an event holding a change anywhere."""
    return event.type in _CHANGE_TYPES or holds_change(event.content)


def change_of(event: RoomEvent) -> Change | None:
    """The change *event* makes (RFC §10.3 rule 1): its content, when its
    type is EDIT and its content an edit, or DELETE and a delete, with no
    change held in it besides; ``None`` otherwise. A type and a content that
    disagree, or a change held anywhere else (an edit's new content, a
    composite's part), name no target that can be checked."""
    content = event.content
    if event.type == EventType.EDIT and isinstance(content, EditContent):
        return None if holds_change(content.new_content) else content
    if event.type == EventType.DELETE and isinstance(content, DeleteContent):
        return content
    return None


def applied(change: Change, target: RoomEvent) -> tuple[RoomEvent, HookTrigger]:
    """*target* as the commit of *change* writes it, its other metadata kept
    (RFC §10.3 state updates 1 and 2), and the mutation trigger fired with
    it: an edit writes its new content and ``metadata.edited``, a delete
    ``metadata.deleted``."""
    if isinstance(change, EditContent):
        edited = target.model_copy(
            update={
                "content": change.new_content,
                "metadata": {**target.metadata, "edited": True},
            }
        )
        return edited, HookTrigger.ON_EVENT_UPDATED
    deleted = target.model_copy(update={"metadata": {**target.metadata, "deleted": True}})
    return deleted, HookTrigger.ON_EVENT_DELETED
