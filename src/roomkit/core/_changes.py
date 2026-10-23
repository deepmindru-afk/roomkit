"""Edits and deletes (RFC §10.3): what the commit of a change writes on its
target, and the trigger it fires once the room lock is released."""

from __future__ import annotations

from roomkit.models.enums import HookTrigger
from roomkit.models.event import DeleteContent, EditContent, RoomEvent

type Change = EditContent | DeleteContent


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
