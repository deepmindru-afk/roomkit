"""An integrator's room transition (RFC §5.1).

One on a room already in the status it leads to does nothing: no write, no
hook, no framework event. One on a room in a status it does not leave from
fails with nothing written. No call takes a room out of ARCHIVED.
"""

from __future__ import annotations

import pytest

from roomkit import (
    FrameworkEvent,
    HookExecution,
    HookTrigger,
    RoomContext,
    RoomEvent,
    RoomKit,
    RoomStatus,
    TransitionRefusedError,
)


async def _room_in(status: RoomStatus) -> tuple[RoomKit, list[str]]:
    """A kit holding room ``r1`` in *status*, and what its transitions report
    from then on: their hooks and their framework events."""
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    if status is RoomStatus.CLOSED:
        await kit.close_room("r1")
    elif status is RoomStatus.ARCHIVED:
        await kit.archive_room("r1")
    heard: list[str] = []

    @kit.hook(HookTrigger.ON_ROOM_CLOSED, execution=HookExecution.ASYNC)
    async def on_closed(event: RoomEvent, ctx: RoomContext) -> None:
        heard.append("ON_ROOM_CLOSED")

    for name in ("room_closed", "room_archived"):

        @kit.on(name)
        async def on_event(event: FrameworkEvent) -> None:
            heard.append(event.type)

    return kit, heard


@pytest.mark.parametrize(
    ("operation", "status"),
    [("close_room", RoomStatus.CLOSED), ("archive_room", RoomStatus.ARCHIVED)],
)
async def test_a_transition_already_done_does_nothing(operation: str, status: RoomStatus) -> None:
    kit, heard = await _room_in(status)
    before = await kit.store.get_room("r1")

    room = await getattr(kit, operation)("r1")

    assert room == before
    assert await kit.store.get_room("r1") == before
    assert heard == []
    await kit.close()


@pytest.mark.parametrize(("operation", "status"), [("close_room", RoomStatus.ARCHIVED)])
async def test_a_transition_from_a_status_it_does_not_leave_fails(
    operation: str, status: RoomStatus
) -> None:
    kit, heard = await _room_in(status)
    before = await kit.store.get_room("r1")

    with pytest.raises(TransitionRefusedError) as refused:
        await getattr(kit, operation)("r1")

    assert (refused.value.room_id, refused.value.status) == ("r1", status)
    assert await kit.store.get_room("r1") == before
    assert heard == []
    await kit.close()
