"""Tests for lifecycle hooks firing on dynamic operations (Area 3.2)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    Access,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent
from tests.test_framework import SimpleChannel


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestRoomLifecycleHooks:
    async def test_hook_fired_on_create_room(self, kit: RoomKit) -> None:
        """ON_ROOM_CREATED fires when a room is created."""
        fired: list[RoomEvent] = []

        async def on_created(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_ROOM_CREATED, on_created))

        await kit.create_room(room_id="r1")
        assert len(fired) == 1
        assert fired[0].room_id == "r1"

    async def test_hook_fired_on_close_room(self, kit: RoomKit) -> None:
        """ON_ROOM_CLOSED fires when a room is closed."""
        fired: list[RoomEvent] = []

        async def on_closed(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_ROOM_CLOSED, on_closed))

        await kit.create_room(room_id="r1")
        await kit.close_room("r1")
        assert len(fired) == 1
        assert fired[0].room_id == "r1"


class TestChannelLifecycleHooks:
    async def test_hook_fired_on_attach(self, kit: RoomKit) -> None:
        """ON_CHANNEL_ATTACHED fires when a channel is attached."""
        fired: list[RoomEvent] = []

        async def on_attached(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_ATTACHED, on_attached))

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        # Filter to just ON_CHANNEL_ATTACHED events (ignoring ON_ROOM_CREATED)
        assert len(fired) == 1
        assert fired[0].room_id == "r1"

    async def test_hook_fired_on_detach(self, kit: RoomKit) -> None:
        """ON_CHANNEL_DETACHED fires when a channel is detached."""
        fired: list[RoomEvent] = []

        async def on_detached(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_DETACHED, on_detached))

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.detach_channel("r1", "sms1")
        assert len(fired) == 1

    async def test_hook_fired_on_mute(self, kit: RoomKit) -> None:
        """ON_CHANNEL_MUTED fires when a channel is muted."""
        fired: list[RoomEvent] = []

        async def on_muted(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_MUTED, on_muted))

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.mute("r1", "sms1")
        assert len(fired) == 1

    async def test_hook_fired_on_unmute(self, kit: RoomKit) -> None:
        """ON_CHANNEL_UNMUTED fires when a channel is unmuted."""
        fired: list[RoomEvent] = []

        async def on_unmuted(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_UNMUTED, on_unmuted))

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.mute("r1", "sms1")
        await kit.unmute("r1", "sms1")
        assert len(fired) == 1

    async def test_detach_nonexistent_no_hook(self, kit: RoomKit) -> None:
        """ON_CHANNEL_DETACHED does not fire if channel was not attached."""
        fired: list[RoomEvent] = []

        async def on_detached(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_DETACHED, on_detached))

        await kit.create_room(room_id="r1")
        result = await kit.detach_channel("r1", "nonexistent")
        assert result is False
        assert len(fired) == 0


_NO_CHANGE = [
    ("mute", (), {"muted": True}),
    ("unmute", (), {}),
    ("mute_output", (), {"output_muted": True}),
    ("unmute_output", (), {}),
    ("set_visibility", ("all",), {}),
    ("set_access", (Access.READ_ONLY,), {"access": Access.READ_ONLY}),
    ("update_binding_metadata", ({"team": "blue"},), {"metadata": {"team": "blue"}}),
]


@pytest.mark.parametrize(("operation", "args", "stored"), _NO_CHANGE)
async def test_a_binding_change_that_changes_nothing_writes_and_fires_nothing(
    operation: str, args: tuple[Any, ...], stored: dict[str, Any], monkeypatch: Any
) -> None:
    """RFC §7.2: a retried call duplicates no hook and no record, and the
    binding is not written again."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1")
    binding = await kit.attach_channel("r1", "sms1")
    await kit.store.update_binding(binding.model_copy(update=stored))
    before = await kit.store.get_binding("r1", "sms1")
    count = (await kit.get_room("r1")).event_count
    fired: list[HookTrigger] = []
    for trigger in (HookTrigger.ON_CHANNEL_MUTED, HookTrigger.ON_CHANNEL_UNMUTED):

        @kit.hook(trigger, execution=HookExecution.ASYNC)
        async def on_change(event: RoomEvent, ctx: RoomContext, trigger: Any = trigger) -> None:
            fired.append(trigger)

    writes: list[ChannelBinding] = []
    monkeypatch.setattr(kit.store, "update_binding", _recording(writes))

    result = await getattr(kit, operation)("r1", "sms1", *args)

    assert result == before
    assert writes == []
    assert fired == []
    assert (await kit.get_room("r1")).event_count == count
    await kit.close()


def _recording(writes: list[ChannelBinding]) -> Any:
    async def update_binding(binding: ChannelBinding) -> ChannelBinding:
        writes.append(binding)
        return binding

    return update_binding


def _async_hook(trigger: HookTrigger, fn: object) -> object:
    """Create an async HookRegistration."""
    from roomkit.core.hooks import HookRegistration

    return HookRegistration(
        trigger=trigger,
        execution=HookExecution.ASYNC,
        fn=fn,
        name=f"test_{trigger}",
    )
