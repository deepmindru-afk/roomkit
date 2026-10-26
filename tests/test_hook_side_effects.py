"""Tests for hook and broadcast side effect persistence (Area 3.4)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.task import Observation, Task
from tests.test_framework import SimpleChannel


class ObserverChannel(Channel):
    """Channel that produces observations on_event."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        obs = Observation(
            id=f"obs_{event.id}",
            room_id=event.room_id,
            channel_id=self.channel_id,
            content="sentiment positive",
            category="sentiment",
        )
        return ChannelOutput(observations=[obs])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class TaskCreatorChannel(Channel):
    """Channel that produces tasks on_event."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        task = Task(
            id=f"task_{event.id}",
            room_id=event.room_id,
            title="Follow up with client",
        )
        return ChannelOutput(tasks=[task])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class UpdatingChannel(Channel):
    """A channel whose output carries room metadata updates: an intelligence
    channel's from on_event, a transport's from deliver."""

    def __init__(self, channel_id: str, updates: dict[str, Any], *, agent: bool = False) -> None:
        super().__init__(channel_id)
        self.channel_type = ChannelType.AI if agent else ChannelType.WEBSOCKET
        self.category = ChannelCategory.INTELLIGENCE if agent else ChannelCategory.TRANSPORT
        self._updates = updates

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if self.category is ChannelCategory.INTELLIGENCE:
            return ChannelOutput(metadata_updates=dict(self._updates))
        return ChannelOutput.empty()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(metadata_updates=dict(self._updates))


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestBroadcastObservationsPersisted:
    async def test_observations_stored_from_broadcast(self, kit: RoomKit) -> None:
        """Observations from channel on_event are persisted to the store."""
        ch = SimpleChannel("sms1")
        observer = ObserverChannel("observer1")
        kit.register_channel(ch)
        kit.register_channel(observer)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "observer1", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        observations = await kit.store.list_observations("r1")
        assert len(observations) == 1
        assert observations[0].category == "sentiment"


class TestBroadcastTasksPersisted:
    async def test_tasks_stored_from_broadcast(self, kit: RoomKit) -> None:
        """Tasks from channel on_event are persisted to the store."""
        ch = SimpleChannel("sms1")
        task_ch = TaskCreatorChannel("task_ai")
        kit.register_channel(ch)
        kit.register_channel(task_ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "task_ai", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        tasks = await kit.store.list_tasks("r1")
        assert len(tasks) == 1
        assert tasks[0].title == "Follow up with client"


class TestHookTasksPersisted:
    async def test_sync_hook_tasks_persisted(self, kit: RoomKit) -> None:
        """Tasks returned by sync hooks are persisted to the store."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="task_hook")
        async def task_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            task = Task(
                id="hook_task_1",
                room_id=event.room_id,
                title="Hook-created task",
            )
            return HookResult(action="allow", tasks=[task])

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        tasks = await kit.store.list_tasks("r1")
        assert len(tasks) == 1
        assert tasks[0].title == "Hook-created task"

    async def test_blocked_hook_tasks_still_persisted(self, kit: RoomKit) -> None:
        """Tasks from blocking hooks are still persisted (even when event blocked)."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="block_with_task")
        async def block_with_task(event: RoomEvent, ctx: RoomContext) -> HookResult:
            task = Task(
                id="blocked_task_1",
                room_id=event.room_id,
                title="Audit: blocked message",
            )
            return HookResult.block(reason="spam", tasks=[task])

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="spammer",
            content=TextContent(body="buy now"),
        )
        result = await kit.process_inbound(msg)
        assert result.blocked

        tasks = await kit.store.list_tasks("r1")
        assert len(tasks) == 1
        assert tasks[0].title == "Audit: blocked message"


class TestHookObservationsPersisted:
    async def test_sync_hook_observations_persisted(self, kit: RoomKit) -> None:
        """Observations returned by sync hooks are persisted."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="obs_hook")
        async def obs_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            obs = Observation(
                id="hook_obs_1",
                room_id=event.room_id,
                channel_id="system",
                content="Content scanned: clean",
                category="content_scan",
            )
            return HookResult(action="allow", observations=[obs])

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        observations = await kit.store.list_observations("r1")
        assert len(observations) == 1
        assert observations[0].category == "content_scan"


class TestOnTaskCreatedHookFires:
    async def test_on_task_hook_fires(self, kit: RoomKit) -> None:
        """ON_TASK_CREATED fires for tasks created during broadcast."""
        fired: list[RoomEvent] = []

        ch = SimpleChannel("sms1")
        task_ch = TaskCreatorChannel("task_ai")
        kit.register_channel(ch)
        kit.register_channel(task_ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "task_ai", category=ChannelCategory.INTELLIGENCE)

        async def on_task(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        from roomkit.core.hooks import HookRegistration

        kit.hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.ON_TASK_CREATED,
                execution=HookExecution.ASYNC,
                fn=on_task,
                name="test_on_task",
            )
        )

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert len(fired) == 1


class TestBroadcastMetadataUpdatesApplied:
    async def test_the_targets_updates_are_written_on_the_room(
        self, kit: RoomKit, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RFC §10.1 step 15 (RMK-976): the room metadata updates of the
        targets' outputs, on_event's and deliver's alike, are merged into the
        room's metadata, which keeps its other keys; a key only the framework
        writes, of its namespace (§5.1) or of the room's private state
        (§14.1), is dropped with a warning, the rest applied."""
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(
            UpdatingChannel("ai1", {"stage": "qualified", "roomkit:forged": True}, agent=True)
        )
        kit.register_channel(
            UpdatingChannel("crm", {"crm_contact": "c_42", "author_register": {"version": 3}})
        )
        await kit.create_room(room_id="r1", metadata={"topic": "loan"})
        for channel_id in ("sms1", "ai1", "crm"):
            await kit.attach_channel("r1", channel_id)

        with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="sms1", sender_id="user1", content=TextContent(body="I want a loan")
                )
            )

        room = await kit.get_room("r1")
        assert room.metadata == {"topic": "loan", "stage": "qualified", "crm_contact": "c_42"}
        dropped = sorted(r.getMessage() for r in caplog.records if "drops" in r.getMessage())
        assert len(dropped) == 2
        assert "'author_register'" in dropped[0] and "crm" in dropped[0]
        assert "'roomkit:forged'" in dropped[1] and "ai1" in dropped[1]
