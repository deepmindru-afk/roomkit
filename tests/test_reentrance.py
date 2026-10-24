"""A call into a room from the room's own work (RFC §9.5.1 Reentrance, §8.3,
§10.5, §10.1 step 18).

Made under the room lock, from an off-lock hook's body or from a message's
parse, a call is adopted by the event whose work made it and draws on that
event's reentry budget; made from a turn or a delivery, it is a new arrival
that draws on the budget of the cascade that delivers, at least as deep as
the turn's response. Each loop a host can write by calling into its room on
every event therefore stops, at the reentry cap or the chain-depth limit.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit import ReentrantCallError, RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import ChannelCategory, EventType, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.store_filter import EventFilter
from tests.test_framework import AILikeChannel, SimpleChannel

# max_chain_depth=1: a reentry budget of 10 passes per cascade.
BUDGET = 10


def _message(body: str = "hi") -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=body))


def _body(event: RoomEvent) -> str | None:
    return event.content.body if isinstance(event.content, TextContent) else None


async def _room(kit: RoomKit, *channels: SimpleChannel) -> RoomKit:
    await kit.create_room(room_id="r1")
    for channel in (SimpleChannel("sms"), *channels):
        kit.register_channel(channel)
        await kit.attach_channel("r1", channel.channel_id)
    return kit


async def _messages(kit: RoomKit) -> list[RoomEvent]:
    events = await kit.store.list_events(
        "r1", limit=1000, event_filter=EventFilter(include_blocked=True)
    )
    return [e for e in events if e.type == EventType.MESSAGE]


def _capped(events: list[RoomEvent]) -> list[RoomEvent]:
    return [e for e in events if e.blocked_by == "reentry_loop_cap"]


class TestAnInjectionMadeFromTheRoomsOwnWork:
    """§9.5.1: under the lock, or from an off-lock hook's body, the
    injection draws on the budget of the event whose work made it."""

    @pytest.mark.parametrize("needs_lock", [True, False], ids=["locked hook", "off-lock hook"])
    async def test_a_hook_that_injects_on_every_event_stops_at_the_cap(
        self, needs_lock: bool
    ) -> None:
        kit = await _room(RoomKit(max_chain_depth=1))

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="echo", needs_lock=needs_lock)
        async def echo(event: RoomEvent, ctx: RoomContext) -> HookResult:
            await kit.send_event("r1", "sms", TextContent(body="again"))
            return HookResult.allow()

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            events = await _messages(kit)
            assert len(events) == 1 + BUDGET + 1
            assert len(_capped(events)) == 1
        finally:
            await kit.close()


class _Parser(SimpleChannel):
    """A transport whose parse sends a message into its room every time."""

    def __init__(self, channel_id: str, kit: RoomKit, *, routed: bool = False) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self.routed = routed
        self.answers: list[InboundResult | BaseException] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        if not self.answers or isinstance(self.answers[-1], InboundResult):
            again = _message("again")
            try:
                room = None if self.routed else "r1"
                self.answers.append(await self._kit.process_inbound(again, room_id=room))
            except ReentrantCallError as refused:
                self.answers.append(refused)
        return await super().handle_inbound(message, context)


class TestACallFromAParse:
    """The parse stands in its message's admission order (§9.5.1): what it
    sends into its room is adopted beside, on the message's budget."""

    async def test_a_parse_that_calls_into_its_room_for_every_message_stops_at_the_cap(
        self,
    ) -> None:
        kit = RoomKit(max_chain_depth=1)
        parser = _Parser("sms", kit)
        await kit.create_room(room_id="r1")
        kit.register_channel(parser)
        await kit.attach_channel("r1", "sms")
        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            refused = [a for a in parser.answers if isinstance(a, InboundResult) and a.blocked]
            assert [a.reason for a in refused] == ["reentry_loop_cap"]
            assert len(await _messages(kit)) == 1 + BUDGET
        finally:
            await kit.close()


class _Looping(SimpleChannel):
    """A transport whose every delivery sends a message into its room."""

    def __init__(self, channel_id: str, kit: RoomKit) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self.refused = asyncio.Event()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        result = await self._kit.process_inbound(_message("again"), room_id="r1")
        if result.reason == "reentry_loop_cap":
            self.refused.set()
        return ChannelOutput.empty()


class TestACallFromADelivery:
    async def test_a_transport_calling_into_its_room_on_every_delivery_stops_at_the_cap(
        self,
    ) -> None:
        """A new arrival draws on the budget of the cascade that delivers:
        with a budget of its own, the room would fill forever."""
        kit = RoomKit(max_chain_depth=1)
        looping = _Looping("loop", kit)
        await _room(kit, looping)
        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(looping.refused.wait(), 5)
            assert len(await _messages(kit)) == 1 + BUDGET
        finally:
            await kit.close()


class _Injecting(AILikeChannel):
    """An agent that, at each turn, sends into its room through the SMS
    channel and answers nothing."""

    def __init__(self, channel_id: str, kit: RoomKit) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self.turns = 0

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.turns += 1
        await self._kit.send_event("r1", "sms", TextContent(body=f"turn {self.turns}"))
        return ChannelOutput.empty()


class TestACallFromATurn:
    @pytest.mark.in_process(
        "the agent calls the kit it holds, whose context an in-memory turn shares with "
        "the owner; across Redis it goes through its TurnHost"
    )
    async def test_an_agent_injecting_at_each_turn_stops_at_the_chain_depth_limit(self) -> None:
        """A call made within a turn is at least as deep as the turn's
        response (RFC §8.3, §10.6): with a budget of 30, only the depth limit
        of 3 stops the agent this soon."""
        kit = RoomKit(max_chain_depth=3)
        agent = _Injecting("ai", kit)
        await _room(kit)
        kit.register_channel(agent)
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert agent.turns == 2
        finally:
            await kit.close()


async def _settled(kit: RoomKit) -> None:
    """Wait until the room's lane has run everything queued."""
    await kit._lanes.lane("r1").drain()  # noqa: SLF001


class TestARoutedMessageBackIntoItsRoom:
    """A message routed from under the lock, from an off-lock hook's body
    or from a parse lands where its caller's room is known only after
    routing: into that room, it fails rather than commit ahead of the event
    its caller works for (RFC §9.5.1, §10.1 step 2)."""

    @pytest.mark.parametrize("needs_lock", [True, False], ids=["locked hook", "off-lock hook"])
    async def test_from_a_hook_it_fails(self, needs_lock: bool) -> None:
        kit = await _room(RoomKit())
        answers: list[object] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="router", needs_lock=needs_lock)
        async def route_back(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) == "hi":
                try:
                    answers.append(await kit.process_inbound(_message("again")))
                except ReentrantCallError as refused:
                    answers.append(refused)
            return HookResult.allow()

        try:
            result = await asyncio.wait_for(kit.process_inbound(_message()), 5)
            assert not result.blocked
            assert [type(a) for a in answers] == [ReentrantCallError]
            assert [_body(e) for e in await _messages(kit)] == ["hi"]
        finally:
            await kit.close()

    async def test_from_a_parse_it_fails(self) -> None:
        kit = RoomKit()
        parser = _Parser("sms", kit, routed=True)
        await kit.create_room(room_id="r1")
        kit.register_channel(parser)
        await kit.attach_channel("r1", "sms")
        try:
            await asyncio.wait_for(kit.process_inbound(_message(), room_id="r1"), 5)
            assert [type(a) for a in parser.answers] == [ReentrantCallError]
            assert [_body(e) for e in await _messages(kit)] == ["hi"]
        finally:
            await kit.close()
