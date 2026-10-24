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
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, ReentrantCallError, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.inbound_router import InboundRoomRouter
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import EditContent, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.conference.test_conference_tool_calls import LOOKUP
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

    async def test_a_replay_draws_nothing_and_returns_its_original(self) -> None:
        """A hook sends with the same idempotency key more times than its
        cascade's budget holds passes, then a new event: a replay is no new
        pass and draws nothing, so each replay returns the event the key
        committed, past the cap too, nothing stored and nothing raised, and
        the new event still has its pass (RFC §13.4, §8.3, §9.5.1)."""
        kit = await _room(RoomKit(max_chain_depth=1))
        sent: list[RoomEvent] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="send")
        async def send(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) == "hi":
                for n in range(BUDGET + 2):
                    sent.append(
                        await kit.send_event(
                            "r1", "sms", TextContent(body=f"sent {n}"), idempotency_key="k"
                        )
                    )
                sent.append(await kit.send_event("r1", "sms", TextContent(body="new")))
            return HookResult.allow()

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            assert len(sent) == BUDGET + 3
            assert {e.id for e in sent[:-1]} == {sent[0].id}
            assert [(_body(e), e.blocked_by) for e in await _messages(kit)] == [
                ("sent 0", None),
                ("new", None),
                ("hi", None),
            ]
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
    """A transport whose every delivery sends a message into its room; once
    one is refused, it sends *then* into the room too, past the budget."""

    def __init__(self, channel_id: str, kit: RoomKit, then: InboundMessage | None = None) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self._then = then
        self.refused = asyncio.Event()
        self.then_answer: InboundResult | Exception | None = None

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        result = await self._kit.process_inbound(_message("again"), room_id="r1")
        if result.reason == "reentry_loop_cap" and not self.refused.is_set():
            if self._then is not None:
                try:
                    self.then_answer = await self._kit.process_inbound(self._then, room_id="r1")
                except Exception as error:
                    self.then_answer = error
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

    @pytest.mark.parametrize("channel_id", ["ghost", "late"], ids=["unregistered", "unattached"])
    async def test_a_call_naming_its_room_past_the_budget_is_refused_before_anything_else(
        self, channel_id: str
    ) -> None:
        """RFC §10.1 step 1, §9.5.1 Past the cap: a message that names its
        room is refused as it starts, before its channel is looked up or
        attached: nothing created or stored."""
        telemetry = MockTelemetryProvider()
        kit = RoomKit(max_chain_depth=1, telemetry=telemetry)
        then = InboundMessage(channel_id=channel_id, sender_id="u", content=TextContent(body="x"))
        looping = _Looping("loop", kit, then=then)
        await _room(kit, looping)
        kit.register_channel(SimpleChannel("late"))
        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(looping.refused.wait(), 5)
            answer = looping.then_answer
            assert isinstance(answer, InboundResult), answer
            assert (answer.blocked, answer.reason, answer.event) == (
                True,
                "reentry_loop_cap",
                None,
            )
            assert await kit.store.get_binding("r1", channel_id) is None
            assert len(await _messages(kit)) == 1 + BUDGET
            # Still traced: a loop the cap cut is visible.
            [span] = [
                span
                for span in telemetry.get_spans(SpanKind.INBOUND_PIPELINE)
                if span.name == "framework.inbound" and span.channel_id == channel_id
            ]
            assert (span.room_id, span.attributes["blocked"]) == ("r1", True)
        finally:
            await kit.close()

    @pytest.mark.parametrize(
        "opens", ["no-room", "a-room-id-the-store-lacks", "a-named-room-the-store-lacks"]
    )
    async def test_a_transport_opening_a_room_on_every_delivery_stops_at_the_cap(
        self, opens: str
    ) -> None:
        """A message the room's work sends that creates a room is that
        work's: it draws on the budget of the cascade that delivers, whether
        routing opens the room (no answer, or an id the store lacks) or the
        call names a room the store lacks, so a transport whose every
        delivery opens a room stops at the cap."""
        router = _OPENING_ROUTERS.get(opens, _NoRoom)()
        kit = RoomKit(max_chain_depth=1, inbound_router=router)
        opening = _Opening("opener", kit, naming=opens == "a-named-room-the-store-lacks")
        kit.register_channel(opening)

        @kit.hook(HookTrigger.ON_ROOM_CREATED)
        async def bind_opener(event: object, ctx: RoomContext) -> None:
            await kit.attach_channel(ctx.room.id, "opener")

        await _room(kit)
        try:
            await asyncio.wait_for(kit.process_inbound(_message(), room_id="r1"), 5)
            await asyncio.wait_for(opening.refused.wait(), 5)
            rooms = await kit.store.list_rooms()
            assert len(rooms) == 1 + BUDGET
        finally:
            await kit.close()

    async def test_a_routed_message_charged_for_a_room_created_meanwhile_is_charged_once(
        self,
    ) -> None:
        """RFC §8.3, §9.5.1: a routed message charged for creating a room
        another call created first, then routed again into the room its work
        delivers in, is not charged a second time there: with one pass left
        on the budget, it lands."""
        kit = RoomKit(
            max_chain_depth=1, store=_CreatedMeanwhile(), inbound_router=_AbsentThenTheRoom()
        )
        spender = _SpendsThenRoutes("spender", kit, named=BUDGET - 1)
        await _room(kit, spender)
        try:
            await asyncio.wait_for(kit.process_inbound(_message("go"), room_id="r1"), 5)
            routed = spender.routed
            assert routed is not None and not routed.blocked, routed
            assert routed.event is not None and routed.event.room_id == "r1"
        finally:
            await kit.close()

    async def test_a_delivery_s_call_into_another_existing_room_has_its_own_budget(
        self,
    ) -> None:
        """A call into an existing room the delivery does not deliver in is
        none of its work: each has a budget of its own."""
        kit = RoomKit(max_chain_depth=1)
        fanning = _Fanning("fan", kit, sends=BUDGET + 1)
        await _room(kit, fanning)
        await kit.create_room(room_id="r2")
        await kit.attach_channel("r2", "sms")
        try:
            await asyncio.wait_for(kit.process_inbound(_message(), room_id="r1"), 5)
            assert [e.blocked_by for e in fanning.sent] == [None] * (BUDGET + 1)
        finally:
            await kit.close()


class _NoRoom(InboundRoomRouter):
    """Finds no room for any message: each one opens a new room."""

    async def route(self, channel_id: str, channel_type: Any, **kwargs: Any) -> str | None:
        return None


class _FreshRoom(InboundRoomRouter):
    """Answers a room id the store does not hold for every message."""

    def __init__(self) -> None:
        self._rooms = 0

    async def route(self, channel_id: str, channel_type: Any, **kwargs: Any) -> str | None:
        self._rooms += 1
        return f"fresh-{self._rooms}"


class _CreatedMeanwhile(InMemoryStore):
    """Another call creates room ``fresh`` right after a routed message read
    it absent, its binding of ``sms`` naming another correspondent: the
    message, charged for the creation, reads it as existing, is no one it
    admits, and is routed again."""

    async def room_exists(self, room_id: str) -> bool:
        exists = await super().room_exists(room_id)
        if room_id == "fresh" and not exists:
            await super().create_room(Room(id="fresh"))
            await super().add_binding(
                ChannelBinding(
                    channel_id="sms",
                    room_id="fresh",
                    channel_type=ChannelType.SMS,
                    participant_id="someone-else",
                )
            )
        return exists


class _AbsentThenTheRoom(InboundRoomRouter):
    """Answers ``fresh``, an id the store lacks, then ``r1``."""

    def __init__(self) -> None:
        self._answers = ["fresh"]

    async def route(self, channel_id: str, channel_type: Any, **kwargs: Any) -> str | None:
        return self._answers.pop() if self._answers else "r1"


class _SpendsThenRoutes(SimpleChannel):
    """A transport whose delivery of ``go`` sends *named* messages naming its
    room, then one routed."""

    def __init__(self, channel_id: str, kit: RoomKit, *, named: int) -> None:
        super().__init__(channel_id)
        self._kit, self._named = kit, named
        self.routed: InboundResult | None = None

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "go":
            for _ in range(self._named):
                await self._kit.process_inbound(_message("named"), room_id="r1")
            self.routed = await self._kit.process_inbound(_message("routed"))
        return ChannelOutput.empty()


_OPENING_ROUTERS: dict[str, Callable[[], InboundRoomRouter]] = {
    "no-room": _NoRoom,
    "a-room-id-the-store-lacks": _FreshRoom,
}


class _Opening(SimpleChannel):
    """A transport whose every delivery sends a message that opens a room:
    routed, or naming a fresh room id when *naming*."""

    def __init__(self, channel_id: str, kit: RoomKit, *, naming: bool = False) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self._naming = naming
        self._named = 0
        self.refused = asyncio.Event()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self._named += 1
        room_id = f"named-{self._named}" if self._naming else None
        result = await self._kit.process_inbound(_message("again"), room_id=room_id)
        if result.reason == "reentry_loop_cap":
            self.refused.set()
        return ChannelOutput.empty()


class _Fanning(SimpleChannel):
    """A transport whose delivery of the first message sends *sends*
    messages into room ``r2``."""

    def __init__(self, channel_id: str, kit: RoomKit, sends: int) -> None:
        super().__init__(channel_id)
        self._kit, self._sends = kit, sends
        self.sent: list[RoomEvent] = []

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "hi":
            for n in range(self._sends):
                self.sent.append(
                    await self._kit.send_event("r2", "sms", TextContent(body=f"there {n}"))
                )
        return ChannelOutput.empty()


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

    @pytest.mark.parametrize(
        "work",
        [
            pytest.param(
                "agent",
                marks=pytest.mark.in_process(
                    "the agent's task calls the kit it holds, whose context an in-memory "
                    "turn shares with the owner; across Redis it goes through its TurnHost"
                ),
            ),
            "tool-handler",
            "conference-handler",
        ],
    )
    async def test_work_a_turn_began_starts_fresh_once_the_turn_is_over(self, work: str) -> None:
        """A task a turn, a tool handler or a conference's tool call spawned,
        sending once that work is over, is work of its own: depth 0 and a
        budget of its own for each call, as a call from outside the room's
        work (RFC §8.3, §9.5.1)."""
        sends = 2 * BUDGET + 1
        kit, start, later = await _WORK[work](sends)
        try:
            await asyncio.wait_for(start(), 5)
            later.over.set()
            sent = await asyncio.wait_for(later.sent, 5)
            assert [(e.chain_depth, e.blocked_by) for e in sent] == [(0, None)] * sends
        finally:
            await kit.close()

    @pytest.mark.in_process(
        "the agent calls the kit it holds, whose context an in-memory turn shares with "
        "the owner; across Redis it goes through its TurnHost"
    )
    async def test_a_call_into_another_room_during_the_turn_has_its_own_budget(self) -> None:
        """A turn's call into another room is none of this room's work: a
        budget of its own for each call; it is still at least as deep as the
        turn's response, as any call made while the turn runs. The agent is
        asked below the depth limit of 2, with a budget of 20 per cascade."""
        kit = await _room(RoomKit(max_chain_depth=2))
        await kit.create_room(room_id="r2")
        await kit.attach_channel("r2", "sms")
        agent = _Elsewhere("ai", kit, sends=2 * BUDGET + 1)
        kit.register_channel(agent)
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        try:
            await asyncio.wait_for(kit.process_inbound(_message(), room_id="r1"), 5)
            assert [(e.chain_depth, e.blocked_by) for e in agent.sent] == [(1, None)] * (
                2 * BUDGET + 1
            )
        finally:
            await kit.close()


class _Later:
    """Work started inside a turn or a tool call: a task that sends *sends*
    messages into *room_id* through *channel_id* once told the work is
    over, each addressed to nobody, so that no agent answers it."""

    def __init__(self, kit: RoomKit, room_id: str, channel_id: str, sends: int) -> None:
        self._kit, self._room_id, self._channel_id, self._sends = kit, room_id, channel_id, sends
        self.over = asyncio.Event()
        self.sent: asyncio.Future[list[RoomEvent]] = asyncio.get_running_loop().create_future()
        self._tasks: set[asyncio.Task[None]] = set()

    def start(self) -> None:
        if self._tasks or self.sent.done():
            return
        task = asyncio.create_task(self._send())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send(self) -> None:
        await self.over.wait()
        sent = [
            await self._kit.send_event(
                self._room_id, self._channel_id, TextContent(body=f"later {n}"), addressed_to=[]
            )
            for n in range(self._sends)
        ]
        self.sent.set_result(sent)


type _Work = tuple[RoomKit, Callable[[], Awaitable[Any]], _Later]


async def _agent_work(sends: int) -> _Work:
    """An agent whose turn starts the work. The agent is asked below the
    depth limit of 2, with a budget of 20 per cascade."""
    kit = await _room(RoomKit(max_chain_depth=2))
    later = _Later(kit, "r1", "sms", sends)
    kit.register_channel(_Starting("ai", later))
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, lambda: kit.process_inbound(_message()), later


async def _tool_work(sends: int) -> _Work:
    """An AI channel's tool handler starts the work, during the turn."""
    kit = await _room(RoomKit(max_chain_depth=2))
    later = _Later(kit, "r1", "sms", sends)

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        later.start()
        return "ok"

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="t1", name="lookup", arguments={})],
            ),
            AIResponse(content="Done.", finish_reason="stop"),
        ]
    )
    agent = AIChannel(
        "ai",
        provider=provider,
        tool_handler=lookup,
        tools=[AITool(name="lookup", description="d")],
    )
    kit.register_channel(agent)
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, lambda: kit.process_inbound(_message()), later


async def _conference_work(sends: int) -> _Work:
    """A conference's tool call starts the work, at the depth 3 of the
    answer that issued it."""
    provider = MockRealtimeProvider()
    holder: list[_Later] = []

    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        holder[0].start()
        return "done"

    config = ConferenceRealtimeConfig(provider=provider, tools=[LOOKUP], tool_handler=handler)
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    later = _Later(kit, ROOM, "src", sends)
    holder.append(later)
    session = await channel._realtime.ensure_session(ROOM)  # noqa: SLF001
    assert session is not None

    async def start() -> None:
        await kit.send_event(ROOM, "src", TextContent(body="agent says"), chain_depth=2)
        await provider.simulate_tool_call(session, "c1", "lookup", {})
        await until(lambda: bool(provider.tool_results))

    return kit, start, later


_WORK: dict[str, Callable[[int], Awaitable[_Work]]] = {
    "agent": _agent_work,
    "tool-handler": _tool_work,
    "conference-handler": _conference_work,
}


class _Starting(AILikeChannel):
    """An agent whose turn on the first message starts *later*."""

    def __init__(self, channel_id: str, later: _Later) -> None:
        super().__init__(channel_id)
        self._later = later

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "hi":
            self._later.start()
        return ChannelOutput.empty()


class _Deferring(AILikeChannel):
    """An agent whose turn on the first message starts a task that sends
    *sends* messages into the room once the turn is over."""

    def __init__(self, channel_id: str, kit: RoomKit, *, sends: int = 1) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self._sends = sends
        self.turn_over = asyncio.Event()
        self.sent: asyncio.Future[list[RoomEvent]] = asyncio.get_running_loop().create_future()
        self._tasks: set[asyncio.Task[None]] = set()

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "hi":
            task = asyncio.create_task(self._later())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return ChannelOutput.empty()

    async def _later(self) -> None:
        await self.turn_over.wait()
        sent = [
            await self._kit.send_event("r1", "sms", TextContent(body=f"later {n}"))
            for n in range(self._sends)
        ]
        self.sent.set_result(sent)


class _Elsewhere(AILikeChannel):
    """An agent whose turn on the first message sends *sends* messages into
    room ``r2``."""

    def __init__(self, channel_id: str, kit: RoomKit, *, sends: int) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self._sends = sends
        self.sent: list[RoomEvent] = []

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "hi":
            for n in range(self._sends):
                self.sent.append(
                    await self._kit.send_event("r2", "sms", TextContent(body=f"there {n}"))
                )
        return ChannelOutput.empty()


async def _settled(kit: RoomKit) -> None:
    """Wait until the room's lane has run everything queued."""
    await kit._lanes.lane("r1").drain()  # noqa: SLF001


class TestACallFromAnAsyncHook:
    """RFC §10.1 step 16, §10.3, §9.5.1: while the body of an ASYNC hook its
    cascade fires runs (``AFTER_BROADCAST`` once an event's delivery set has
    run, a change's mutation trigger), a call it makes into the room is a
    new arrival that draws on that cascade's budget, as a delivery's does; a
    task it starts calls as work of its own once the body has returned.

    Each looping hook stops itself well past the cap, so that a loop the cap
    does not stop fails at once rather than run forever."""

    async def test_a_hook_posting_after_every_event_stops_at_the_cap(self) -> None:
        """No agent in the room: the hook's own post fires it again."""
        kit = await _room(RoomKit(max_chain_depth=1))
        posted: list[RoomEvent] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="notice")
        async def notice(event: RoomEvent, ctx: RoomContext) -> None:
            if event.type is EventType.MESSAGE and len(posted) <= 2 * BUDGET:
                posted.append(await kit.post("r1", f"Noted {len(posted)}."))

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert [e.blocked_by for e in posted] == [None] * BUDGET + ["reentry_loop_cap"]
        finally:
            await kit.close()

    async def test_a_hook_posting_to_the_agent_after_each_answer_stops_at_the_cap(
        self,
    ) -> None:
        """The agent's answer fires the hook, whose post the agent answers:
        each round trip draws on the budget of the cascade the first message
        opened, its post and the answer's reentry: ten round trips of the
        budget of 20 passes, the eleventh answer capped. The agent is asked
        below the depth limit of 2."""
        kit = await _room(RoomKit(max_chain_depth=2))
        kit.register_channel(AILikeChannel("ai", response="Answer."))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        posted: list[RoomEvent] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="judge")
        async def judge(event: RoomEvent, ctx: RoomContext) -> None:
            if event.source.channel_id == "ai" and len(posted) <= 4 * BUDGET:
                posted.append(await kit.post("r1", "Try again.", addressed_to=["ai"]))

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_settled(kit), 5)
            events = await _messages(kit)
            answers = [e for e in events if e.source.channel_id == "ai" and e.blocked_by is None]
            assert (len(posted), len(answers)) == (10, 10)
            assert [(e.source.channel_id, e.blocked_by) for e in _capped(events)] == [
                ("ai", "reentry_loop_cap")
            ]
        finally:
            await kit.close()

    async def test_a_mutation_hook_editing_its_event_again_stops_at_the_cap(self) -> None:
        """A change's ``ON_EVENT_UPDATED`` is its cascade's work too: a hook
        that edits the event again each time it is edited stops at the cap."""
        kit = await _room(RoomKit(max_chain_depth=1))
        edits: list[RoomEvent] = []

        @kit.hook(HookTrigger.ON_EVENT_UPDATED, execution=HookExecution.ASYNC, name="reword")
        async def reword(event: RoomEvent, ctx: RoomContext) -> None:
            if len(edits) <= 2 * BUDGET:
                edits.append(await _send_edit(kit, event, len(edits)))

        try:
            first = await kit.send_event("r1", "sms", TextContent(body="v"), participant_id="u")
            await asyncio.wait_for(_send_edit(kit, first, 0), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert [e.blocked_by for e in edits] == [None] * BUDGET + ["reentry_loop_cap"]
        finally:
            await kit.close()

    async def test_a_hook_posting_once_per_message_posts_for_every_message(self) -> None:
        """A notice, one post for each message, draws one pass of that
        message's cascade: more messages than a budget holds passes each get
        theirs."""
        kit = await _room(RoomKit(max_chain_depth=1))
        posted: list[RoomEvent] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="receipt")
        async def receipt(event: RoomEvent, ctx: RoomContext) -> None:
            if event.source.channel_id == "sms":
                posted.append(await kit.post("r1", "Received."))

        try:
            for n in range(BUDGET + 1):
                await asyncio.wait_for(kit.process_inbound(_message(f"message {n}")), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert [e.blocked_by for e in posted] == [None] * (BUDGET + 1)
        finally:
            await kit.close()

    async def test_a_task_the_hook_starts_calls_as_work_of_its_own(self) -> None:
        """Once the hook's body has returned, a task it started sends at
        depth 0 with a budget of its own for each call, as a call from
        outside the room's work (RFC §9.5.1, as for a turn's task)."""
        kit = await _room(RoomKit(max_chain_depth=1))
        sends = 2 * BUDGET + 1
        later = _Later(kit, "r1", "sms", sends)

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="start")
        async def start(event: RoomEvent, ctx: RoomContext) -> None:
            if _body(event) == "hi":
                later.start()

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_settled(kit), 5)
            later.over.set()
            sent = await asyncio.wait_for(later.sent, 5)
            assert [(e.chain_depth, e.blocked_by) for e in sent] == [(0, None)] * sends
        finally:
            await kit.close()


async def _send_edit(kit: RoomKit, event: RoomEvent, n: int) -> RoomEvent:
    """Its sender's edit of *event* in room ``r1``."""
    edit = EditContent(target_event_id=event.id, new_content=TextContent(body=f"v{n}"))
    return await kit.send_event("r1", "sms", edit, EventType.EDIT, participant_id="u")


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
