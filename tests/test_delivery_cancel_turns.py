"""Cancelling a message's delivery stops its turns, never its delivery (RFC
§10.1, *Explicit delivery cancellation*): a turn not started yet never starts,
one handed to its executor is cut as it arrives, a running one is cut, and the
other turns of the room go on. The message still reaches the room's other
channels and fires its AFTER_BROADCAST hooks. A caller names the delivery by a
key of its own (``InboundMessage.delivery_key``), so code without its handle
cancels it: ``kit.cancel_delivery(key)``.

Every test runs on the three kinds of intelligence channel a turn can run on:
an ``Agent`` (a tool loop), an ACP agent (its own process, one prompt at a time
per room) and a channel with no steering at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from acp.schema import PromptResponse

from roomkit import Agent, ChannelCategory, HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import ChannelMediaType, ChannelType, EventType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import InMemoryTurnBackend
from roomkit.turns.base import TurnLink
from roomkit.turns.wire import TurnRequest
from tests.test_channels.test_acp import _channel as acp_channel
from tests.test_framework import SimpleChannel

WAIT = 2.0


@dataclass
class Seen:
    """What the agent did with each message it was asked about: started
    working on it, and finished."""

    started: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    gates: dict[str, asyncio.Event] = field(default_factory=dict)

    def gate(self, body: str) -> asyncio.Event:
        return self.gates.setdefault(body, asyncio.Event())

    async def work(self, body: str) -> None:
        self.started.append(body)
        await self.gate(body).wait()
        self.finished.append(body)

    async def until_started(self, body: str) -> None:
        async with asyncio.timeout(WAIT):
            while body not in self.started:
                await asyncio.sleep(0.01)

    async def until_finished(self, body: str) -> None:
        async with asyncio.timeout(WAIT):
            while body not in self.finished:
                await asyncio.sleep(0.01)


def _body(context: AIContext) -> str:
    content = context.messages[-1].content
    return (
        content if isinstance(content, str) else "".join(getattr(p, "text", "") for p in content)
    )


class _Working(MockAIProvider):
    def __init__(self, seen: Seen) -> None:
        super().__init__()
        self._seen = seen

    async def generate(self, context: AIContext) -> AIResponse:
        await self._seen.work(_body(context))
        return AIResponse(content="done")


class _Steerless(Channel):
    """An intelligence channel with no ``steer()``: its owner cuts its turns."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, seen: Seen) -> None:
        super().__init__("agent")
        self._seen = seen

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(media_types=[ChannelMediaType.TEXT])

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.type != EventType.MESSAGE or event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        assert isinstance(event.content, TextContent)
        await self._seen.work(event.content.body)
        return ChannelOutput.empty()


def _agent(seen: Seen, tmp_path: Path) -> Channel:
    return Agent("agent", provider=_Working(seen))


TASKS = ("task A", "task B", "task C")


def _acp(seen: Seen, tmp_path: Path) -> Channel:
    channel, connection, _ = acp_channel(tmp_path, emit_updates=False)
    prompting: dict[str, str] = {}
    cut: set[str] = set()

    async def prompt(session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        text = "".join(getattr(block, "text", "") for block in prompt)
        # The prompt catches the agent up on the room first: its task is the last one named.
        body = max((t for t in TASKS if t in text), key=text.rfind)
        prompting[session_id] = body
        seen.started.append(body)
        await seen.gate(body).wait()
        if body in cut:
            return PromptResponse(stop_reason="cancelled")
        seen.finished.append(body)
        return PromptResponse(stop_reason="end_turn")

    async def cancel(session_id: str) -> None:
        # The agent stops the prompt it runs, which then returns "cancelled".
        body = prompting.get(session_id)
        if body is not None and body not in seen.finished:
            cut.add(body)
            seen.gate(body).set()

    connection.prompt = prompt  # type: ignore[method-assign]
    connection.cancel = cancel  # type: ignore[method-assign]
    return channel


def _steerless(seen: Seen, tmp_path: Path) -> Channel:
    return _Steerless(seen)


PATHS = {"agent": _agent, "acp": _acp, "steerless": _steerless}


class _Transport(SimpleChannel):
    """A transport whose delivery of a message waits while it is held."""

    def __init__(self) -> None:
        super().__init__("ws")
        self.held: dict[str, asyncio.Event] = {}

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if isinstance(event.content, TextContent) and event.content.body in self.held:
            await self.held[event.content.body].wait()
        return await super().deliver(event, binding, context)


@dataclass
class Room:
    kit: RoomKit
    seen: Seen
    ws: _Transport
    broadcast: list[str]

    async def send(
        self, body: str, key: str | None = None, *, visibility: str = "all"
    ) -> InboundResult:
        message = InboundMessage(
            channel_id="sms",
            sender_id="user",
            content=TextContent(body=body),
            delivery_key=key,
            visibility=visibility,
        )
        return await self.kit.process_inbound(message, room_id="r", defer_delivery=True)

    def delivered(self) -> list[str]:
        return [e.content.body for e in self.ws.delivered if isinstance(e.content, TextContent)]

    async def until_broadcast(self, body: str) -> None:
        async with asyncio.timeout(WAIT):
            while body not in self.broadcast:
                await asyncio.sleep(0.01)


async def _room(
    path: str, tmp_path: Path, *, turn_backend: InMemoryTurnBackend | None = None
) -> Room:
    seen = Seen()
    kit = RoomKit(turn_backend=turn_backend)
    ws = _Transport()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(ws)
    agent = PATHS[path](seen, tmp_path)
    kit.register_channel(agent)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", agent.channel_id, category=ChannelCategory.INTELLIGENCE)
    broadcast: list[str] = []

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def record(event: RoomEvent, ctx: Any) -> None:
        if event.type == EventType.MESSAGE and isinstance(event.content, TextContent):
            broadcast.append(event.content.body)

    return Room(kit, seen, ws, broadcast)


@pytest.fixture(params=sorted(PATHS))
def path(request: pytest.FixtureRequest) -> str:
    return str(request.param)


async def _closed(room: Room) -> None:
    for gate in room.seen.gates.values():
        gate.set()
    await room.kit.close()


# -- queued: it never starts ----------------------------------------------------------


async def test_a_turn_queued_behind_the_room_s_turn_never_starts(
    path: str, tmp_path: Path
) -> None:
    room = await _room(path, tmp_path)
    held = room.ws.held["held"] = asyncio.Event()
    try:
        a = await room.send("task A", "a")
        await room.seen.until_started("task A")
        if path == "agent":
            # An agent runs a room's turns side by side: B waits in the
            # room's delivery lane, behind a message a transport still holds.
            await room.send("held", visibility="transport")
        b = await room.send("task B", "b")
        await asyncio.sleep(0.1)
        assert room.seen.started == ["task A"]

        assert await room.kit.cancel_delivery("b") == 1
        room.seen.gate("task A").set()
        held.set()
        assert a.delivery is not None and b.delivery is not None
        done_a = await asyncio.wait_for(a.delivery.wait(), WAIT)
        done_b = await asyncio.wait_for(b.delivery.wait(), WAIT)
        await room.until_broadcast("task B")

        # B's agent never started; A, the room's other turn, ran to its end.
        assert room.seen.started == ["task A"]
        assert room.seen.finished == ["task A"]
        assert done_a.cancellation_reason is None
        assert done_b.cancellation_reason == "caller_cancelled"
        # B itself reached the room: its transport, its AFTER_BROADCAST.
        assert "task B" in room.delivered()
    finally:
        await _closed(room)


# -- running: it is cut, the queued one runs after -------------------------------


async def test_a_running_turn_is_cut_and_the_room_s_next_turn_runs(
    path: str, tmp_path: Path
) -> None:
    room = await _room(path, tmp_path)
    try:
        a = await room.send("task A", "a")
        await room.seen.until_started("task A")
        b = await room.send("task B", "b")

        assert await room.kit.cancel_delivery("a") == 1
        assert a.delivery is not None and b.delivery is not None
        done_a = await asyncio.wait_for(a.delivery.wait(), WAIT)
        await room.seen.until_started("task B")
        room.seen.gate("task B").set()
        done_b = await asyncio.wait_for(b.delivery.wait(), WAIT)
        await room.until_broadcast("task A")

        assert "task A" not in room.seen.finished
        assert "task B" in room.seen.finished
        assert done_a.cancellation_reason == "caller_cancelled"
        assert done_b.cancellation_reason is None
        assert room.delivered()[:2] == ["task A", "task B"]
    finally:
        await _closed(room)


# -- handed: cut as it arrives, never lost --------------------------------------------


class _HeldHandOver(InMemoryTurnBackend):
    """Holds the hand-over of the turns of one message, as a distributed
    backend waits for its executor's answer: the executor has them, their
    owner does not hold them yet."""

    def __init__(self, body: str) -> None:
        super().__init__()
        self._body = body
        self.handing = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, request: TurnRequest) -> TurnLink:
        link = await super().submit(request)
        content = request.event.content
        if isinstance(content, TextContent) and content.body == self._body:
            self.handing.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                # As a distributed backend does: a turn its owner gave up
                # while handed is released, and its executor cuts it.
                link.release()
                raise
        return link


async def test_a_turn_being_handed_over_is_cut_as_it_arrives(path: str, tmp_path: Path) -> None:
    backend = _HeldHandOver("task B")
    room = await _room(path, tmp_path, turn_backend=backend)
    try:
        b = await room.send("task B", "b")
        await asyncio.wait_for(backend.handing.wait(), WAIT)
        c = await room.send("task C", "c")

        assert await room.kit.cancel_delivery("b") == 1
        backend.release.set()
        room.seen.gate("task C").set()
        assert b.delivery is not None and c.delivery is not None
        done_b = await asyncio.wait_for(b.delivery.wait(), WAIT)
        done_c = await asyncio.wait_for(c.delivery.wait(), WAIT)

        assert "task B" not in room.seen.finished
        assert "task C" in room.seen.finished
        assert done_b.cancellation_reason == "caller_cancelled"
        assert done_c.cancellation_reason is None
        assert "task B" in room.delivered()
    finally:
        backend.release.set()
        await _closed(room)


# -- the key ----------------------------------------------------------------------------


async def test_an_awaited_call_reads_its_cancelled_end(tmp_path: Path) -> None:
    room = await _room("agent", tmp_path)
    try:
        message = InboundMessage(
            channel_id="sms",
            sender_id="user",
            content=TextContent(body="task A"),
            delivery_key="a",
        )
        call: asyncio.Task[InboundResult] = asyncio.create_task(
            room.kit.process_inbound(message, room_id="r")
        )
        await room.seen.until_started("task A")

        assert await room.kit.cancel_delivery("a", reason="stopped") == 1
        result = await asyncio.wait_for(call, WAIT)

        assert result.cancellation_reason == "stopped"
        assert room.seen.finished == []
    finally:
        await _closed(room)


async def test_a_key_names_every_delivery_in_flight_and_nothing_after(tmp_path: Path) -> None:
    room = await _room("agent", tmp_path)
    try:
        assert await room.kit.cancel_delivery("never sent") == 0
        a = await room.send("task A", "task")
        await room.seen.until_started("task A")
        b = await room.send("task B", "task")

        assert await room.kit.cancel_delivery("task") == 2
        # Already cancelled: nothing more to cancel.
        assert await room.kit.cancel_delivery("task") == 0
        assert a.delivery is not None and b.delivery is not None
        await asyncio.wait_for(a.delivery.wait(), WAIT)
        await asyncio.wait_for(b.delivery.wait(), WAIT)
        room.seen.gate("task C").set()
        c = await room.send("task C", "c")
        assert c.delivery is not None
        await asyncio.wait_for(c.delivery.wait(), WAIT)

        # Over: a key names nothing any more, and is held by nothing.
        assert await room.kit.cancel_delivery("c") == 0
        await _settled(lambda: not room.kit._delivery_keys._named)
        assert room.seen.finished == ["task C"]
    finally:
        await _closed(room)


async def test_a_cancel_under_the_room_lock_returns_at_once(path: str, tmp_path: Path) -> None:
    """A sync hook runs under the room lock, where waiting on the room's
    delivery would never end: the cancel returns at once, and the turn it
    names never starts."""
    room = await _room(path, tmp_path)
    cancelled: list[int] = []

    @room.kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def stop(event: RoomEvent, ctx: Any) -> HookResult:
        if isinstance(event.content, TextContent) and event.content.body == "stop B":
            cancelled.append(await room.kit.cancel_delivery("b"))
        return HookResult.allow()

    try:
        a = await room.send("task A", "a")
        await room.seen.until_started("task A")
        b = await room.send("task B", "b")
        await room.send("stop B")
        room.seen.gate("task A").set()
        room.seen.gate("stop B").set()
        assert a.delivery is not None and b.delivery is not None
        done_b = await asyncio.wait_for(b.delivery.wait(), WAIT)

        assert cancelled == [1]
        assert "task B" not in room.seen.started
        assert done_b.cancellation_reason == "caller_cancelled"
    finally:
        await _closed(room)


async def _settled(ready: Callable[[], Any]) -> None:
    async with asyncio.timeout(WAIT):
        while not ready():
            await asyncio.sleep(0.01)


async def test_a_task_delegated_from_a_delivery_is_not_its_turn_work(tmp_path: Path) -> None:
    """A task an AFTER_BROADCAST hook delegates runs on its own: cancelling
    the delivery that fired the hook stops the delivery's turns, never the
    delegated task's (RFC §10.1: no task detached by application code is
    owned)."""
    room = await _room("agent", tmp_path)
    worker = Seen()
    room.kit.register_channel(Agent("worker", provider=_Working(worker)))
    delegated: list[Any] = []

    @room.kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def delegate(event: RoomEvent, ctx: Any) -> None:
        if isinstance(event.content, TextContent) and event.content.body == "task A":
            delegated.append(await room.kit.delegate("r", "worker", "child task"))

    try:
        a = await room.send("task A", "a")
        await room.seen.until_started("task A")
        await _settled(lambda: worker.started)

        assert await room.kit.cancel_delivery("a") == 1
        assert a.delivery is not None
        done_a = await asyncio.wait_for(a.delivery.wait(), WAIT)
        worker.gate(worker.started[0]).set()
        result = await asyncio.wait_for(delegated[0].wait(), WAIT)

        assert done_a.cancellation_reason == "caller_cancelled"
        assert "task A" not in room.seen.finished
        assert worker.finished == worker.started
        assert result.output == "done"
    finally:
        for gate in worker.gates.values():
            gate.set()
        await _closed(room)
