"""Work a turn began reaches the room owner after the turn ended (RFC
§10.6.3, §10.6.5): the host verbs that write the room, at the turn's chain
depth, and the channel's publications, from the executor of the room's last
turn only."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import Agent, ChannelDeclaration, RoomKit, current_turn_host
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.store.memory import InMemoryStore
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from tests.test_framework import SimpleChannel

LATER = AITool(name="later", description="Starts work that outlives the turn.", parameters={})


class _Later:
    """A host tool starting work that waits for the test to let it go, after
    the turn ended, then runs *work* with the host the turn left it."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.done: asyncio.Future[Any] | None = None
        self.work: Any = None

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        async def run() -> Any:
            await self.release.wait()
            return await self.work()

        self.done = asyncio.ensure_future(run())
        return "started"


def _calls_later() -> AIResponse:
    call = AIToolCall(id="c1", name="later", arguments={})
    return AIResponse(content="", finish_reason="tool_calls", tool_calls=[call])


class _Host:
    def __init__(self, later: _Later) -> None:
        self.store = InMemoryStore()
        self.backend = InMemoryTurnBackend()
        provider = MockAIProvider(
            ai_responses=[_calls_later(), AIResponse(content="On it.")]
            + [AIResponse(content="Again.")] * 4
        )
        self.agent = Agent("ana", provider=provider, tools=[LATER], tool_handler=later.handler)
        self.executor = TurnExecutor(
            {"ana": self.agent}.get, hosts=lambda cid: cid == "ana", store=self.store
        )

    async def kit(self) -> RoomKit:
        await self.backend.serve(self.executor)
        kit = RoomKit(store=self.store, turn_backend=self.backend)
        kit.register_channel(SimpleChannel("sms"))
        kit.register_channel(ChannelDeclaration(channel_id="ana"))
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "sms")
        await kit.attach_channel("r", "ana", category=ChannelCategory.INTELLIGENCE)
        return kit

    async def close(self, kit: RoomKit) -> None:
        await kit.close()
        await self.executor.close()


async def _say(kit: RoomKit, text: str = "Go.") -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="marie", content=TextContent(body=text)),
        room_id="r",
    )


async def _after_the_turn(later: _Later) -> Any:
    assert later.done is not None
    later.release.set()
    return await asyncio.wait_for(later.done, 5)


async def test_work_a_turn_began_writes_the_room_after_the_turn_ended() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()
    posted: list[Any] = []

    async def on_status(entry: Any) -> None:
        posted.append(entry)

    await kit.status_bus.subscribe(on_status)

    async def work() -> Any:
        turn_host = current_turn_host()
        assert turn_host is not None and not turn_host.open
        await turn_host.post_status("ana", "thought", "info", detail="Still thinking")
        return await turn_host.send_event(
            room_id="r", channel_id="ana", content=TextContent(body="A thought, later.")
        )

    later.work = work
    await _say(kit)
    sent = await _after_the_turn(later)
    stored = [e for e in await kit.store.list_events("r") if e.id == sent.id]
    await host.close(kit)

    assert [
        (p.action, p.detail, p.metadata.get("room_id")) for p in posted if p.action == "thought"
    ] == [("thought", "Still thinking", "r")]
    assert stored and stored[0].chain_depth >= 1


async def test_a_verb_that_needs_its_turn_is_refused_after_it() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()

    async def work() -> Any:
        turn_host = current_turn_host()
        assert turn_host is not None
        with pytest.raises(RuntimeError, match="needs a turn"):
            await turn_host.delegate("r", "ana", "More.")
        return True

    later.work = work
    await _say(kit)
    refused = await _after_the_turn(later)
    await host.close(kit)

    assert refused is True


async def test_work_of_a_turn_that_is_no_longer_the_room_s_last_is_refused() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()

    async def work() -> Any:
        turn_host = current_turn_host()
        assert turn_host is not None
        with pytest.raises(PermissionError, match="no longer the room's last"):
            await turn_host.post_status("ana", "stale", "info")
        return True

    later.work = work
    await _say(kit)
    await _say(kit, "And again.")  # a later turn of ana in r
    refused = await _after_the_turn(later)
    await host.close(kit)

    assert refused is True


async def test_a_publication_after_the_turn_reaches_the_owner_for_its_room_only() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()
    seen: list[str] = []

    async def on_event(event: EphemeralEvent) -> None:
        if event.type is EphemeralEventType.TYPING_START:
            seen.append(f"{event.room_id}:{event.type}")

    await kit.realtime.subscribe_to_room("r", on_event)
    await kit.realtime.subscribe_to_room("elsewhere", on_event)

    async def work() -> Any:
        realtime = host.agent._realtime
        for room_id in ("r", "elsewhere"):
            await realtime.publish_to_room(
                room_id,
                EphemeralEvent(
                    room_id=room_id,
                    type=EphemeralEventType.TYPING_START,
                    user_id="ana",
                    channel_id="ana",
                ),
            )
        return True

    later.work = work
    await _say(kit)
    await _after_the_turn(later)
    await asyncio.sleep(0.1)
    await host.close(kit)

    assert seen == ["r:typing_start"]
