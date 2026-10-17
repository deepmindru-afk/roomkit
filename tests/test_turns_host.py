"""Code on a turn's executor reaches a room through the turn's host (RFC §10.6.5).

A host's tool reads ``current_turn_host()``; each verb crosses the turn
backend and runs on the room owner, in a task of its own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import RoomKit, TurnHost, current_turn_host
from roomkit.channels.agent import Agent
from roomkit.core._turn_host import KitHost
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.task import Observation
from roomkit.models.tool_call import ToolCallEvent
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

USE = AITool(name="use_host", description="Use the host.", parameters={"type": "object"})

HostUse = Callable[[TurnHost], Awaitable[Any]]


def _call(call_id: str = "c1") -> AIResponse:
    return AIResponse(
        content="", tool_calls=[AIToolCall(id=call_id, name="use_host", arguments={})]
    )


async def _kit(
    use: HostUse, *more: Agent, then: tuple[str, ...] = ("done",)
) -> tuple[RoomKit, Agent, dict[str, Any]]:
    """A kit whose agent's one host tool runs *use* with the turn's host,
    its answers after the call *then*."""
    seen: dict[str, Any] = {}

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        host = current_turn_host()
        task = asyncio.current_task()
        seen["host"] = host
        seen["tool_task"] = task.get_name() if task else ""
        assert host is not None
        seen["result"] = await use(host)
        return "used"

    provider = MockAIProvider(ai_responses=[_call(), *(AIResponse(content=a) for a in then)])
    agent = Agent("agent", provider=provider, role="agent", tools=[USE], tool_handler=handler)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    for other in more:
        kit.register_channel(other)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, agent, seen


async def _say(kit: RoomKit, body: str = "go") -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body)),
        room_id="r1",
    )


@pytest.fixture
def verbs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Each verb the owner ran: its name, the task that ran it."""
    seen: list[tuple[str, str]] = []
    call = KitHost.call

    async def spy(self: KitHost, request: Any, verb: str, arguments: dict[str, Any]) -> Any:
        task = asyncio.current_task()
        seen.append((verb, task.get_name() if task else ""))
        return await call(self, request, verb, arguments)

    monkeypatch.setattr(KitHost, "call", spy)
    return seen


def _on_owner(verbs: list[tuple[str, str]], verb: str) -> bool:
    return any(v == verb and task.startswith("roomkit-owner-call-") for v, task in verbs)


# -- Where the host is ----------------------------------------------------------


async def test_a_host_tool_reads_the_turns_host_and_it_ends_with_the_turn() -> None:
    async def use(host: TurnHost) -> Any:
        return host.open

    kit, _, seen = await _kit(use)
    assert current_turn_host() is None

    await _say(kit)

    host = seen["host"]
    assert seen["result"] is True
    assert not host.open
    with pytest.raises(RuntimeError):
        await host.post_status("agent", "late", "info")
    await kit.close()


async def test_a_hook_the_executor_fires_during_a_turn_reads_its_host() -> None:
    async def use(host: TurnHost) -> Any:
        return None

    kit, _, _ = await _kit(use)
    hosts: list[TurnHost | None] = []

    @kit.turn_executor.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
    async def report(event: ToolCallEvent, ctx: Any) -> None:
        hosts.append(current_turn_host())

    await _say(kit)

    assert len(hosts) == 1
    assert hosts[0] is not None
    await kit.close()


# -- The verbs --------------------------------------------------------------------


async def test_send_event_runs_on_the_owner_and_the_room_moves_on(
    verbs: list[tuple[str, str]],
) -> None:
    async def use(host: TurnHost) -> Any:
        return await host.send_event(
            "r1", "ops", TextContent(body="from the tool"), addressed_to=[]
        )

    kit, _, seen = await _kit(use, then=("done", "again"))

    await asyncio.wait_for(_say(kit), 5)

    event = seen["result"]
    assert event.content.body == "from the tool"
    assert await kit.store.get_event(event.id) is not None
    assert _on_owner(verbs, "send_event")
    # The room's lane went on: a later message is answered.
    result = await asyncio.wait_for(_say(kit, "next"), 5)
    assert [e.content.body for e in result.response_events] == ["again"]
    await kit.close()


async def test_delegate_answers_a_task_id_that_wait_task_waits(
    verbs: list[tuple[str, str]],
) -> None:
    async def use(host: TurnHost) -> Any:
        task_id = await host.delegate("r1", "worker", "Look it up")
        return task_id, await host.wait_task(task_id, timeout=5)

    worker = Agent(
        "worker",
        provider=MockAIProvider(ai_responses=[AIResponse(content="found")]),
        role="worker",
    )
    kit, _, seen = await _kit(use, worker)

    await asyncio.wait_for(_say(kit), 10)

    task_id, result = seen["result"]
    assert isinstance(task_id, str)
    assert result is not None
    assert result.task_id == task_id
    assert result.output == "found"
    assert _on_owner(verbs, "delegate")
    assert _on_owner(verbs, "wait_task")
    await kit.close()


async def test_wait_task_answers_none_for_a_task_it_does_not_know() -> None:
    async def use(host: TurnHost) -> Any:
        return await host.wait_task("no-such-task", timeout=0.1)

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert seen["result"] is None
    await kit.close()


async def test_room_metadata_status_and_observations_are_written_on_the_owner(
    verbs: list[tuple[str, str]],
) -> None:
    async def use(host: TurnHost) -> Any:
        room = await host.update_room_metadata("r1", {"stage": "billing"})
        await host.post_status("agent", "lookup", "info", detail="looked")
        await host.add_observation(
            Observation(id="o1", room_id="r1", channel_id="agent", content="noted")
        )
        return room

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert seen["result"].metadata["stage"] == "billing"
    assert (await kit.get_room("r1")).metadata["stage"] == "billing"
    [entry] = await kit.status_bus.recent(1, agent_id="agent")
    assert entry.action == "lookup"
    assert [o.content for o in await kit.store.list_observations("r1")] == ["noted"]
    for verb in ("update_room_metadata", "post_status", "add_observation"):
        assert _on_owner(verbs, verb)
    await kit.close()


async def test_a_verb_that_raises_raises_in_the_tool_with_its_class() -> None:
    async def use(host: TurnHost) -> Any:
        try:
            await host.update_room_metadata("no-such-room", {"x": 1})
        except RoomNotFoundError as exc:
            return exc
        return None

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert isinstance(seen["result"], RoomNotFoundError)
    await kit.close()


async def test_the_owner_refuses_a_verb_or_arguments_it_does_not_know() -> None:
    kit = RoomKit()
    host = KitHost(kit)
    request: Any = None

    with pytest.raises(ValueError, match="No verb"):
        await host.call(request, "close_kit", {})
    with pytest.raises(ValueError, match="not valid"):
        await host.call(request, "send_event", {"room_id": "r1"})
    await kit.close()


async def test_send_event_from_a_turn_the_lane_itself_reads_returns_detached() -> None:
    """A supervisor's first pass is read inside the room's lane, by its
    runner: an event its host tool sends into the room comes back detached,
    not awaited on the delivery the lane is busy with (RFC §10.1 step 18)."""
    sent: list[Any] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        host = current_turn_host()
        assert host is not None
        # Addressed to no agent: its delivery set, queued behind the one the
        # lane runs, holds no turn. Awaited, it would wait for the lane.
        sent.append(
            await host.send_event("r1", "ops", TextContent(body="posted"), addressed_to=[])
        )
        return "posted"

    boss = Agent(
        "boss",
        provider=MockAIProvider(
            ai_responses=[
                _call(),
                AIResponse(content="Look it up."),
                AIResponse(content="Presented."),
            ]
        ),
        role="boss",
        tools=[USE],
        tool_handler=handler,
    )
    worker = Agent(
        "worker", provider=MockAIProvider(ai_responses=[AIResponse(content="found")]), role="w"
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(boss)
    kit.register_channel(worker)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.install_strategy(
        "r1",
        Supervisor(
            supervisor=boss,
            workers=[worker],
            strategy="parallel",
            auto_delegate=True,
            refine_task=True,
        ),
    )

    result = await asyncio.wait_for(_say(kit, "Why?"), 10)

    assert [e.content.body for e in sent] == ["posted"]
    texts = [e.content.body for e in result.response_events if isinstance(e.content, TextContent)]
    assert texts == ["Presented."]
    await kit.close()
