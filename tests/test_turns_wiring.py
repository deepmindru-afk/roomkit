"""An executor wires the channels it runs, the kit's own as a host's (RFC
§10.6.2, §10.6.3, §10.6.6, RMK-821).

A host builds an executor with a factory; its channels fire its hooks,
against a room context built from the shared store, and what they publish
and emit during a turn reaches the room's owner as frames. Room hooks on the
triggers a turn fires are defined by name on the executors and enabled per
room on the owner.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import TurnLinkError
from roomkit.core.hooks import HookRegistration
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.scoring import MockScorer, ScoringHook
from roomkit.store.memory import InMemoryStore
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.turns import InMemoryTurnBackend, TurnExecutor, memory
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="Look it up.", parameters={"type": "object"})


def _agent(*, calls: bool = True) -> Agent:
    """An agent that calls ``lookup`` once, then answers ``done``."""
    first = (
        AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})])
        if calls
        else AIResponse(content="done")
    )

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    provider = MockAIProvider(ai_responses=[first, AIResponse(content="done")])
    return Agent("agent", provider=provider, role="agent", tools=[LOOKUP], tool_handler=handler)


async def _kit(backend: InMemoryTurnBackend | None = None, **kit: Any) -> RoomKit:
    room = RoomKit(turn_backend=backend, **kit)
    room.register_channel(SimpleChannel("ops"))
    room.register_channel(_agent())
    await room.create_room(room_id="r1")
    await room.attach_channel("r1", "ops")
    await room.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    return room


async def _say(kit: RoomKit, body: str = "go") -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body)),
        room_id="r1",
    )


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _elsewhere(store: InMemoryStore, **options: Any) -> tuple[InMemoryTurnBackend, Any]:
    """A backend served first by an executor a host built, with its own copy
    of the agent: the room's turns run there."""
    backend = InMemoryTurnBackend()
    built: list[Agent] = []

    def factory(channel_id: str) -> Agent | None:
        if channel_id != "agent":
            return None
        built.append(_agent())
        return built[-1]

    executor = TurnExecutor(
        factory, hosts=lambda c: c == "agent", executor_id="elsewhere", store=store, **options
    )
    executor.built = built  # ty: ignore[unresolved-attribute]
    await backend.serve(executor)
    return backend, executor


# -- An executor a host built -------------------------------------------------------


async def test_a_host_s_executor_fires_its_own_hooks() -> None:
    store = InMemoryStore()
    backend, executor = await _elsewhere(store)
    kit = await _kit(backend, store=store)
    seen: list[str] = []

    @executor.hook(HookTrigger.BEFORE_TOOL_USE)
    async def gate(event: Any, context: Any) -> Any:
        seen.append(f"gate:{context.room.id}")

    @executor.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def report(event: Any, context: Any) -> None:
        seen.append(f"tool:{event.name}")

    @executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def response(event: Any, context: Any) -> None:
        seen.append(f"response:{event.response_content}")

    @kit.turn_executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def kits(event: Any, context: Any) -> None:
        seen.append("kit's executor")

    await _say(kit)
    await _settle()

    assert seen == ["gate:r1", "tool:lookup", "response:done"]
    await kit.close()
    await executor.close()


async def test_a_host_s_executor_builds_a_channel_once_and_closes_it() -> None:
    store = InMemoryStore()
    backend, executor = await _elsewhere(store)
    kit = await _kit(backend, store=store)
    await _say(kit, "one")
    await _say(kit, "two")

    (agent,) = executor.built
    closed: list[str] = []
    original = agent.close

    async def close() -> None:
        closed.append(agent.channel_id)
        await original()

    agent.close = close  # ty: ignore[invalid-assignment]
    await executor.close()
    assert closed == ["agent"]
    await kit.close()


async def test_what_a_turn_elsewhere_publishes_reaches_the_owner() -> None:
    store = InMemoryStore()
    backend, executor = await _elsewhere(store)
    kit = await _kit(backend, store=store)
    published: list[str] = []
    emitted: list[str] = []

    async def on_ephemeral(event: Any) -> None:
        published.append(str(event.type))

    await kit.realtime.subscribe_to_room("r1", on_ephemeral)

    @kit.on("tool_call")
    async def on_tool_call(event: Any) -> None:
        emitted.append(f"tool_call:{event.data['tool_name']}")

    @kit.on("ai_response")
    async def on_response(event: Any) -> None:
        emitted.append(f"ai_response:{event.channel_id}")

    await _say(kit)
    await _settle()

    assert "tool_call_start" in published and "tool_call_end" in published
    assert emitted == ["tool_call:lookup", "ai_response:agent"]
    await kit.close()
    await executor.close()


async def test_the_kit_s_own_turns_publish_through_the_backend_too() -> None:
    kit = await _kit()
    frames: list[str] = []
    executor = kit.turn_executor
    original = executor.take

    def take(link: Any) -> None:
        send = link.send

        async def recorded(frame: Any) -> int:
            frames.append(frame.type)
            return await send(frame)

        link.send = recorded
        original(link)

    executor.take = take  # ty: ignore[invalid-assignment]
    await _say(kit)
    await _settle()

    assert "ephemeral" in frames and "framework_event" in frames
    await kit.close()


async def test_a_turn_in_a_context_of_its_own_continues_the_owner_s_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An executor in another process starts a turn in a fresh context: the
    request's trace is what parents the turn's spans to the broadcast's."""
    monkeypatch.setattr(memory._ExecutorLink, "turn_context", lambda self: contextvars.Context())
    telemetry = MockTelemetryProvider()
    kit = await _kit(telemetry=telemetry)
    requests: list[Any] = []
    take = kit.turn_executor.take

    def record(link: Any) -> None:
        requests.append(link.request)
        take(link)

    kit.turn_executor.take = record  # ty: ignore[invalid-assignment]
    await _say(kit)

    (request,) = requests
    assert request.trace is not None
    span_id = request.trace["span_id"]
    assert span_id in {span.id for span in telemetry.get_spans(SpanKind.BROADCAST)}
    generations = telemetry.get_spans(SpanKind.LLM_GENERATE)
    assert generations and all(span.parent_id == span_id for span in generations)
    await kit.close()


async def test_a_scoring_hook_attaches_to_a_host_s_executor() -> None:
    store = InMemoryStore()
    backend, executor = await _elsewhere(store)
    kit = await _kit(backend, store=store)
    hook = ScoringHook(scorers=[MockScorer()])
    hook.attach(executor)

    await _say(kit)
    await _settle()

    assert hook.recent_scores
    assert await store.list_observations("r1")
    await kit.close()
    await executor.close()


def test_a_scoring_hook_needs_a_store() -> None:
    executor = TurnExecutor(lambda channel_id: None)
    with pytest.raises(ValueError, match="store"):
        ScoringHook(scorers=[MockScorer()]).attach(executor)


# -- Room hooks by name (RFC §10.6.6) -------------------------------------------------


async def test_a_room_hook_runs_for_the_rooms_that_enable_it() -> None:
    kit = await _kit()
    await kit.create_room(room_id="r2")
    await kit.attach_channel("r2", "ops")
    await kit.attach_channel("r2", "agent", category=ChannelCategory.INTELLIGENCE)
    seen: list[str] = []

    @kit.turn_executor.room_hook("audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)
    async def audit(event: Any, context: Any) -> None:
        seen.append(context.room.id)

    kit.enable_room_hook("r1", "audit")
    await _say(kit)
    await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body="go")),
        room_id="r2",
    )
    await _settle()
    assert seen == ["r1"]

    assert kit.disable_room_hook("r1", "audit") is True
    assert kit.disable_room_hook("r1", "audit") is False
    await kit.close()


async def test_a_room_s_turn_goes_to_an_executor_defining_its_hooks() -> None:
    store = InMemoryStore()
    backend, elsewhere = await _elsewhere(store)
    kit = await _kit(backend, store=store)
    seen: list[str] = []

    @kit.turn_executor.room_hook("audit", HookTrigger.ON_AI_RESPONSE, HookExecution.ASYNC)
    async def audit(event: Any, context: Any) -> None:
        seen.append("kit's executor")

    taken: list[str] = []
    take = elsewhere.take

    def record(link: Any) -> None:
        taken.append(link.request.turn_id)
        take(link)

    elsewhere.take = record  # ty: ignore[invalid-assignment]
    kit.enable_room_hook("r1", "audit")
    await _say(kit)
    await _settle()

    # The executor served first does not define "audit": the kit's own takes it.
    assert seen == ["kit's executor"]
    assert taken == []
    await kit.close()
    await elsewhere.close()


async def test_a_room_hook_no_executor_defines_leaves_the_room_unanswered() -> None:
    kit = await _kit()
    kit.enable_room_hook("r1", "missing")

    await _say(kit)

    assert kit._turns.room_hook_names("r1") == ["missing"]
    timeline = await kit.store.list_events("r1")
    assert not [e for e in timeline if e.source.channel_id == "agent"]
    with pytest.raises(TurnLinkError, match="room hooks"):
        kit._turns.backend._place("agent", "r1", ["missing"])
    await kit.close()


def test_a_room_hook_on_an_executor_trigger_is_refused_by_callable() -> None:
    executor = TurnExecutor(lambda channel_id: None)

    async def fn(event: Any, context: Any) -> None:
        return None

    registration = HookRegistration(
        trigger=HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, fn=fn, name="x"
    )
    with pytest.raises(ValueError, match="room_hook"):
        executor.hooks.add_room_hook("r1", registration)
