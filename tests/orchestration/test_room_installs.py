"""What a strategy installs for a room runs for that room (RFC §19.7; RMK-307).

One agent serves every room it is attached to. Two rooms may install the same
kind of strategy on it with different configurations: each room's call runs its
own install, whichever room was installed first. The tools orchestration sets
up are served through the channel's dispatch, so the channel's rules (the
repeat guard) hold on them as on any other tool.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import RoomKit, ToolNameCollisionError
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.orchestration.strategies import loop as loop_module
from roomkit.orchestration.strategies.loop import Loop
from roomkit.orchestration.strategies.pipeline import Pipeline
from roomkit.orchestration.strategies.supervisor import Supervisor, _install_auto
from roomkit.orchestration.strategies.swarm import Swarm
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delegate import DelegateHandler, setup_delegation
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conftest import make_event
from tests.test_framework import SimpleChannel
from tests.tool_loop_modes import respond
from tests.tool_room import room_tool_names

TENANTS = ("bank-A", "clinic-B")


def _calling(
    name: str, arguments: dict[str, Any], *, times: int, streaming: bool
) -> MockAIProvider:
    call = AIResponse(content="", tool_calls=[AIToolCall(id="c", name=name, arguments=arguments)])
    return MockAIProvider(
        ai_responses=[*(call for _ in range(times)), AIResponse(content="done")],
        streaming=streaming,
    )


def _tool_results(provider: MockAIProvider) -> list[Any]:
    return [
        part.result
        for call in provider.calls
        for message in call.messages
        if message.role == "tool"
        for part in message.content
    ]


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def _two_rooms(sup: Agent, installs: dict[str, Supervisor]) -> RoomKit:
    kit = RoomKit()
    for tenant, orchestration in installs.items():
        kit.register_channel(SimpleChannel(f"sms-{tenant}"))
        await kit.create_room(room_id=tenant, orchestration=orchestration)
        await kit.attach_channel(tenant, f"sms-{tenant}")
    return kit


async def test_each_room_s_per_worker_tools_run_with_its_own_install(streaming: bool) -> None:
    """Room B's delegation runs B's worker with B's settings, though room A
    installed a supervisor on the same agent first."""
    worker_a = Agent("worker_a", provider=MockAIProvider(responses=["a"]), tool_search=False)
    worker_b = Agent("worker_b", provider=MockAIProvider(responses=["b"]), tool_search=False)
    sup_model = _calling("delegate_to_worker_b", {"task": "file it"}, times=1, streaming=streaming)
    # The failure handed back gets an answer, not a second dispatch.
    sup_model._ai_responses.append(AIResponse(content="Noted."))
    sup = Agent("sup", provider=sup_model, tool_search=False)
    kit = await _two_rooms(
        sup,
        {
            "bank-A": Supervisor(sup, [worker_a], wait_for_result=False, share_channels=["a"]),
            "clinic-B": Supervisor(sup, [worker_b], wait_for_result=False, share_channels=["b"]),
        },
    )
    delegated: list[tuple[str, str, list[str] | None]] = []

    async def recording(room_id: str, agent_id: str, task: str, **kwargs: Any) -> Any:
        delegated.append((room_id, agent_id, kwargs.get("share_channels")))
        raise RuntimeError("stop here")

    kit.delegate = recording  # type: ignore[method-assign]

    await kit.process_inbound(
        InboundMessage(channel_id="sms-clinic-B", sender_id="u", content=TextContent(body="go"))
    )
    await _until(lambda: bool(delegated))

    await asyncio.sleep(0.05)  # the failure is handed back and answered
    assert delegated == [("clinic-B", "worker_b", ["b"])]
    await kit.close()


async def test_each_room_s_team_tool_runs_its_own_team(streaming: bool) -> None:
    """``delegate_workers`` declares room B's team and runs it."""
    worker_a = Agent("worker_a", provider=MockAIProvider(responses=["a"]), tool_search=False)
    worker_b = Agent("worker_b", provider=MockAIProvider(responses=["b"]), tool_search=False)
    sup_model = _calling("delegate_workers", {"task": "file it"}, times=1, streaming=streaming)
    sup = Agent("sup", provider=sup_model, tool_search=False)
    team = {"strategy": "parallel", "async_delivery": True}
    kit = await _two_rooms(
        sup,
        {
            "bank-A": Supervisor(sup, [worker_a], **team),  # type: ignore[arg-type]
            "clinic-B": Supervisor(sup, [worker_b], **team),  # type: ignore[arg-type]
        },
    )

    await kit.process_inbound(
        InboundMessage(channel_id="sms-clinic-B", sender_id="u", content=TextContent(body="go"))
    )
    await _until(lambda: bool(_tool_results(sup_model)))

    declared = {t.name: t.description for t in sup_model.calls[0].tools or []}
    assert "worker_b" in declared["delegate_workers"]
    assert "worker_a" not in declared["delegate_workers"]
    assert json.loads(_tool_results(sup_model)[0])["workers"] == ["worker_b"]
    await kit.close()


async def test_the_repeat_guard_holds_on_an_orchestration_tool(streaming: bool) -> None:
    """The third identical ``delegate_task`` of a turn is stopped like any
    other tool's: orchestration tools are served through the channel's
    dispatch, not around it."""
    delegations: list[dict[str, Any]] = []

    class Recording(DelegateHandler):
        async def handle(self, **kwargs: Any) -> dict[str, Any]:
            delegations.append(kwargs)
            return {"status": "delegated"}

    provider = _calling(
        "delegate_task", {"agent": "w", "task": "same"}, times=5, streaming=streaming
    )
    channel = AIChannel("ai1", provider=provider, tool_search=False)
    setup_delegation(channel, Recording(RoomKit()))

    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    event = make_event(room_id="r1", body="go", channel_id="sms1")
    await respond(channel, event, binding, RoomContext(room=Room(id="r1")))

    assert len(delegations) == 2


async def test_a_room_without_the_install_gets_the_supervisor_s_own_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supervisor whose delegation passes take room A's turns answers as
    itself in a room where nothing was installed: no worker runs there."""
    passes: list[str] = []

    async def one_pass(kit: Any, rid: str, *args: Any, **kwargs: Any) -> ChannelOutput:
        passes.append(rid)
        return ChannelOutput.empty()

    monkeypatch.setattr(_install_auto, "_one_pass_delegate", one_pass)
    monkeypatch.setattr(_install_auto, "_two_pass_delegate", one_pass)
    sup_model = MockAIProvider(responses=["answered as myself"])
    sup = Agent("sup", provider=sup_model, tool_search=False)
    worker = Agent("worker", provider=MockAIProvider(responses=["w"]), tool_search=False)
    kit = RoomKit()
    for room_id in ("room-A", "room-C"):
        kit.register_channel(SimpleChannel(f"sms-{room_id}"))
    orchestration = Supervisor(sup, [worker], strategy="sequential", auto_delegate=True)
    await kit.create_room(room_id="room-A", orchestration=orchestration)
    await kit.attach_channel("room-A", "sms-room-A")
    await kit.create_room(room_id="room-C")
    await kit.attach_channel("room-C", "sms-room-C")
    await kit.attach_channel("room-C", "sup", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="sms-room-C", sender_id="u", content=TextContent(body="hi"))
    )
    await kit.process_inbound(
        InboundMessage(channel_id="sms-room-A", sender_id="u", content=TextContent(body="go"))
    )

    assert passes == ["room-A"]
    assert len(sup_model.calls) == 1
    await kit.close()


async def test_each_room_s_loop_runs_with_its_own_reviewers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two rooms loop the same producer past different reviewers: each room's
    turn runs its own."""
    ran: list[tuple[str, list[str]]] = []

    async def run_loop(*, room_id: str, reviewers: list[Agent], **kwargs: Any) -> ChannelOutput:
        ran.append((room_id, [r.channel_id for r in reviewers]))
        return ChannelOutput.empty()

    monkeypatch.setattr(loop_module, "_run_loop", run_loop)
    producer = Agent("writer", provider=MockAIProvider(responses=["draft"]), tool_search=False)
    kit = await _two_rooms(
        producer,
        {
            tenant: Loop(
                agent=producer,
                reviewer=Agent(f"editor-{tenant}", provider=MockAIProvider(responses=["ok"])),
            )
            for tenant in TENANTS
        },
    )

    await kit.process_inbound(
        InboundMessage(channel_id="sms-clinic-B", sender_id="u", content=TextContent(body="go"))
    )

    assert ran == [("clinic-B", ["editor-clinic-B"])]
    await kit.close()


# -- Realtime ------------------------------------------------------------------


def _connected(provider: MockRealtimeProvider, session: Any) -> dict[str, Any]:
    """What *session* was given when it connected."""
    return [
        c.args
        for c in provider.calls
        if c.method == "connect" and c.args["session_id"] == session.id
    ][-1]


async def test_each_room_s_voice_supervisor_runs_its_own_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Room B's call runs B's workers though room A installed a supervisor on
    the same voice channel first; a room with no install declares nothing."""
    ran: list[tuple[str, list[str]]] = []

    async def run_and_deliver(**kwargs: Any) -> None:
        ran.append((kwargs["room_id"], [w.channel_id for w in kwargs["workers"]]))
        kwargs["on_done"]()

    monkeypatch.setattr(_install_auto, "_async_run_and_deliver", run_and_deliver)
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel("voice", provider=provider, transport=MockRealtimeTransport())
    kit = RoomKit()
    kit.register_channel(voice)
    sup = Agent("sup", provider=MockAIProvider(responses=["ok"]), tool_search=False)
    sessions = {}
    for tenant in (*TENANTS, "plain"):
        worker = Agent(f"worker-{tenant}", provider=MockAIProvider(responses=["w"]))
        orchestration = (
            None
            if tenant == "plain"
            else Supervisor(
                sup, [worker], strategy="sequential", auto_delegate=True, async_delivery=True
            )
        )
        await kit.create_room(room_id=tenant, orchestration=orchestration)
        await kit.attach_channel(tenant, "voice")
        sessions[tenant] = await voice.start_session(tenant, f"user-{tenant}", "ws")

    await provider.simulate_tool_call(
        sessions["clinic-B"], "c1", "delegate_workers", {"task": "B"}
    )
    await _until(lambda: len(provider.tool_results) == 1)

    assert ran == [("clinic-B", ["worker-clinic-B"])]
    assert [t["name"] for t in _connected(provider, sessions["bank-A"])["tools"]] == [
        "delegate_workers"
    ]
    assert not _connected(provider, sessions["plain"])["tools"]
    await kit.close()


async def test_a_handoff_in_one_room_leaves_another_room_s_sessions() -> None:
    """A pipeline's handoff in room A reconfigures A's sessions; a session that
    starts in room B starts with B's active agent, and the channel's own
    configuration is left as it was."""
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        system_prompt="I am the channel",
        tool_handler=AsyncMock(return_value="host"),
    )
    triage = Agent(
        "triage",
        system_prompt="I am TRIAGE",
        tools=[AITool(name="lookup", description="l", parameters={})],
    )
    billing = Agent(
        "billing",
        system_prompt="I am BILLING",
        tools=[AITool(name="refund", description="r", parameters={})],
    )
    kit = RoomKit()
    for channel in (voice, triage, billing):
        kit.register_channel(channel)
    ConversationPipeline(
        stages=[
            PipelineStage(phase="triage", agent_id="triage", next="billing"),
            PipelineStage(phase="billing", agent_id="billing", next=None),
        ]
    ).install(kit, [triage, billing], voice_channel_id="voice")
    for room_id in ("room-A", "room-B"):
        room = await kit.create_room(room_id=room_id)
        triage_state = ConversationState(phase="triage", active_agent_id="triage")
        await kit.store.update_room(set_conversation_state(room, triage_state))
        await kit.attach_channel(room_id, "voice")
    in_a = await voice.start_session("room-A", "caller-a", "ws")

    handed = {"target": "billing", "reason": "r", "summary": "s"}
    await provider.simulate_tool_call(in_a, "c1", "handoff_conversation", handed)
    await _until(lambda: len(provider.tool_results) == 1)
    in_b = await voice.start_session("room-B", "caller-b", "ws")

    assert json.loads(provider.tool_results[0][2])["accepted"] is True
    # Room A's session was reconfigured to billing, a new one there starts with it.
    in_a_again = await voice.start_session("room-A", "caller-a2", "ws")
    assert _connected(provider, in_a_again)["system_prompt"].startswith("I am BILLING")

    connected = _connected(provider, in_b)
    assert connected["system_prompt"].startswith("I am TRIAGE")
    assert [t["name"] for t in connected["tools"]] == ["lookup", "handoff_conversation"]
    assert (voice._system_prompt, voice._tools) == ("I am the channel", None)
    await kit.close()


async def test_reconfiguring_a_session_changes_that_session_only() -> None:
    """An application changing one call's instructions leaves the channel's
    configuration, so the next session of another room starts with it (RFC §12.4)."""
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        system_prompt="default",
        tools=[{"name": "lookup", "description": "l", "parameters": {}}],
    )
    kit = RoomKit()
    kit.register_channel(voice)
    for room_id in ("room-A", "room-B"):
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "voice")
    in_a = await voice.start_session("room-A", "caller-a", "ws")

    await voice.reconfigure_session(
        in_a,
        system_prompt="a playful attitude",
        voice="other",
        tools=[{"name": "joke", "description": "j", "parameters": {}}],
    )
    in_b = await voice.start_session("room-B", "caller-b", "ws")

    connected = _connected(provider, in_b)
    assert connected["system_prompt"] == "default"
    assert [t["name"] for t in connected["tools"]] == ["lookup"]
    assert in_a.metadata["system_prompt"] == "a playful attitude"
    await kit.close()


async def test_what_the_room_s_agent_leaves_unset_is_the_channel_s() -> None:
    """A pipeline agent with no voice nor prompt of its own: the session speaks
    with the channel's (RFC §12.4)."""
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        system_prompt="the channel's prompt",
        voice="Kore",
    )
    triage, billing = Agent("triage"), Agent("billing")
    kit = RoomKit()
    for channel in (voice, triage, billing):
        kit.register_channel(channel)
    ConversationPipeline(
        stages=[
            PipelineStage(phase="triage", agent_id="triage", next="billing"),
            PipelineStage(phase="billing", agent_id="billing", next=None),
        ]
    ).install(kit, [triage, billing], voice_channel_id="voice")
    await kit.create_room(room_id="room-A")
    await kit.attach_channel("room-A", "voice")

    session = await voice.start_session("room-A", "caller", "ws")

    connected = _connected(provider, session)
    assert connected["voice"] == "Kore"
    assert connected["system_prompt"] == "the channel's prompt"
    await kit.close()


async def test_a_handoff_to_an_agent_with_many_tools_hides_them_behind_search() -> None:
    """Tool Search is decided on the catalogue the handoff brings, before the
    session is reconfigured with it."""
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel(
        "voice", provider=provider, transport=MockRealtimeTransport(), tool_search_threshold=3
    )
    many = [AITool(name=f"op_{i}", description=f"op {i}", parameters={}) for i in range(10)]
    triage = Agent("triage", system_prompt="I am TRIAGE")
    billing = Agent("billing", system_prompt="I am BILLING", tools=many)
    kit = RoomKit()
    for channel in (voice, triage, billing):
        kit.register_channel(channel)
    ConversationPipeline(
        stages=[
            PipelineStage(phase="triage", agent_id="triage", next="billing"),
            PipelineStage(phase="billing", agent_id="billing", next=None),
        ]
    ).install(kit, [triage, billing], voice_channel_id="voice")
    room = await kit.create_room(room_id="room-A")
    triage_state = ConversationState(phase="triage", active_agent_id="triage")
    await kit.store.update_room(set_conversation_state(room, triage_state))
    await kit.attach_channel("room-A", "voice")
    session = await voice.start_session("room-A", "caller", "ws")
    assert "find_tools" not in [t["name"] for t in _connected(provider, session)["tools"]]

    handed = {"target": "billing", "reason": "r", "summary": "s"}
    await provider.simulate_tool_call(session, "h1", "handoff_conversation", handed)
    await _until(lambda: len(provider.tool_results) == 1)
    await _until(lambda: _connected(provider, session)["system_prompt"].startswith("I am BILLING"))

    declared = [t["name"] for t in _connected(provider, session)["tools"]]
    assert declared[:2] == ["find_tools", "list_tools"]
    assert "handoff_conversation" in declared
    assert len(declared) < 10
    await kit.close()


def _strategies() -> dict[str, Callable[[], tuple[Any, list[Any]]]]:
    """Each strategy with the channels it sets tools or turns up on."""

    def agent(name: str, **kwargs: Any) -> Agent:
        return Agent(name, provider=MockAIProvider(responses=["ok"]), tool_search=False, **kwargs)

    def supervisor(**kwargs: Any) -> tuple[Any, list[Any]]:
        sup = agent("sup")
        return Supervisor(sup, [agent("w1"), agent("w2")], **kwargs), [sup]

    def loop() -> tuple[Any, list[Any]]:
        writer = agent("writer")
        return Loop(agent=writer, reviewer=agent("editor")), [writer]

    def swarm() -> tuple[Any, list[Any]]:
        agents = [agent("a"), agent("b")]
        return Swarm(agents=agents), agents

    def pipeline() -> tuple[Any, list[Any]]:
        agents = [agent("a"), agent("b")]
        return Pipeline(agents=agents), agents

    def voice(orchestration: Callable[[], Any]) -> Callable[[], tuple[Any, list[Any]]]:
        def build() -> tuple[Any, list[Any]]:
            rtv = RealtimeVoiceChannel(
                "voice", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
            )
            return orchestration(), [rtv]

        return build

    return {
        "supervisor-per-worker": lambda: supervisor(),
        "supervisor-team": lambda: supervisor(strategy="parallel"),
        "supervisor-auto": lambda: supervisor(strategy="parallel", auto_delegate=True),
        "loop": loop,
        "swarm": swarm,
        "pipeline": pipeline,
        "voice-supervisor": voice(
            lambda: Supervisor(
                agent("sup"),
                [agent("w1")],
                strategy="parallel",
                auto_delegate=True,
                async_delivery=True,
            )
        ),
        "voice-loop": voice(
            lambda: Loop(agent=agent("writer"), reviewer=agent("editor"), async_delivery=True)
        ),
    }


@pytest.mark.parametrize("kind", list(_strategies()))
async def test_installing_a_strategy_again_in_its_room_replaces_its_own(kind: str) -> None:
    """RFC §21.1: the same strategy installed again for a room replaces what it
    set up there, declaring nothing twice and refusing nothing."""
    orchestration, channels = _strategies()[kind]()
    kit = RoomKit()
    for channel in channels:
        if channel.channel_id not in kit.channels:
            kit.register_channel(channel)
    await kit.create_room(room_id="room-A", orchestration=orchestration)
    before = {c.channel_id: room_tool_names(c, "room-A") for c in channels}

    await orchestration.install(kit, "room-A")

    assert {c.channel_id: room_tool_names(c, "room-A") for c in channels} == before
    await kit.close()


async def test_an_install_refused_halfway_sets_nothing_up() -> None:
    """A per-worker install whose second tool collides with a host tool of the
    supervisor sets up neither: an install is all or nothing."""
    sup = Agent(
        "sup",
        provider=MockAIProvider(responses=["ok"]),
        tools=[AITool(name="delegate_to_w2", description="host", parameters={})],
    )
    workers = [Agent(w, provider=MockAIProvider(responses=["ok"])) for w in ("w1", "w2")]
    kit = RoomKit()
    kit.register_channel(sup)
    await kit.create_room(room_id="room-A")

    with pytest.raises(ToolNameCollisionError, match="delegate_to_w2"):
        await Supervisor(sup, workers).install(kit, "room-A")

    assert room_tool_names(sup, "room-A") == []
    await kit.close()


async def test_a_structured_delegation_to_an_agent_with_its_own_result_tool_fails_naming_it() -> (
    None
):
    """The result tool is set up for the child room when the delegation runs: an
    agent whose host tool carries its name fails the delegation, and the task's
    error names the tool to rename (RFC §21.1)."""
    worker = Agent(
        "worker",
        provider=MockAIProvider(responses=["x"]),
        tools=[AITool(name="submit_result", description="host", parameters={})],
        tool_handler=AsyncMock(return_value="host"),
    )
    kit = RoomKit()
    kit.register_channel(worker)
    await kit.create_room(room_id="parent")

    task = await kit.delegate(
        "parent", "worker", "do it", wait=True, require_structured_result=True
    )

    assert task.result is not None
    assert task.result.status == "failed"
    assert "'submit_result'" in (task.result.error or "")
    await kit.close()
