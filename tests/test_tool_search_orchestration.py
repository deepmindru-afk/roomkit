"""The tools orchestration injects stay declared under Tool Search (RFC §21.1; RMK-277).

Tool Search hides a large catalogue behind ``find_tools``. A handoff, a
delegation, a delegation's result tool or a strategy's tool is one the agent is
told to call: it stays declared as a pinned tool does, and it does not count
toward the catalogue whose size decides the collapse.
"""

from __future__ import annotations

from typing import Any

from roomkit import RoomKit
from roomkit.channels._tool_search import estimate_tool_tokens
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.mixins._result_capture import capture_result
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.orchestration.handoff import setup_handoff
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.result import SUBMIT_RESULT
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delegate import DelegateHandler, setup_delegation, setup_realtime_delegation
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conftest import make_event
from tests.tool_loop_modes import respond

ORCHESTRATION = {"handoff_conversation", "delegate_task", "submit_result"}
CRM = [
    AITool(
        name=f"crm_action_{i}",
        description=f"Perform CRM operation number {i} on a customer record, "
        "with filters, pagination and audit metadata.",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
    )
    for i in range(30)
]


class _Unused:
    async def handle(self, **kwargs: Any) -> Any:
        raise AssertionError("no call is made")


async def _crm(name: str, arguments: dict[str, Any]) -> str:
    return "ok"


def _orchestrated(streaming: bool, tools: list[AITool], **kwargs: Any) -> AIChannel:
    """A channel with a handoff and a delegation wired on it."""
    provider = MockAIProvider(responses=["hi"], streaming=streaming)
    channel = AIChannel("ai1", provider=provider, tools=tools, tool_handler=_crm, **kwargs)
    setup_handoff(channel, _Unused())
    setup_delegation(channel, _Unused())
    return channel


async def _declared_in(channel: AIChannel, room_id: str) -> set[str]:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    provider = channel.provider
    assert isinstance(provider, MockAIProvider)
    before = len(provider.calls)
    await respond(
        channel,
        make_event(room_id=room_id, body="go", channel_id="sms1"),
        binding,
        RoomContext(room=Room(id=room_id)),
    )
    return {tool.name for tool in provider.calls[before].tools or []}


async def test_the_tools_orchestration_injects_stay_declared(streaming: bool) -> None:
    channel = _orchestrated(streaming, CRM)

    with capture_result(channel, "r1", SUBMIT_RESULT):
        declared = await _declared_in(channel, "r1")

    assert declared >= ORCHESTRATION
    assert "find_tools" in declared  # Tool Search is on
    assert "crm_action_0" not in declared


async def test_a_result_tool_stays_in_its_child_room(streaming: bool) -> None:
    channel = _orchestrated(streaming, CRM)

    with capture_result(channel, "parent::task-1", SUBMIT_RESULT):
        elsewhere = await _declared_in(channel, "r1")
        child = await _declared_in(channel, "parent::task-1")

    assert "submit_result" not in elsewhere
    assert "submit_result" in child


async def test_orchestration_tools_do_not_tip_a_catalogue_into_tool_search(
    streaming: bool,
) -> None:
    host = CRM[:5]
    # A budget the host tools fit under, and that the orchestration tools
    # would overflow if they counted.
    budget_pct = (sum(estimate_tool_tokens(t) for t in host) + 1) * 100 / 8192
    channel = _orchestrated(streaming, host, tool_search_threshold_pct=budget_pct)

    declared = await _declared_in(channel, "r1")

    assert "find_tools" not in declared
    assert {"crm_action_0", "handoff_conversation", "delegate_task"} <= declared


def _voice(provider: MockRealtimeProvider) -> RealtimeVoiceChannel:
    """A voice channel whose catalogue is behind Tool Search."""
    tools = [
        {"name": t.name, "description": t.description, "parameters": t.parameters} for t in CRM
    ]
    return RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools,
        tool_handler=_crm,
        tool_search=True,
    )


def _connected(provider: MockRealtimeProvider) -> set[str]:
    connected = [c.args for c in provider.calls if c.method == "connect"][-1]
    return {tool["name"] for tool in connected["tools"] or []}


async def test_a_voice_supervisors_tool_stays_declared() -> None:
    provider = MockRealtimeProvider()
    voice = _voice(provider)
    kit = RoomKit()
    kit.register_channel(voice)
    supervisor = Supervisor(
        Agent("sup", provider=MockAIProvider(responses=["ok"])),
        [Agent("worker", provider=MockAIProvider(responses=["done"]))],
        strategy="sequential",
        auto_delegate=True,
        async_delivery=True,
    )
    await kit.create_room(room_id="r1", orchestration=supervisor)
    await kit.attach_channel("r1", "voice")

    await voice.start_session("r1", "caller", "ws")

    declared = _connected(provider)
    assert "delegate_workers" in declared
    assert "crm_action_0" not in declared
    await kit.close()


async def test_a_realtime_delegation_and_a_voice_pipelines_handoff_stay_declared() -> None:
    provider = MockRealtimeProvider()
    voice = _voice(provider)
    kit = RoomKit()
    triage, billing = (Agent(a, provider=MockAIProvider()) for a in ("triage", "billing"))
    for channel in (voice, triage, billing):
        kit.register_channel(channel)
    setup_realtime_delegation(voice, DelegateHandler(kit))
    ConversationPipeline(
        stages=[
            PipelineStage(phase="triage", agent_id="triage", next="billing"),
            PipelineStage(phase="billing", agent_id="billing", next=None),
        ]
    ).install(kit, [triage, billing], voice_channel_id="voice")
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice")

    await voice.start_session("r1", "caller", "ws")

    declared = _connected(provider)
    assert {"delegate_task", "handoff_conversation"} <= declared
    assert "crm_action_0" not in declared
    await kit.close()
