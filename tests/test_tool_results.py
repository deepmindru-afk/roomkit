"""What a tool answers, and the outcomes the channel decides (RFC §9.3, §21.4; RMK-278).

A handler's return value outside the contract (a mapping, a list of values, a
number, ``None``) reaches the model as JSON, on both tool loops and in realtime,
without failing the turn; so does a hook's replacement. An unknown, unavailable
or repeated call is a refusal carrying the failure marker; a declared tool no
handler serves goes to ON_TOOL_CALL's hooks with no result, and fails once if
none serves it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomContext, RoomKit, ToolCallEvent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, AIToolResultPart
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.compose import extract_tools
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.video.vision.screen_tool import DescribeScreenTool
from roomkit.video.vision.webcam_tool import ListWebcamsTool
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conftest import make_event
from tests.tool_loop_modes import respond

SCHEMA = {"type": "object", "properties": {}}
T = AITool(name="t", description="a tool", parameters=SCHEMA)
VALUES = [["a", "b"], [{"id": 1}], {"ok": True, "n": None}, None, 3]


class _Round:
    """One tool round on an AIChannel attached to a kit, with its observers."""

    def __init__(self, *calls: AIToolCall, streaming: bool, vision: bool = False) -> None:
        self.provider = MockAIProvider(
            vision=vision,
            streaming=streaming,
            ai_responses=[
                AIResponse(content="", finish_reason="tool_calls", tool_calls=list(calls)),
                AIResponse(content="done"),
            ],
        )
        self.kit = RoomKit()
        self.observed: list[ToolCallEvent] = []
        self.hook_saw: list[Any] = []

    def channel(self, **kwargs: Any) -> AIChannel:
        channel = AIChannel("ai-1", provider=self.provider, **kwargs)
        self.kit.register_channel(channel)

        @self.kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="spy")
        async def spy(event: ToolCallEvent, ctx: RoomContext) -> None:
            self.observed.append(event)

        return channel

    async def run(self, channel: AIChannel) -> list[AIToolResultPart]:
        room = await self.kit.create_room()
        await self.kit.attach_channel(room.id, "ai-1")
        binding = ChannelBinding(channel_id="ai-1", room_id=room.id, channel_type=ChannelType.AI)
        await respond(
            channel,
            make_event(room_id=room.id, body="go", channel_id="sms-1"),
            binding,
            await self.kit._build_context(room.id),
        )
        return [
            part
            for message in self.provider.calls[-1].messages
            if isinstance(message.content, list)
            for part in message.content
            if isinstance(part, AIToolResultPart)
        ]


def _call(call_id: str = "c1", name: str = "t", **arguments: Any) -> AIToolCall:
    return AIToolCall(id=call_id, name=name, arguments=arguments)


@pytest.mark.parametrize("vision", [True, False], ids=["vision", "text-only"])
@pytest.mark.parametrize("value", VALUES, ids=["list-str", "list-dict", "dict", "none", "int"])
async def test_a_return_value_outside_the_contract_reaches_the_model_as_json(
    value: Any, vision: bool, streaming: bool
) -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> Any:
        return value

    round_ = _Round(_call(), streaming=streaming, vision=vision)
    parts = await round_.run(round_.channel(tools=[T], tool_handler=handler))

    assert [(p.result, p.is_error) for p in parts] == [(json.dumps(value), False)]


@pytest.mark.parametrize("value", VALUES, ids=["list-str", "list-dict", "dict", "none", "int"])
async def test_a_realtime_call_reads_the_same_json(value: Any) -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> Any:
        return value

    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[{"name": "t", "description": "a tool", "parameters": SCHEMA}],
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(voice)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt")
    session = await voice.start_session(room.id, "u1", "ws")

    await provider.simulate_tool_call(session, "c1", "t", {})
    await asyncio.sleep(0.05)

    assert provider.tool_results[0][2] == json.dumps(value)
    await kit.close()


async def test_a_hooks_replacement_outside_the_contract_does_not_fail_the_turn(
    streaming: bool,
) -> None:
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    round_ = _Round(_call("c1"), _call("c2", "u"), streaming=streaming, vision=True)
    u = AITool(name="u", description="another tool", parameters=SCHEMA)
    channel = round_.channel(tools=[T, u], tool_handler=handler)

    @round_.kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="rewrite")
    async def rewrite(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        if event.name == "t":
            return HookResult(action="allow", metadata={"result": {"temp": 99}})
        return HookResult.allow()

    parts = await round_.run(channel)

    assert {p.tool_call_id: p.result for p in parts} == {"c1": '{"temp": 99}', "c2": "ok"}
    assert ran == ["t", "u"]


async def test_a_declared_tool_nothing_serves_fails_once(streaming: bool) -> None:
    round_ = _Round(_call(), streaming=streaming)
    channel = round_.channel(tools=[T])

    @round_.kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="look")
    async def look(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        round_.hook_saw.append(event.result)
        return HookResult.allow()

    parts = await round_.run(channel)

    assert round_.hook_saw == [None]  # the hooks' chance to serve it
    assert [(json.loads(p.result), p.is_error) for p in parts] == [
        ({"error": "No handler for tool t"}, True)
    ]
    assert [e.is_error for e in round_.observed] == [True]


async def test_a_hook_serves_a_declared_tool_nothing_else_serves(streaming: bool) -> None:
    round_ = _Round(_call(), streaming=streaming)
    channel = round_.channel(tools=[T])

    @round_.kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
    async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        round_.hook_saw.append(event.result)
        if event.result is None:
            return HookResult(action="allow", metadata={"result": {"temp": 22}})
        return HookResult.allow()

    parts = await round_.run(channel)

    assert round_.hook_saw == [None]
    assert [(p.result, p.is_error) for p in parts] == [('{"temp": 22}', False)]
    assert [(e.result, e.is_error) for e in round_.observed] == [({"temp": 22}, False)]


async def test_the_channels_own_outcomes_are_refusals_no_sync_hook_serves(
    streaming: bool,
) -> None:
    """A repeat the guard stops, and a tool outside the turn's toolset."""

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    same = [_call(f"c{i}", x=1) for i in range(3)]
    round_ = _Round(*same, _call("c3", "ghost"), streaming=streaming)
    channel = round_.channel(tools=[T], tool_handler=handler)
    # "ghost" gets past the declared check, as a recovered deferred tool does,
    # but is outside the turn's toolset: the dispatcher refuses it.
    channel._recover_deferred_tool = lambda name: T if name == "ghost" else None  # ty: ignore

    @round_.kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="look")
    async def look(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        round_.hook_saw.append(event.tool_call_id)
        return HookResult.allow()

    parts = {p.tool_call_id: p for p in await round_.run(channel)}

    assert [parts[c].is_error for c in ("c0", "c1", "c2", "c3")] == [False, False, True, True]
    assert "already called" in parts["c2"].result
    assert "not available in the current turn" in parts["c3"].result
    assert sorted(round_.hook_saw) == ["c0", "c1"]


async def test_composed_vision_tools_serve_each_their_own_call() -> None:
    """``compose_tool_handlers`` falls through on the JSON unknown-tool
    envelope: the second built-in tool is reachable."""
    definitions, handler = extract_tools(
        [DescribeScreenTool(MockVisionProvider()), ListWebcamsTool()]
    )

    answer = await handler(definitions[1].name, {})

    assert "Unknown tool" not in str(answer)
