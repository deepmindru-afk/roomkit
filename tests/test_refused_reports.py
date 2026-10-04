"""A refused call reaches ON_TOOL_CALL's observers only, marked ``refused``,
on every door (RMK-432, RFC §9.3).

A gate's refusal, an external handler's refusal (through its
``on_tool_refused``), a call the external door refused itself and a rejected
ACP permission never reach a SYNC hook. A handler that raised is a failure
the channel reports with what failed. An ACP call reports the same body with
and without an external handler.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import acp
import pytest
from acp import PromptResponse

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import PolicyExternalToolHandler, ToolDecision
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_channels.test_acp import _channel
from tests.test_framework import SimpleChannel

SECRET = "postgres://admin:hunter2@db"


class _Denying(PolicyExternalToolHandler):
    async def process_tool_call(self, tool_name: str, tool_input: Any, **kw: Any) -> ToolDecision:
        return ToolDecision(approved=False, reason="not on this host")


class _Raising(PolicyExternalToolHandler):
    async def process_tool_call(self, tool_name: str, tool_input: Any, **kw: Any) -> ToolDecision:
        raise RuntimeError(SECRET)


class _Heard:
    """What the SYNC hooks and the ASYNC observers of ON_TOOL_CALL received."""

    def __init__(self, kit: RoomKit) -> None:
        self.sync: list[ToolCallEvent] = []
        self.observed: list[ToolCallEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: Any) -> HookResult:
            self.sync.append(event)
            return HookResult.allow()

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: ToolCallEvent, ctx: Any) -> None:
            self.observed.append(event)


async def _room(kit: RoomKit, agent_id: str) -> None:
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "sms")
    await kit.attach_channel("room-1", agent_id, category=ChannelCategory.INTELLIGENCE)


async def _ask(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.2)


async def _external_door(handler: Any, call: AIToolCall) -> _Heard:
    kit = RoomKit()
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ]
    )
    kit.register_channel(AIChannel("ai1", provider=provider, external_tool_handler=handler))
    heard = _Heard(kit)
    await _room(kit, "ai1")
    await _ask(kit)
    await kit.close()
    return heard


BASH = AIToolCall(id="p1", name="Bash", arguments={"cmd": "ls"})
CUT = AIToolCall(id="p1", name="Bash", arguments={"raw": "[1"}, partial=True)


@pytest.mark.parametrize(
    ("handler", "call"),
    [(_Denying(), BASH), (PolicyExternalToolHandler(), CUT)],
    ids=["handler-refuses", "channel-refuses-a-cut-call"],
)
async def test_an_external_door_refusal_reaches_the_observers_only(
    handler: Any, call: AIToolCall
) -> None:
    heard = await _external_door(handler, call)

    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused, event.cancelled) == (True, True, False)


async def test_an_external_handler_that_raises_is_a_failure_with_its_detail() -> None:
    heard = await _external_door(_Raising(), BASH)

    [event] = [*heard.sync, *heard.observed][:1]
    assert (event.is_error, event.refused) == (True, False)
    assert event.error_detail == f"RuntimeError: {SECRET}"
    assert SECRET not in str(event.result)


async def _acp(handler: Any, *, status: str, raw_output: Any) -> _Heard:
    async def prompt(connection: Any, session_id: str, *a: Any, **k: Any) -> PromptResponse:
        await connection.client.session_update(
            session_id,
            acp.start_tool_call("tool-1", "Read file", kind="read", status="in_progress"),
        )
        await connection.client.session_update(
            session_id, acp.update_tool_call("tool-1", status=status, raw_output=raw_output)
        )
        return PromptResponse(stop_reason="end_turn")

    with tempfile.TemporaryDirectory() as tmp:
        kit = RoomKit()
        channel, connection, _ = _channel(Path(tmp), handler=handler, emit_updates=False)
        connection.prompt = lambda *a, **k: prompt(connection, *a, **k)  # type: ignore[method-assign]
        kit.register_channel(channel)
        heard = _Heard(kit)
        await _room(kit, "acp-agent")
        await _ask(kit)
        await kit.close()
        return heard


async def _acp_permission(handler: Any) -> _Heard:
    with tempfile.TemporaryDirectory() as tmp:
        kit = RoomKit()
        channel, connection, _ = _channel(Path(tmp), handler=handler)
        connection.tool_status = "failed"
        connection.tool_raw_output = {"error": "permission rejected"}
        kit.register_channel(channel)
        heard = _Heard(kit)
        await _room(kit, "acp-agent")
        await _ask(kit)
        await kit.close()
        return heard


@pytest.mark.parametrize("handler", [None, _Denying()], ids=["no-handler", "denying-handler"])
async def test_a_rejected_acp_permission_reaches_the_observers_only(handler: Any) -> None:
    heard = await _acp_permission(handler)

    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused) == (True, True)


async def test_an_acp_handler_that_raises_is_a_failure_with_its_detail() -> None:
    heard = await _acp_permission(_Raising())

    [event] = [*heard.sync, *heard.observed][:1]
    assert (event.is_error, event.refused) == (True, False)
    assert event.error_detail == f"RuntimeError: {SECRET}"


@pytest.mark.parametrize(
    "raw_output",
    [[{"type": "image", "mimeType": "image/png", "data": "A" * (600 * 1024)}], None],
    ids=["image", "nothing"],
)
async def test_an_acp_failure_reports_one_body_with_or_without_a_handler(raw_output: Any) -> None:
    bodies = []
    for handler in (None, PolicyExternalToolHandler()):
        heard = await _acp(handler, status="failed", raw_output=raw_output)
        [event] = [*heard.sync, *heard.observed][:1]
        bodies.append(event.result)

    assert bodies[0] == bodies[1]
    assert len(str(bodies[0])) < 1000


async def test_a_gate_refusal_is_marked_refused_on_the_local_door() -> None:
    kit = RoomKit()
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[BASH]),
            AIResponse(content="done"),
        ]
    )
    kit.register_channel(AIChannel("ai1", provider=provider, tool_handler=lambda *a: "ran"))
    heard = _Heard(kit)
    await _room(kit, "ai1")
    await _ask(kit)
    await kit.close()

    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused, event.cancelled) == (True, True, False)


async def test_a_gate_refusal_is_marked_refused_on_a_realtime_session() -> None:
    kit = RoomKit()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[{"name": "lookup", "description": "lookup", "parameters": {"type": "object"}}],
        tool_handler=lambda *a: "ran",
    )
    kit.register_channel(channel)
    heard = _Heard(kit)
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "rt")
    session = await channel.start_session("room-1", "u", "ws")

    await provider.simulate_tool_call(session, "c1", "nope", {})
    for _ in range(100):
        if heard.observed:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused) == (True, True)
