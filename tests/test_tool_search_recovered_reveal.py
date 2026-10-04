"""A hidden tool called by its exact name stays revealed only once the tool
answered (RMK-461, RFC §6.4).

Tool Search recovers an exact-name call to a catalogue tool it hides: the
name joins the turn's reveal window while the call runs. The reveal then
follows the room's tool memory, which re-reveals every tool used: a call the
tool answered (served, failed, withheld by an ON_TOOL_CALL hook) keeps it
revealed; a call refused before it ran (BEFORE_TOOL_USE, its handler's
refusal) reveals nothing, for the turn's next rounds or for later turns.
Before, the recovery recorded the reveal for the session before any gate ran.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import ToolFailedError, ToolRefusedError
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_SMS_TOOL = {
    "name": "send_sms",
    "description": "Send an SMS text message to a phone number.",
    "parameters": {
        "type": "object",
        "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
        "required": ["to", "body"],
    },
}
_CATALOGUE = [
    *({"name": f"widget_{i}", "description": f"Operate widget number {i}."} for i in range(5)),
    _SMS_TOOL,
]


def _provider(streaming: bool) -> MockAIProvider:
    """Calls send_sms directly, without find_tools, then stops."""
    call = AIToolCall(id="t1", name="send_sms", arguments={"to": "+15551234567", "body": "hi"})
    return MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done", finish_reason="stop"),
        ],
        streaming=streaming,
    )


async def _turn(provider: MockAIProvider, handler: Any, hooks: Any = None) -> AIChannel:
    channel = AIChannel("ai1", provider=provider, tool_search=True, tool_handler=handler)
    kit = RoomKit()
    kit.register_channel(channel)
    if hooks is not None:
        hooks(kit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1")
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": _CATALOGUE},
    )
    event = make_event(room_id="r1", body="go", channel_id="sms1")
    await respond(channel, event, binding, await kit._build_context("r1"))
    await kit.close()
    return channel


async def _served(name: str, arguments: dict[str, Any]) -> str:
    return '{"sent": true}'


async def _refusing(name: str, arguments: dict[str, Any]) -> str:
    raise ToolRefusedError("Not to that number.")


async def _failing(name: str, arguments: dict[str, Any]) -> str:
    raise ToolFailedError("The SMS gateway is down.")


def _deny_before_use(kit: RoomKit) -> None:
    @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
    async def _deny(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("not now")


def _withhold_result(kit: RoomKit) -> None:
    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="withhold")
    async def _withhold(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("withheld")


def _revealed(provider: MockAIProvider, channel: AIChannel) -> tuple[bool, bool]:
    """Whether send_sms is declared on the turn's next round, and kept for
    the room's later turns."""
    next_round = {tool.name for tool in provider.calls[1].tools}
    return "send_sms" in next_round, "send_sms" in channel._tool_usage.tool_names("r1")


@pytest.mark.parametrize(
    ("handler", "hooks"),
    [(_refusing, None), (_served, _deny_before_use)],
    ids=["handler-refusal", "before-tool-use-block"],
)
async def test_a_call_refused_before_it_ran_reveals_nothing(
    streaming: bool, handler: Any, hooks: Any
) -> None:
    provider = _provider(streaming)
    channel = await _turn(provider, handler, hooks)

    assert _revealed(provider, channel) == (False, False)


@pytest.mark.parametrize(
    ("handler", "hooks"),
    [(_served, None), (_failing, None), (_served, _withhold_result)],
    ids=["served", "failed", "on-tool-call-block"],
)
async def test_a_call_the_tool_answered_keeps_it_revealed(
    streaming: bool, handler: Any, hooks: Any
) -> None:
    provider = _provider(streaming)
    channel = await _turn(provider, handler, hooks)

    assert _revealed(provider, channel) == (True, True)
