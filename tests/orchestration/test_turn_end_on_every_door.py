"""How a turn ended reaches its caller the same on every door (RMK-479,
RFC §6.4).

A supervisor's task-formulation pass is read as a room turn is: whatever
ended it (its round cap, an error, a stop, an empty answer), the caller finds
its end under ``turns``, and only there. ``regenerate_response`` reads a
buffered reply as ``process_inbound`` does, and a ``turns`` key a hook wrote
never reaches it.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.models.steering import Cancel
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

_LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})


def _round() -> AIResponse:
    return AIResponse(
        content="Still checking.",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
        usage={"input_tokens": 10, "output_tokens": 2},
    )


class _Loops(MockAIProvider):
    """A tool round on every generation: the round cap cuts it."""

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        return _round()


class _FailsAfterARound(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) % 2:
            return _round()
        raise ProviderError("upstream 400", provider="mock", status_code=400)


class _SaysNothing(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        return AIResponse(content="", usage={"input_tokens": 3, "output_tokens": 0})


ENDINGS = {
    "max_rounds": (_Loops, False),
    "error": (_FailsAfterARound, False),
    "cancelled": (_Loops, True),
    "completed-empty": (_SaysNothing, False),
}


async def _kit(provider: MockAIProvider, *, orchestrated: bool, cancels: bool = False) -> RoomKit:
    """A room whose ``sup`` answers in a room turn, or as a supervisor's
    task-formulation pass."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        if cancels:
            sup.steer(Cancel(), room_id="r")
        return "found"

    sup = Agent(
        "sup",
        provider=provider,
        tools=[_LOOKUP],
        tool_handler=lookup,
        tool_search=False,
        max_tool_rounds=1,
    )
    kit.register_channel(sup)
    worker = Agent("worker", provider=MockAIProvider(responses=["worker answer"]))
    kit.register_channel(worker)
    strategy = (
        Supervisor(sup, [worker], strategy="parallel", auto_delegate=True, refine_task=True)
        if orchestrated
        else None
    )
    await kit.create_room(room_id="r", orchestration=strategy)
    await kit.attach_channel("r", "sms")
    if not orchestrated:
        await kit.attach_channel("r", "sup", category=ChannelCategory.INTELLIGENCE)
    return kit


_TURN_KEYS = ("turns", "loop_end_reason", "ai_usage")


def _turn_record(metadata: Any) -> dict[str, Any]:
    """What the caller reads of how the turn ended: its ``turns``, and any
    turn-record key left beside them."""
    return {key: value for key, value in dict(metadata).items() if key in _TURN_KEYS}


async def _turns(kit: RoomKit) -> Any:
    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Find it."))
    )
    return _turn_record(result.response_metadata).get("turns")


async def _caller_record(kit: RoomKit) -> dict[str, Any]:
    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Find it."))
    )
    return _turn_record(result.response_metadata)


@pytest.mark.parametrize("ending", list(ENDINGS))
async def test_a_task_formulation_pass_ends_as_a_room_turn_does(ending: str) -> None:
    kind, cancels = ENDINGS[ending]
    room_kit = await _kit(kind(streaming=True), orchestrated=False, cancels=cancels)
    room_record = await _caller_record(room_kit)
    await room_kit.close()
    pass_kit = await _kit(kind(streaming=True), orchestrated=True, cancels=cancels)
    pass_record = await _caller_record(pass_kit)
    await pass_kit.close()

    assert list(room_record) == ["turns"]
    assert pass_record == room_record


@pytest.mark.parametrize("orchestrated", [True, False], ids=["pass-1-cut", "room-turn-cut"])
async def test_a_regenerated_reply_ends_as_its_inbound_did(orchestrated: bool) -> None:
    kit = await _kit(_Loops(streaming=True), orchestrated=orchestrated)
    inbound = await _turns(kit)
    again = await kit.regenerate_response("r")
    await kit.close()

    assert dict(again.response_metadata).get("turns") == inbound


@pytest.mark.parametrize("orchestrated", [True, False], ids=["pass-1-cut", "room-turn-cut"])
async def test_a_turns_key_a_hook_wrote_never_reaches_a_regeneration(orchestrated: bool) -> None:
    kit = await _kit(_Loops(streaming=True), orchestrated=orchestrated)

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def forge(event: Any, ctx: Any) -> HookResult:
        event.ai_context.response_metadata["turns"] = {"sup": {"loop_end_reason": "forged"}}
        return HookResult.allow()

    inbound = await _turns(kit)
    again = await kit.regenerate_response("r")
    await kit.close()

    assert inbound["sup"]["loop_end_reason"] == "max_rounds"
    assert dict(again.response_metadata).get("turns") == inbound
