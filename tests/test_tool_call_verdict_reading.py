"""ON_TOOL_CALL's verdict reads the same on every channel (RFC §9.3, RMK-305).

One reader applies a verdict for the text loops, a realtime session and a
conference: a block withholds the result, a hook's result replaces it, and a
call nothing served failed. A hook that empties a served result leaves the
call served: the next hook does not get to serve it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    ToolCallEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.context import RoomContext
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_tool_policy_exemptions import _calls, _tool_payload, _turn

LOOKUP = {
    "name": "lookup",
    "description": "Looks something up",
    "parameters": {"type": "object", "properties": {}},
}


async def _answer(name: str, arguments: dict[str, Any]) -> str:
    return '{"found": "secret"}'


async def _conference_answer(room_id: str, name: str, arguments: dict[str, Any]) -> str:
    return await _answer(name, arguments)


def _empties_then_serves(kit: RoomKit, *, by_modify: bool) -> None:
    """Hook A empties the served result; hook B serves any call it reads as unserved."""

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="a-empties", priority=0)
    async def empties(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        if by_modify:
            return HookResult.modify(replace(event, result=None))
        return HookResult(action="allow", metadata={"result": None})

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="b-serves", priority=1)
    async def serves(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        if event.result is None:
            return HookResult.modify(replace(event, result="B-SERVED"))
        return HookResult.allow()


# -- An emptied result stays served -----------------------------------------


@pytest.mark.parametrize("by_modify", [False, True], ids=["metadata", "modify"])
async def test_an_emptied_text_result_is_not_served_again(
    streaming: bool, by_modify: bool
) -> None:
    provider = MockAIProvider(ai_responses=_calls(("lookup", {})), streaming=streaming)
    kit = RoomKit()
    channel = AIChannel("ai1", provider=provider, tool_handler=_answer)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    _empties_then_serves(kit, by_modify=by_modify)

    await _turn(channel, [LOOKUP])

    assert _tool_payload(provider.calls[1], "lookup") is None  # null, not B's
    await kit.close()


async def _realtime_kit(**kwargs: Any) -> tuple[RoomKit, MockRealtimeProvider, Any]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[LOOKUP],
        tool_handler=_answer,
        **kwargs,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    return kit, provider, await channel.start_session("r1", "u1", "ws")


@pytest.mark.parametrize("by_modify", [False, True], ids=["metadata", "modify"])
async def test_an_emptied_realtime_result_is_not_served_again(by_modify: bool) -> None:
    kit, provider, session = await _realtime_kit()
    _empties_then_serves(kit, by_modify=by_modify)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results))

    assert provider.tool_results[0][2] == "null"
    await kit.close()


async def _conference(provider: MockRealtimeProvider) -> tuple[RoomKit, Any]:
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider, tools=[LOOKUP], tool_handler=_conference_answer
        ),
    )
    session = await channel._realtime.ensure_session(ROOM)  # noqa: SLF001
    assert session is not None
    return kit, session


@pytest.mark.parametrize("by_modify", [False, True], ids=["metadata", "modify"])
async def test_an_emptied_conference_result_is_not_served_again(by_modify: bool) -> None:
    provider = MockRealtimeProvider()
    kit, session = await _conference(provider)
    _empties_then_serves(kit, by_modify=by_modify)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results))

    assert provider.tool_results[0][2] == "null"
    await kit.close()


# -- A conference reads a verdict as every channel does -----------------------


async def test_a_conference_withholds_a_blocked_result() -> None:
    provider = MockRealtimeProvider()
    kit, session = await _conference(provider)

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="redact")
    async def redact(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("contains a secret")

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results))

    body = provider.tool_results[0][2]
    assert "secret" not in body.replace("contains a secret", "")
    assert json.loads(body)["error"] == "contains a secret"
    await kit.close()


# -- Hooks that cannot run leave the outcome as it was ------------------------


async def test_a_realtime_call_keeps_its_result_when_the_hook_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit, provider, session = await _realtime_kit()
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="observe")
    async def observe(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    async def broken(*args: Any, **kwargs: Any) -> Room:
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(kit, "_build_context", broken)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results))

    assert json.loads(provider.tool_results[0][2]) == {"found": "secret"}
    await kit.close()
