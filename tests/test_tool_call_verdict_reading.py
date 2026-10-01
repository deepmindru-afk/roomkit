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
    HookResult,
    HookTrigger,
    RoomKit,
    ToolCallEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
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


async def test_a_hook_engine_fold_keeps_its_two_argument_signature() -> None:
    """``run_sync_hooks``' public ``fold`` is ``fold(event, metadata)``, run
    after a ``modify`` too, its metadata then empty."""
    kit = RoomKit()
    await kit.create_room(room_id="r1")

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="rewrites")
    async def rewrites(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.modify(replace(event, result="rewritten"))

    seen: list[tuple[Any, dict[str, Any]]] = []

    def fold(event: Any, metadata: dict[str, Any]) -> Any:
        seen.append((event.result, metadata))
        return event

    call = ToolCallEvent(
        channel_id="ai1",
        channel_type=ChannelType.AI,
        tool_call_id="c1",
        name="lookup",
        arguments={},
        result="original",
        room_id="r1",
    )
    context = await kit._build_context("r1")  # noqa: SLF001
    await kit.hook_engine.run_sync_hooks(
        "r1", HookTrigger.ON_TOOL_CALL, call, context, skip_event_filter=True, fold=fold
    )

    assert seen == [("rewritten", {})]
    await kit.close()


async def test_a_blocked_call_is_remembered_with_its_reason_not_its_eviction() -> None:
    provider = MockAIProvider(ai_responses=_calls(("lookup", {})))
    kit = RoomKit()
    channel = AIChannel("ai1", provider=provider, tool_handler=_answer, evict_threshold_tokens=20)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="block")
    async def block(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("withheld: " + "policy says no. " * 20)

    await _turn(channel, [LOOKUP])

    digest = channel._tool_usage.render_digest("r1")  # noqa: SLF001
    assert "policy says no" in digest
    assert "too large" not in digest
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


async def _context_fails(kit: RoomKit, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Break the hooks' context; return the ``tool_call`` framework events."""
    reported: list[Any] = []

    @kit.on("tool_call")
    async def on_report(event: Any) -> None:
        reported.append(event.data)

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="allows")
    async def allows(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.allow()

    async def broken(*args: Any, **kwargs: Any) -> Room:
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(kit, "_build_context", broken)
    return reported


async def test_a_text_call_keeps_its_result_when_the_hook_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockAIProvider(ai_responses=_calls(("lookup", {})))
    kit = RoomKit()
    channel = AIChannel("ai1", provider=provider, tool_handler=_answer)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    reported = await _context_fails(kit, monkeypatch)

    await _turn(channel, [LOOKUP])

    assert _tool_payload(provider.calls[1], "lookup") == {"found": "secret"}
    assert [r["tool_name"] for r in reported] == ["lookup"]
    await kit.close()


async def test_a_realtime_call_keeps_its_result_when_the_hook_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit, provider, session = await _realtime_kit()
    reported = await _context_fails(kit, monkeypatch)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results) and bool(reported))

    assert json.loads(provider.tool_results[0][2]) == {"found": "secret"}
    assert [r["tool_name"] for r in reported] == ["lookup"]
    await kit.close()


async def test_a_conference_call_keeps_its_result_when_the_hook_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockRealtimeProvider()
    kit, session = await _conference(provider)
    reported = await _context_fails(kit, monkeypatch)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results) and bool(reported))

    assert json.loads(provider.tool_results[0][2]) == {"found": "secret"}
    assert [r["tool_name"] for r in reported] == ["lookup"]
    await kit.close()
