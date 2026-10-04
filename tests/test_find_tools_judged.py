"""A ``find_tools`` reveal counts once its call is served, on every door
(RMK-447, RFC §6.4, §9.3).

An ON_TOOL_CALL hook judges a Tool Search call before the model reads its
result, as any other call: a block reveals nothing (not this turn, not the
next), a replacement is what the model reads, and a search that finds nothing
keeps the reveal window as it was. A realtime session declares the matches
only after the call's result went out.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, ToolCallEvent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conftest import make_event
from tests.test_toolset_edges import (
    SPOTIFY,
    _call,
    _calling,
    _declared,
    _FixedProvider,
    _Recorder,
    _schema,
    _session,
    _tool,
)
from tests.tool_loop_modes import respond

FOUND = {"spotify_play", "spotify_search"}


def _judge_search(kit: RoomKit, verdict: str) -> list[ToolCallEvent]:
    """Block or replace every Tool Search call; the observers' reports."""

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="judge")
    async def judge(event: ToolCallEvent, ctx: Any) -> HookResult:
        if event.name not in ("find_tools", "list_tools"):
            return HookResult.allow()
        if verdict == "block":
            return HookResult.block("search is not allowed here")
        return HookResult.modify(replace(event, result='{"matches": []}'))

    seen: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        seen.append(event)

    return seen


async def _text_kit(responses: list[AIResponse]) -> tuple[RoomKit, AIChannel, MockAIProvider]:
    provider = MockAIProvider(ai_responses=responses)
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[_tool(n) for n in SPOTIFY],
        tool_handler=_Recorder(),
        tool_search=True,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1")
    return kit, channel, provider


async def _text_turn(kit: RoomKit, channel: AIChannel) -> None:
    binding = ChannelBinding(channel_id="ai1", room_id="r1", channel_type=ChannelType.AI)
    event = make_event(room_id="r1", body="go", channel_id="sms1")
    await respond(channel, event, binding, await kit._build_context("r1"))


def _round_declares(provider: MockAIProvider, index: int) -> set[str]:
    return {t.name for t in provider.calls[index].tools or [] if not t.defer_loading}


class TestText:
    async def test_a_blocked_search_reveals_nothing_this_turn_or_the_next(self) -> None:
        kit, channel, provider = await _text_kit(
            [
                _calling("find_tools", query="spotify"),
                AIResponse(content="done"),
                AIResponse(content="next turn"),
            ]
        )
        _judge_search(kit, "block")

        await _text_turn(kit, channel)
        await _text_turn(kit, channel)

        [answer] = [
            json.loads(str(part.result))
            for message in provider.calls[1].messages
            if message.role == "tool"
            for part in message.content
        ]
        assert answer == {"error": "search is not allowed here"}
        assert not FOUND & _round_declares(provider, 1)
        assert not FOUND & _round_declares(provider, 2)
        assert not FOUND & channel._tool_usage.tool_names("r1")
        await kit.close()

    async def test_a_served_search_reveals_for_the_next_round_and_turn(self) -> None:
        kit, channel, provider = await _text_kit(
            [
                _calling("find_tools", query="spotify"),
                AIResponse(content="done"),
                AIResponse(content="next turn"),
            ]
        )

        await _text_turn(kit, channel)
        await _text_turn(kit, channel)

        assert _round_declares(provider, 1) >= FOUND
        assert _round_declares(provider, 2) >= FOUND
        await kit.close()

    async def test_a_search_that_finds_nothing_keeps_the_window(self) -> None:
        kit, channel, provider = await _text_kit(
            [
                _calling("find_tools", query="spotify"),
                _calling("find_tools", query="zzzz-nothing-matches"),
                AIResponse(content="done"),
            ]
        )

        await _text_turn(kit, channel)

        assert _round_declares(provider, 2) >= FOUND
        await kit.close()


async def _realtime_kit(
    provider: MockRealtimeProvider,
) -> tuple[RoomKit, RealtimeVoiceChannel, Any]:
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema(n) for n in SPOTIFY],
        tool_handler=_Recorder(),
        tool_search=True,
    )
    kit, session = await _session(channel)
    return kit, channel, session


class TestRealtime:
    async def test_a_blocked_search_reconfigures_nothing(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        seen = _judge_search(kit, "block")
        before = len(provider.calls)

        result = await _call(channel, provider, session, "find_tools", {"query": "spotify"})

        assert json.loads(result) == {"error": "search is not allowed here"}
        assert [c.method for c in provider.calls[before:]] == ["submit_tool_result"]
        assert not channel._tool_search_support._exposed.get(session.id)
        assert [(e.name, e.is_error) for e in seen] == [("find_tools", True)]
        await kit.close()

    async def test_a_served_search_is_declared_after_its_result_went_out(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        before = len(provider.calls)

        await _call(channel, provider, session, "find_tools", {"query": "spotify"})

        # The mock reconfigures by reconnecting, after the result went out.
        methods = [c.method for c in provider.calls[before:]]
        assert methods == ["submit_tool_result", "disconnect", "connect"]
        assert {t.get("name") for t in _declared(provider)} >= FOUND
        await kit.close()

    async def test_a_replaced_search_result_is_what_the_model_reads(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        seen = _judge_search(kit, "replace")

        result = await _call(channel, provider, session, "find_tools", {"query": "spotify"})

        assert result == '{"matches": []}'
        assert [(e.name, e.is_error, e.result) for e in seen] == [
            ("find_tools", False, '{"matches": []}')
        ]
        await kit.close()

    async def test_a_search_that_finds_nothing_keeps_what_was_revealed(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        await _call(channel, provider, session, "find_tools", {"query": "spotify"})
        before = len(provider.calls)

        await _call(channel, provider, session, "find_tools", {"query": "zzzz-nothing-matches"})

        assert [c.method for c in provider.calls[before:]] == ["submit_tool_result"]
        assert channel._tool_search_support._exposed[session.id] >= FOUND
        await kit.close()

    async def test_a_blocked_list_tools_reads_the_block(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        _judge_search(kit, "block")

        result = await _call(channel, provider, session, "list_tools")

        assert json.loads(result) == {"error": "search is not allowed here"}
        await kit.close()

    async def test_a_fixed_declaration_provider_is_never_reconfigured(self) -> None:
        provider = _FixedProvider()
        kit, channel, session = await _realtime_kit(provider)
        before = len(provider.calls)

        result = await _call(channel, provider, session, "find_tools", {"query": "spotify"})

        assert {m["name"] for m in json.loads(result)["matches"]} >= FOUND
        assert [c.method for c in provider.calls[before:]] == ["submit_tool_result"]
        await kit.close()
