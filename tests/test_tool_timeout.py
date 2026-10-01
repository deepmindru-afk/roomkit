"""A tool handler that never answers costs its call, never the turn (RFC §21.6, RMK-366).

Every wait on a handler goes through one bound, on every path a call takes:
both text loops, a speech-to-speech channel's provider calls and the calls it
recovers from speech, and a conference's provider calls. Past the bound the
handler is cancelled and the call fails as a raise (RFC §9.3). A tool that
waits on another agent or a person keeps its own bound.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, ToolTimeoutError
from roomkit.channels._tool_registry import ToolEntry, ToolSource, schema_tool
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.framework import RoomKit
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.tools.timeout import ToolTimeouts, answer_within
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_tool_policy_exemptions import _calls, _tool_payload, _turn

SLOW = {
    "name": "slow",
    "description": "Answers slowly",
    "parameters": {"type": "object", "properties": {}},
}
BOUND = 0.05


class _Hung:
    """A handler that takes *delay* seconds to answer, and records a cancel."""

    def __init__(self, delay: float = 60.0) -> None:
        self.delay = delay
        self.cancelled = False

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return '{"ok": true}'

    async def conference(self, room_id: str, name: str, arguments: dict[str, Any]) -> str:
        return await self.handler(name, arguments)


# -- The bound itself -------------------------------------------------------


class TestToolTimeouts:
    def test_a_call_takes_the_default(self) -> None:
        assert ToolTimeouts(10.0).for_call("lookup") == 10.0

    def test_a_tool_bound_overrides_the_default(self) -> None:
        timeouts = ToolTimeouts(10.0, {"report": 120.0, "export": None})
        assert timeouts.for_call("report") == 120.0
        assert timeouts.for_call("export") is None

    def test_a_tool_that_waits_keeps_its_own_bound_unless_named(self) -> None:
        assert ToolTimeouts(10.0).for_call("delegate", waits=True) is None
        assert ToolTimeouts(10.0, {"delegate": 30.0}).for_call("delegate", waits=True) == 30.0

    @pytest.mark.parametrize("bounds", [(0.0, {}), (-1.0, {}), (10.0, {"slow": 0.0})])
    def test_a_bound_that_is_not_positive_is_refused(
        self, bounds: tuple[float, dict[str, float | None]]
    ) -> None:
        with pytest.raises(ValueError, match="must be positive or None"):
            ToolTimeouts(*bounds)


class TestAnswerWithin:
    async def test_an_expired_bound_cancels_the_handler_and_raises(self) -> None:
        hung = _Hung()

        with pytest.raises(ToolTimeoutError, match="'slow' did not answer within 0.05 s"):
            await answer_within(BOUND, "slow", hung.handler("slow", {}))

        assert hung.cancelled

    async def test_a_handler_timeout_is_its_own_failure(self) -> None:
        async def times_out() -> str:
            raise TimeoutError("upstream API")

        with pytest.raises(TimeoutError, match="upstream API") as raised:
            await answer_within(10.0, "slow", times_out())
        assert not isinstance(raised.value, ToolTimeoutError)

    async def test_no_bound_waits_for_the_answer(self) -> None:
        assert await answer_within(None, "slow", _Hung(delay=0.01).handler("slow", {})) == (
            '{"ok": true}'
        )


# -- Text: both generation loops -------------------------------------------


def _text_channel(provider: MockAIProvider, hung: _Hung, **kwargs: Any) -> AIChannel:
    return AIChannel(
        "ai1", provider=provider, tool_handler=hung.handler, tool_timeout_seconds=BOUND, **kwargs
    )


async def test_a_hung_text_tool_fails_its_call_and_the_turn_goes_on(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=_calls(("slow", {})), streaming=streaming)
    hung = _Hung()

    await _turn(_text_channel(provider, hung), [SLOW])

    assert hung.cancelled
    assert "ToolTimeoutError" in _tool_payload(provider.calls[1], "slow")["error"]
    assert len(provider.calls) == 2  # the model answered after the failed call


async def test_a_text_tool_bound_lets_a_slow_tool_finish(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=_calls(("slow", {})), streaming=streaming)
    hung = _Hung(delay=0.2)

    await _turn(_text_channel(provider, hung, tool_timeouts={"slow": None}), [SLOW])

    assert not hung.cancelled
    assert _tool_payload(provider.calls[1], "slow") == {"ok": True}


def test_a_human_input_tool_keeps_its_own_bound() -> None:
    human = HumanInputToolHandler({"ask_user"}, timeout=300)
    channel = _text_channel(MockAIProvider(responses=["ok"]), _Hung(), human_input_handler=human)

    assert channel._call_timeout("ask_user", None) is None  # noqa: SLF001
    assert channel._call_timeout("slow", None) == BOUND  # noqa: SLF001


def test_an_orchestration_tool_keeps_its_own_bound() -> None:
    channel = _text_channel(MockAIProvider(responses=["ok"]), _Hung())
    channel._registry.register(  # noqa: SLF001
        ToolEntry(
            schema_tool({"name": "delegate_task"}),
            serve=None,
            source=ToolSource.ORCHESTRATION,
        ),
        room_id="r1",
        owner=object(),
    )

    assert channel._call_timeout("delegate_task", "r1") is None  # noqa: SLF001
    assert channel._call_timeout("delegate_task", "r2") == BOUND  # noqa: SLF001


# -- Speech-to-speech: provider calls and calls recovered from speech --------


async def _realtime(hung: _Hung) -> tuple[RoomKit, MockRealtimeProvider, VoiceSession]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[SLOW],
        tool_handler=hung.handler,
        tool_timeout_seconds=BOUND,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    return kit, provider, await channel.start_session("r1", "u1", "ws")


async def test_a_hung_realtime_tool_fails_its_call() -> None:
    hung = _Hung()
    kit, provider, session = await _realtime(hung)

    await provider.simulate_tool_call(session, "c1", "slow", {})
    await until(lambda: bool(provider.tool_results))

    assert hung.cancelled
    assert "ToolTimeoutError" in json.loads(provider.tool_results[0][2])["error"]
    await kit.close()


async def test_a_hung_tool_recovered_from_speech_fails_its_call() -> None:
    hung = _Hung()
    kit, provider, session = await _realtime(hung)

    await provider.simulate_transcription(session, "call:slow{}", "assistant")
    await until(lambda: any("ToolTimeoutError" in t for _s, t, _r in provider.injected_texts))

    assert hung.cancelled
    await kit.close()


# -- Conference ----------------------------------------------------------------


async def test_a_hung_conference_tool_fails_its_call() -> None:
    hung = _Hung()
    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider,
            tools=[SLOW],
            tool_handler=hung.conference,
            tool_timeout_seconds=BOUND,
        ),
    )
    session = await channel._realtime.ensure_session(ROOM)  # noqa: SLF001
    assert session is not None

    await provider.simulate_tool_call(session, "c1", "slow", {})
    await until(lambda: bool(provider.tool_results))

    assert hung.cancelled
    assert "ToolTimeoutError" in json.loads(provider.tool_results[0][2])["error"]
    await kit.close()


def test_a_conference_bound_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive or None"):
        ConferenceRealtimeConfig(provider=MockRealtimeProvider(), tool_timeout_seconds=0)
