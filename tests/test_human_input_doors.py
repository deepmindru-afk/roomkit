"""The tools that ask a person keep their rules on every door (RMK-481, RFC
§9.3, §21.6).

``human_input_handler=`` on an AIChannel and a RealtimeVoiceChannel: the
channel declares the tools and serves them before the host's handler, which
replacing that handler leaves alone; ON_USER_INPUT_REQUIRED's BLOCK rejects
the request, the handler's own timeout bounds the call rather than the
channel's default bound, the request names the door's channel type, and the
channel's close settles the requests still open.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import ChannelType
from roomkit.providers.ai.base import AITool, AIToolCall
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.tool_doors import DOORS, Hooks, run_door

EVERY_DOOR = pytest.mark.parametrize("door", [d for d in DOORS if d != "conference"])

ASK = AITool(
    name="ask",
    description="Ask the person",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)
ASK_DICT = {"name": "ask", "description": ASK.description, "parameters": ASK.parameters}
CALL = AIToolCall(id="c1", name="ask", arguments={"q": "Which room?"})

DOOR_TYPE = {
    "text-stream": ChannelType.AI,
    "text-nostream": ChannelType.AI,
}


def _human(timeout: float = 3.0) -> HumanInputToolHandler:
    return HumanInputToolHandler({"ask"}, timeout=timeout, tool_definitions=[ASK])


async def _host(name: str, arguments: dict[str, Any]) -> str:
    raise AssertionError(f"the host's handler served {name!r}")


def _on_request(decide: Callable[[Any], HookResult], seen: list[Any]) -> Callable[[RoomKit], None]:
    """An ON_USER_INPUT_REQUIRED hook that records each request and returns
    what *decide* says of it."""

    def setup(kit: RoomKit) -> None:
        @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC, name="ask")
        async def ask(event: Any, ctx: Any) -> HookResult:
            seen.append(event)
            return decide(event)

    return setup


async def _ask(door: str, human: HumanInputToolHandler, decide: Any, **channel: Any) -> Any:
    seen: list[Any] = []
    hooks = Hooks(setup=_on_request(decide, seen))
    options = {"human_input_handler": human, **channel}
    return await run_door(door, _host, call=CALL, hooks=hooks, channel=options), seen


@EVERY_DOOR
async def test_the_person_s_answer_is_the_call_s_result(door: str) -> None:
    human = _human()

    def answer(event: Any) -> HookResult:
        human.handler.resolve(event.pending_id, "the blue room")
        return HookResult.allow()

    seen, requests = await _ask(door, human, answer)

    assert "the blue room" in str(seen.model_read)
    assert [(e.is_error, e.refused) for e in seen.reports] == [(False, False)]
    assert [r.channel_type for r in requests] == [DOOR_TYPE.get(door, ChannelType.REALTIME_VOICE)]


@EVERY_DOOR
async def test_a_blocked_request_is_a_refusal(door: str) -> None:
    seen, requests = await _ask(door, _human(), lambda event: HookResult.block("no questions"))

    assert len(requests) == 1
    assert "Denied by ON_USER_INPUT_REQUIRED hook" in str(seen.model_read)
    assert [(e.is_error, e.refused) for e in seen.reports] == [(True, True)]


@EVERY_DOOR
async def test_the_handler_s_timeout_bounds_the_call_not_the_channel_s(door: str) -> None:
    seen, _ = await _ask(
        door, _human(timeout=0.4), lambda event: HookResult.allow(), tool_timeout_seconds=0.1
    )

    assert "Human input timed out after 0.4s" in str(seen.model_read)
    assert [(e.is_error, e.refused) for e in seen.reports] == [(True, False)]


@EVERY_DOOR
async def test_a_bound_named_for_the_tool_still_applies(door: str) -> None:
    seen, _ = await _ask(
        door, _human(timeout=3.0), lambda event: HookResult.allow(), tool_timeouts={"ask": 0.1}
    )

    assert "ToolTimeoutError" in str(seen.model_read)


def _replace_host_handler(kit: RoomKit) -> None:
    channel = kit.get_channel("ai1") or kit.get_channel("rt")
    assert channel is not None
    channel.tool_handler = _host  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize("door", ["text-stream", "text-nostream", "rt-provider"])
async def test_replacing_the_host_s_handler_keeps_the_person_s_tools(door: str) -> None:
    human = _human()
    requests: list[Any] = []

    def setup(kit: RoomKit) -> None:
        _on_request(lambda event: HookResult.block("no"), requests)(kit)
        _replace_host_handler(kit)

    options = {"human_input_handler": human}
    seen = await run_door(door, _host, call=CALL, hooks=Hooks(setup=setup), channel=options)

    assert len(requests) == 1
    assert "Denied by ON_USER_INPUT_REQUIRED hook" in str(seen.model_read)


async def _until_asked(human: HumanInputToolHandler) -> str:
    for _ in range(300):
        if human.handler.pending:
            return next(iter(human.handler.pending))
        await asyncio.sleep(0.01)
    raise AssertionError("no request was raised")


async def test_closing_a_realtime_channel_settles_its_requests() -> None:
    human = _human()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=MockRealtimeTransport(), human_input_handler=human
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")
    await provider.simulate_tool_call(session, "c1", "ask", {"q": "Which room?"})
    pending = await _until_asked(human)

    await channel.close()

    assert human.handler.pending == {}
    assert not human.handler.resolve(pending, "too late")
    assert provider.tool_results == []
    await kit.close()


async def test_a_host_tool_under_a_person_s_tool_name_is_refused() -> None:
    with pytest.raises(ValueError, match="serves itself"):
        RealtimeVoiceChannel(
            "rt",
            provider=MockRealtimeProvider(),
            transport=MockRealtimeTransport(),
            tools=[ASK_DICT],
            tool_handler=_host,
            human_input_handler=_human(),
        )
