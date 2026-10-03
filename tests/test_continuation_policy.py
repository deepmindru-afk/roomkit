"""An AI channel's continuation policy goes on an answer that did not act (RFC §6.4).

The loop consults the policy for a round the model ended itself on text,
without a call, a tool declared, whatever the provider's word for the stop;
the policy shares the empty round's bound, the loop keeps its guards, and a
turn whose policy still asks once the bound has run out ends ``unfinished``,
its marker and ON_AI_RESPONSE included. Each case runs on a provider that
streams and on one that only has ``generate()``.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import AIResponse, AITool
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_LOOKUP = AITool(name="lookup", description="Look it up", parameters={})
_GO_ON = "You announced a check: run it now."


class _Policy:
    """Asks to go on when the round only announced a check, and records what it read."""

    def __init__(self) -> None:
        self.read: list[str] = []

    def __call__(self, text: str) -> str | None:
        self.read.append(text)
        return _GO_ON if text.startswith("I will check") else None


def _said(text: str, finish_reason: str | None = "stop", **usage: int) -> AIResponse:
    return AIResponse(content=text, finish_reason=finish_reason, usage=usage)


async def _turn(
    responses: list[AIResponse], streaming: bool, **kwargs: Any
) -> tuple[LoopRun, MockAIProvider, _Policy, list[AIResponseEvent]]:
    provider = MockAIProvider(ai_responses=responses, streaming=streaming)
    policy = _Policy()
    kwargs.setdefault("tools", [_LOOKUP])
    ch = AIChannel("ai1", provider=provider, continuation=policy, **kwargs)
    reported: list[AIResponseEvent] = []

    async def report(event: AIResponseEvent) -> None:
        reported.append(event)

    ch._after_response_hook = report
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    run = await respond(
        ch, make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )
    return run, provider, policy, reported


@pytest.mark.parametrize("stop", ["stop", "end_turn", "STOP"])
async def test_an_announcement_is_continued_once_whatever_the_providers_stop(
    streaming: bool, stop: str
) -> None:
    run, provider, policy, reported = await _turn(
        [_said("I will check the run.", stop), _said("The run finished at noon.")], streaming
    )

    assert len(provider.calls) == 2
    assert policy.read == ["I will check the run.", "The run finished at noon."]
    roles = [(m.role, m.content) for m in provider.calls[1].messages[-2:]]
    assert roles == [("assistant", "I will check the run."), ("user", _GO_ON)]
    assert run.reason == "completed"
    assert [event.loop_end_reason for event in reported] == ["completed"]


async def test_a_second_announcement_ends_the_turn_unfinished(streaming: bool) -> None:
    run, provider, _policy, reported = await _turn(
        [_said("I will check the run."), _said("I will check it now.")], streaming
    )

    assert len(provider.calls) == 2
    assert run.reason == "unfinished"
    assert [event.loop_end_reason for event in reported] == ["unfinished"]


async def test_the_bound_is_the_empty_rounds(streaming: bool) -> None:
    run, provider, _policy, _reported = await _turn(
        [_said("I will check the run.")] * 3, streaming, max_empty_retries=2
    )

    assert len(provider.calls) == 3
    assert run.reason == "unfinished"


async def test_no_continuation_past_the_turns_budget(streaming: bool) -> None:
    """The budget is a limit the loop keeps: an unfinished answer past it ends
    on the budget, and nothing more is generated."""
    run, provider, _policy, _reported = await _turn(
        [_said("I will check the run.", input_tokens=10, output_tokens=5)],
        streaming,
        turn_budget_tokens=1,
    )

    assert len(provider.calls) == 1
    assert run.reason == "budget_exceeded"


@pytest.mark.parametrize("stop", ["length", "MAX_TOKENS", "content_filter", "SAFETY", None])
async def test_a_round_the_model_did_not_end_itself_is_not_the_policys(
    streaming: bool, stop: str | None
) -> None:
    _run, provider, policy, _reported = await _turn(
        [_said("I will check the run.", stop)], streaming
    )

    assert policy.read == []
    assert len(provider.calls) == 1


async def test_without_a_declared_tool_there_is_nothing_to_go_on_with(streaming: bool) -> None:
    run, provider, policy, _reported = await _turn(
        [_said("I will check the run.")], streaming, tools=[]
    )

    assert policy.read == []
    assert len(provider.calls) == 1
    assert run.reason == "completed"


async def test_no_continuation_past_the_turns_deadline(streaming: bool) -> None:
    run, provider, _policy, _reported = await _turn(
        [_said("I will check the run.")], streaming, tool_loop_timeout_seconds=1e-9
    )

    assert len(provider.calls) == 1
    assert run.reason == "timeout"
