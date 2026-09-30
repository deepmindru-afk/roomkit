"""A turn's budget stops its tool loop at a round boundary (RFC §6.4, RMK-320).

A token budget counts every token the provider bills for the turn, a cost
budget prices each generation at the model's catalogue rate. A turn that has
reached either ends ``budget_exceeded``: the calls its last generation asked
for do not run, and no further generation is asked for.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from roomkit.channels._turn_budget import TurnBudget
from roomkit.channels._turn_config import AIChannelTurnConfig
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, ModelInfo, ModelPricing
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_LOOKUP = AITool(name="lookup", description="Look up an item", parameters={})
# 500 billed tokens per generation.
_USAGE = {"input_tokens": 300, "cache_read_input_tokens": 100, "output_tokens": 100}
_PRICING = ModelPricing(
    input_per_million=10.0,
    output_per_million=50.0,
    cache_read_per_million=1.0,
    long_context_threshold_tokens=1000,
    long_context_input_multiplier=2.0,
    verified=date(2026, 9, 30),
)


class _Priced(MockAIProvider):
    """A mock whose model has a catalogue price."""

    def catalog_entry(self) -> ModelInfo | None:
        return ModelInfo(id="priced", pricing=_PRICING)


def _round(n: int) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"c{n}", name="lookup", arguments={"n": n})],
        usage=dict(_USAGE),
    )


_DONE = AIResponse(content="done", finish_reason="stop", usage=dict(_USAGE))


def _script(rounds: int = 6) -> list[AIResponse]:
    return [*(_round(n) for n in range(rounds)), _DONE]


async def _turn(ch: AIChannel, metadata: dict[str, Any] | None = None) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_LOOKUP.model_dump()], **(metadata or {})},
    )
    return await respond(
        ch, make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )


def _channel(provider: MockAIProvider, **options: Any) -> tuple[AIChannel, list[int]]:
    ran: list[int] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        ran.append(arguments["n"])
        return "ok"

    return AIChannel("ai1", provider=provider, tool_handler=lookup, **options), ran


async def test_a_token_budget_stops_the_turn_where_it_is_reached(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=_script(), streaming=streaming)
    ch, ran = _channel(provider, turn_budget_tokens=1200)

    run = await _turn(ch)

    # 500 billed tokens a generation: the third reaches 1,200, and its call does not run.
    assert run.reason == "budget_exceeded"
    assert ran == [0, 1]
    assert len(provider.calls) == 3


async def test_a_cost_budget_stops_the_turn_at_the_catalogue_price(streaming: bool) -> None:
    provider = _Priced(ai_responses=_script(), streaming=streaming)
    per_generation = _PRICING.cost_for(_USAGE)
    ch, ran = _channel(provider, turn_budget_usd=per_generation * 2.5)

    run = await _turn(ch)

    assert run.reason == "budget_exceeded"
    assert ran == [0, 1]
    assert len(provider.calls) == 3


async def test_a_turn_under_its_budget_completes(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=_script(rounds=2), streaming=streaming)
    ch, ran = _channel(provider, turn_budget_tokens=100_000)

    run = await _turn(ch)

    assert (run.reason, run.text, ran) == ("completed", "done", [0, 1])


async def test_the_turn_config_and_the_room_outrank_the_channel(streaming: bool) -> None:
    async def per_turn(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(turn_budget_tokens=700)

    provider = MockAIProvider(ai_responses=_script(), streaming=streaming)
    ch, ran = _channel(provider, turn_budget_tokens=100_000, config_provider=per_turn)
    assert (await _turn(ch)).reason == "budget_exceeded" and ran == [0]

    provider = MockAIProvider(ai_responses=_script(), streaming=streaming)
    ch, ran = _channel(provider, turn_budget_tokens=100_000, config_provider=per_turn)
    run = await _turn(ch, {"turn_budget_tokens": 100_000})
    assert run.reason == "completed" and ran == [0, 1, 2, 3, 4, 5]


def test_a_cost_budget_needs_a_priced_model() -> None:
    with pytest.raises(ValueError, match="turn_budget_usd needs a model with a known price"):
        AIChannel("ai1", provider=MockAIProvider(), turn_budget_usd=0.05)


def test_each_generation_is_priced_as_one_response() -> None:
    """Two generations under the long-context threshold cost their own price
    each; pricing their sum as one response would apply the multiplier."""
    budget = TurnBudget(usd=1.0, pricing=_PRICING)
    usage = {"input_tokens": 600}

    per_generation = budget.cost_of(usage)

    assert budget.cost_of({"input_tokens": 1200}) > 2 * per_generation
    assert budget.tokens_of(_USAGE) == 500
    assert not budget.reached(0, 2 * per_generation)


def test_fifty_rounds_by_default() -> None:
    ch = AIChannel("ai1", provider=MockAIProvider())

    assert ch._max_tool_rounds == 50
