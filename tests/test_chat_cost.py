"""Offline contracts for the cost suite: fixed rounds, honest billing, readable totals."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from benchmarks.chat.cost import (
    BilledScript,
    cost_scenarios,
    first_change,
    scenario_options,
)
from benchmarks.chat.cost_report import cost_markdown, cost_summary, read_rate, turn_totals
from benchmarks.chat.harness import Harness
from benchmarks.chat.scenarios import Scenario

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITool,
    ModelInfo,
    ModelPricing,
    StreamDone,
)
from roomkit.providers.ai.mock import MockAIProvider

USAGE = {"input_tokens": 100, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0}
PRICING = ModelPricing(
    input_per_million=1.0,
    output_per_million=5.0,
    cache_read_per_million=0.1,
    cache_write_per_million=1.25,
    verified=date(2026, 9, 29),
)


class PricedBilling(MockAIProvider):
    """A billing provider with a price and fixed usage, recording what it was sent."""

    def __init__(self) -> None:
        super().__init__(ai_responses=[AIResponse(content="ignored", usage=USAGE)])
        self.sent: list[AIContext] = []

    @property
    def model_name(self) -> str:
        return "priced"

    def catalog_entry(self) -> ModelInfo | None:
        return ModelInfo(id="priced", pricing=PRICING)

    async def generate(self, context: AIContext) -> AIResponse:
        self.sent.append(context)
        return await super().generate(context)


def _scenario(name: str) -> Scenario:
    return next(s for s in cost_scenarios() if s.name == name)


async def _run(scenario: Scenario, billing: MockAIProvider | None) -> Harness:
    model = scenario.model_for(billing or MockAIProvider(), billed=billing is not None)
    h = Harness(
        model, streaming=scenario.streaming, **scenario_options(scenario.name, "fixed-nonce")
    )
    try:
        await h.add_room("main")
        await scenario.run(h)
        await h.validate()
    finally:
        await h.close()
    return h


def _requests(h: Harness) -> list[Any]:
    return [
        (
            [(t.name, t.description, t.parameters) for t in c.tools],
            c.system_prompt,
            [m.model_dump(mode="json") for m in c.messages],
        )
        for c in h.provider.contexts
    ]


@pytest.mark.parametrize("name", [s.name for s in cost_scenarios()])
async def test_two_runs_make_the_same_requests_over_the_same_rounds(name: str) -> None:
    first = await _run(_scenario(name), None)
    second = await _run(_scenario(name), None)

    assert all(first.checks.values()), first.checks
    assert _requests(first) == _requests(second)
    rows = first.details["cost"]
    assert [(r["turn"], r["round"]) for r in rows] == [
        (r["turn"], r["round"]) for r in second.details["cost"]
    ]
    # An offline run bills nothing: the rows carry the requests, not a price.
    assert all(r["cost"] is None and r["input_tokens"] == 0 for r in rows)


async def test_every_round_is_billed_by_the_real_provider_at_its_price() -> None:
    billing = PricedBilling()
    h = await _run(_scenario("tool_cost"), billing)

    assert all(h.checks.values()), h.checks
    rows = h.details["cost"]
    assert len(billing.sent) == len(rows) == 11
    assert all(context.max_tokens == 1 for context in billing.sent)
    assert [c.tools for c in billing.sent] == [c.tools for c in h.provider.contexts]
    assert all(r["cache_read_input_tokens"] == 900 for r in rows)
    assert rows[0]["cost"] == pytest.approx(PRICING.cost_for(USAGE))
    assert [(r["turn"], r["round"]) for r in rows] == [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (4, 0),
        (4, 1),
    ]


async def test_a_streamed_round_carries_the_billed_usage_to_its_done_event() -> None:
    script = BilledScript(PricedBilling(), [AIResponse(content="scripted")])
    events = [e async for e in script.generate_structured_stream(AIContext())]

    done = events[-1]
    assert isinstance(done, StreamDone) and done.usage == USAGE


def _context(tools: list[str], system: str, messages: list[str]) -> AIContext:
    return AIContext(
        tools=[AITool(name=n, description=n) for n in tools],
        system_prompt=system,
        messages=[AIMessage(role="user", content=m) for m in messages],
    )


def test_first_change_names_where_the_cached_prefix_stops() -> None:
    base = _context(["a", "b"], "sys", ["hi"])

    assert first_change(None, base) == "first"
    assert first_change(base, _context(["a", "b"], "sys", ["hi", "more"])) == "append"
    assert first_change(base, _context(["a", "c", "b"], "sys", ["hi", "more"])) == "tools"
    assert first_change(base, _context(["a", "b", "c"], "sys", ["hi"])) == "tools"
    assert first_change(base, _context(["b", "a"], "sys", ["hi"])) == "tools"
    assert first_change(base, _context(["a", "b"], "sys + digest", ["hi"])) == "system"
    assert first_change(base, _context(["a", "b"], "sys", ["rebuilt", "more"])) == "messages"


def _row(turn: int, round_: int, read: int, cost: float | None) -> dict[str, Any]:
    return {
        "turn": turn,
        "round": round_,
        "change": "append",
        "tools": 3,
        "input_tokens": 100,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": 100,
        "output_tokens": 1,
        "cost": cost,
    }


def test_turn_totals_sum_the_rounds_and_read_the_cache_share() -> None:
    rows = [_row(1, 0, 0, 0.001), _row(1, 1, 800, 0.002), _row(2, 0, 300, None)]

    first, second = turn_totals(rows)

    assert (first["rounds"], first["cache_read_input_tokens"]) == (2, 800)
    assert first["cost"] == pytest.approx(0.003)
    assert first["read_rate"] == pytest.approx(800 / 1200)
    # A round with no price leaves its turn unpriced rather than cheap.
    assert second["cost"] is None
    assert (
        read_rate({**_row(1, 0, 0, None), "input_tokens": 0, "cache_creation_input_tokens": 0})
        is None
    )


def test_the_report_gives_the_medians_per_turn_and_every_round() -> None:
    sample = {
        "scenario": "tool_cost",
        "status": "passed",
        "metrics": {"details": {"cost": [_row(1, 0, 0, 0.001), _row(1, 1, 800, 0.002)]}},
    }
    failed = {**sample, "status": "failed"}
    document = {"samples": [sample, sample, failed]}
    document["cost_summary"] = cost_summary(document["samples"])

    (row,) = document["cost_summary"]
    assert (row["samples"], row["rounds"]) == (2, 2)
    text = cost_markdown(document)
    assert "| tool_cost | 1 | 2 | 200 | 800 | 200 | 2 | 66.7% | 0.00300 |" in text
    assert text.count("| tool_cost | 1 | 1 | 3 | `append` |") == 1
    json.dumps(document)
