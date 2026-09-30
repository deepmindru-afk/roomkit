"""What a turn may spend (RFC §6.4).

A token budget over every token the provider bills for the turn, a cost
budget at the model's catalogue price, or both. The tool loop counts each
generation's usage as it comes and stops at the first round boundary where
the turn has reached either.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from roomkit.providers.ai.base import AIProvider, ModelPricing

# The counters a chat provider bills, disjoint (RFC §6.7): a token is counted
# under one of them only.
_BILLED_COUNTERS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
)


@dataclass(frozen=True)
class TurnBudget:
    """A turn's spending cap: billed tokens, a cost in the price's currency, or both."""

    tokens: int | None = None
    usd: float | None = None
    pricing: ModelPricing | None = None

    def tokens_of(self, usage: Mapping[str, Any]) -> int:
        """The tokens one generation's *usage* bills."""
        return sum(int(usage.get(counter) or 0) for counter in _BILLED_COUNTERS)

    def cost_of(self, usage: Mapping[str, Any]) -> float:
        """What one generation's *usage* costs, priced as one response."""
        return self.pricing.cost_for(usage) if self.pricing is not None else 0.0

    def reached(self, tokens: int, cost: float) -> bool:
        """Whether a turn that billed *tokens* and cost *cost* so far has reached it."""
        if self.tokens is not None and tokens >= self.tokens:
            return True
        return self.usd is not None and cost >= self.usd


def turn_budget(tokens: int | None, usd: float | None, provider: AIProvider) -> TurnBudget | None:
    """The budget *tokens* and *usd* set for a turn on *provider*, or ``None``
    when neither is set.

    Raises:
        ValueError: *usd* is set and the provider's model has no known price,
            so what the turn costs cannot be counted.
    """
    if tokens is None and usd is None:
        return None
    entry = provider.catalog_entry()
    pricing = entry.pricing if entry is not None else None
    if usd is not None and pricing is None:
        raise ValueError(
            f"turn_budget_usd needs a model with a known price; {provider.model_name!r} "
            "has none in the catalogue. Set turn_budget_tokens instead."
        )
    return TurnBudget(tokens=tokens, usd=usd, pricing=pricing)
