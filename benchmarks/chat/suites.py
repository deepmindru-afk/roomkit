"""The benchmark's suites: the scenarios each one runs and what its report adds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from benchmarks.chat.cost import cost_scenarios
from benchmarks.chat.cost_report import cost_markdown, cost_summary
from benchmarks.chat.quality import quality_scenarios
from benchmarks.chat.quality_report import quality_markdown, quality_summary
from benchmarks.chat.scenarios import Scenario, scenarios

Rows = list[dict[str, Any]]


@dataclass(frozen=True)
class Suite:
    """A suite: its scenarios, from the run's seed and variant count, and the
    summary rows (``<suite>_summary`` in the results, ``<suite>.csv``) and
    report section it adds to the common ones, if any."""

    scenarios: Callable[[int, int], list[Scenario]]
    summarize: Callable[[Rows], Rows] | None = None
    render: Callable[[dict[str, Any]], str] | None = None


SUITES: dict[str, Suite] = {
    "chat": Suite(lambda seed, variants: scenarios()),
    "quality": Suite(quality_scenarios, quality_summary, quality_markdown),
    "cost": Suite(lambda seed, variants: cost_scenarios(), cost_summary, cost_markdown),
}
