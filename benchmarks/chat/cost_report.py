"""What the cost suite's turns were billed, and how much the cache read back."""

from __future__ import annotations

import statistics
from typing import Any

_INPUT = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
_COUNTERS = (*_INPUT, "output_tokens")


def read_rate(row: dict[str, Any]) -> float | None:
    """The share of a request's input the cache read back (counters are disjoint)."""
    total = sum(row[counter] for counter in _INPUT)
    return row["cache_read_input_tokens"] / total if total else None


def turn_totals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One sample's rounds summed per turn."""
    totals: dict[int, dict[str, Any]] = {}
    for row in rows:
        turn = totals.setdefault(
            row["turn"],
            {"turn": row["turn"], "rounds": 0, "cost": 0.0, **dict.fromkeys(_COUNTERS, 0)},
        )
        turn["rounds"] += 1
        for counter in _COUNTERS:
            turn[counter] += row[counter]
        turn["cost"] = (
            None if row["cost"] is None or turn["cost"] is None else turn["cost"] + row["cost"]
        )
    return [{**turn, "read_rate": read_rate(turn)} for turn in totals.values()]


def cost_summary(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per scenario and turn, the median over the passed samples."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sample in samples:
        if sample["status"] != "passed":
            continue
        for turn in turn_totals(sample["metrics"]["details"].get("cost", [])):
            grouped.setdefault((sample["scenario"], turn["turn"]), []).append(turn)
    summary = []
    for (scenario, turn), group in grouped.items():
        row: dict[str, Any] = {"scenario": scenario, "turn": turn, "samples": len(group)}
        for key in ("rounds", *_COUNTERS, "read_rate", "cost"):
            values = [t[key] for t in group if t[key] is not None]
            row[key] = statistics.median(values) if values else None
        summary.append(row)
    return summary


def _fmt(value: Any, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def cost_markdown(document: dict[str, Any]) -> str:
    """The per-turn medians, then every round of each scenario's first passed sample."""
    lines = [
        "## Cost per turn",
        "",
        "Median over passed samples. Counters are disjoint: input is what was billed "
        "at the full rate, cache read and write at theirs. Read rate is cache read over "
        "the three input counters.",
        "",
        "| Scenario | Turn | Rounds | Input | Cache read | Cache write | Output | "
        "Read rate | Cost $ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in document.get("cost_summary", []):
        lines.append(
            f"| {row['scenario']} | {row['turn']} | {row['rounds']:.0f} | "
            f"{row['input_tokens']:.0f} | {row['cache_read_input_tokens']:.0f} | "
            f"{row['cache_creation_input_tokens']:.0f} | {row['output_tokens']:.0f} | "
            f"{_fmt(row['read_rate'], '.1%')} | {_fmt(row['cost'], '.5f')} |"
        )
    lines += [
        "",
        "## Rounds",
        "",
        "First passed sample of each scenario. Changed names the first block of the "
        "request that differs from the previous request: `append` is a request whose "
        "whole predecessor is its prefix; `tools`, `system` or `messages` is where the "
        "cached prefix stops.",
        "",
        "| Scenario | Turn | Round | Tools | Changed | Input | Cache read | Cache write | "
        "Cost $ |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    seen: set[str] = set()
    for sample in document["samples"]:
        if sample["status"] != "passed" or sample["scenario"] in seen:
            continue
        seen.add(sample["scenario"])
        for row in sample["metrics"]["details"].get("cost", []):
            lines.append(
                f"| {sample['scenario']} | {row['turn']} | {row['round']} | {row['tools']} | "
                f"`{row['change']}` | {row['input_tokens']} | "
                f"{row['cache_read_input_tokens']} | {row['cache_creation_input_tokens']} | "
                f"{_fmt(row['cost'], '.5f')} |"
            )
    return "\n".join(lines) + "\n"
