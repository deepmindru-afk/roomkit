"""Verdicts, the capacity search and the summary table: pure functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

OK = "ok"
FAIL = "fail"
INVALID = "invalid"


@dataclass(frozen=True)
class Limits:
    """What a point must hold to count as carried.

    Loss counts packets that never came back. Jitter is the RTT spread
    (p99 - p50). Queueing is how far the median RTT rose above the floor,
    the median of a lightly loaded point: a saturated engine keeps every
    packet but delivers it late. Both are measured against the engine's
    own delay, so a design that buffers on purpose is not penalised for
    it. The other limits void a point the system under test is not alone
    to blame for: a late or saturated generator, or other processes taking
    CPU on the server's own CPUs or across the machine.
    """

    max_loss_pct: float = 0.1
    max_jitter_ms: float = 40.0
    max_queueing_ms: float = 20.0
    max_send_lateness_ms: float = 2.0
    max_generator_cpu: float = 0.8
    max_background_cores: float = 0.10
    max_machine_background_cores: float = 2.0


class InvalidMeasurementError(RuntimeError):
    """A point stayed invalid: the generator or other processes limited it."""


def generator_problem(gen: dict[str, Any], limits: Limits) -> str | None:
    """Why the generator cannot vouch for this point, or None."""
    # Lateness already counts ticks the generator woke up too late for.
    lateness = gen["send_lateness_ms"]["p99"]
    if lateness is not None and lateness > limits.max_send_lateness_ms:
        return f"generator sent late (p99 {lateness:.2f} ms)"
    if gen["send_dropped"]:
        return f"generator send buffers dropped {gen['send_dropped']} packets"
    if gen["recv_errors"]:
        return f"generator receive errors: {gen['recv_errors']}"
    load = gen["cpu_seconds"] / gen["window_seconds"] / gen["threads"]
    if load > limits.max_generator_cpu:
        return f"generator threads at {100 * load:.0f}% CPU"
    return None


def background_problem(point: dict[str, Any], limits: Limits) -> str | None:
    """Why other processes void this point, or None."""
    background = point.get("background_cores")
    if background is not None and background > limits.max_background_cores:
        return f"other processes took {100 * background:.0f}% of a server core"
    machine = point.get("machine_background_cores")
    if machine is not None and machine > limits.max_machine_background_cores:
        return f"other processes kept {machine:.1f} cores busy"
    return None


def check(point: dict[str, Any], limits: Limits) -> str | None:
    """Why the point cannot be blamed on the system under test alone, or None."""
    return generator_problem(point["generator"], limits) or background_problem(point, limits)


def verdict(point: dict[str, Any], limits: Limits, floor_ms: float | None) -> str:
    """Judge a point against the RTT floor; None makes it the floor itself."""
    gen = point["generator"]
    if check(point, limits):
        return INVALID
    rtt = gen["rtt_ms"]
    if gen["loss_pct"] > limits.max_loss_pct or rtt["p50"] is None or rtt["p99"] is None:
        return FAIL
    if rtt["p99"] - rtt["p50"] > limits.max_jitter_ms:
        return FAIL
    if floor_ms is not None and rtt["p50"] - floor_ms > limits.max_queueing_ms:
        return FAIL
    return OK


def retry_while_disturbed(
    run_once: Callable[[], dict[str, Any]],
    limits: Limits,
    retries: int,
    pause: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Run a point again while it cannot be blamed on the system under test
    alone (other processes, a late generator), up to *retries* more times,
    calling *pause* before each retry so a passing load can end; the last
    attempt stands, with the attempt count recorded."""
    for attempt in range(1, retries + 2):
        if attempt > 1:
            pause()
        point = run_once()
        point["attempts"] = attempt
        if check(point, limits) is None:
            break
    return point


def find_max(
    run: Callable[[int], str],
    start: int,
    cap: int,
    resolution: float = 0.05,
    granularity: int = 50,
) -> int | None:
    """Largest call count that passes: doubling from *start*, then bisection
    until the bracket is within *resolution* of the answer.

    Returns None when even *start* fails. Raises InvalidMeasurementError as
    soon as a point is invalid: a bigger generator is needed, not a verdict.
    """

    def passes(calls: int) -> bool:
        outcome = run(calls)
        if outcome == INVALID:
            raise InvalidMeasurementError(
                f"{calls} calls: the system under test was not alone to blame"
            )
        return outcome == OK

    best: int | None = None
    calls = start
    while passes(calls):
        best = calls
        if calls >= cap:
            return best
        calls = min(calls * 2, cap)
    if best is None:
        return None
    worst = calls
    while worst - best > max(granularity, best * resolution):
        mid = (best + worst) // 2 // granularity * granularity
        if mid <= best:
            break
        if passes(mid):
            best = mid
        else:
            worst = mid
    return best


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def row(point: dict[str, Any], limits: Limits) -> dict[str, Any]:
    """The summary fields of one point."""
    server, gen = point["server"], point["generator"]
    # Packets the generator saw carried in the window, and the server's CPU
    # read from outside: both hold when a saturated server runs late.
    packets = gen["sent"] + gen["received"]
    cpu = point.get("server_cpu_seconds", server["cpu_seconds"])
    return {
        "calls": point["calls"],
        "verdict": point["verdict"],
        "cpu_pct": 100 * cpu / server["window_seconds"],
        "us_per_packet": 1e6 * cpu / packets if packets else None,
        "loss_pct": gen["loss_pct"],
        "rtt_p50_ms": gen["rtt_ms"]["p50"],
        "rtt_p99_ms": gen["rtt_ms"]["p99"],
        "gaps_40ms": gen["gaps_40ms"],
        "kernel_drops": sum(point["kernel_udp"].values()),
        "rss_mb": server["rss_mb"],
        "background_pct": (
            None if point.get("background_cores") is None else 100 * point["background_cores"]
        ),
        "check": check(point, limits) or "ok",
    }


def markdown_table(points: list[dict[str, Any]], limits: Limits) -> str:
    header = (
        "| Calls | Verdict | Server CPU | µs/packet | Loss | RTT p50 / p99 | Gaps >40 ms "
        "| Kernel drops | RSS | Other load | Check |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    lines = []
    for point in points:
        r = row(point, limits)
        lines.append(
            f"| {r['calls']} | {r['verdict']} | {_fmt(r['cpu_pct'])} % "
            f"| {_fmt(r['us_per_packet'], 2)} | {_fmt(r['loss_pct'], 3)} % "
            f"| {_fmt(r['rtt_p50_ms'])} / {_fmt(r['rtt_p99_ms'])} ms | {r['gaps_40ms']} "
            f"| {r['kernel_drops']} | {_fmt(r['rss_mb'])} MB "
            f"| {_fmt(r['background_pct'])} % | {r['check']} |"
        )
    return header + "\n".join(lines) + "\n"


def parse_cpus(spec: str) -> list[int]:
    """A CPU list such as ``2``, ``4-7`` or ``2,8-11``."""
    cpus: list[int] = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus
