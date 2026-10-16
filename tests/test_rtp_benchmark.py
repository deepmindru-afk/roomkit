"""Offline contracts for the RTP benchmark: verdicts, the search and the commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from benchmarks.rtp import __main__ as cli
from benchmarks.rtp.analysis import (
    FAIL,
    INVALID,
    OK,
    InvalidMeasurementError,
    Limits,
    find_max,
    generator_problem,
    markdown_table,
    parse_cpus,
    retry_while_disturbed,
    verdict,
)
from benchmarks.rtp.processes import (
    PointConfig,
    cpu_busy_seconds,
    generator_command,
    process_cpu_seconds,
    server_command,
)

LIMITS = Limits()


def point(**gen: Any) -> dict[str, Any]:
    generator = {
        "window_seconds": 15.0,
        "threads": 4,
        "cpu_seconds": 6.0,
        "loss_pct": 0.0,
        "missed_ticks": 0,
        "send_dropped": 0,
        "recv_errors": 0,
        "gaps_40ms": 0,
        "sent": 1_500_000,
        "received": 1_500_000,
        "rtt_ms": {"p50": 12.0, "p99": 22.0, "p999": 23.0, "max": 30.0},
        "send_lateness_ms": {"p50": 0.01, "p99": 0.05, "max": 1.0},
    }
    generator.update(gen)
    server = {"cpu_seconds": 7.5, "window_seconds": 15.0, "rx": 1_500_000, "tx": 1_500_000}
    server["rss_mb"] = 24.0
    return {
        "calls": 2000,
        "server": server,
        "generator": generator,
        "kernel_udp": {"RcvbufErrors": 0, "SndbufErrors": 0, "MemErrors": 0},
    }


def test_a_clean_point_passes() -> None:
    assert verdict(point(), LIMITS, None) == OK


@pytest.mark.parametrize(
    "gen",
    [
        {"loss_pct": 0.5},
        {"rtt_ms": {"p50": 12.0, "p99": 80.0, "p999": 90.0, "max": 95.0}},
        {"rtt_ms": {"p50": None, "p99": None, "p999": None, "max": 0.0}},
    ],
    ids=["loss", "jitter", "nothing-came-back"],
)
def test_loss_and_added_jitter_fail(gen: dict[str, Any]) -> None:
    assert verdict(point(**gen), LIMITS, 12.0) == FAIL


def test_a_median_risen_above_the_floor_is_queueing() -> None:
    # Saturation keeps every packet but delivers it late, with a tight spread.
    rtt = {"p50": 110.0, "p99": 140.0, "p999": 150.0, "max": 160.0}
    assert verdict(point(rtt_ms=rtt), LIMITS, 80.0) == FAIL
    assert verdict(point(rtt_ms=rtt), LIMITS, 95.0) == OK


def test_a_fixed_buffering_delay_is_not_jitter() -> None:
    # aiortp's jitter prefetch holds ~80 ms on purpose: spread is what counts.
    rtt = {"p50": 80.0, "p99": 82.0, "p999": 85.0, "max": 90.0}
    assert verdict(point(rtt_ms=rtt), LIMITS, 80.0) == OK


@pytest.mark.parametrize(
    ("gen", "reason"),
    [
        ({"send_lateness_ms": {"p50": 0.1, "p99": 5.0, "max": 9.0}}, "sent late"),
        ({"send_dropped": 3}, "dropped"),
        ({"recv_errors": 1}, "receive errors"),
        ({"cpu_seconds": 55.0}, "CPU"),
    ],
)
def test_a_struggling_generator_voids_the_point(gen: dict[str, Any], reason: str) -> None:
    p = point(loss_pct=5.0, **gen)  # the loss must not be blamed on the server
    assert verdict(p, LIMITS, 12.0) == INVALID
    assert reason in (generator_problem(p["generator"], LIMITS) or "")


def test_other_processes_on_the_server_core_void_the_point() -> None:
    p = point()
    p["background_cores"] = 0.30
    assert verdict(p, LIMITS, 12.0) == INVALID
    p["background_cores"] = 0.01
    assert verdict(p, LIMITS, 12.0) == OK
    p["machine_background_cores"] = 3.0  # shared cache and turbo budget
    assert verdict(p, LIMITS, 12.0) == INVALID


def test_a_disturbed_point_runs_again_until_quiet() -> None:
    noise = iter([0.4, 0.2, 0.0])

    def run_once() -> dict[str, Any]:
        p = point()
        p["background_cores"] = next(noise)
        return p

    pauses: list[int] = []
    p = retry_while_disturbed(run_once, LIMITS, retries=2, pause=lambda: pauses.append(1))
    assert (p["attempts"], p["background_cores"], len(pauses)) == (3, 0.0, 2)

    still = retry_while_disturbed(lambda: {**point(), "background_cores": 0.5}, LIMITS, retries=1)
    assert still["attempts"] == 2  # the last attempt stands, judged invalid


def test_busy_time_counts_the_named_cpus_without_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stat = (
        "cpu  900 0 900 9000 0 0 0 0 0 0\n"
        "intr 5 6 7\n"
        "cpu2 100 5 40 1000 3 7 11 2 0 0\n"
        "cpu3 10 0 10 1000 0 0 0 0 0 0\n"
        "cpu4 999 0 999 0 0 0 0 0 0 0\n"
    )
    monkeypatch.setattr("benchmarks.rtp.processes.os.sysconf", lambda _name: 100)
    # user + nice + system + steal: (100+5+40+2) + (10+0+10+0) = 167 ticks.
    assert cpu_busy_seconds([2, 3], stat) == pytest.approx(1.67)
    # Every CPU, the aggregate "cpu" line left out: 167 + 1998 ticks.
    assert cpu_busy_seconds(None, stat) == pytest.approx(21.65)


def test_server_cpu_is_read_from_outside(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchmarks.rtp.processes.os.sysconf", lambda _name: 100)
    # A command name with spaces and parentheses must not shift the fields.
    stat = "4242 (python (x) y) R 1 2 3 4 5 6 7 8 9 10 250 50 0 0 20 0 3 0"
    assert process_cpu_seconds(4242, stat) == pytest.approx(3.0)


def test_cost_uses_the_outside_cpu_reading() -> None:
    p = point()
    p["verdict"] = OK
    p["server_cpu_seconds"] = 15.0  # the server's own report said 7.5 s
    assert "| 2000 | ok | 100.0 % | 5.00 |" in markdown_table([p], LIMITS)


def threshold_run(limit: int, seen: list[int]) -> Any:
    def run(calls: int) -> str:
        seen.append(calls)
        return OK if calls <= limit else FAIL

    return run


def test_find_max_brackets_the_limit() -> None:
    seen: list[int] = []
    best = find_max(threshold_run(3700, seen), start=500, cap=30000)
    assert best is not None and 3700 * 0.95 <= best <= 3700
    assert seen[:4] == [500, 1000, 2000, 4000]


def test_find_max_edges() -> None:
    assert find_max(threshold_run(100, []), start=500, cap=30000) is None
    seen: list[int] = []
    assert find_max(threshold_run(10**9, seen), start=500, cap=3000) == 3000
    assert seen == [500, 1000, 2000, 3000]


def test_find_max_stops_on_an_invalid_point() -> None:
    with pytest.raises(InvalidMeasurementError):
        find_max(lambda calls: INVALID if calls > 1000 else OK, start=500, cap=30000)


def test_parse_cpus() -> None:
    assert parse_cpus("2") == [2]
    assert parse_cpus("4-7,10") == [4, 5, 6, 7, 10]


def test_table_reports_cost_per_packet() -> None:
    p = point()
    p["verdict"] = OK
    table = markdown_table([p], LIMITS)
    # 7.5 CPU seconds over 3 M packets: 2.5 µs each; 7.5 s over 15 s: 50 %.
    assert "| 2000 | ok | 50.0 % | 2.50 |" in table


def test_generator_cpus_skip_the_server_core_and_its_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(cli, "_siblings", lambda cpu: {cpu, cpu ^ 1})
    assert cli.default_gen_cpus([2]) == [0, 1, 4, 5, 6, 7]


def test_both_sides_share_the_ports_the_server_bound(tmp_path: Path) -> None:
    ports = tmp_path / "ports"
    cfg = PointConfig(server="native", mode="shared", server_threads=2, server_cpus=[2, 4])
    echo = server_command(cfg, 3000, ports)
    load = generator_command(cfg, 3000, start=10.0, end=25.0, ports_file=ports)
    assert echo[1:] == [
        "echo", "--calls", "3000", "--mode", "shared", "--threads", "2",
        "--base-port", "20000", "--ports-file", str(ports), "--cpus", "2,4",
    ]  # fmt: skip
    assert load[load.index("--ports-file") + 1] == str(ports)

    py = server_command(PointConfig(server="aiortp", playout=True), 400, ports)
    assert py[1:3] == ["-m", "benchmarks.rtp.aiortp_echo"] and "--playout" in py
    assert py[py.index("--ports-file") + 1] == str(ports)


def test_cli_rejects_options_that_do_not_apply(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--server", "aiortp", "--mode", "shared", "--calls", "10"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--server", "native", "--playout", "--calls", "10"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--server", "native", "--calls", "10", "--output", str(tmp_path)])
