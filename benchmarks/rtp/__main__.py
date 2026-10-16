"""Run with ``uv run --extra rtp python -m benchmarks.rtp --help`` from the repository."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import subprocess  # nosec B404
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.rtp.analysis import (
    INVALID,
    InvalidMeasurementError,
    Limits,
    find_max,
    markdown_table,
    parse_cpus,
    retry_while_disturbed,
    row,
    verdict,
)
from benchmarks.rtp.processes import PointConfig, build_native, run_point


def _siblings(cpu: int) -> set[int]:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    try:
        return set(parse_cpus(path.read_text().strip()))
    except OSError:
        return {cpu}


def server_cores(server_cpus: list[int]) -> list[int]:
    """The server's CPUs with their hyperthread siblings."""
    return sorted(set().union(*(_siblings(c) for c in server_cpus))) if server_cpus else []


def default_gen_cpus(server_cpus: list[int]) -> list[int]:
    """Every allowed CPU except the server's cores, hyperthread siblings included."""
    return sorted(os.sched_getaffinity(0) - set(server_cores(server_cpus)))


def _command_output(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(  # nosec B603 B607
            cmd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    cpuinfo = Path("/proc/cpuinfo").read_text()
    model = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in cpuinfo.splitlines()
            if ln.startswith("model name")
        ),
        None,
    )
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "kernel": platform.release(),
        "cpu_model": model,
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "aiortp": _version("aiortp"),
        "uvloop": _version("uvloop"),
        "rustc": _command_output(["rustc", "--version"]),
        "git": _command_output(["git", "rev-parse", "--short", "HEAD"]),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m benchmarks.rtp", description=__doc__)
    p.add_argument("--server", choices=["native", "aiortp"], required=True)
    p.add_argument(
        "--mode",
        choices=["percall", "shared"],
        default="percall",
        help="native only: one socket per call, or one per thread (SSRC demux)",
    )
    p.add_argument("--server-threads", type=int, default=1, help="native only")
    p.add_argument("--server-cpus", default="2", help="CPU list the server is pinned to")
    p.add_argument("--playout", action="store_true", help="aiortp only: adaptive playout clock")
    p.add_argument("--uvloop", action="store_true", help="aiortp only: needs --with uvloop")
    calls = p.add_mutually_exclusive_group(required=True)
    calls.add_argument(
        "--calls", help="comma-separated call counts, lowest first: it sets the RTT floor"
    )
    calls.add_argument("--find-max", action="store_true", help="search the largest passing count")
    p.add_argument("--start-calls", type=int, default=250, help="--find-max first point")
    p.add_argument("--max-calls", type=int, default=30000, help="--find-max ceiling")
    p.add_argument("--gen-threads", type=int, default=4)
    p.add_argument("--gen-sockets", type=int, default=8, help="sockets per generator thread")
    p.add_argument("--gen-cpus", help="default: every CPU but the server's cores")
    p.add_argument("--seconds", type=float, default=15.0, help="measurement window")
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--max-loss-pct", type=float, default=Limits.max_loss_pct)
    p.add_argument("--max-jitter-ms", type=float, default=Limits.max_jitter_ms)
    p.add_argument("--max-queueing-ms", type=float, default=Limits.max_queueing_ms)
    p.add_argument("--output", type=Path, help="new directory for results.json and summary.md")
    p.add_argument(
        "--retries", type=int, default=2, help="re-runs of a point other processes disturbed"
    )
    p.add_argument(
        "--retry-wait",
        type=float,
        default=20.0,
        help="seconds to wait before re-running a disturbed point",
    )
    p.add_argument("--no-build", action="store_true", help="skip cargo build --release")
    args = p.parse_args(argv)
    if args.server == "aiortp" and (args.mode != "percall" or args.server_threads != 1):
        p.error("aiortp runs one socket per call in one thread")
    if args.server == "native" and (args.playout or args.uvloop):
        p.error("--playout and --uvloop apply to aiortp")
    if args.output and args.output.exists():
        p.error(f"{args.output} exists; choose a fresh directory")
    return args


def point_config(args: argparse.Namespace) -> PointConfig:
    server_cpus = parse_cpus(args.server_cpus) if args.server_cpus else []
    return PointConfig(
        server=args.server,
        mode=args.mode,
        server_threads=args.server_threads,
        server_cpus=server_cpus,
        playout=args.playout,
        uvloop=args.uvloop,
        gen_threads=args.gen_threads,
        gen_sockets=args.gen_sockets,
        gen_cpus=parse_cpus(args.gen_cpus) if args.gen_cpus else default_gen_cpus(server_cpus),
        warmup=args.warmup,
        seconds=args.seconds,
    )


def write_results(
    output: Path, meta: dict[str, Any], points: list[dict[str, Any]], limits: Limits
) -> None:
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps({**meta, "points": points}, indent=2) + "\n")
    summary = [
        f"# RTP benchmark: {meta['label']}",
        "",
        f"`{meta['command']}`",
        "",
        "Environment: " + ", ".join(f"{k} {v}" for k, v in meta["environment"].items() if v),
        "",
        markdown_table(points, limits),
    ]
    if "max_calls" in meta:
        summary.append(f"Largest passing count: **{meta['max_calls']}**\n")
    (output / "summary.md").write_text("\n".join(summary))


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.no_build:
        build_native()
    cfg = point_config(args)
    limits = Limits(
        max_loss_pct=args.max_loss_pct,
        max_jitter_ms=args.max_jitter_ms,
        max_queueing_ms=args.max_queueing_ms,
    )
    points: list[dict[str, Any]] = []

    def measure(calls: int) -> str:
        point = retry_while_disturbed(
            lambda: run_point(cfg, calls),
            limits,
            args.retries,
            pause=lambda: time.sleep(args.retry_wait),
        )
        # The first point, lightly loaded, sets the RTT floor for the rest.
        floor = points[0]["generator"]["rtt_ms"]["p50"] if points else None
        point["rtt_floor_ms"] = floor
        point["verdict"] = verdict(point, limits, floor)
        points.append(point)
        r = row(point, limits)
        print(
            f"{calls:>6} calls  {r['verdict']:<7}  cpu {r['cpu_pct']:5.1f} %  "
            f"{r['us_per_packet'] or 0:5.2f} µs/pkt  loss {r['loss_pct']:.3f} %  "
            f"rtt p50/p99 {r['rtt_p50_ms']}/{r['rtt_p99_ms']} ms  "
            f"other {r['background_pct'] or 0:.0f} %  attempts {point['attempts']}  {r['check']}",
            flush=True,
        )
        return point["verdict"]

    label = args.server + (f"-{args.mode}" if args.server == "native" else "")
    label += "-playout" * args.playout + "-uvloop" * args.uvloop
    meta: dict[str, Any] = {
        "label": label,
        "command": "python -m benchmarks.rtp " + " ".join(argv),
        "environment": environment(),
        "config": {k: str(v) for k, v in dataclasses.asdict(cfg).items()},
        "limits": dataclasses.asdict(limits),
    }
    status = 0
    try:
        if args.find_max:
            meta["max_calls"] = find_max(measure, args.start_calls, args.max_calls)
            print(f"largest passing count: {meta['max_calls']}")
        else:
            for calls in (int(c) for c in args.calls.split(",")):
                status |= measure(calls) == INVALID
    except InvalidMeasurementError as exc:
        print(f"stopped: {exc}; see the point's check column", file=sys.stderr)
        status = 1
    print()
    print(markdown_table(points, limits))
    if args.output:
        write_results(args.output, meta, points, limits)
        print(f"results in {args.output}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
