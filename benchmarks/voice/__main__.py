"""Run with ``uv run python -m benchmarks.voice --help`` from the repository."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from benchmarks.voice.frame_cost import measure


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.voice",
        description="CPU RoomKit spends per inbound 20 ms voice frame, the transport excluded.",
    )
    parser.add_argument("--calls", type=_positive, default=100, help="sessions, one room each")
    parser.add_argument("--seconds", type=_positive, default=10, help="audio per run")
    parser.add_argument("--vad", choices=["energy", "none"], default="energy")
    parser.add_argument("--stt", choices=["batch", "stream"], default="batch")
    parser.add_argument("--repeat", type=_positive, default=3, help="runs, each a fresh kit")
    parser.add_argument("--cpu", type=int, help="pin the process to this CPU (Linux)")
    parser.add_argument("--dsn", help="put the rooms in this Postgres instead of memory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    for _ in range(args.repeat):
        cost = asyncio.run(
            measure(
                args.calls,
                args.seconds,
                vad=args.vad == "energy",
                streaming_stt=args.stt == "stream",
                dsn=args.dsn,
            )
        )
        store = "postgres" if args.dsn else "memory"
        sys.stdout.write(
            f"store={store} vad={args.vad} stt={args.stt} calls={cost.calls} "
            f"frames={cost.frames}: "
            f"{cost.us_per_frame:.1f} µs/frame, {cost.calls_per_core:.0f} calls/core "
            f"for this layer, {cost.stt_requests} STT requests\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
