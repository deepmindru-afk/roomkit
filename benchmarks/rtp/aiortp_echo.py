"""aiortp echo server: the Python system under test.

One asyncio process, one ``aiortp.RTPSession`` per call, each echoing the
decoded audio back re-encoded: the RTP work a RoomKit SIP call does before
any voice pipeline stage. Speaks the driver's protocol: prints ``READY``,
reads ``<start> <end>`` (epoch seconds) on stdin, prints what it did inside
that window as one JSON line, then keeps echoing until it is stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import logging
import os
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiortp

try:  # optional: uv run --with uvloop
    import uvloop
except ImportError:
    uvloop = None

# Each call owns a port pair (RTP even, RTCP odd), as the native engine.
PORT_STRIDE = 2


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _echo(session: aiortp.RTPSession) -> Callable[[bytes, int], None]:
    def on_audio(pcm: bytes, _timestamp: int) -> None:
        session.send_audio_pcm_auto(pcm)

    return on_audio


def _totals(sessions: list[aiortp.RTPSession]) -> dict[str, int]:
    totals = {"rx": 0, "tx": 0, "lost": 0}
    for session in sessions:
        stats: dict[str, Any] = session.stats
        totals["rx"] += stats.get("packets_received", 0)
        totals["tx"] += stats.get("packets_sent", 0)
        totals["lost"] += stats.get("packets_lost", 0)
    return totals


async def _open_sessions(args: argparse.Namespace) -> list[aiortp.RTPSession]:
    """One session per call from the base port up, skipping port pairs
    another program holds; the generator reads the ports file."""
    sessions: list[aiortp.RTPSession] = []
    ports: list[int] = []
    port = args.base_port
    while len(sessions) < args.calls:
        if port + 1 > 65535:
            raise SystemExit(f"only {len(sessions)} free port pairs from {args.base_port}")
        try:
            session = await aiortp.RTPSession.create(
                local_addr=(args.bind, port),
                # Symmetric RTP answers wherever the media comes from; the
                # initial address only receives RTCP until then.
                remote_addr=(args.peer_host, 9),
                payload_type=0,
                symmetric_rtp=True,
                playout=args.playout,
            )
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
        else:
            session.on_audio = _echo(session)
            sessions.append(session)
            ports.append(port)
        port += PORT_STRIDE
    listing = "".join(f"{p}\n" for p in ports)
    await asyncio.to_thread(Path(args.ports_file).write_text, listing)
    return sessions


async def serve(args: argparse.Namespace) -> None:
    sessions = await _open_sessions(args)
    print("READY", flush=True)

    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    start, end = (float(x) for x in line.split())
    await asyncio.sleep(max(0.0, start - time.time()))
    cpu0, before = _cpu_seconds(), _totals(sessions)
    await asyncio.sleep(max(0.0, end - time.time()))
    cpu1, after = _cpu_seconds(), _totals(sessions)
    report = {
        "cpu_seconds": round(cpu1 - cpu0, 4),
        "window_seconds": round(end - start, 4),
        "threads": 1,
        "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        **{key: after[key] - before[key] for key in after},
    }
    print(json.dumps(report), flush=True)
    await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--peer-host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=20000)
    parser.add_argument("--ports-file", required=True)
    parser.add_argument("--cpus", type=lambda s: {int(c) for c in s.split(",")})
    parser.add_argument("--playout", action="store_true")
    parser.add_argument("--uvloop", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.cpus:
        os.sched_setaffinity(0, args.cpus)
    if args.uvloop:
        if uvloop is None:
            parser.error("--uvloop needs uvloop: uv run --with uvloop ...")
        uvloop.run(serve(args))
    else:
        asyncio.run(serve(args))


if __name__ == "__main__":
    main()
