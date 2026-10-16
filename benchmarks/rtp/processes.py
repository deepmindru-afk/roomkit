"""Runs one point: a system under test, the generator, one shared window."""

from __future__ import annotations

import json
import os
import select
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NATIVE_DIR = Path(__file__).parent / "native"
NATIVE_BINARY = NATIVE_DIR / "target" / "release" / "rtp-bench"
# /proc/net/snmp counters that mean the kernel dropped a datagram.
KERNEL_DROPS = ("RcvbufErrors", "SndbufErrors", "MemErrors")


@dataclass(frozen=True)
class PointConfig:
    server: str  # "native" or "aiortp"
    mode: str = "percall"
    server_threads: int = 1
    server_cpus: list[int] = field(default_factory=list)
    playout: bool = False
    uvloop: bool = False
    gen_threads: int = 4
    gen_sockets: int = 8
    gen_cpus: list[int] = field(default_factory=list)
    base_port: int = 20000
    warmup: float = 3.0
    seconds: float = 15.0
    grace: float = 1.5
    setup_timeout: float = 120.0
    binary: Path = NATIVE_BINARY


def build_native() -> None:
    subprocess.run(  # nosec B603 B607
        [
            "cargo",
            "build",
            "--release",
            "--quiet",
            "--manifest-path",
            str(NATIVE_DIR / "Cargo.toml"),
        ],
        check=True,
    )


def _cpu_list(cpus: list[int]) -> list[str]:
    return [",".join(map(str, cpus))] if cpus else []


def server_command(cfg: PointConfig, calls: int, ports_file: Path) -> list[str]:
    """The server binds its ports and lists them in *ports_file*."""
    cpus = _cpu_list(cfg.server_cpus)
    if cfg.server == "native":
        return [
            str(cfg.binary), "echo",
            "--calls", str(calls),
            "--mode", cfg.mode,
            "--threads", str(cfg.server_threads),
            "--base-port", str(cfg.base_port),
            "--ports-file", str(ports_file),
            *(["--cpus", *cpus] if cpus else []),
        ]  # fmt: skip
    return [
        sys.executable, "-m", "benchmarks.rtp.aiortp_echo",
        "--calls", str(calls),
        "--base-port", str(cfg.base_port),
        "--ports-file", str(ports_file),
        *(["--cpus", *cpus] if cpus else []),
        *(["--playout"] if cfg.playout else []),
        *(["--uvloop"] if cfg.uvloop else []),
    ]  # fmt: skip


def generator_command(
    cfg: PointConfig, calls: int, start: float, end: float, ports_file: Path
) -> list[str]:
    """The generator sends call i to the i-th port listed (modulo the list)."""
    cpus = _cpu_list(cfg.gen_cpus)
    return [
        str(cfg.binary), "load",
        "--calls", str(calls),
        "--ports-file", str(ports_file),
        "--threads", str(cfg.gen_threads),
        "--sockets", str(cfg.gen_sockets),
        "--start", f"{start:.6f}",
        "--end", f"{end:.6f}",
        "--grace", str(cfg.grace),
        *(["--cpus", *cpus] if cpus else []),
    ]  # fmt: skip


def udp_counters() -> dict[str, int]:
    lines = [line.split() for line in Path("/proc/net/snmp").read_text().splitlines()]
    udp = [line[1:] for line in lines if line and line[0] == "Udp:"]
    return dict(zip(udp[0], map(int, udp[1]), strict=True))


def cpu_busy_seconds(cpus: list[int] | None, stat: str | None = None) -> float:
    """User, nice, system and steal time spent on *cpus* (None: every CPU)
    since boot.

    Interrupt and softirq time are left out: they are not charged to the
    processes they serve, so they would read as someone else's load.
    """
    text = stat if stat is not None else Path("/proc/stat").read_text()
    wanted = None if cpus is None else {f"cpu{c}" for c in cpus}
    ticks = 0
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] == "cpu" or not fields[0].startswith("cpu"):
            continue
        if wanted is None or fields[0] in wanted:
            user, nice, system, _idle, _iowait, _irq, _softirq, steal = map(int, fields[1:9])
            ticks += user + nice + system + steal
    return ticks / os.sysconf("SC_CLK_TCK")


def process_cpu_seconds(pid: int, stat: str | None = None) -> float:
    """User plus system time of a process, all threads, read from outside.

    The driver measures the server this way because a saturated asyncio
    server wakes its own measuring code seconds late and under-reports.
    """
    text = stat if stat is not None else Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def _sleep_until(t: float) -> None:
    time.sleep(max(0.0, t - time.time()))


def _readline(proc: subprocess.Popen[str], timeout: float) -> str:
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    line = proc.stdout.readline() if ready else ""
    if not line:
        raise RuntimeError(f"system under test gave no output (exit code {proc.poll()})")
    return line


def run_point(cfg: PointConfig, calls: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rtp-bench-") as tmp:
        return _run_point(cfg, calls, Path(tmp) / "ports")


def _run_point(cfg: PointConfig, calls: int, ports_file: Path) -> dict[str, Any]:
    before = udp_counters()
    server = subprocess.Popen(  # nosec B603
        server_command(cfg, calls, ports_file),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        if _readline(server, cfg.setup_timeout).strip() != "READY":
            raise RuntimeError("system under test did not report READY")
        start = time.time() + cfg.warmup
        end = start + cfg.seconds
        assert server.stdin is not None
        server.stdin.write(f"{start} {end}\n")
        server.stdin.flush()
        load = subprocess.Popen(  # nosec B603
            generator_command(cfg, calls, start, end, ports_file),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _sleep_until(start)
        server_start = process_cpu_seconds(server.pid)
        cores_start, machine_start = cpu_busy_seconds(cfg.server_cpus), cpu_busy_seconds(None)
        _sleep_until(end)
        server_cpu = process_cpu_seconds(server.pid) - server_start
        cores_busy = cpu_busy_seconds(cfg.server_cpus) - cores_start
        machine_busy = cpu_busy_seconds(None) - machine_start
        out, err = load.communicate(timeout=cfg.grace + 30)
        if load.returncode != 0:
            raise RuntimeError(f"generator failed: {err.strip()}")
        generator = json.loads(out.strip().splitlines()[-1])
        report = json.loads(_readline(server, timeout=10))
    finally:
        server.kill()
        server.wait()
    after = udp_counters()
    return {
        "calls": calls,
        "server": report,
        "generator": generator,
        "kernel_udp": {key: after[key] - before[key] for key in KERNEL_DROPS},
        "loadavg_1m": os.getloadavg()[0],
        "server_cpu_seconds": server_cpu,
        # Cores' worth of CPU other processes took on the server's own CPUs,
        # and on the whole machine (shared cache, turbo budget). Hyperthread
        # siblings are left to the machine-wide figure: the scheduler fills
        # them with short tasks that measured no effect on the server's cost.
        "background_cores": (
            max(0.0, cores_busy - server_cpu) / cfg.seconds if cfg.server_cpus else None
        ),
        "machine_background_cores": max(0.0, machine_busy - server_cpu - generator["cpu_seconds"])
        / cfg.seconds,
    }
