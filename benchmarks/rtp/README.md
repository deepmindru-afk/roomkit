# RTP media-path benchmark

How many G.711 calls can one core carry, and what does Python cost in the
packet path? This suite measures the RTP layer alone, the work a RoomKit
SIP call does before any voice pipeline stage, on two systems under test
with one instrument:

| System under test | What runs per call |
|---|---|
| `aiortp` | One `aiortp.RTPSession` per call in one asyncio process: parse, jitter buffer, decode, re-encode, send. Optional `--playout` (adaptive playout clock per call) and `--uvloop`. |
| `native` | `rtp-bench echo`, a Rust engine with a media server's hot path: parse, jitter buffer, decode on receive; one 1 ms clock per thread plays each call once per 20 ms (encode, send). `--mode percall` keeps one socket per call; `--mode shared` puts a thread's calls on one socket, demultiplexed by SSRC, so receives and sends batch through `recvmmsg`/`sendmmsg`. |

Both echo the audio back. The generator, `rtp-bench load`, sends G.711
every 20 ms per call, with call phases spread over the frame like
independent callers. Each payload carries its send time and call index in
bytes that survive a µ-law decode and re-encode. The echo therefore gives
loss, round-trip time and per-call gaps without trusting the counters of
the system under test.

What RoomKit adds per frame past RTP (audio pipeline, VAD, STT hand-off)
is measured by `benchmarks/voice`.

Linux only (epoll, timerfd, `recvmmsg`/`sendmmsg`). Needs a Rust toolchain
(`cargo`); the driver builds the native binary on each run.

```bash
# One point list on the native engine, one socket per call.
uv run --extra rtp python -m benchmarks.rtp --server native --calls 1000,4000

# Largest passing count for aiortp, results kept.
uv run --extra rtp python -m benchmarks.rtp --server aiortp --find-max \
  --output benchmark-results/rtp-aiortp

# Same, with uvloop (not a project dependency).
uv run --extra rtp --with uvloop python -m benchmarks.rtp --server aiortp --uvloop --find-max

# Native engine, shared socket, more generator threads for high counts.
uv run --extra rtp python -m benchmarks.rtp --server native --mode shared \
  --find-max --start-calls 1000 --gen-threads 12

# Native unit tests (codec, RTP parsing, jitter buffer, payload stamp).
cargo test --release --manifest-path benchmarks/rtp/native/Cargo.toml
```

The server binds its ports from `--base-port` up, skipping any that
another program holds, as a media server's port pool does, and lists them
in a ports file the generator reads. The server is pinned to
`--server-cpus` (default CPU 2). The generator gets
every other CPU except the server core's hyperthread siblings, unless
`--gen-cpus` says otherwise. Outputs go to a fresh `--output` directory:
`results.json` (environment, configuration, every point) and `summary.md`.
Both are gitignored under `benchmark-results/`.

## Reading a point

Each point runs a fresh server, then 3 s of warm-up, a 15 s window and a
1.5 s grace period. The generator keeps sending through the grace period,
uncounted, so audio still buffered at the end of the window comes back
instead of being counted as lost.

- **Verdict `ok`**: loss ≤ 0.1 %, RTT p99 − p50 ≤ 40 ms (jitter), and
  RTT p50 no more than 20 ms above the floor (queueing). The floor is the
  median RTT of the first point, so start low: `--find-max` starts at
  `--start-calls`, and `--calls` should list the lightest count first.
  Both limits measure what load adds to the engine's own delay. aiortp's
  jitter prefetch holds ~80 ms and the native engine's clock up to 20 ms;
  neither is penalised for buffering on purpose. A saturated engine keeps
  every packet and delivers it late, so the queueing limit is what
  usually fails first.
- **Verdict `invalid`**: the point cannot be blamed on the system under
  test alone. Either the generator sent late (send lateness p99 > 2 ms,
  which counts the 1 ms ticks it woke up too late for), dropped sends, or
  ran its threads above 80 % CPU. Or other processes took more than 10 % of
  the server's own CPUs, or kept more than two cores busy across the
  machine; that load shares the cache and the turbo budget. Such a point runs again after `--retry-wait` seconds
  (default 20), up to `--retries` (default 2) more times. If it is still
  invalid, `--find-max` stops there: add `--gen-threads` or CPUs, or quiet
  the machine.
- **Other load**: CPU time other processes spent on the server's own CPUs
  during the window (`/proc/stat` user, nice, system and steal, minus the
  server's own time), as a share of one core. `results.json` also keeps
  the machine-wide figure.
- **µs/packet**: server CPU time (user + system, all threads) divided by
  the packets the generator sent and got back in the window. The driver
  reads the server's CPU from `/proc/<pid>/stat` at the window's edges: a
  saturated asyncio server wakes its own measuring code seconds late and
  would under-report. This is the number to compare across engines.
- **Kernel drops**: `RcvbufErrors + SndbufErrors + MemErrors` from
  `/proc/net/snmp`, system-wide over the point.

## Limits of a loopback run

- Over loopback, the kernel's delivery work runs in the *sender's* system
  call. The server pays for delivering its echoes and the generator pays
  for delivering its sends. On a real NIC, receive processing (softirq)
  lands on whichever core handles the interrupt. A loopback µs/packet is
  therefore a floor: measure the target on two machines before quoting
  capacity.
- Saturation shows up as latency before loss. Kernel socket queues absorb
  a slow consumer for seconds, so the RTT spread fails first.
- The machine is shared by whatever else runs on it. Disturbed points
  are detected and retried (above), but compare runs from one machine
  only, never absolutes across machines.
- Hybrid CPUs (performance and efficiency cores) need the server pinned to
  a performance core. The default CPU 2 is one on the reference machine,
  but check yours.
