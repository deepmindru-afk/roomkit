//! The load generator: G.711 every 20 ms for N calls, phases spread over
//! the frame like independent callers, and an exact account of the echo.
//!
//! Every packet carries its send time and call index in the payload (see
//! `stamp`), so the echo gives loss, round-trip time and per-call gaps
//! without trusting the system under test's own counters.

use std::io::{self, Write};
use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::sys::{self, Epoll, RecvBatch, SendBatch, Timer};
use crate::{rtp, stamp, Args, FRAME, PACKET_LEN, PCMU, SLOTS};

const SOCKET_BUFFER: libc::c_int = 4 << 20;
const TIMER: u64 = u64::MAX;
/// An inter-arrival gap a listener hears as a dropout: two frames missing.
const GAP: Duration = Duration::from_millis(40);

pub struct Config {
    target: Ipv4Addr,
    /// The ports the echo server bound: one per call, or one per server
    /// thread when calls share sockets. Call i goes to ports[i % len].
    ports: Vec<u16>,
    calls: usize,
    threads: usize,
    sockets: usize,
    cpus: Vec<usize>,
    start: f64,
    end: f64,
    grace: f64,
}

impl Config {
    pub fn from_args(args: &Args) -> Result<Self, String> {
        let ports_file: String = args.req("ports-file")?;
        let ports = std::fs::read_to_string(&ports_file)
            .map_err(|e| format!("{ports_file}: {e}"))?
            .lines()
            .map(|line| line.trim().parse().map_err(|_| format!("{ports_file}: bad port {line:?}")))
            .collect::<Result<Vec<u16>, String>>()?;
        let cfg = Self {
            target: args.get("target", Ipv4Addr::LOCALHOST)?,
            ports,
            calls: args.req("calls")?,
            threads: args.get("threads", 1)?,
            sockets: args.get("sockets", 8)?,
            cpus: args.cpus("cpus")?,
            start: args.req("start")?,
            end: args.req("end")?,
            grace: args.get("grace", 1.5)?,
        };
        if cfg.calls == 0 || cfg.threads == 0 || cfg.sockets == 0 {
            return Err("--calls, --threads and --sockets must be positive".into());
        }
        if cfg.ports.is_empty() {
            return Err("the ports file lists no port".into());
        }
        if cfg.end <= cfg.start {
            return Err("--end must come after --start".into());
        }
        Ok(cfg)
    }

    fn port_for(&self, call: usize) -> u16 {
        self.ports[call % self.ports.len()]
    }
}

struct Histogram {
    width_us: u32,
    buckets: Vec<u64>,
    count: u64,
    max_us: u32,
}

impl Histogram {
    fn new(width_us: u32, max_us: u32) -> Self {
        Self { width_us, buckets: vec![0; (max_us / width_us + 1) as usize], count: 0, max_us: 0 }
    }

    fn add(&mut self, us: u32) {
        let i = ((us / self.width_us) as usize).min(self.buckets.len() - 1);
        self.buckets[i] += 1;
        self.count += 1;
        self.max_us = self.max_us.max(us);
    }

    fn merge(&mut self, other: &Histogram) {
        self.buckets.iter_mut().zip(&other.buckets).for_each(|(a, b)| *a += b);
        self.count += other.count;
        self.max_us = self.max_us.max(other.max_us);
    }

    /// Upper edge of the bucket holding the p-th percentile, in ms.
    fn percentile_ms(&self, p: f64) -> Option<f64> {
        let target = ((p / 100.0) * self.count as f64).ceil().max(1.0) as u64;
        let mut seen = 0;
        for (i, &n) in self.buckets.iter().enumerate() {
            seen += n;
            if seen >= target && n > 0 {
                return Some(f64::from(self.width_us) * (i + 1) as f64 / 1000.0);
            }
        }
        None
    }

    fn json(&self, percentiles: &[(&str, f64)]) -> String {
        let mut parts: Vec<String> = percentiles
            .iter()
            .map(|(name, p)| {
                let value = self.percentile_ms(*p).map_or("null".into(), |v| format!("{v:.2}"));
                format!("\"{name}\": {value}")
            })
            .collect();
        parts.push(format!("\"max\": {:.2}", f64::from(self.max_us) / 1000.0));
        format!("{{{}}}", parts.join(", "))
    }
}

struct Report {
    sent: u64,
    send_dropped: u64,
    received: u64,
    corrupt: u64,
    foreign: u64,
    gaps: u64,
    missed_ticks: u64,
    recv_errors: u64,
    rtt: Histogram,
    lateness: Histogram,
}

impl Report {
    fn new() -> Self {
        Self {
            sent: 0,
            send_dropped: 0,
            received: 0,
            corrupt: 0,
            foreign: 0,
            gaps: 0,
            missed_ticks: 0,
            recv_errors: 0,
            rtt: Histogram::new(100, 10_000_000),
            lateness: Histogram::new(10, 100_000),
        }
    }

    fn merge(&mut self, o: &Report) {
        self.sent += o.sent;
        self.send_dropped += o.send_dropped;
        self.received += o.received;
        self.corrupt += o.corrupt;
        self.foreign += o.foreign;
        self.gaps += o.gaps;
        self.missed_ticks += o.missed_ticks;
        self.recv_errors += o.recv_errors;
        self.rtt.merge(&o.rtt);
        self.lateness.merge(&o.lateness);
    }
}

struct Caller {
    dest: libc::sockaddr_in,
    ssrc: u32,
    seq: u16,
    timestamp: u32,
    index: u32,
    socket: usize,
    last_rx: Option<Instant>,
}

/// What came back, apart from the I/O buffers so a received batch can be
/// read while callers are updated.
struct Ledger {
    callers: Vec<Caller>,
    report: Report,
    thread: usize,
    threads: usize,
    start_us: u32,
    end_us: u32,
}

impl Ledger {
    fn account(&mut self, data: &[u8], now: Instant, now_us: u32) {
        let Some(packet) = rtp::parse(data) else {
            if !rtp::is_rtcp(data) {
                self.report.corrupt += 1;
            }
            return;
        };
        let Some((sent_us, index)) = stamp::read(packet.payload) else {
            self.report.corrupt += 1; // concealment or a mangled payload
            return;
        };
        let local = index as usize / self.threads;
        if index as usize % self.threads != self.thread || local >= self.callers.len() {
            self.report.foreign += 1;
            return;
        }
        if sent_us < self.start_us || sent_us > self.end_us {
            return; // sent during warm-up or after the window
        }
        self.report.received += 1;
        self.report.rtt.add(now_us.wrapping_sub(sent_us));
        let caller = &mut self.callers[local];
        if caller.last_rx.is_some_and(|prev| now.duration_since(prev) > GAP) {
            self.report.gaps += 1;
        }
        caller.last_rx = Some(now);
    }
}

struct Generator {
    sockets: Vec<UdpSocket>,
    batches: Vec<SendBatch>,
    rx: RecvBatch,
    packet: [u8; PACKET_LEN],
    ledger: Ledger,
    base: Instant,
    start: Instant,
    end: Instant,
}

impl Generator {
    fn us(&self, t: Instant) -> u32 {
        t.saturating_duration_since(self.base).as_micros() as u32
    }

    fn run(mut self, grace: Duration) -> io::Result<Report> {
        let mut epoll = Epoll::new()?;
        let timer = Timer::periodic(Duration::from_millis(1))?;
        let first_tick = Instant::now() + Duration::from_millis(1);
        epoll.add(timer.fd(), TIMER)?;
        for (i, sock) in self.sockets.iter().enumerate() {
            epoll.add(sock.as_raw_fd(), i as u64)?;
        }
        let stop = self.end + grace;
        let mut ready = Vec::with_capacity(64);
        let mut tick: u64 = 0;
        loop {
            epoll.wait(&mut ready)?;
            let now = Instant::now();
            if now >= stop {
                return Ok(self.ledger.report);
            }
            for &token in &ready {
                if token != TIMER {
                    self.drain(token as usize);
                    continue;
                }
                let due = timer.expirations();
                let in_window = now >= self.start && now <= self.end;
                if in_window {
                    self.ledger.report.missed_ticks += due.saturating_sub(1);
                }
                // Sending goes on through the grace period, uncounted, so
                // audio a system under test still buffers at the end of the
                // window is pushed out instead of counted as lost.
                for _ in 0..due {
                    if in_window {
                        let scheduled = first_tick + Duration::from_millis(tick);
                        let late = now.saturating_duration_since(scheduled);
                        self.ledger.report.lateness.add(late.as_micros() as u32);
                    }
                    self.send_slot(tick as usize % SLOTS, now, in_window);
                    tick += 1;
                }
            }
        }
    }

    fn send_slot(&mut self, slot: usize, now: Instant, in_window: bool) {
        let sent_us = self.us(now);
        for c in (slot..self.ledger.callers.len()).step_by(SLOTS) {
            let caller = &mut self.ledger.callers[c];
            rtp::write_header(
                &mut self.packet[..rtp::HEADER_LEN],
                PCMU,
                caller.seq,
                caller.timestamp,
                caller.ssrc,
            );
            let payload: &mut [u8; FRAME] =
                (&mut self.packet[rtp::HEADER_LEN..]).try_into().expect("one frame");
            stamp::write(payload, sent_us, caller.index);
            caller.seq = caller.seq.wrapping_add(1);
            caller.timestamp = caller.timestamp.wrapping_add(FRAME as u32);
            let (b, dest) = (caller.socket, caller.dest);
            self.batches[b].push(&dest, &self.packet);
            if self.batches[b].is_full() {
                self.flush(b, in_window);
            }
        }
        for b in 0..self.batches.len() {
            self.flush(b, in_window);
        }
    }

    fn flush(&mut self, b: usize, in_window: bool) {
        let (sent, dropped) = self.batches[b].flush(self.sockets[b].as_raw_fd());
        if in_window {
            self.ledger.report.sent += sent;
            self.ledger.report.send_dropped += dropped;
        }
    }

    fn drain(&mut self, socket: usize) {
        let fd = self.sockets[socket].as_raw_fd();
        loop {
            let n = match self.rx.recv(fd) {
                Ok(n) => n,
                Err(_) => {
                    self.ledger.report.recv_errors += 1;
                    return;
                }
            };
            let now = Instant::now();
            let now_us = self.us(now);
            for i in 0..n {
                let (data, _) = self.rx.get(i);
                self.ledger.account(data, now, now_us);
            }
            if n < self.rx.capacity() {
                return;
            }
        }
    }
}

fn generate(thread: usize, cfg: &Config, base: Instant) -> io::Result<Report> {
    if !cfg.cpus.is_empty() {
        let _ = sys::pin_current_thread(cfg.cpus[thread % cfg.cpus.len()]);
    }
    let sockets: Vec<UdpSocket> = (0..cfg.sockets)
        .map(|_| sys::udp_socket(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0), SOCKET_BUFFER))
        .collect::<io::Result<_>>()?;
    let callers: Vec<Caller> = (thread..cfg.calls)
        .step_by(cfg.threads)
        .enumerate()
        .map(|(local, i)| Caller {
            dest: sys::sockaddr(SocketAddrV4::new(cfg.target, cfg.port_for(i))),
            ssrc: 0x1000_0000 | i as u32,
            seq: 0,
            timestamp: 0,
            index: i as u32,
            socket: local % cfg.sockets,
            last_rx: None,
        })
        .collect();
    let (start, end) = (sys::instant_at(cfg.start), sys::instant_at(cfg.end));
    let us = |t: Instant| t.saturating_duration_since(base).as_micros() as u32;
    let generator = Generator {
        batches: sockets.iter().map(|_| SendBatch::new(256)).collect(),
        sockets,
        rx: RecvBatch::new(64),
        packet: [0; PACKET_LEN],
        ledger: Ledger {
            callers,
            report: Report::new(),
            thread,
            threads: cfg.threads,
            start_us: us(start),
            end_us: us(end),
        },
        base,
        start,
        end,
    };
    generator.run(Duration::from_secs_f64(cfg.grace))
}

pub fn run(cfg: Config) -> Result<(), String> {
    sys::raise_nofile_limit();
    let base = Instant::now();
    let cfg = Arc::new(cfg);
    let handles = (0..cfg.threads)
        .map(|t| {
            let cfg = cfg.clone();
            thread::Builder::new()
                .name(format!("load-{t}"))
                .spawn(move || generate(t, &cfg, base))
        })
        .collect::<io::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?;
    sys::sleep_until(sys::instant_at(cfg.start));
    let cpu0 = sys::process_cpu_seconds();
    sys::sleep_until(sys::instant_at(cfg.end));
    let cpu1 = sys::process_cpu_seconds();
    let mut total = Report::new();
    for handle in handles {
        let report = handle.join().map_err(|_| "a generator thread panicked")?;
        total.merge(&report.map_err(|e| e.to_string())?);
    }
    let lost = total.sent.saturating_sub(total.received);
    let loss_pct = if total.sent > 0 { 100.0 * lost as f64 / total.sent as f64 } else { 0.0 };
    println!(
        "{{\"calls\": {}, \"sent\": {}, \"received\": {}, \"lost\": {lost}, \"loss_pct\": {loss_pct:.4}, \
         \"send_dropped\": {}, \"corrupt\": {}, \"foreign\": {}, \"gaps_40ms\": {}, \
         \"missed_ticks\": {}, \"recv_errors\": {}, \"rtt_ms\": {}, \"send_lateness_ms\": {}, \
         \"cpu_seconds\": {:.4}, \"window_seconds\": {:.4}, \"threads\": {}}}",
        cfg.calls,
        total.sent,
        total.received,
        total.send_dropped,
        total.corrupt,
        total.foreign,
        total.gaps,
        total.missed_ticks,
        total.recv_errors,
        total.rtt.json(&[("p50", 50.0), ("p99", 99.0), ("p999", 99.9)]),
        total.lateness.json(&[("p50", 50.0), ("p99", 99.0)]),
        cpu1 - cpu0,
        cfg.end - cfg.start,
        cfg.threads,
    );
    io::stdout().flush().map_err(|e| e.to_string())
}
