//! The native system under test: an RTP echo engine shaped like a media
//! server's hot path. Per inbound packet: parse, jitter buffer, G.711
//! decode. One 1 ms clock per thread plays every call once per 20 ms
//! (G.711 encode, send), batched with sendmmsg when calls share a socket.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{self, BufRead, Write};
use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use crate::g711::Ulaw;
use crate::jitter::Jitter;
use crate::sys::{self, Epoll, RecvBatch, SendBatch, Timer};
use crate::{rtp, Args, Mode, FRAME, PACKET_LEN, PCMU, PORT_STRIDE, SLOTS};

/// Frames a call may queue before the oldest is dropped (160 ms).
const MAX_DEPTH: usize = 8;
const SOCKET_BUFFER: libc::c_int = 4 << 20;
const TIMER: u64 = u64::MAX;

const STATS: [&str; 9] = [
    "rx",
    "tx",
    "late",
    "concealed",
    "overflow",
    "malformed",
    "send_dropped",
    "recv_errors",
    "missed_ticks",
];
const RX: usize = 0;
const TX: usize = 1;
const LATE: usize = 2;
const CONCEALED: usize = 3;
const OVERFLOW: usize = 4;
const MALFORMED: usize = 5;
const SEND_DROPPED: usize = 6;
const RECV_ERRORS: usize = 7;
const MISSED_TICKS: usize = 8;

type Stats = [u64; STATS.len()];

pub struct Config {
    bind: Ipv4Addr,
    base_port: u16,
    calls: usize,
    mode: Mode,
    threads: usize,
    cpus: Vec<usize>,
    ports_file: PathBuf,
}

impl Config {
    pub fn from_args(args: &Args) -> Result<Self, String> {
        let cfg = Self {
            bind: args.get("bind", Ipv4Addr::LOCALHOST)?,
            base_port: args.get("base-port", 20000)?,
            calls: args.get("calls", 0)?,
            mode: args.get("mode", Mode::PerCall)?,
            threads: args.get("threads", 1)?,
            cpus: args.cpus("cpus")?,
            ports_file: args.req("ports-file")?,
        };
        if cfg.threads == 0 {
            return Err("--threads must be at least 1".into());
        }
        Ok(cfg)
    }
}

/// Counters a worker publishes once per tick for the measuring thread.
#[derive(Default)]
struct Published([AtomicU64; STATS.len()]);

impl Published {
    fn store(&self, stats: &Stats) {
        for (cell, value) in self.0.iter().zip(stats) {
            cell.store(*value, Relaxed);
        }
    }

    fn load(&self) -> Stats {
        let mut out = [0; STATS.len()];
        for (value, cell) in out.iter_mut().zip(&self.0) {
            *value = cell.load(Relaxed);
        }
        out
    }
}

/// SSRCs are random already: a multiplicative mix is enough.
#[derive(Default)]
struct SsrcHasher(u64);

impl Hasher for SsrcHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8 | u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.0 = u64::from(v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

struct Call {
    jitter: Jitter,
    peer: Option<libc::sockaddr_in>,
    socket: usize,
    seq: u16,
    timestamp: u32,
    ssrc: u32,
}

impl Call {
    fn new(index: usize, socket: usize) -> Self {
        let mix = (index as u32).wrapping_mul(2_654_435_761);
        Self {
            jitter: Jitter::new(),
            peer: None,
            socket,
            seq: mix as u16,
            timestamp: mix,
            ssrc: 0x5EED_0000 ^ mix,
        }
    }
}

/// Call state and the work done per packet, apart from the I/O buffers so
/// a received batch can be read while calls are updated.
struct Calls {
    ulaw: Arc<Ulaw>,
    calls: Vec<Call>,
    by_ssrc: HashMap<u32, usize, BuildHasherDefault<SsrcHasher>>,
    mode: Mode,
    stats: Stats,
}

impl Calls {
    fn handle(&mut self, data: &[u8], from: &libc::sockaddr_in, socket: usize) {
        let Some(packet) = rtp::parse(data) else {
            if !rtp::is_rtcp(data) {
                self.stats[MALFORMED] += 1;
            }
            return;
        };
        if packet.payload.len() != FRAME {
            self.stats[MALFORMED] += 1;
            return;
        }
        let index = match self.mode {
            Mode::PerCall => socket,
            Mode::Shared => self.call_for(packet.ssrc, socket),
        };
        let call = &mut self.calls[index];
        call.peer = Some(*from); // symmetric RTP: answer where the media comes from
        self.stats[RX] += 1;
        match call.jitter.push(packet.seq) {
            Some(slot) => self.ulaw.decode_into(packet.payload, slot),
            None => self.stats[LATE] += 1,
        }
        while call.jitter.depth() > MAX_DEPTH {
            let _ = call.jitter.pop();
            self.stats[OVERFLOW] += 1;
        }
    }

    fn call_for(&mut self, ssrc: u32, socket: usize) -> usize {
        if let Some(&index) = self.by_ssrc.get(&ssrc) {
            return index;
        }
        let index = self.calls.len();
        self.calls.push(Call::new(index, socket));
        self.by_ssrc.insert(ssrc, index);
        index
    }

    /// Builds the call's next outbound packet; None when it has nothing to play.
    fn play(&mut self, index: usize, packet: &mut [u8; PACKET_LEN]) -> Option<libc::sockaddr_in> {
        let call = &mut self.calls[index];
        let peer = call.peer?;
        let (frame, skipped) = call.jitter.pop();
        self.stats[CONCEALED] += u64::from(skipped);
        let pcm = frame?;
        rtp::write_header(&mut packet[..rtp::HEADER_LEN], PCMU, call.seq, call.timestamp, call.ssrc);
        self.ulaw.encode_into(pcm, &mut packet[rtp::HEADER_LEN..]);
        call.seq = call.seq.wrapping_add(1);
        call.timestamp = call.timestamp.wrapping_add(FRAME as u32);
        Some(peer)
    }
}

struct Worker {
    sockets: Vec<UdpSocket>,
    rx: RecvBatch,
    tx: SendBatch,
    packet: [u8; PACKET_LEN],
    state: Calls,
    published: Arc<Published>,
}

impl Worker {
    fn run(mut self) -> io::Result<()> {
        let mut epoll = Epoll::new()?;
        let timer = Timer::periodic(Duration::from_millis(1))?;
        epoll.add(timer.fd(), TIMER)?;
        for (i, sock) in self.sockets.iter().enumerate() {
            epoll.add(sock.as_raw_fd(), i as u64)?;
        }
        let mut ready = Vec::with_capacity(1024);
        let mut slot = 0;
        loop {
            epoll.wait(&mut ready)?;
            for &token in &ready {
                if token != TIMER {
                    self.drain(token as usize);
                    continue;
                }
                let due = timer.expirations();
                self.state.stats[MISSED_TICKS] += due.saturating_sub(1);
                for _ in 0..due.min(SLOTS as u64) {
                    self.tick(slot);
                    slot = (slot + 1) % SLOTS;
                }
                self.published.store(&self.state.stats);
            }
        }
    }

    fn drain(&mut self, socket: usize) {
        let fd = self.sockets[socket].as_raw_fd();
        loop {
            let n = match self.rx.recv(fd) {
                Ok(n) => n,
                Err(_) => {
                    self.state.stats[RECV_ERRORS] += 1;
                    return;
                }
            };
            for i in 0..n {
                let (data, from) = self.rx.get(i);
                self.state.handle(data, from, socket);
            }
            if n < self.rx.capacity() {
                return;
            }
        }
    }

    /// Plays the calls of one slot; new calls learned meanwhile wait a round.
    fn tick(&mut self, slot: usize) {
        for index in (slot..self.state.calls.len()).step_by(SLOTS) {
            let Some(peer) = self.state.play(index, &mut self.packet) else {
                continue;
            };
            let socket = self.state.calls[index].socket;
            match self.state.mode {
                Mode::PerCall => {
                    let fd = self.sockets[socket].as_raw_fd();
                    let sent = sys::send_to(fd, &self.packet, &peer);
                    self.state.stats[if sent { TX } else { SEND_DROPPED }] += 1;
                }
                Mode::Shared => {
                    self.tx.push(&peer, &self.packet);
                    if self.tx.is_full() {
                        self.flush();
                        // Keep reading between bursts so the receive queue
                        // never waits a whole tick behind a large send.
                        self.drain(0);
                    }
                }
            }
        }
        if self.state.mode == Mode::Shared {
            self.flush();
        }
    }

    fn flush(&mut self) {
        let (sent, dropped) = self.tx.flush(self.sockets[0].as_raw_fd());
        self.state.stats[TX] += sent;
        self.state.stats[SEND_DROPPED] += dropped;
    }
}

/// Binds one port per call (or per thread when shared) from `base_port`
/// up, skipping ports another program holds, as a media server's port pool
/// does; the generator reads the result from the ports file.
fn allocate_ports(cfg: &Config) -> Result<Vec<(u16, UdpSocket)>, String> {
    let wanted = match cfg.mode {
        Mode::PerCall => cfg.calls,
        Mode::Shared => cfg.threads,
    };
    let mut bound = Vec::with_capacity(wanted);
    let mut port = cfg.base_port as usize;
    while bound.len() < wanted {
        if port > u16::MAX as usize {
            return Err(format!("only {} free ports from {}", bound.len(), cfg.base_port));
        }
        match sys::udp_socket(SocketAddrV4::new(cfg.bind, port as u16), SOCKET_BUFFER) {
            Ok(sock) => bound.push((port as u16, sock)),
            Err(e) if e.kind() == io::ErrorKind::AddrInUse => {}
            Err(e) => return Err(format!("bind {}:{port}: {e}", cfg.bind)),
        }
        port += PORT_STRIDE;
    }
    let list: String = bound.iter().map(|(port, _)| format!("{port}\n")).collect();
    std::fs::write(&cfg.ports_file, list)
        .map_err(|e| format!("{}: {e}", cfg.ports_file.display()))?;
    Ok(bound)
}

pub fn run(cfg: Config) -> Result<(), String> {
    sys::raise_nofile_limit();
    let ulaw = Arc::new(Ulaw::new());
    let (failed_tx, failed_rx) = mpsc::channel::<String>();
    let mut published = Vec::new();
    let mut per_thread: Vec<Vec<UdpSocket>> = (0..cfg.threads).map(|_| Vec::new()).collect();
    for (i, (_, sock)) in allocate_ports(&cfg)?.into_iter().enumerate() {
        per_thread[i % cfg.threads].push(sock);
    }
    for (t, sockets) in per_thread.into_iter().enumerate() {
        let cpu = (!cfg.cpus.is_empty()).then(|| cfg.cpus[t % cfg.cpus.len()]);
        let counters = Arc::new(Published::default());
        published.push(counters.clone());
        let (ulaw, mode, failed) = (ulaw.clone(), cfg.mode, failed_tx.clone());
        let calls: Vec<Call> = match mode {
            Mode::PerCall => (0..sockets.len()).map(|i| Call::new(i, i)).collect(),
            Mode::Shared => Vec::new(),
        };
        thread::Builder::new()
            .name(format!("echo-{t}"))
            .spawn(move || {
                if let Some(cpu) = cpu {
                    let _ = sys::pin_current_thread(cpu);
                }
                let worker = Worker {
                    sockets,
                    rx: RecvBatch::new(64),
                    tx: SendBatch::new(256),
                    packet: [0; PACKET_LEN],
                    state: Calls {
                        ulaw,
                        calls,
                        by_ssrc: HashMap::default(),
                        mode,
                        stats: [0; STATS.len()],
                    },
                    published: counters,
                };
                if let Err(e) = worker.run() {
                    let _ = failed.send(format!("echo-{t}: {e}"));
                }
            })
            .map_err(|e| e.to_string())?;
    }
    println!("READY");
    io::stdout().flush().map_err(|e| e.to_string())?;
    measure(&published)?;
    // Keep echoing in-flight packets until the driver stops the process.
    match failed_rx.recv() {
        Ok(err) => Err(err),
        Err(_) => Ok(()),
    }
}

/// Reads the window `<start> <end>` (epoch seconds) from stdin and prints
/// what the engine did inside it as one JSON line.
fn measure(published: &[Arc<Published>]) -> Result<(), String> {
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line).map_err(|e| e.to_string())?;
    let window: Vec<f64> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    let &[start, end] = window.as_slice() else {
        return Err(format!("expected '<start> <end>' on stdin, got {line:?}"));
    };
    let total = || {
        published.iter().map(|p| p.load()).fold([0u64; STATS.len()], |mut acc, s| {
            acc.iter_mut().zip(s).for_each(|(a, v)| *a += v);
            acc
        })
    };
    sys::sleep_until(sys::instant_at(start));
    let (cpu0, s0) = (sys::process_cpu_seconds(), total());
    sys::sleep_until(sys::instant_at(end));
    let (cpu1, s1) = (sys::process_cpu_seconds(), total());
    let counts: Vec<String> =
        STATS.iter().enumerate().map(|(i, name)| format!("\"{name}\": {}", s1[i] - s0[i])).collect();
    println!(
        "{{\"cpu_seconds\": {:.4}, \"window_seconds\": {:.4}, \"threads\": {}, \"rss_mb\": {:.1}, {}}}",
        cpu1 - cpu0,
        end - start,
        published.len(),
        sys::max_rss_mb(),
        counts.join(", ")
    );
    io::stdout().flush().map_err(|e| e.to_string())
}
