//! Native side of the RoomKit RTP benchmark.
//!
//! `rtp-bench echo` is a native RTP echo engine, the system under test that
//! answers "how many calls can one core carry without Python in the packet
//! path". `rtp-bench load` is the load generator every system under test is
//! measured with, aiortp included. Both are driven by `python -m
//! benchmarks.rtp`; see benchmarks/rtp/README.md.

mod echo;
mod g711;
mod jitter;
mod load;
mod rtp;
mod stamp;
mod sys;

use std::collections::HashMap;
use std::process::ExitCode;
use std::str::FromStr;

/// One G.711 frame: 20 ms at 8 kHz.
pub const FRAME: usize = 160;
pub const PACKET_LEN: usize = rtp::HEADER_LEN + FRAME;
/// Both sides tick every 1 ms and serve one twentieth of their calls per
/// tick, so each call sends once per 20 ms with phases spread like
/// independent callers instead of one burst.
pub const SLOTS: usize = 20;
pub const PCMU: u8 = 0;
/// Each call owns a port pair (RTP even, RTCP odd), as SDP offers do.
pub const PORT_STRIDE: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// One socket per call, the classic SIP media layout.
    PerCall,
    /// One socket per thread; calls are told apart by SSRC.
    Shared,
}

impl FromStr for Mode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "percall" => Ok(Mode::PerCall),
            "shared" => Ok(Mode::Shared),
            _ => Err(format!("unknown mode {s:?} (percall, shared)")),
        }
    }
}

/// `--key value` options.
pub struct Args(HashMap<String, String>);

impl Args {
    fn parse(argv: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut map = HashMap::new();
        let mut argv = argv.peekable();
        while let Some(key) = argv.next() {
            let name = key.strip_prefix("--").ok_or(format!("unexpected argument {key:?}"))?;
            let value = argv.next().ok_or(format!("{key} needs a value"))?;
            map.insert(name.to_string(), value);
        }
        Ok(Self(map))
    }

    pub fn get<T: FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.0.get(key) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{key}: invalid value {v:?}")),
        }
    }

    pub fn req<T: FromStr>(&self, key: &str) -> Result<T, String> {
        let v = self.0.get(key).ok_or(format!("--{key} is required"))?;
        v.parse().map_err(|_| format!("--{key}: invalid value {v:?}"))
    }

    /// A CPU list such as `2`, `4-7` or `2,8-11`; empty when absent.
    pub fn cpus(&self, key: &str) -> Result<Vec<usize>, String> {
        let Some(spec) = self.0.get(key) else {
            return Ok(Vec::new());
        };
        let bad = || format!("--{key}: invalid CPU list {spec:?}");
        let mut cpus = Vec::new();
        for part in spec.split(',') {
            match part.split_once('-') {
                Some((a, b)) => {
                    let (a, b): (usize, usize) =
                        (a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?);
                    cpus.extend(a..=b);
                }
                None => cpus.push(part.parse().map_err(|_| bad())?),
            }
        }
        Ok(cpus)
    }
}

fn main() -> ExitCode {
    let mut argv = std::env::args().skip(1);
    let command = argv.next();
    let result = Args::parse(argv).and_then(|args| match command.as_deref() {
        Some("echo") => echo::Config::from_args(&args).and_then(echo::run),
        Some("load") => load::Config::from_args(&args).and_then(load::run),
        _ => Err("usage: rtp-bench echo|load --option value ...".to_string()),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("rtp-bench: {err}");
            ExitCode::FAILURE
        }
    }
}
