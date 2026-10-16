//! Thin wrappers over the Linux calls the benchmark relies on: epoll,
//! timerfd, recvmmsg/sendmmsg, CPU affinity and getrusage.

use std::io;
use std::mem::{size_of, zeroed};
use std::net::{SocketAddrV4, UdpSocket};
use std::os::fd::{AsRawFd, RawFd};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Largest datagram either side reads.
const RECV_SLOT: usize = 1500;
/// Largest datagram either side sends: one RTP header and one G.711 frame fit.
const SEND_SLOT: usize = 256;

fn check(ret: libc::c_int) -> io::Result<libc::c_int> {
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret)
    }
}

/// Binds a non-blocking UDP socket with enlarged kernel buffers.
pub fn udp_socket(addr: SocketAddrV4, buffer_bytes: libc::c_int) -> io::Result<UdpSocket> {
    let sock = UdpSocket::bind(addr)?;
    sock.set_nonblocking(true)?;
    for opt in [libc::SO_RCVBUF, libc::SO_SNDBUF] {
        // SAFETY: valid descriptor; the value pointer and length describe a live c_int.
        check(unsafe {
            libc::setsockopt(
                sock.as_raw_fd(),
                libc::SOL_SOCKET,
                opt,
                (&buffer_bytes as *const libc::c_int).cast(),
                size_of::<libc::c_int>() as libc::socklen_t,
            )
        })?;
    }
    Ok(sock)
}

pub fn sockaddr(addr: SocketAddrV4) -> libc::sockaddr_in {
    // SAFETY: sockaddr_in is plain data and all-zero is a valid value.
    let mut sa: libc::sockaddr_in = unsafe { zeroed() };
    sa.sin_family = libc::AF_INET as libc::sa_family_t;
    sa.sin_port = addr.port().to_be();
    sa.sin_addr.s_addr = u32::from(*addr.ip()).to_be();
    sa
}

pub fn send_to(fd: RawFd, packet: &[u8], to: &libc::sockaddr_in) -> bool {
    // SAFETY: the buffer and address pointers are valid for the given lengths.
    let sent = unsafe {
        libc::sendto(
            fd,
            packet.as_ptr().cast(),
            packet.len(),
            0,
            (to as *const libc::sockaddr_in).cast(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        )
    };
    sent >= 0
}

pub struct Epoll {
    fd: RawFd,
    events: Vec<libc::epoll_event>,
}

impl Epoll {
    pub fn new() -> io::Result<Self> {
        // SAFETY: plain syscall; the result is checked.
        let fd = check(unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) })?;
        Ok(Self { fd, events: vec![libc::epoll_event { events: 0, u64: 0 }; 1024] })
    }

    pub fn add(&self, fd: RawFd, token: u64) -> io::Result<()> {
        let mut ev = libc::epoll_event { events: libc::EPOLLIN as u32, u64: token };
        // SAFETY: both descriptors are open and ev outlives the call.
        check(unsafe { libc::epoll_ctl(self.fd, libc::EPOLL_CTL_ADD, fd, &mut ev) })?;
        Ok(())
    }

    /// Blocks until a descriptor is readable; fills `ready` with their tokens.
    pub fn wait(&mut self, ready: &mut Vec<u64>) -> io::Result<()> {
        ready.clear();
        // SAFETY: the events buffer is valid for its full length.
        let n = unsafe {
            libc::epoll_wait(self.fd, self.events.as_mut_ptr(), self.events.len() as libc::c_int, -1)
        };
        if n < 0 {
            let err = io::Error::last_os_error();
            return if err.kind() == io::ErrorKind::Interrupted { Ok(()) } else { Err(err) };
        }
        ready.extend(self.events[..n as usize].iter().map(|e| e.u64));
        Ok(())
    }
}

impl Drop for Epoll {
    fn drop(&mut self) {
        // SAFETY: the descriptor is owned and closed once.
        unsafe { libc::close(self.fd) };
    }
}

/// A periodic monotonic timer readable through epoll.
pub struct Timer {
    fd: RawFd,
}

impl Timer {
    pub fn periodic(period: Duration) -> io::Result<Self> {
        // SAFETY: plain syscall; the result is checked.
        let fd = check(unsafe {
            libc::timerfd_create(libc::CLOCK_MONOTONIC, libc::TFD_NONBLOCK | libc::TFD_CLOEXEC)
        })?;
        // SAFETY: itimerspec is plain data and all-zero is a valid value.
        let mut spec: libc::itimerspec = unsafe { zeroed() };
        spec.it_interval.tv_sec = period.as_secs() as libc::time_t;
        spec.it_interval.tv_nsec = period.subsec_nanos() as libc::c_long;
        spec.it_value = spec.it_interval;
        // SAFETY: fd is the timer just created; spec outlives the call.
        check(unsafe { libc::timerfd_settime(fd, 0, &spec, std::ptr::null_mut()) })?;
        Ok(Self { fd })
    }

    pub fn fd(&self) -> RawFd {
        self.fd
    }

    /// Expirations since the last read; 0 when none is pending.
    pub fn expirations(&self) -> u64 {
        let mut count: u64 = 0;
        // SAFETY: reads exactly 8 bytes into a live u64.
        let n = unsafe { libc::read(self.fd, (&mut count as *mut u64).cast(), size_of::<u64>()) };
        if n == size_of::<u64>() as isize {
            count
        } else {
            0
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        // SAFETY: the descriptor is owned and closed once.
        unsafe { libc::close(self.fd) };
    }
}

/// Preallocated recvmmsg buffers: one syscall reads up to `capacity` datagrams.
///
/// The headers point into the boxed slices, whose heap storage never moves.
pub struct RecvBatch {
    bufs: Box<[[u8; RECV_SLOT]]>,
    addrs: Box<[libc::sockaddr_in]>,
    _iovs: Box<[libc::iovec]>,
    hdrs: Box<[libc::mmsghdr]>,
}

impl RecvBatch {
    pub fn new(capacity: usize) -> Self {
        let mut bufs = vec![[0u8; RECV_SLOT]; capacity].into_boxed_slice();
        // SAFETY: sockaddr_in is plain data and all-zero is a valid value.
        let mut addrs = vec![unsafe { zeroed::<libc::sockaddr_in>() }; capacity].into_boxed_slice();
        let mut iovs: Box<[libc::iovec]> = bufs
            .iter_mut()
            .map(|b| libc::iovec { iov_base: b.as_mut_ptr().cast(), iov_len: RECV_SLOT })
            .collect();
        let hdrs = iovs
            .iter_mut()
            .zip(addrs.iter_mut())
            .map(|(iov, addr)| {
                // SAFETY: mmsghdr is plain data and all-zero is a valid value.
                let mut h: libc::mmsghdr = unsafe { zeroed() };
                h.msg_hdr.msg_name = (addr as *mut libc::sockaddr_in).cast();
                h.msg_hdr.msg_iov = iov;
                h.msg_hdr.msg_iovlen = 1;
                h
            })
            .collect();
        Self { bufs, addrs, _iovs: iovs, hdrs }
    }

    pub fn capacity(&self) -> usize {
        self.hdrs.len()
    }

    /// Reads what is queued without blocking; returns the datagram count.
    pub fn recv(&mut self, fd: RawFd) -> io::Result<usize> {
        for h in self.hdrs.iter_mut() {
            h.msg_hdr.msg_namelen = size_of::<libc::sockaddr_in>() as libc::socklen_t;
        }
        // SAFETY: every header points into buffers owned by self.
        let n = unsafe {
            libc::recvmmsg(
                fd,
                self.hdrs.as_mut_ptr(),
                self.hdrs.len() as libc::c_uint,
                libc::MSG_DONTWAIT,
                std::ptr::null_mut(),
            )
        };
        if n >= 0 {
            return Ok(n as usize);
        }
        let err = io::Error::last_os_error();
        if err.kind() == io::ErrorKind::WouldBlock {
            Ok(0)
        } else {
            Err(err)
        }
    }

    pub fn get(&self, i: usize) -> (&[u8], &libc::sockaddr_in) {
        (&self.bufs[i][..self.hdrs[i].msg_len as usize], &self.addrs[i])
    }
}

/// Preallocated sendmmsg buffers: datagrams queue, one syscall sends them all.
pub struct SendBatch {
    bufs: Box<[[u8; SEND_SLOT]]>,
    addrs: Box<[libc::sockaddr_in]>,
    iovs: Box<[libc::iovec]>,
    hdrs: Box<[libc::mmsghdr]>,
    len: usize,
}

impl SendBatch {
    pub fn new(capacity: usize) -> Self {
        let mut bufs = vec![[0u8; SEND_SLOT]; capacity].into_boxed_slice();
        // SAFETY: sockaddr_in is plain data and all-zero is a valid value.
        let mut addrs = vec![unsafe { zeroed::<libc::sockaddr_in>() }; capacity].into_boxed_slice();
        let mut iovs: Box<[libc::iovec]> = bufs
            .iter_mut()
            .map(|b| libc::iovec { iov_base: b.as_mut_ptr().cast(), iov_len: 0 })
            .collect();
        let hdrs = iovs
            .iter_mut()
            .zip(addrs.iter_mut())
            .map(|(iov, addr)| {
                // SAFETY: mmsghdr is plain data and all-zero is a valid value.
                let mut h: libc::mmsghdr = unsafe { zeroed() };
                h.msg_hdr.msg_name = (addr as *mut libc::sockaddr_in).cast();
                h.msg_hdr.msg_namelen = size_of::<libc::sockaddr_in>() as libc::socklen_t;
                h.msg_hdr.msg_iov = iov;
                h.msg_hdr.msg_iovlen = 1;
                h
            })
            .collect();
        Self { bufs, addrs, iovs, hdrs, len: 0 }
    }

    pub fn is_full(&self) -> bool {
        self.len == self.hdrs.len()
    }

    pub fn push(&mut self, to: &libc::sockaddr_in, packet: &[u8]) {
        let i = self.len;
        self.bufs[i][..packet.len()].copy_from_slice(packet);
        self.iovs[i].iov_len = packet.len();
        self.addrs[i] = *to;
        self.len += 1;
    }

    /// Sends everything queued; returns (sent, dropped). A full socket
    /// buffer drops the rest, as a media engine would rather than block.
    pub fn flush(&mut self, fd: RawFd) -> (u64, u64) {
        let mut off = 0;
        while off < self.len {
            // SAFETY: headers [off, len) point into buffers owned by self.
            let n = unsafe {
                libc::sendmmsg(
                    fd,
                    self.hdrs.as_mut_ptr().add(off),
                    (self.len - off) as libc::c_uint,
                    0,
                )
            };
            if n > 0 {
                off += n as usize;
            } else if n < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            } else {
                break;
            }
        }
        let result = (off as u64, (self.len - off) as u64);
        self.len = 0;
        result
    }
}

pub fn pin_current_thread(cpu: usize) -> io::Result<()> {
    // SAFETY: cpu_set_t is plain data; CPU_SET bounds-checks the index.
    unsafe {
        let mut set: libc::cpu_set_t = zeroed();
        libc::CPU_SET(cpu, &mut set);
        check(libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set))?;
    }
    Ok(())
}

fn rusage() -> libc::rusage {
    // SAFETY: rusage is plain data; getrusage fills it.
    unsafe {
        let mut ru: libc::rusage = zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        ru
    }
}

/// User plus system CPU time of the whole process (all threads), seconds.
pub fn process_cpu_seconds() -> f64 {
    let ru = rusage();
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

pub fn max_rss_mb() -> f64 {
    rusage().ru_maxrss as f64 / 1024.0
}

/// Lifts the open-file soft limit to the hard limit: one socket per call.
pub fn raise_nofile_limit() {
    // SAFETY: rlimit is plain data; both calls take a valid pointer.
    unsafe {
        let mut lim: libc::rlimit = zeroed();
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) == 0 {
            lim.rlim_cur = lim.rlim_max;
            libc::setrlimit(libc::RLIMIT_NOFILE, &lim);
        }
    }
}

pub fn epoch_seconds() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Converts a wall-clock epoch time shared with the driver to an Instant.
pub fn instant_at(epoch: f64) -> Instant {
    let now = Instant::now();
    let delta = epoch - epoch_seconds();
    if delta >= 0.0 {
        now + Duration::from_secs_f64(delta)
    } else {
        now.checked_sub(Duration::from_secs_f64(-delta)).unwrap_or(now)
    }
}

pub fn sleep_until(t: Instant) {
    let now = Instant::now();
    if t > now {
        std::thread::sleep(t - now);
    }
}
