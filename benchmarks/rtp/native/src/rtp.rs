//! RTP header parsing and writing (RFC 3550), as much as a G.711 call needs.

pub const HEADER_LEN: usize = 12;

pub struct Packet<'a> {
    pub seq: u16,
    pub ssrc: u32,
    pub payload: &'a [u8],
}

/// RTCP shares the port in the shared mode; its packet types are 200-204.
pub fn is_rtcp(buf: &[u8]) -> bool {
    buf.len() >= 2 && (200..=204).contains(&buf[1])
}

pub fn parse(buf: &[u8]) -> Option<Packet<'_>> {
    if buf.len() < HEADER_LEN || buf[0] >> 6 != 2 || is_rtcp(buf) {
        return None;
    }
    let mut start = HEADER_LEN + 4 * (buf[0] & 0x0F) as usize;
    if buf[0] & 0x10 != 0 {
        let words = u16::from_be_bytes([*buf.get(start + 2)?, *buf.get(start + 3)?]) as usize;
        start += 4 + 4 * words;
    }
    let mut end = buf.len();
    if buf[0] & 0x20 != 0 {
        end = end.checked_sub(*buf.last()? as usize)?;
    }
    if start > end {
        return None;
    }
    Some(Packet {
        seq: u16::from_be_bytes([buf[2], buf[3]]),
        ssrc: u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]),
        payload: &buf[start..end],
    })
}

pub fn write_header(buf: &mut [u8], payload_type: u8, seq: u16, timestamp: u32, ssrc: u32) {
    buf[0] = 0x80;
    buf[1] = payload_type & 0x7F;
    buf[2..4].copy_from_slice(&seq.to_be_bytes());
    buf[4..8].copy_from_slice(&timestamp.to_be_bytes());
    buf[8..12].copy_from_slice(&ssrc.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let mut buf = [0u8; HEADER_LEN + 4];
        write_header(&mut buf, 0, 0xBEEF, 160, 0x1234_5678);
        buf[HEADER_LEN..].copy_from_slice(b"abcd");
        let p = parse(&buf).unwrap();
        assert_eq!((p.seq, p.ssrc, p.payload), (0xBEEF, 0x1234_5678, &b"abcd"[..]));
    }

    #[test]
    fn csrc_extension_and_padding_are_skipped() {
        // V=2, P=1, X=1, CC=1; extension of one word; two bytes of padding.
        let mut buf = vec![0xB1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7];
        buf.extend([0; 4]); // CSRC
        buf.extend([0xBE, 0xDE, 0, 1, 0, 0, 0, 0]); // extension header + one word
        buf.extend(b"pay");
        buf.extend([0, 2]); // padding, count in the last byte
        let p = parse(&buf).unwrap();
        assert_eq!(p.payload, b"pay");
    }

    #[test]
    fn rtcp_and_short_buffers_are_rejected() {
        assert!(parse(&[0x80, 200, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0]).is_none());
        assert!(parse(&[0x80, 0, 0]).is_none());
    }
}
