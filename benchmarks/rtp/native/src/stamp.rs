//! The probe payload: send time and call index, written in bytes that
//! survive a G.711 µ-law decode and re-encode, so any echo engine that
//! decodes and re-encodes the audio returns them intact.

use crate::FRAME;

/// µ-law codes 0x10-0x1F all round-trip; one nibble per byte.
fn put_u32(buf: &mut [u8], value: u32) {
    for (k, byte) in buf[..8].iter_mut().enumerate() {
        *byte = 0x10 + ((value >> (28 - 4 * k)) & 0xF) as u8;
    }
}

fn get_u32(buf: &[u8]) -> Option<u32> {
    let mut value = 0u32;
    for &byte in buf.get(..8)? {
        if !(0x10..=0x1F).contains(&byte) {
            return None;
        }
        value = (value << 4) | u32::from(byte - 0x10);
    }
    Some(value)
}

pub fn write(payload: &mut [u8; FRAME], sent_us: u32, call: u32) {
    put_u32(&mut payload[..8], sent_us);
    put_u32(&mut payload[8..16], call);
    payload[16..].fill(0x55);
}

/// Returns (sent_us, call), or None for a payload that is not a probe.
pub fn read(payload: &[u8]) -> Option<(u32, u32)> {
    Some((get_u32(payload)?, get_u32(payload.get(8..)?)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g711::Ulaw;

    #[test]
    fn survives_a_codec_round_trip() {
        let ulaw = Ulaw::new();
        let mut payload = [0u8; FRAME];
        write(&mut payload, 0xDEAD_BEEF, 12_345);
        let mut pcm = [0i16; FRAME];
        let mut back = [0u8; FRAME];
        ulaw.decode_into(&payload, &mut pcm);
        ulaw.encode_into(&pcm, &mut back);
        assert_eq!(read(&back), Some((0xDEAD_BEEF, 12_345)));
    }

    #[test]
    fn foreign_payloads_are_rejected() {
        assert_eq!(read(&[0xFF; FRAME]), None);
        assert_eq!(read(&[0x10; 4]), None);
    }
}
