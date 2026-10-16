//! G.711 µ-law, table driven: one lookup per sample each way.

const BIAS: i32 = 0x84;
const CLIP: i32 = 32635;

pub struct Ulaw {
    decode: [i16; 256],
    encode: Box<[u8]>,
}

impl Ulaw {
    pub fn new() -> Self {
        let mut decode = [0i16; 256];
        for (code, slot) in decode.iter_mut().enumerate() {
            *slot = decode_sample(code as u8);
        }
        let encode = (0..=u16::MAX).map(|raw| encode_sample(raw as i16)).collect();
        Self { decode, encode }
    }

    pub fn decode_into(&self, payload: &[u8], pcm: &mut [i16]) {
        for (sample, &code) in pcm.iter_mut().zip(payload) {
            *sample = self.decode[code as usize];
        }
    }

    pub fn encode_into(&self, pcm: &[i16], payload: &mut [u8]) {
        for (code, &sample) in payload.iter_mut().zip(pcm) {
            *code = self.encode[sample as u16 as usize];
        }
    }
}

fn decode_sample(code: u8) -> i16 {
    let u = !code;
    let exponent = (u >> 4) & 0x07;
    let mantissa = (u & 0x0F) as i32;
    let magnitude = (((mantissa << 3) + BIAS) << exponent) - BIAS;
    if u & 0x80 != 0 {
        -magnitude as i16
    } else {
        magnitude as i16
    }
}

fn encode_sample(pcm: i16) -> u8 {
    let mut s = pcm as i32;
    let sign = if s < 0 {
        s = -s;
        0x80
    } else {
        0
    };
    s = s.min(CLIP) + BIAS;
    let mut exponent = 7;
    while exponent > 0 && s & (0x80 << exponent) == 0 {
        exponent -= 1;
    }
    let mantissa = (s >> (exponent + 3)) & 0x0F;
    !(sign | (exponent << 4) | mantissa) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_but_negative_zero_survives_a_round_trip() {
        let ulaw = Ulaw::new();
        let codes: Vec<u8> = (0..=255u8).filter(|&c| c != 0x7F).collect();
        let mut pcm = vec![0i16; codes.len()];
        let mut back = vec![0u8; codes.len()];
        ulaw.decode_into(&codes, &mut pcm);
        ulaw.encode_into(&pcm, &mut back);
        assert_eq!(codes, back);
    }

    #[test]
    fn silence_and_extremes() {
        assert_eq!(decode_sample(0xFF), 0);
        assert_eq!(encode_sample(0), 0xFF);
        assert_eq!(encode_sample(i16::MAX), 0x80);
        assert_eq!(encode_sample(i16::MIN), 0x00);
    }
}
