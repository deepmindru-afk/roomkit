//! A per-call jitter buffer drained by the engine's clock: frames are
//! stored decoded, by sequence number, and played one per 20 ms.

use crate::FRAME;

const SLOTS: usize = 16;

pub struct Jitter {
    seqs: [u16; SLOTS],
    full: [bool; SLOTS],
    pcm: Box<[[i16; FRAME]; SLOTS]>,
    next: Option<u16>,
    depth: usize,
}

impl Jitter {
    pub fn new() -> Self {
        Self {
            seqs: [0; SLOTS],
            full: [false; SLOTS],
            pcm: Box::new([[0; FRAME]; SLOTS]),
            next: None,
            depth: 0,
        }
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Reserves the slot for `seq` and returns it to decode into, or None
    /// for a packet that arrives after its turn or twice.
    pub fn push(&mut self, seq: u16) -> Option<&mut [i16; FRAME]> {
        let next = *self.next.get_or_insert(seq);
        let ahead = seq.wrapping_sub(next) as i16;
        if ahead < 0 {
            return None;
        }
        if ahead as usize >= SLOTS {
            // The stream jumped (restart, long outage): start over from it.
            self.full = [false; SLOTS];
            self.depth = 0;
            self.next = Some(seq);
        }
        let i = seq as usize % SLOTS;
        if self.full[i] {
            return None;
        }
        self.full[i] = true;
        self.seqs[i] = seq;
        self.depth += 1;
        Some(&mut self.pcm[i])
    }

    /// Plays the next frame in order, skipping lost ones when later frames
    /// wait; returns the frame and the count of skipped (concealed) frames.
    pub fn pop(&mut self) -> (Option<&[i16; FRAME]>, u32) {
        let Some(mut next) = self.next else {
            return (None, 0);
        };
        let mut skipped = 0;
        while self.depth > 0 {
            let i = next as usize % SLOTS;
            if self.full[i] && self.seqs[i] == next {
                self.full[i] = false;
                self.depth -= 1;
                self.next = Some(next.wrapping_add(1));
                return (Some(&self.pcm[i]), skipped);
            }
            next = next.wrapping_add(1);
            skipped += 1;
        }
        (None, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(j: &mut Jitter, seq: u16) -> bool {
        j.push(seq).map(|slot| slot[0] = seq as i16).is_some()
    }

    fn play(j: &mut Jitter) -> (Option<i16>, u32) {
        let (frame, skipped) = j.pop();
        (frame.map(|f| f[0]), skipped)
    }

    #[test]
    fn reorders_and_conceals() {
        let mut j = Jitter::new();
        assert!(store(&mut j, 10));
        assert!(store(&mut j, 12));
        assert!(store(&mut j, 11));
        assert_eq!(play(&mut j), (Some(10), 0));
        assert_eq!(play(&mut j), (Some(11), 0));
        assert_eq!(play(&mut j), (Some(12), 0));
        assert!(store(&mut j, 14)); // 13 lost
        assert_eq!(play(&mut j), (Some(14), 1));
        assert_eq!(play(&mut j), (None, 0)); // underrun waits, does not skip
    }

    #[test]
    fn late_duplicates_and_wrap() {
        let mut j = Jitter::new();
        assert!(store(&mut j, u16::MAX));
        assert!(!store(&mut j, u16::MAX)); // duplicate
        assert_eq!(play(&mut j), (Some(-1), 0));
        assert!(!store(&mut j, u16::MAX)); // late
        assert!(store(&mut j, 0)); // wraps
        assert_eq!(play(&mut j), (Some(0), 0));
    }

    #[test]
    fn a_jump_restarts_the_buffer() {
        let mut j = Jitter::new();
        assert!(store(&mut j, 1));
        assert!(store(&mut j, 1000));
        assert_eq!(j.depth(), 1);
        assert_eq!(play(&mut j), (Some(1000), 0));
    }
}
