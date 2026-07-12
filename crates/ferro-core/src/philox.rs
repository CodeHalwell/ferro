//! Counter-based RNG: Philox-4x32-10 (Salmon, Moraes, Dror, Shaw, "Parallel
//! Random Numbers: As Easy as 1, 2, 3", SC'11). Unlike `Rng` (xorshift128+,
//! stateful and sequential), a Philox stream is a pure function of (key,
//! counter): `value = bijection(key, counter)`. Op-level randomness (starting
//! with dropout, see docs/CAPABILITY.md S7) must be counter-based so that
//! element i of op instance n reads counter (n, i) and gets the same value
//! regardless of thread count, device, or evaluation order - required for
//! bitwise-reproducible runs and for a checkpoint-recomputed mask to equal
//! the original.
//!
//! `uniform_at(offset, i)` packs the counter as `(offset << 64) | (i / 4)`:
//! one Philox block yields 4 lanes of u32, so 4 consecutive elements share a
//! block and only every 4th element pays for a fresh 10-round mix.

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64) * (b as u64);
    ((p >> 32) as u32, p as u32)
}

fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(M0, ctr[0]);
    let (hi1, lo1) = mulhilo(M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

fn bump(key: [u32; 2]) -> [u32; 2] {
    [key[0].wrapping_add(W0), key[1].wrapping_add(W1)]
}

/// A Philox-4x32-10 stream keyed by a 64-bit seed. Stateless: `block` and
/// `uniform_at` are pure functions of `self` and their arguments.
pub struct Philox {
    key: [u32; 2],
}

impl Philox {
    pub fn new(seed: u64) -> Self {
        Philox { key: [seed as u32, (seed >> 32) as u32] }
    }

    /// Ten Philox rounds over a 128-bit counter split into 4 little-endian
    /// u32 words, returning the mixed 4-word block.
    pub fn block(&self, counter: u128) -> [u32; 4] {
        let mut ctr =
            [counter as u32, (counter >> 32) as u32, (counter >> 64) as u32, (counter >> 96) as u32];
        let mut key = self.key;
        for _ in 0..10 {
            ctr = round(ctr, key);
            key = bump(key);
        }
        ctr
    }

    /// Uniform f32 in [0, 1) for element `i` of op instance `offset`. Top 24
    /// bits of the lane give a value with full f32 mantissa precision (same
    /// construction as `Rng::uniform`).
    pub fn uniform_at(&self, offset: u64, i: u64) -> f32 {
        let counter = ((offset as u128) << 64) | ((i >> 2) as u128);
        let lane = self.block(counter)[(i & 3) as usize];
        ((lane >> 8) as f32) / ((1u32 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::Philox;

    #[test]
    fn known_answer_zero_key_zero_counter() {
        let p = Philox::new(0);
        assert_eq!(p.block(0), [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]);
    }

    #[test]
    fn deterministic() {
        let p = Philox::new(42);
        assert_eq!(p.block(12345), p.block(12345));
        assert_eq!(p.uniform_at(7, 100), p.uniform_at(7, 100));
    }

    #[test]
    fn key_sensitivity() {
        let a = Philox::new(0);
        let b = Philox::new(1);
        assert_ne!(a.block(0), b.block(0));
    }

    #[test]
    fn counter_sensitivity() {
        let p = Philox::new(1);
        assert_ne!(p.block(0), p.block(1));
        assert_ne!(p.uniform_at(0, 0), p.uniform_at(0, 1));
        assert_ne!(p.uniform_at(0, 0), p.uniform_at(1, 0));
    }

    #[test]
    fn uniform_moments() {
        let p = Philox::new(9);
        let n = 1_000_000u64;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for i in 0..n {
            let u = p.uniform_at(0, i) as f64;
            assert!((0.0..1.0).contains(&u));
            sum += u;
            sum_sq += u * u;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
        assert!((var - 1.0 / 12.0).abs() < 0.01, "var {var}");
    }
}
