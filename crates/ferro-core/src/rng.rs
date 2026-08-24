use std::cell::Cell;

/// Tiny splitmix64-seeded xorshift128+ PRNG so `ferro-core` can produce random
/// tensors without pulling in the `rand` crate. Deterministic given a seed;
/// good enough for weight init and tests, not for cryptography.
pub struct Rng {
    s0: Cell<u64>,
    s1: Cell<u64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut split = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng {
            s0: Cell::new(split() | 1),
            s1: Cell::new(split() | 1),
        }
    }

    fn next_u64(&self) -> u64 {
        let mut x = self.s0.get();
        let y = self.s1.get();
        self.s0.set(y);
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        self.s1.set(x);
        x.wrapping_add(y)
    }

    /// Uniform f32 in [0, 1).
    pub fn uniform(&self) -> f32 {
        // Top 24 bits give a uniform value with full f32 mantissa precision.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Standard-normal f32 via Box-Muller.
    pub fn normal(&self) -> f32 {
        let u1 = self.uniform().max(1e-9);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (std::f32::consts::TAU * u2).cos()
    }
}
