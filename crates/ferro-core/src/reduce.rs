//! Fixed-shape pairwise (blocked-tree) summation, the default reduction
//! kernel for every host-side sum in ferro-core (`sum`, `mean`, `raw_sum_dim`,
//! softmax/log_softmax). Naive left-to-right summation of n terms has error
//! bounded by (n-1) * u * sum|x_i| (u = 2^-24 for f32): at n = 1e6 the
//! worst-case relative error is ~6e-2. Recursive halving down to a bounded
//! base case turns that into a tree of depth ~log2(n), giving error
//! ~log2(n) * u (~1.2e-6 at n = 1e6). The base case's independent
//! accumulators, combined in a fixed pairwise order, are exactly what SIMD
//! wants - the accurate version is also the fast version.
//!
//! The tree shape (base size, split point, accumulator lane count) is a pure
//! function of n and the stride pattern, never of thread count or
//! `available_parallelism`: floating-point addition is not associative, so
//! reproducibility requires the reduction order itself to be fixed.

const BASE: usize = 128;
const LANES: usize = 8;

/// Sum a contiguous slice via fixed-shape pairwise reduction.
pub(crate) fn pairwise_sum(x: &[f32]) -> f32 {
    pairwise_sum_strided(x, 0, x.len(), 1)
}

/// Sum `n` elements of `x` starting at `offset`, spaced `stride` apart
/// (`stride == 1` is the contiguous case). Recurses by halving `n` until a
/// base case of at most `BASE` elements, which is summed with `LANES`
/// independent accumulators combined in a fixed pairwise order.
pub(crate) fn pairwise_sum_strided(x: &[f32], offset: usize, n: usize, stride: usize) -> f32 {
    if n <= BASE {
        return base_sum(x, offset, n, stride);
    }
    let half = n / 2;
    let left = pairwise_sum_strided(x, offset, half, stride);
    let right = pairwise_sum_strided(x, offset + half * stride, n - half, stride);
    left + right
}

fn base_sum(x: &[f32], offset: usize, n: usize, stride: usize) -> f32 {
    let mut acc = [0.0f32; LANES];
    let chunks = n / LANES;
    for c in 0..chunks {
        for (l, a) in acc.iter_mut().enumerate() {
            *a += x[offset + (c * LANES + l) * stride];
        }
    }
    let s01 = acc[0] + acc[1];
    let s23 = acc[2] + acc[3];
    let s45 = acc[4] + acc[5];
    let s67 = acc[6] + acc[7];
    let mut sum = (s01 + s23) + (s45 + s67);
    for k in (chunks * LANES)..n {
        sum += x[offset + k * stride];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(pairwise_sum(&[]), 0.0);
    }

    #[test]
    fn matches_exact_sum_for_small_input() {
        let x: Vec<f32> = (1..=10).map(|v| v as f32).collect();
        assert_eq!(pairwise_sum(&x), 55.0);
    }

    #[test]
    fn strided_matches_manual_gather() {
        let x: Vec<f32> = (0..40).map(|v| v as f32).collect();
        let got = pairwise_sum_strided(&x, 3, 6, 4);
        let want: f32 = (0..6).map(|k| x[3 + k * 4]).sum();
        assert_eq!(got, want);
    }

    #[test]
    fn deterministic_across_repeated_calls() {
        let x: Vec<f32> = (0..10_000).map(|v| (v as f32).sin()).collect();
        let a = pairwise_sum(&x);
        let b = pairwise_sum(&x);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn tree_shape_independent_of_base_alignment() {
        // Crossing the BASE boundary (127, 128, 129 elements) must not change
        // the recursion structure in a way that depends on anything but n.
        for n in [1usize, 2, 127, 128, 129, 255, 256, 257, 1000] {
            let x: Vec<f32> = (0..n).map(|v| 1.0 + (v as f32) * 1e-3).collect();
            let a = pairwise_sum(&x);
            let b = pairwise_sum(&x);
            assert_eq!(a.to_bits(), b.to_bits(), "n = {n}");
        }
    }

    #[test]
    fn pairwise_beats_naive_on_adversarial_input() {
        // Same construction as the accuracy gate: 1e6 log-uniform magnitudes
        // in [1e-6, 1e6] with mixed signs. A plain left-to-right f32 sum
        // violates a 1e-6 relative-error bound here; pairwise does not. This
        // is the "has teeth" check for the base case / recursion shape.
        let n = 1_000_000usize;
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let x: Vec<f32> = (0..n)
            .map(|_| {
                let mag_bits = next();
                let mag_u = ((mag_bits >> 40) as f32) / ((1u32 << 24) as f32);
                let mag = 10f32.powf(-6.0 + 12.0 * mag_u);
                let sign_bits = next();
                if sign_bits & 1 == 0 { mag } else { -mag }
            })
            .collect();

        let naive: f32 = x.iter().fold(0.0f32, |a, &b| a + b);
        let pairwise = pairwise_sum(&x);
        let reference: f64 = x.iter().map(|&v| v as f64).sum();

        let rel = |v: f32| ((v as f64) - reference).abs() / reference.abs().max(1.0);
        assert!(rel(naive) > 1e-6, "naive sum unexpectedly accurate: rel = {}", rel(naive));
        assert!(rel(pairwise) <= 1e-6, "pairwise sum too inaccurate: rel = {}", rel(pairwise));
    }
}
