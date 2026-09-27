//! Independent, unpacked row arithmetic for contained BMM.
pub(super) fn row(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len();
    // Each mutable tile owns distinct columns. K is never split or reduced:
    // each lane starts at +0 and sees one separate multiply/add per input.
    let mut tiles = out.chunks_exact_mut(32);
    for (tile, dst) in tiles.by_ref().enumerate() {
        let col = tile * 32;
        let mut sums = [0.0f32; 32];
        for (p, &av) in a.iter().enumerate() {
            let bv: &[f32; 32] = b[p*n+col..p*n+col+32].try_into().unwrap();
            for j in 0..32 { sums[j] += av * bv[j]; }
        }
        dst.copy_from_slice(&sums);
    }
    let tail = tiles.into_remainder();
    let col = n - tail.len();
    for (j, dst) in tail.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (p, &av) in a.iter().enumerate() { sum += av * b[p*n+col+j]; }
        *dst = sum;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn distributions_and_specials_keep_exact_per_output_contract() {
        for n in [31, 32, 33, 65] {
            for seed in [1u64, 11, 2556] {
                for distribution in 0..3 {
                    let mut state = seed;
                    let mut sample = || {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let x = ((state >> 40) as f32 / 16777216.0 - 0.5) * 4.0;
                        match distribution {
                            0 => x,
                            1 => x * 10.0f32.powi((state % 13) as i32 - 6),
                            _ => [f32::from_bits(1), -f32::from_bits(1), -0.0, 0.0,
                                f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MAX, 1.0][state as usize % 9],
                        }
                    };
                    let a: Vec<_> = (0..257).map(|_| sample()).collect();
                    let b: Vec<_> = (0..257*n).map(|_| sample()).collect();
                    let got = crate::matmul_batch(&a,&b,1,1,257,n);
                    for j in 0..n {
                        let mut want = 0.0f32;
                        for p in 0..257 { want += a[p]*b[p*n+j]; }
                        assert!(got[j].is_nan() && want.is_nan() || got[j].to_bits() == want.to_bits(),
                            "n={n} seed={seed} distribution={distribution} col={j}");
                    }
                }
            }
        }
    }

    #[test]
    fn row_overwrites_every_lane_in_ascending_k_order() {
        for n in [1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129] {
            for k in [0, 1, 7, 90, 128, 255, 256, 257] {
                let a: Vec<_> = (0..k).map(|i| ((i*17%71) as f32-35.0)/13.0).collect();
                let b: Vec<_> = (0..k*n).map(|i| ((i*31%73) as f32-36.0)/17.0).collect();
                let mut out = vec![f32::NAN; n];
                super::row(&a, &b, &mut out);
                for j in 0..n {
                    let mut want = 0.0f32;
                    for p in 0..k { want += a[p]*b[p*n+j]; }
                    assert_eq!(out[j].to_bits(),want.to_bits(),"n={n} k={k} col={j}");
                }
            }
        }
    }
}
