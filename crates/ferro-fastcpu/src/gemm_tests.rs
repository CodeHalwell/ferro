//! Exact and bounded oracles for the packed SGEMM. The SIMD paths promise
//! one ascending-p fused chain per output, so `f32::mul_add` in a plain loop
//! is a bitwise oracle for them; f64 with the gamma(k) bound checks accuracy.
use super::*;
use crate::gemm::{sgemm_batch, sgemm_isa, Isa};
use ferro_core::dispatch::Backend;

fn fill(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    (0..len).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale
    }).collect()
}

fn oracle(isa: Isa, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let (x, y) = (a[i * k + p], b[p * n + j]);
                acc = if isa == Isa::Scalar { acc + x * y } else { x.mul_add(y, acc) };
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn isas() -> Vec<Isa> {
    [Isa::Avx512, Isa::Avx2, Isa::Scalar].into_iter().filter(|i| i.supported()).collect()
}

fn same(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}");
    for (i, (x, y)) in got.iter().zip(want).enumerate() {
        assert!(x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan(), "{ctx}: index {i}: {x} vs oracle {y}");
    }
}

fn run(isa: Isa, a: &[f32], b: &[f32], m: usize, k: usize, n: usize, threads: usize) -> Vec<f32> {
    // Poisoned destination and packing scratch: every output must be written
    // (never accumulated into) and no stale packed value may be consumed.
    crate::gemm::poison_scratch();
    let mut c = vec![f32::NAN; m * n];
    sgemm_isa(isa, a, b, &mut c, m, k, n, threads);
    c
}

#[test]
fn every_isa_matches_its_exact_oracle_across_tile_and_block_edges() {
    // Edges of MR (4, 6, 14), NR (16, 32), MC (64, 96, 196) and the K
    // blocking (one block up to 320, then balanced blocks).
    let ms = [1, 5, 6, 7, 13, 14, 15, 63, 97, 197];
    let ns = [1, 15, 16, 17, 31, 32, 33, 65];
    let ks = [1, 2, 255, 256, 257, 321, 513, 700];
    for isa in isas() {
        for &m in &ms {
            for &n in &ns {
                for &k in &ks {
                    let a = fill((m * 1000 + k) as u64, m * k, 4.0);
                    let b = fill((n * 7 + k) as u64, k * n, 4.0);
                    same(&run(isa, &a, &b, m, k, n, 1), &oracle(isa, &a, &b, m, k, n), &format!("{isa:?} {m}x{k}x{n}"));
                }
            }
        }
    }
}

#[test]
fn results_are_bitwise_independent_of_threads_and_split_direction() {
    // Row split (tall), column split (short-wide, private buffers), NC=3072
    // edge, and thread counts above the CPU count.
    for (m, k, n) in [(203, 300, 77), (5, 129, 1000), (3, 40, 3100), (64, 288, 1024), (1, 513, 33)] {
        let a = fill(m as u64, m * k, 4.0);
        let b = fill(n as u64, k * n, 4.0);
        for isa in isas() {
            let want = oracle(isa, &a, &b, m, k, n);
            for threads in [1, 2, 3, 4, 7, 32] {
                for repeat in 0..2 {
                    same(&run(isa, &a, &b, m, k, n, threads), &want, &format!("{isa:?} {m}x{k}x{n} threads={threads} repeat={repeat}"));
                }
            }
        }
    }
}

#[test]
fn simd_paths_agree_bitwise_with_each_other() {
    if !Isa::Avx512.supported() {
        return;
    }
    for (m, k, n) in [(97, 777, 131), (12, 256, 32), (6, 1, 16)] {
        let a = fill(3, m * k, 4.0);
        let b = fill(4, k * n, 4.0);
        same(&run(Isa::Avx512, &a, &b, m, k, n, 3), &run(Isa::Avx2, &a, &b, m, k, n, 2), &format!("{m}x{k}x{n}"));
    }
}

#[test]
fn error_is_within_gamma_k_bound_of_f64_reference() {
    for (m, k, n, scale) in [(33, 4096, 47, 1.0), (40, 1000, 70, 1e-3), (17, 300, 19, 1e4)] {
        let a = fill(5, m * k, scale);
        let b: Vec<f32> = fill(6, k * n, scale).into_iter().map(|x| x + 0.25 * scale).collect();
        for isa in isas() {
            check::check(&a, &b, &run(isa, &a, &b, m, k, n, 4), 1, m, k, n).unwrap();
        }
    }
}

#[test]
fn special_values_follow_the_oracle() {
    let specials = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, f32::from_bits(1), f32::MAX, -1.0, 1.0];
    let (m, k, n) = (13, 9, 37);
    let a: Vec<f32> = (0..m * k).map(|i| specials[i * 7 % 9]).collect();
    let b: Vec<f32> = (0..k * n).map(|i| specials[i * 5 % 9]).collect();
    for isa in isas() {
        same(&run(isa, &a, &b, m, k, n, 2), &oracle(isa, &a, &b, m, k, n), &format!("{isa:?}"));
    }
}

#[test]
fn batched_matches_per_slab_for_every_thread_count() {
    for (batch, m, k, n) in [(16, 128, 128, 128), (3, 49, 257, 17), (7, 5, 300, 70), (2, 1, 9, 1)] {
        let a = fill(2556, batch * m * k, 4.0);
        let b = fill(3556, batch * k * n, 4.0);
        let isa = Isa::detect();
        let want: Vec<f32> = (0..batch).flat_map(|bi| oracle(isa, &a[bi * m * k..][..m * k], &b[bi * k * n..][..k * n], m, k, n)).collect();
        for threads in [1, 2, 3, 4, 16, 33] {
            let mut got = vec![f32::NAN; batch * m * n];
            sgemm_batch(&a, &b, &mut got, batch, m, k, n, threads);
            same(&got, &want, &format!("{batch},{m},{k},{n} threads={threads}"));
        }
        ferro_core::pool::give(vec![f32::NAN; batch * m * n]);
        same(&matmul_batch(&a, &b, batch, m, k, n), &want, "matmul_batch");
        same(&elementwise::FastCpuBackend.matmul_batch(&a, &b, batch, m, k, n), &want, "backend");
    }
}

#[test]
fn concurrent_callers_get_exact_results() {
    let (m, k, n) = (150, 300, 170);
    let a = fill(1, m * k, 4.0);
    let b = fill(2, k * n, 4.0);
    let want = oracle(Isa::detect(), &a, &b, m, k, n);
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| for _ in 0..4 { same(&matmul(&a, &b, m, k, n), &want, "concurrent") });
        }
    });
}

#[test]
fn historical_fixture_is_exact_and_its_omission_fingerprint_breaches_the_check() {
    // Seeds, shape and coordinate of the 343812-ULP incident: (16,128,128,128),
    // flat index 231244 = batch 14, row 14, col 76.
    let (batch, m, k, n) = (16, 128, 128, 128);
    let a = fill(2556, batch * m * k, 4.0);
    let b = fill(3556, batch * k * n, 4.0);
    let got = matmul_batch(&a, &b, batch, m, k, n);
    check::check(&a, &b, &got, batch, m, k, n).unwrap();
    let (ar, bc) = (&a[(14 * m + 14) * k..][..k], (0..k).map(|p| b[(14 * k + p) * n + 76]).collect::<Vec<_>>());
    let mut seq = 0.0f32;
    let mut omit = 0.0f32;
    for p in 0..k {
        seq += ar[p] * bc[p];
        if p != 90 { omit += ar[p] * bc[p]; }
    }
    assert_eq!((seq.to_bits(), omit.to_bits()), (1094681645, 1094337833));
    let mut bad = got.clone();
    bad[231244] = omit;
    let report = check::check(&a, &b, &bad, batch, m, k, n).unwrap_err();
    assert!(report.starts_with("1 of 262144 outputs breach") && report.contains("[14,14,76]"), "{report}");
    for v in [f32::NAN, f32::INFINITY, got[231244] + 0.01] {
        bad[231244] = v;
        assert!(check::check(&a, &b, &bad, batch, m, k, n).is_err(), "{v}");
    }
}

#[test]
fn check_skips_k_beyond_gamma_domain() {
    // At k = 2^24 + 1, gamma(k) is negative; a sequential f32 sum of ones
    // stalls at 2^24 and is still a correct fused-chain result.
    let k = (1 << 24) + 1;
    let ones = vec![1.0f32; k];
    assert!(check::check(&ones, &ones, &[16777216.0], 1, 1, k, 1).is_ok());
}

#[test]
fn shape_validation() {
    for shape in [(0, 3, 4, 5), (2, 0, 4, 5), (2, 3, 4, 0)] {
        assert!(matmul_batch(&[], &[], shape.0, shape.1, shape.2, shape.3).is_empty());
    }
    assert_eq!(matmul_batch(&[], &[], 2, 3, 0, 5), vec![0.0; 30]);
    for shape in [(1, 1, 1, 1), (usize::MAX, 2, 1, 1), (1, 1, usize::MAX, 2), (1, 2, 0, usize::MAX)] {
        assert!(std::panic::catch_unwind(|| matmul_batch(&[], &[], shape.0, shape.1, shape.2, shape.3)).is_err());
    }
}

#[test]
fn install_routes_default_backend_bmm_through_fast_gemm() {
    let _registry = crate::registry_tests::lock();
    install();
    let (batch, m, k, n) = (3, 7, 130, 33);
    let a = fill(8, batch * m * k, 4.0);
    let b = fill(9, batch * k * n, 4.0);
    same(&ferro_core::CpuBackend.matmul_batch(&a, &b, batch, m, k, n), &matmul_batch(&a, &b, batch, m, k, n), "installed");
}
