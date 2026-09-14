//! Independent scalar oracle: no registry, pool, packing, or backend reference.
use ferro_fastcpu::{matmul_batch, matmul_with_threads};
#[path = "support/bmm_diagnostics.rs"]
mod diagnostics;

#[test]
fn injected_omission_report_is_bounded_and_counts_all_mismatches() {
    // Artificial TEST output only: not a naturally reproduced kernel failure.
    let (batch, m, k, n) = (1, 1, 128, 300);
    let a = input(2556, batch*m*k);
    let b = input(3556, batch*k*n);
    let want = oracle(&a, &b, batch, m, k, n);
    let mut got = want.clone();
    for col in 0..n {
        let mut sum = 0.0f32;
        for p in 0..k {
            if p != 90 { sum += a[p] * b[p*n+col]; }
        }
        got[col] = sum;
    }
    let count = got.iter().zip(&want).filter(|(x,y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(count, 300);
    let report = diagnostics::injected_report(&a, &b, &got, &want);
    assert!(report.contains("\"schema\":1"), "missing bounded report schema");
    assert!(report.contains("\"origin\":\"injected_omission_p90_test_buffer\""));
    assert!(report.contains("\"mismatch_count\":300"));
    assert!(report.contains("\"retained_count\":256"));
    assert_eq!(report.matches("\"record\":\"mismatch\"").count(), 256);
    assert!(report.contains("\"p90\":{\"a_bits\":"));
    assert!(report.len() < 200_000);
}

#[test]
fn injected_historical_coordinate_omission_retains_operands() {
    let (batch, m, k, n) = (16, 128, 128, 128);
    let a = input(2556, batch*m*k);
    let b = input(3556, batch*k*n);
    let mut evidence = diagnostics::Capture::new(&a, &b, [batch, m, k, n], [2556, 3556], 0, None);
    let want = oracle(&a, &b, batch, m, k, n);
    evidence.before_fast(&a, &b, &want);
    // Deliberately omit p=90 in ONE test output coordinate; never modify a kernel.
    let mut got = want.clone();
    let mut omitted = 0.0f32;
    for p in 0..k {
        if p != 90 { omitted += a[(14*m+14)*k+p] * b[(14*k+p)*n+76]; }
    }
    got[231244] = omitted;
    assert_eq!(omitted.to_bits(), 1094337833);
    assert_eq!(want[231244].to_bits(), 1094681645);
    let path = evidence.capture_failure(&a, &b, &got, &want, "historical_coordinate_test_only", "injected_omission_p90_test_buffer").unwrap();
    let report = std::fs::read_to_string(path).unwrap();
    assert!(report.contains("\"mismatch_count\":1,"));
    assert!(report.contains("\"coordinate\":[14,14,76]"));
    assert!(report.contains("\"a_bits\":1047347712,\"b_bits\":1068835684,\"product_bits\":1051189367"));
    assert!(report.contains("\"independent_scalar_bits\":1094681645"));
    assert!(report.contains("\"counterfactual_skip90_bits\":1094337833"));
}

#[test]
fn injected_input_byte_changes_are_distinguished_by_stage() {
    let mut a = input(2556, 128);
    let b = input(3556, 128);
    // An injected corruption BEFORE the first snapshot must differ from the seed.
    a[90] = f32::from_bits(a[90].to_bits() ^ 1);
    let mut evidence = diagnostics::Capture::new(&a, &b, [1, 1, 128, 1], [2556, 3556], 0, Some(1));
    let want = oracle(&a, &b, 1, 1, 128, 1);
    evidence.before_fast(&a, &b, &want);
    // A different injected corruption AFTER before_fast must appear in that delta.
    a[91] = f32::from_bits(a[91].to_bits() ^ 1);
    evidence.stage("after_injected_input_byte_flip", &a, &b);
    let report = evidence.report(&[f32::NAN], &want, "byte_flip_test_only", "injected_input_byte_flip_test_buffer");
    assert!(report.lines().find(|l| l.contains("seed_to_first_snapshot")).unwrap().contains("\"changed_bytes\":1"));
    assert!(report.lines().find(|l| l.contains("after_injected_input_byte_flip")).unwrap().contains("\"changed_bytes\":1"));
    assert!(report.lines().find(|l| l.contains("before_fast")).unwrap().contains("\"changed_bytes\":0"));
    diagnostics::persist(&report, "injected-byte-flip").unwrap();
    let mut unchanged = diagnostics::Capture::new(&a, &b, [1, 1, 128, 1], [2556, 3556], 0, Some(1));
    assert!(unchanged.capture_failure(&a, &b, &want, &want, "no_failure", "injected_no_failure").is_none());
}

fn input(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    (0..len).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 4.0
    }).collect()
}

fn oracle(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch * m * n];
    for bi in 0..batch {
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[(bi * m + row) * k + p] * b[(bi * k + p) * n + col];
                }
                out[(bi * m + row) * n + col] = sum;
            }
        }
    }
    out
}

fn fingerprint(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf29ce484222325, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3))
}

fn exact(got: &[f32], want: &[f32], context: &str, evidence: &mut diagnostics::Capture, a: &[f32], b: &[f32]) {
    evidence.on_failure(a, b, got, want, context);
    assert_eq!(got.len(), want.len());
    for (i, (x, y)) in got.iter().zip(want).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{context}: index={i} got={x} oracle={y}");
    }
}

#[test]
fn original_failure_fixture_matches_independent_oracle() {
    let (batch, m, k, n) = (16, 128, 128, 128);
    // Original elementwise test's i=4 and batch=16 seeds.
    let a = input(2556, batch * m * k);
    let b = input(3556, batch * k * n);
    let baseline = diagnostics::Capture::new(&a, &b, [batch, m, k, n], [2556, 3556], 0, None);
    let want = oracle(&a, &b, batch, m, k, n);
    eprintln!("original fixture a={:016x} b={:016x} oracle={:016x} index231244={} bits={} threads={:?}",
        fingerprint(&a), fingerprint(&b), fingerprint(&want), want[231244], want[231244].to_bits(),
        std::thread::available_parallelism());
    let repeats = std::env::var("FERRO_BMM_REPEATS").map_or(1, |s| s.parse::<usize>().unwrap());
    for repeat in 0..repeats {
        // Force actual recycled output contents, including NaNs in release.
        ferro_core::pool::give(vec![f32::NAN; want.len()]);
        let mut evidence = baseline.clone();
        evidence.before_fast(&a, &b, &want);
        let got = matmul_batch(&a, &b, batch, m, k, n);
        exact(&got, &want, &format!("original repeat={repeat}"), &mut evidence, &a, &b);
        for threads in [1, 3, 7, 32] {
            // Independently exercise the originally failing batch slab.
            let bi = 14;
            let (sa, sb, sw) = (&a[bi*m*k..(bi+1)*m*k], &b[bi*k*n..(bi+1)*k*n], &want[bi*m*n..(bi+1)*m*n]);
            let mut evidence = baseline.slab(bi, threads);
            evidence.before_fast(sa, sb, sw);
            let slab = matmul_with_threads(sa, sb, m, k, n, threads);
            exact(&slab, sw, &format!("slab threads={threads} repeat={repeat}"), &mut evidence, sa, sb);
        }
    }
}

#[test]
fn batch_tails_and_k_blocks_match_independent_oracle() {
    for (batch, m, k, n) in [(3, 7, 0, 17), (3, 5, 255, 15), (3, 7, 256, 17),
        (3, 49, 257, 17), (16, 65, 257, 33), (3, 129, 300, 65)] {
        for seed in [1, 11, 2556] {
            let a = input(seed, batch*m*k);
            let b = input(seed+1000, batch*k*n);
            let mut evidence = diagnostics::Capture::new(&a, &b, [batch, m, k, n], [seed, seed+1000], 0, None);
            let want = oracle(&a, &b, batch, m, k, n);
            evidence.before_fast(&a, &b, &want);
            ferro_core::pool::give(vec![f32::NAN; want.len()]);
            let got = matmul_batch(&a, &b, batch, m, k, n);
            exact(&got, &want, &format!("shape={batch},{m},{k},{n} seed={seed}"), &mut evidence, &a, &b);
        }
    }
}

#[test]
fn concurrent_batches_match_independent_oracle() {
    let (batch, m, k, n) = (3, 49, 257, 65);
    let a = input(2556, batch*m*k);
    let b = input(3556, batch*k*n);
    let baseline = diagnostics::Capture::new(&a, &b, [batch, m, k, n], [2556, 3556], 0, None);
    let want = oracle(&a, &b, batch, m, k, n);
    std::thread::scope(|scope| {
        for worker in 0..4 {
            let (a, b, want, baseline) = (&a, &b, &want, &baseline);
            scope.spawn(move || {
                for repeat in 0..4 {
                    ferro_core::pool::give(vec![f32::NAN; want.len()]);
                    let mut evidence = baseline.clone();
                    evidence.before_fast(a, b, want);
                    exact(&matmul_batch(a, b, batch, m, k, n), want, &format!("worker={worker} repeat={repeat}"), &mut evidence, a, b);
                }
            });
        }
    });
}
