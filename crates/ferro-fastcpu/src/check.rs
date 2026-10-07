//! Independent forward-error check of GEMM outputs against an f64 dot.
//!
//! Bound (Higham, dot products): |fl(a.b) - a.b| <= gamma(k) * sum|a_p b_p|
//! with gamma(k) = k*u / (1 - k*u), u = 2^-24, for any summation order and
//! with or without FMA; one extra k*2^-126 absorbs gradual underflow. An
//! omitted, duplicated or corrupted product of ordinary magnitude breaches it
//! (the historical 343812-ULP output is 314x over this bound).
//!
//! `FERRO_GEMM_CHECK=1` makes every fastcpu matmul verify its own output and
//! panic with a bounded evidence report on breach. It costs one f64 GEMM per
//! call: a diagnostic for chasing intermittent faults, not a default.

use std::fmt::Write;
use std::sync::OnceLock;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FERRO_GEMM_CHECK").is_ok_and(|v| v != "0" && !v.is_empty()))
}

pub(crate) fn verify(a: &[f32], b: &[f32], out: &[f32], batch: usize, m: usize, k: usize, n: usize) {
    if enabled() {
        if let Err(report) = check(a, b, out, batch, m, k, n) {
            panic!("FERRO_GEMM_CHECK breach\n{report}");
        }
    }
}

/// Ok, or a report naming the first breaches. Dots whose inputs are
/// non-finite, or whose |products| sum past f32::MAX, cannot be bounded and
/// are skipped.
pub fn check(a: &[f32], b: &[f32], out: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Result<(), String> {
    let u = f64::powi(2.0, -24);
    let gamma = k as f64 * u / (1.0 - k as f64 * u);
    let (mut count, mut report) = (0usize, String::new());
    for bi in 0..batch {
        for i in 0..m {
            let arow = &a[(bi * m + i) * k..][..k];
            for j in 0..n {
                let (mut dot, mut mag) = (0.0f64, 0.0f64);
                for (p, &av) in arow.iter().enumerate() {
                    let prod = av as f64 * b[(bi * k + p) * n + j] as f64;
                    dot += prod;
                    mag += prod.abs();
                }
                if !mag.is_finite() || mag > f32::MAX as f64 {
                    continue;
                }
                let got = out[(bi * m + i) * n + j];
                let bound = gamma * mag * (1.0 + 4.0 * u) + k as f64 * f32::MIN_POSITIVE as f64;
                if got.is_finite() && (got as f64 - dot).abs() <= bound {
                    continue;
                }
                count += 1;
                if count <= 16 {
                    let _ = writeln!(report, "  [{bi},{i},{j}] got={got:e} bits={} f64_dot={dot:e} err={:e} bound={bound:e}",
                        got.to_bits(), (got as f64 - dot).abs());
                }
            }
        }
    }
    if count == 0 {
        return Ok(());
    }
    Err(format!("{count} of {} outputs breach the gamma(k) bound; shape (batch,m,k,n)=({batch},{m},{k},{n}) isa={:?} cpus={:?}\n{report}",
        batch * m * n, crate::gemm::Isa::detect(), std::thread::available_parallelism().map(|p| p.get())))
}
