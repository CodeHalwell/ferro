//! Gate G3 (docs/CAPABILITY.md 3.2): pairwise reductions must be accurate
//! (<= 1e-6 relative error vs an f64 reference on adversarial magnitude
//! spreads) and bitwise deterministic (the reduction tree is a pure function
//! of shape, never of thread count or call order). Only public API is
//! available here; the pairwise tree itself has unit tests in reduce.rs.

use ferro_core::{Rng, Tensor};

// Deterministic log-uniform magnitude in [1e-6, 1e6] with a mixed sign,
// driven by the repo's own Rng so the dataset is reproducible without a
// crate dependency.
fn adversarial_data(n: usize, seed: u64) -> Vec<f32> {
    let rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let mag = 10f32.powf(-6.0 + 12.0 * rng.uniform());
            if rng.uniform() < 0.5 {
                mag
            } else {
                -mag
            }
        })
        .collect()
}

fn relative_error(got: f32, reference: f64) -> f64 {
    ((got as f64) - reference).abs() / reference.abs().max(1.0)
}

#[test]
fn adversarial_sum_matches_f64_reference() {
    let n = 1_000_000usize;
    let data = adversarial_data(n, 0xC0FF_EE00_1234_5678);
    let reference: f64 = data.iter().map(|&v| v as f64).sum();

    // Has-teeth check: plain left-to-right f32 summation of this data
    // violates the 1e-6 relative-error bound the pairwise path must meet.
    // Naive error is bounded by (n-1) * u * sum|x_i| ~ n * u here since the
    // magnitudes span 12 decades; at n = 1e6, u = 2^-24, that is ~6e-2,
    // several orders of magnitude looser than pairwise's ~log2(n) * u.
    let naive: f32 = data.iter().fold(0.0f32, |acc, &v| acc + v);
    let naive_rel = relative_error(naive, reference);
    assert!(
        naive_rel > 1e-6,
        "naive f32 sum unexpectedly within bound: rel = {naive_rel}"
    );

    let t = Tensor::from_vec(data, &[n]).unwrap();
    let got = t.sum().item();
    let rel = relative_error(got, reference);
    assert!(
        rel <= 1e-6,
        "pairwise sum outside bound: rel = {rel} (naive was {naive_rel})"
    );
}

#[test]
fn cancellation_recovers_small_residual() {
    // 1e8, then 1e6 ones, then -1e8: exact answer is 1e6, but left-to-right
    // summation loses precision the instant 1.0 is added to 1e8 in f32.
    let mut data = vec![1e8f32];
    data.extend(std::iter::repeat(1.0f32).take(1_000_000));
    data.push(-1e8f32);
    let reference: f64 = data.iter().map(|&v| v as f64).sum();
    assert_eq!(reference, 1_000_000.0);

    let t = Tensor::from_vec(data, &[1_000_002]).unwrap();
    let got = t.sum().item();
    let rel = relative_error(got, reference);
    assert!(
        rel <= 1e-3,
        "cancellation case too inaccurate: got {got}, rel = {rel}"
    );
}

#[test]
fn sum_is_bitwise_deterministic_across_calls() {
    let data = adversarial_data(200_000, 42);
    let t = Tensor::from_vec(data, &[200_000]).unwrap();
    let a = t.sum().item();
    let b = t.sum().item();
    assert_eq!(a.to_bits(), b.to_bits());
}

#[test]
fn sum_dim_is_bitwise_deterministic_across_calls() {
    let data = adversarial_data(4 * 4096, 7);
    let t = Tensor::from_vec(data, &[4, 4096]).unwrap();

    let r0a = t.sum_dim(0, false).unwrap();
    let r0b = t.sum_dim(0, false).unwrap();
    assert_eq!(
        r0a.to_vec().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        r0b.to_vec().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let r1a = t.sum_dim(1, false).unwrap();
    let r1b = t.sum_dim(1, false).unwrap();
    assert_eq!(
        r1a.to_vec().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        r1b.to_vec().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
