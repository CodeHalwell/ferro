//! Exact-erf GELU: values against standard-normal-CDF constants, the
//! finite-difference gradient check, and the deliberate (small) divergence
//! from the tanh approximation.

use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn matches_normal_cdf_constants() {
    // gelu_erf(x) = x * Phi(x); Phi values are textbook constants.
    let x = Tensor::from_vec(vec![0.0, 1.0, -1.0, 0.5, 2.0, -2.0], &[6]).unwrap();
    let y = x.gelu_erf().to_vec();
    let want = [
        0.0,
        1.0 * 0.8413447,
        -1.0 * 0.15865526,
        0.5 * 0.6914625,
        2.0 * 0.9772499,
        -2.0 * 0.02275013,
    ];
    for (i, (&got, &w)) in y.iter().zip(&want).enumerate() {
        assert!((got - w).abs() < 2e-6, "x[{i}]: {got} vs {w}");
    }
}

#[test]
fn diverges_from_the_tanh_approximation_but_only_slightly() {
    // The two forms differ by up to ~1e-3 around |x| ~ 2; identical results
    // would mean one of them is routing to the other's kernel.
    let x = Tensor::from_vec(vec![-3.0, -2.0, -1.0, 0.5, 1.5, 2.5], &[6]).unwrap();
    let exact = x.gelu_erf().to_vec();
    let approx = x.gelu().to_vec();
    let max_diff = exact
        .iter()
        .zip(&approx)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-5, "forms are suspiciously identical");
    assert!(max_diff < 5e-3, "forms diverged too far: {max_diff}");
}

#[test]
fn grad_matches_finite_differences() {
    let a = Tensor::from_vec(vec![0.9, -0.7, 0.3, 1.8, -1.4, 0.05], &[2, 3]).unwrap();
    grad_check(&[a], |t| t[0].gelu_erf().sum());
}

#[test]
fn backward_value_is_phi_plus_x_phi() {
    // d/dx at x = 1: Phi(1) + 1 * phi(1) = 0.8413447 + 0.2419707 = 1.0833154.
    let x = Tensor::from_vec(vec![1.0], &[1])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    x.gelu_erf().sum().backward();
    let g = x.grad().unwrap().to_vec()[0];
    assert!((g - 1.0833154).abs() < 2e-6, "grad {g}");
}
