use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn xlogy_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![0.0, std::f32::consts::E, 4.0], &[3]).unwrap();
    let got = a.xlogy(&b).unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 1.0).abs() < 1e-5);
    assert!((got[2] - 2.0 * 4.0f32.ln()).abs() < 1e-5);
}

#[test]
fn xlogy_zero_zero_is_zero() {
    let a = Tensor::from_vec(vec![0.0], &[1]).unwrap();
    let b = Tensor::from_vec(vec![0.0], &[1]).unwrap();
    let got = a.xlogy(&b).unwrap().to_vec();
    assert_eq!(got[0], 0.0);
}

#[test]
fn xlogy_nan_b_propagates() {
    let a = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let b = Tensor::from_vec(vec![f32::NAN, f32::NAN], &[2]).unwrap();
    let got = a.xlogy(&b).unwrap().to_vec();
    assert!(got[0].is_nan());
    assert!(got[1].is_nan());
}

#[test]
fn xlogy_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3]).unwrap();
    let got = a.xlogy(&b).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    let av = a.to_vec();
    let bv = vec![1.0f32, 2.0, 4.0, 1.0, 2.0, 4.0];
    for i in 0..6 {
        let expected = av[i] * bv[i].ln();
        assert!((got.to_vec()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn xlogy_grad_at_zero_is_masked_not_nan() {
    // The forward defines xlogy(0, b) = 0 for any b, including b = 0. The
    // naive partials ln(b) and a/b are -inf/NaN there; the backward must
    // mask both to 0 at a == 0 instead of propagating them.
    let a = Tensor::from_vec(vec![0.0, 0.0], &[2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let b = Tensor::from_vec(vec![0.0, 5.0], &[2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    a.xlogy(&b).unwrap().sum().backward();
    let ga = a.grad().unwrap().to_vec();
    let gb = b.grad().unwrap().to_vec();
    assert_eq!(ga, vec![0.0, 0.0]);
    assert_eq!(gb, vec![0.0, 0.0]);
}

#[test]
fn xlogy_grad() {
    let a = Tensor::from_vec(vec![0.5, 1.5, 2.0, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.2, 2.5, 0.8, 3.3], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].xlogy(&t[1]).unwrap().sum());
}
