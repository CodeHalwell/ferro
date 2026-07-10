use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn clamp_values() {
    let a = Tensor::from_vec(vec![-1.0, 0.5, 2.0], &[3]).unwrap();
    let got = a.clamp(0.0, 1.0).to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 0.5).abs() < 1e-5);
    assert!((got[2] - 1.0).abs() < 1e-5);
}

#[test]
fn clamp_min_gt_max() {
    // torch semantics: min > max yields max everywhere, no panic.
    let a = Tensor::from_vec(vec![-1.0, 0.5, 3.0], &[3]).unwrap();
    assert_eq!(a.clamp(2.0, 1.0).to_vec(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn clamp_grad() {
    let a = Tensor::from_vec(vec![0.2, 0.4, 0.6, 0.8], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].clamp(0.0, 1.0).sum());
}

#[test]
fn clamp_boundary_passes_gradient() {
    let x = Tensor::from_vec(vec![0.0, 0.5, 1.0, -0.1, 1.1], &[5]).unwrap().requires_grad_(true);
    x.clamp(0.0, 1.0).sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![1.0, 1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn clamp_propagates_nan() {
    let a = Tensor::from_vec(vec![f32::NAN, -1.0, 2.0], &[3]).unwrap();
    let y = a.clamp(0.0, 1.0).to_vec();
    assert!(y[0].is_nan());
    assert_eq!(&y[1..], &[0.0, 1.0]);
}
