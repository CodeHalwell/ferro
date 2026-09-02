use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn smooth_l1_loss_values() {
    let beta = 0.5f32;
    let input = Tensor::from_vec(vec![1.4, -1.1, 2.0], &[3]).unwrap();
    let target = Tensor::from_vec(vec![1.0, -1.0, 0.0], &[3]).unwrap();
    let got = input.smooth_l1_loss(&target, beta).unwrap().to_vec();
    // d = [0.4, -0.1, 2.0]; losses = [0.16, 0.01, 1.75]; mean = 0.64
    assert!((got[0] - 0.64).abs() < 1e-5);
}

#[test]
fn smooth_l1_loss_grad() {
    let beta = 0.5f32;
    let a = Tensor::from_vec(vec![0.3, -0.3, 2.0, -1.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].smooth_l1_loss(&t[1], beta).unwrap());
}
