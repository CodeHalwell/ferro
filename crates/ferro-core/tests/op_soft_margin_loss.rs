use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn soft_margin_loss_values() {
    let x = Tensor::from_vec(vec![1.0, -2.0, 0.5], &[3]).unwrap();
    let t = Tensor::from_vec(vec![1.0, -1.0, 1.0], &[3]).unwrap();
    let got = x.soft_margin_loss(&t).unwrap().item();
    let pairs = [(1.0f32, 1.0f32), (-2.0, -1.0), (0.5, 1.0)];
    let expected: f32 = pairs.iter().map(|&(xi, ti)| (1.0 + (-ti * xi).exp()).ln()).sum::<f32>() / 3.0;
    assert!((got - expected).abs() < 1e-5);
}

#[test]
fn soft_margin_loss_grad() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 0.5, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.0, -1.0, 1.0, -1.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].soft_margin_loss(&t[1]).unwrap());
}
