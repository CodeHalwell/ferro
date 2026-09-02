use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn mse_loss_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![1.5, 2.5, 2.0], &[3]).unwrap();
    let got = a.mse_loss(&b).unwrap().item();
    assert!((got - 0.5).abs() < 1e-5);
}

#[test]
fn mse_loss_grad() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 0.5, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.5, -1.0, 1.5, 2.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].mse_loss(&t[1]).unwrap());
}
