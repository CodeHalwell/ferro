use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn l1_loss_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![0.0, 2.0, 5.0], &[3]).unwrap();
    let got = a.l1_loss(&b).unwrap().item();
    assert!((got - 1.0).abs() < 1e-5);
}

#[test]
fn l1_loss_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, -1.0, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.4, 3.2, -2.5, 1.1], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].l1_loss(&t[1]).unwrap());
}
