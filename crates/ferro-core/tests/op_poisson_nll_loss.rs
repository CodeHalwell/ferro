use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn poisson_nll_loss_values() {
    let input = Tensor::from_vec(vec![0.0, 0.5, -0.5], &[3]).unwrap();
    let target = Tensor::from_vec(vec![1.0, 2.0, 0.0], &[3]).unwrap();
    let got = input.poisson_nll_loss(&target).unwrap().item();
    assert!((got - 0.7517506).abs() < 1e-4);
}

#[test]
fn poisson_nll_loss_grad() {
    let input = Tensor::from_vec(vec![0.3, -0.2, 0.1, -0.4], &[2, 2]).unwrap();
    let target = Tensor::from_vec(vec![1.0, 0.5, 2.0, 1.5], &[2, 2]).unwrap();
    grad_check(&[input, target], |t| t[0].poisson_nll_loss(&t[1]).unwrap());
}
