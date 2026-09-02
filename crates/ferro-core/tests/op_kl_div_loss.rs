use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn kl_div_loss_values() {
    let input = Tensor::from_vec(vec![-1.0, -0.5], &[2]).unwrap();
    let target = Tensor::from_vec(vec![0.4, 0.6], &[2]).unwrap();
    let got = input.kl_div_loss(&target).unwrap().item();
    assert!((got - 0.0134942).abs() < 1e-4);
}

#[test]
fn kl_div_loss_grad() {
    let input = Tensor::from_vec(vec![-0.3, -0.7, -1.2, -0.1], &[2, 2]).unwrap();
    let target = Tensor::from_vec(vec![0.2, 0.5, 0.3, 0.6], &[2, 2]).unwrap();
    grad_check(&[input, target], |t| t[0].kl_div_loss(&t[1]).unwrap());
}
