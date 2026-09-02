use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn huber_loss_values() {
    let input = Tensor::from_vec(vec![2.0, -1.0, 0.0, 3.0], &[4]).unwrap();
    let target = Tensor::from_vec(vec![0.5, -1.5, 0.2, 0.1], &[4]).unwrap();
    // d = [1.5, 0.5, -0.2, 2.9], delta = 1.0
    // |d| > delta: delta*(|d|-0.5*delta); |d| <= delta: 0.5*d^2
    // per-element: [1.0, 0.125, 0.02, 2.4], mean = 3.545 / 4 = 0.88625
    let got = input.huber_loss(&target, 1.0).unwrap().to_vec();
    assert!((got[0] - 0.88625).abs() < 1e-5);
}

#[test]
fn huber_loss_grad() {
    let input = Tensor::from_vec(vec![0.3, 2.5, -0.5, -2.0], &[2, 2]).unwrap();
    let target = Tensor::from_vec(vec![0.0, 1.0, 0.2, 0.0], &[2, 2]).unwrap();
    grad_check(&[input, target], |t| t[0].huber_loss(&t[1], 1.0).unwrap());
}
