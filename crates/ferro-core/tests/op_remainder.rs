use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn remainder_values() {
    let a = Tensor::from_vec(vec![-7.0, 7.0, -5.5, 5.5], &[4]).unwrap();
    let b = Tensor::from_vec(vec![3.0, 3.0, 2.0, 2.0], &[4]).unwrap();
    let got = a.remainder(&b).unwrap().to_vec();
    // -7 rem 3 = 2 (sign of the divisor), whereas fmod(-7, 3) = -1: the
    // classic sign difference between torch remainder and fmod.
    assert!((got[0] - 2.0).abs() < 1e-5);
    assert!((got[1] - 1.0).abs() < 1e-5);
    assert!((got[2] - 0.5).abs() < 1e-5);
    assert!((got[3] - 1.5).abs() < 1e-5);
}

#[test]
fn remainder_grad() {
    let a = Tensor::from_vec(vec![1.3, -2.6, 5.7, 3.2], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![2.0, 3.0, 2.5, 4.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].remainder(&t[1]).unwrap().sum());
}

#[test]
fn remainder_grad_broadcast() {
    let a = Tensor::from_vec(vec![1.3, -2.6, 5.7, 3.2, -4.1, 6.3], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![2.0, 3.0, 2.5], &[3]).unwrap();
    grad_check(&[a, b], |t| t[0].remainder(&t[1]).unwrap().sum());
}
