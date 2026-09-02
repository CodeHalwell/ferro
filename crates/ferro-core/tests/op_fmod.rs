use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn fmod_values() {
    let a = Tensor::from_vec(vec![7.5, -7.5, 5.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![2.0, 2.0, 3.0], &[3]).unwrap();
    let got = a.fmod(&b).unwrap().to_vec();
    assert!((got[0] - 1.5).abs() < 1e-5);
    assert!((got[1] - -1.5).abs() < 1e-5);
    assert!((got[2] - 2.0).abs() < 1e-5);
}

#[test]
fn fmod_grad() {
    let a = Tensor::from_vec(vec![1.2, 5.3, -2.6, 7.4], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![3.0, 2.0, 4.0, 3.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].fmod(&t[1]).unwrap().sum());
}
