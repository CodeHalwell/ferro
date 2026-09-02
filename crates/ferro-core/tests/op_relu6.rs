use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn relu6_values() {
    let a = Tensor::from_vec(vec![-3.0, 0.0, 2.5, 6.0, 8.0], &[5]).unwrap();
    let got = a.relu6().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 0.0).abs() < 1e-5);
    assert!((got[2] - 2.5).abs() < 1e-5);
    assert!((got[3] - 6.0).abs() < 1e-5);
    assert!((got[4] - 6.0).abs() < 1e-5);
}

#[test]
fn relu6_grad() {
    let a = Tensor::from_vec(vec![-1.5, 1.0, 4.0, 3.2], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].relu6().unwrap().sum());
}
