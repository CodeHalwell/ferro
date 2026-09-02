use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn softmin_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 0.5, 3.0, -1.0, 0.0], &[2, 3]).unwrap();
    let got = a.softmin(1).unwrap().to_vec();
    let row0 = got[0] + got[1] + got[2];
    let row1 = got[3] + got[4] + got[5];
    assert!((row0 - 1.0).abs() < 1e-5);
    assert!((row1 - 1.0).abs() < 1e-5);
    // smallest input in each row gets the largest softmin weight.
    assert!(got[2] > got[0] && got[2] > got[1]);
    assert!(got[4] > got[3] && got[4] > got[5]);
}

#[test]
fn softmin_grad() {
    let a = Tensor::from_vec(vec![0.3, 1.2, -0.7, 2.1], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].softmin(1).unwrap().sum());
}
