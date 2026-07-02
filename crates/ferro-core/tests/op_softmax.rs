use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn softmax_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]).unwrap();
    let got = a.softmax(1).unwrap().to_vec();
    let row0: f32 = got[0..3].iter().sum();
    let row1: f32 = got[3..6].iter().sum();
    assert!((row0 - 1.0).abs() < 1e-5);
    assert!((row1 - 1.0).abs() < 1e-5);
    for k in 3..6 {
        assert!((got[k] - 1.0 / 3.0).abs() < 1e-6);
    }
}

#[test]
fn softmax_dim_out_of_range_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]).unwrap();
    assert!(a.softmax(2).is_err());
    assert!(a.log_softmax(2).is_err());
}

#[test]
fn softmax_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.0, 0.3, 1.2, 0.1, -0.4], &[2, 3]).unwrap();
    grad_check(&[a.clone()], |t| t[0].softmax(1).unwrap().sum());
    grad_check(&[a.clone()], |t| t[0].softmax(0).unwrap().sum());
    // sum() of a full softmax is constant per slice; multiply by the input to
    // get a loss with nonzero gradients.
    grad_check(&[a], |t| t[0].softmax(1).unwrap().mul(&t[0]).unwrap().sum());
}
