use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn std_dim_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]).unwrap();
    let got = a.std_dim(1, 1, false).unwrap().to_vec();
    // row 0: mean 2, deviations -1,0,1, sum sq 2, /(3-1) = 1.0, sqrt = 1.0
    // row 1: mean 4, deviations -2,0,2, sum sq 8, /(3-1) = 4.0, sqrt = 2.0
    assert!((got[0] - 1.0).abs() < 1e-5);
    assert!((got[1] - 2.0).abs() < 1e-5);
}

#[test]
fn std_dim_keepdim_shape() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]).unwrap();
    assert_eq!(a.std_dim(1, 1, true).unwrap().shape(), &[2, 1]);
    assert_eq!(a.std_dim(1, 1, false).unwrap().shape(), &[2]);
}

#[test]
fn std_dim_out_of_range_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!(a.std_dim(2, 1, false).is_err());
}

#[test]
fn std_dim_correction_too_large_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!(a.std_dim(1, 3, false).is_err());
    assert!(a.std_dim(1, 4, false).is_err());
}

#[test]
fn std_dim_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.5, -1.0, 3.0, 0.5, 2.0], &[2, 3]).unwrap();
    grad_check(&[a.clone()], |t| t[0].std_dim(1, 1, false).unwrap().sum());
    grad_check(&[a], |t| t[0].std_dim(1, 1, true).unwrap().sum());
}
