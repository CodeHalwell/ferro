use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn var_dim_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let biased = a.var_dim(1, 0, false).unwrap().to_vec();
    assert!((biased[0] - 2.0 / 3.0).abs() < 1e-5);
    assert!((biased[1] - 2.0 / 3.0).abs() < 1e-5);
    let unbiased = a.var_dim(1, 1, false).unwrap().to_vec();
    assert!((unbiased[0] - 1.0).abs() < 1e-5);
    assert!((unbiased[1] - 1.0).abs() < 1e-5);
}

#[test]
fn var_dim_keepdim_shape() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert_eq!(a.var_dim(1, 1, true).unwrap().shape(), &[2, 1]);
    assert_eq!(a.var_dim(1, 1, false).unwrap().shape(), &[2]);
}

#[test]
fn var_dim_out_of_range_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!(a.var_dim(2, 0, false).is_err());
}

#[test]
fn var_dim_correction_too_large_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!(a.var_dim(1, 3, false).is_err());
    assert!(a.var_dim(1, 4, false).is_err());
}

#[test]
fn var_dim_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 4.0, 0.5, -1.0, 3.0], &[2, 3]).unwrap();
    grad_check(&[a], |t| t[0].var_dim(1, 1, false).unwrap().sum());
}
