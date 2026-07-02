use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn mean_dim_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert_eq!(a.mean_dim(0, false).unwrap().to_vec(), vec![2.5, 3.5, 4.5]);
    assert_eq!(a.mean_dim(1, false).unwrap().to_vec(), vec![2.0, 5.0]);
}

#[test]
fn mean_dim_out_of_range_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!(a.mean_dim(2, false).is_err());
}

#[test]
fn mean_dim_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    grad_check(&[a.clone()], |t| t[0].mean_dim(1, false).unwrap().sum());
    grad_check(&[a], |t| t[0].mean_dim(0, true).unwrap().sum());
}
