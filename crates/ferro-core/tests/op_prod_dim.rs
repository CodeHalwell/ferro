use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn prod_dim_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let p0 = a.prod_dim(0, false).unwrap();
    assert_eq!(p0.shape(), &[3]);
    assert_eq!(p0.to_vec(), vec![4.0, 10.0, 18.0]);

    let p1 = a.prod_dim(1, true).unwrap();
    assert_eq!(p1.shape(), &[2, 1]);
    assert_eq!(p1.to_vec(), vec![6.0, 120.0]);
}

#[test]
fn prod_dim_out_of_range_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!(a.prod_dim(2, false).is_err());
}

#[test]
fn prod_dim_grad() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 3.0, -0.5, 2.0, -1.5], &[2, 3]).unwrap();
    grad_check(&[a], |t| t[0].prod_dim(1, false).unwrap().sum());

    let b = Tensor::from_vec(vec![1.0, -2.0, 3.0, -0.5, 2.0, -1.5], &[2, 3]).unwrap();
    grad_check(&[b], |t| t[0].prod_dim(0, true).unwrap().sum());
}
