use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn cumsum_values_1d() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    assert_eq!(a.cumsum(0).unwrap().to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn cumsum_values_2d_both_dims() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert_eq!(
        a.cumsum(0).unwrap().to_vec(),
        vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
    );
    assert_eq!(
        a.cumsum(1).unwrap().to_vec(),
        vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
    );
}

#[test]
fn cumsum_rejects_bad_dim() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    assert!(a.cumsum(1).is_err());
}

#[test]
fn cumsum_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.2, 0.8, 1.4, -0.3, 0.9], &[2, 3]).unwrap();
    grad_check(&[a.clone()], |t| {
        t[0].cumsum(1).unwrap().mul(&t[0]).unwrap().sum()
    });
    grad_check(&[a], |t| t[0].cumsum(0).unwrap().mul(&t[0]).unwrap().sum());
}
