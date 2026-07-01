use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn sum_dim_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let s0 = a.sum_dim(0, false);
    assert_eq!(s0.shape(), &[3]);
    assert_eq!(s0.to_vec(), vec![5.0, 7.0, 9.0]);

    let s1 = a.sum_dim(1, true);
    assert_eq!(s1.shape(), &[2, 1]);
    assert_eq!(s1.to_vec(), vec![6.0, 15.0]);
}

#[test]
fn sum_dim_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    grad_check(&[a], |t| t[0].sum_dim(1, false).sum());

    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    grad_check(&[b], |t| t[0].sum_dim(0, true).sum());
}
