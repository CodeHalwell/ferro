use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn topk_values_and_indices() {
    let a = Tensor::from_vec(vec![1.0, 4.0, 2.0, 5.0, 0.0, 3.0], &[2, 3]).unwrap();
    let (v, i) = a.topk(2, 1).unwrap();
    assert_eq!(v.shape(), &[2, 2]);
    assert_eq!(v.to_vec(), vec![4.0, 2.0, 5.0, 3.0]);
    assert_eq!(i.to_vec_i64(), vec![1, 2, 0, 2]);
}

#[test]
fn topk_along_dim0() {
    let a = Tensor::from_vec(vec![1.0, 4.0, 2.0, 5.0, 0.0, 3.0], &[2, 3]).unwrap();
    let (v, i) = a.topk(1, 0).unwrap();
    assert_eq!(v.shape(), &[1, 3]);
    assert_eq!(v.to_vec(), vec![5.0, 4.0, 3.0]);
    assert_eq!(i.to_vec_i64(), vec![1, 0, 1]);
}

#[test]
fn topk_ties_take_lowest_index() {
    let a = Tensor::from_vec(vec![2.0, 3.0, 2.0], &[3]).unwrap();
    let (v, i) = a.topk(3, 0).unwrap();
    assert_eq!(v.to_vec(), vec![3.0, 2.0, 2.0]);
    assert_eq!(i.to_vec_i64(), vec![1, 0, 2]);
}

#[test]
fn topk_rejects_bad_args() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    assert!(a.topk(3, 0).is_err());
    assert!(a.topk(1, 1).is_err());
}

#[test]
fn topk_grad() {
    let a = Tensor::from_vec(vec![0.9, -0.4, 1.6, 0.1, -1.2, 0.7], &[2, 3]).unwrap();
    grad_check(&[a], |t| {
        let (v, _) = t[0].topk(2, 1).unwrap();
        v.mul(&v).unwrap().sum()
    });
}
