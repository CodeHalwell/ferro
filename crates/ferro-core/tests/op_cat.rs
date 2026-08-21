use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn cat_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let c0 = Tensor::cat(&[a.clone(), b.clone()], 0).unwrap();
    assert_eq!(c0.shape(), &[4, 2]);
    assert_eq!(c0.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let c1 = Tensor::cat(&[a.clone(), b.clone()], 1).unwrap();
    assert_eq!(c1.shape(), &[2, 4]);
    assert_eq!(c1.to_vec(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);

    let d = Tensor::from_vec(vec![9.0, 10.0], &[1, 2]).unwrap();
    let c3 = Tensor::cat(&[a, b, d], 0).unwrap();
    assert_eq!(c3.shape(), &[5, 2]);
    assert_eq!(c3.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
}

#[test]
fn cat_shape_errors() {
    let a = Tensor::from_vec(vec![0.0; 4], &[2, 2]).unwrap();
    // non-cat dim mismatch: [2,2] vs [2,3] along dim 0
    let b = Tensor::from_vec(vec![0.0; 6], &[2, 3]).unwrap();
    assert!(Tensor::cat(&[a.clone(), b], 0).is_err());
    // dim out of range
    assert!(Tensor::cat(&[a.clone(), a.clone()], 2).is_err());
    // empty list
    assert!(Tensor::cat(&[], 0).is_err());
}

#[test]
fn cat_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.0, 2.0, 1.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.3, -0.7, 1.0, -0.2], &[2, 2]).unwrap();
    // Weight makes the loss position-dependent so per-slot grads differ.
    let w1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
    grad_check(&[a.clone(), b.clone()], |t| {
        Tensor::cat(&[t[0].clone(), t[1].clone()], 1).unwrap().mul(&w1).unwrap().sum()
    });

    let w0 = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], &[4, 2]).unwrap();
    grad_check(&[a, b], |t| {
        Tensor::cat(&[t[0].clone(), t[1].clone()], 0).unwrap().mul(&w0).unwrap().sum()
    });
}

#[test]
fn cat_rejects_non_f32() {
    let ids = Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap();
    let more = Tensor::from_vec_i64(vec![3], &[1]).unwrap();
    assert!(Tensor::cat(&[ids, more], 0).is_err());
}
