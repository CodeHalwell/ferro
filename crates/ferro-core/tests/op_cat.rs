use ferro_core::testkit::grad_check;
use ferro_core::{DType, Error, Tensor};

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
    assert_eq!(
        c3.to_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    );
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
        Tensor::cat(&[t[0].clone(), t[1].clone()], 1)
            .unwrap()
            .mul(&w1)
            .unwrap()
            .sum()
    });

    let w0 = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], &[4, 2]).unwrap();
    grad_check(&[a, b], |t| {
        Tensor::cat(&[t[0].clone(), t[1].clone()], 0)
            .unwrap()
            .mul(&w0)
            .unwrap()
            .sum()
    });
}

#[test]
fn cat_preserves_large_i64_exactly() {
    let ids = Tensor::from_vec_i64(vec![i64::MAX, 16_777_217], &[2, 1]).unwrap();
    let more = Tensor::from_vec_i64(vec![-9_007_199_254_740_993, i64::MIN], &[2, 1]).unwrap();
    let out = Tensor::cat(&[ids, more], 1).unwrap();
    assert_eq!(out.dtype(), DType::I64);
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.to_vec_i64(), vec![i64::MAX, -9_007_199_254_740_993, 16_777_217, i64::MIN]);
}

#[test]
fn cat_rejects_mismatched_dtypes() {
    let ids = Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap();
    let floats = Tensor::from_vec(vec![3.], &[1]).unwrap();
    assert!(matches!(Tensor::cat(&[ids.clone(), floats.clone()], 0),
        Err(Error::DtypeMismatch { op: "cat", expected: DType::I64, got: DType::F32 })));
    assert!(matches!(Tensor::cat(&[floats, ids], 0),
        Err(Error::DtypeMismatch { op: "cat", expected: DType::F32, got: DType::I64 })));
}
