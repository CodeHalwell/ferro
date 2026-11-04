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

#[test]
fn cat_only_saves_metadata_for_ordinary_and_functional_backwards() {
    for dtype in [DType::F32, DType::F64] { for functional in [false, true] { for output in [false, true] {
        let x = Tensor::from_vec_f64(vec![1., 2., 3., 4.], &[2, 2]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let y = Tensor::cat(&[x.clone(), x.clone()], 1).unwrap();
        if output { y._bump_version_for_test(); } else { x._bump_version_for_test(); }
        let seed = Tensor::from_vec_f64((1..=8).map(|x| x as f64).collect(), &[2, 4]).unwrap().to_dtype(dtype);
        let gradient = if functional { y.vjp_wrt(&[&x], &seed, false).unwrap().remove(0) }
            else { y.backward_with(&seed); x.grad().unwrap() };
        assert_eq!(gradient.to_dtype(DType::F64).to_vec_f64(), vec![4., 6., 12., 14.]);
        if functional { assert!(x.grad().is_none()); }
    } } }
}

#[test]
fn cat_metadata_omission_preserves_higher_derivatives_and_live_cotangents() {
    let x = Tensor::from_vec(vec![0.5, -2.], &[2]).unwrap().requires_grad_(true).unwrap();
    let y = Tensor::cat(&[x.clone(), x.clone()], 0).unwrap();
    let loss = y.square().unwrap().sum();
    x._bump_version_for_test();
    let first = loss.grad_wrt(&[&x], true).unwrap().remove(0);
    assert_eq!(first.to_vec(), vec![2., -8.]);
    assert_eq!(first.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec(), vec![4., 4.]);
    let other = Tensor::from_vec(vec![3., 4.], &[2]).unwrap().requires_grad_(true).unwrap();
    let y = Tensor::cat(&[x.clone(), other], 0).unwrap();
    let seed = Tensor::from_vec(vec![1., 2., 3., 4.], &[4]).unwrap().requires_grad_(true).unwrap();
    x._bump_version_for_test();
    let first = y.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
    assert!(first.requires_grad());
    assert_eq!(first.to_vec(), vec![1., 2.]);
    assert_eq!(first.sum().grad_wrt(&[&seed], false).unwrap()[0].to_vec(), vec![1., 1., 0., 0.]);
    assert!(x.grad().is_none()); assert!(seed.grad().is_none());
}

#[test]
fn cat_does_not_weaken_nonlinear_producer_or_consumer_guards() {
    for producer in [false, true] { for modify_source in [false, true] {
        let x = Tensor::from_vec(vec![0.5, -2.], &[2]).unwrap().requires_grad_(true).unwrap();
        let (middle, loss) = if producer {
            let middle = x.square().unwrap();
            let loss = Tensor::cat(&[middle.clone(), middle.clone()], 0).unwrap().sum();
            (middle, loss)
        } else {
            let middle = Tensor::cat(&[x.clone(), x.clone()], 0).unwrap();
            let loss = middle.square().unwrap().sum();
            (middle, loss)
        };
        if modify_source { x._bump_version_for_test(); } else { middle._bump_version_for_test(); }
        let result = loss.grad_wrt(&[&x], false);
        assert_eq!(result.is_err(), producer == modify_source);
        if let Ok(gradient) = result { assert_eq!(gradient[0].to_vec(), vec![2., -8.]); }
        assert!(x.grad().is_none());
    } }
}
