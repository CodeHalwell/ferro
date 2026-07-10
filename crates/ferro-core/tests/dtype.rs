use ferro_core::ops_ext::embedding;
use ferro_core::testkit::grad_check;
use ferro_core::{DType, Error, Tensor};

#[test]
fn creation_and_dtype() {
    let f = Tensor::from_vec(vec![1.5, -2.0], &[2]).unwrap();
    assert_eq!(f.dtype(), DType::F32);

    let d = Tensor::from_vec_f64(vec![1.5, -2.0], &[2]).unwrap();
    assert_eq!(d.dtype(), DType::F64);
    assert_eq!(d.shape(), &[2]);

    let i = Tensor::from_vec_i64(vec![3, -4], &[2]).unwrap();
    assert_eq!(i.dtype(), DType::I64);
    assert_eq!(i.shape(), &[2]);

    assert!(Tensor::from_vec_f64(vec![1.0], &[2]).is_err());
    assert!(Tensor::from_vec_i64(vec![1, 2, 3], &[2]).is_err());
}

#[test]
fn to_vec_casts_between_dtypes() {
    let f = Tensor::from_vec(vec![1.9, -2.7, 3.0], &[3]).unwrap();
    assert_eq!(f.to_vec(), vec![1.9, -2.7, 3.0]);
    assert_eq!(f.to_vec_f64(), vec![1.9f32 as f64, -2.7f32 as f64, 3.0]);
    // Float -> i64 truncates toward zero.
    assert_eq!(f.to_vec_i64(), vec![1, -2, 3]);

    let d = Tensor::from_vec_f64(vec![0.5, -1.25, 2.0], &[3]).unwrap();
    assert_eq!(d.to_vec_f64(), vec![0.5, -1.25, 2.0]);
    assert_eq!(d.to_vec(), vec![0.5f32, -1.25, 2.0]);
    assert_eq!(d.to_vec_i64(), vec![0, -1, 2]);

    let i = Tensor::from_vec_i64(vec![7, -8, 0], &[3]).unwrap();
    assert_eq!(i.to_vec_i64(), vec![7, -8, 0]);
    assert_eq!(i.to_vec(), vec![7.0f32, -8.0, 0.0]);
    assert_eq!(i.to_vec_f64(), vec![7.0f64, -8.0, 0.0]);
}

#[test]
fn arange_builds_i64_range() {
    let t = Tensor::arange(5);
    assert_eq!(t.dtype(), DType::I64);
    assert_eq!(t.shape(), &[5]);
    assert_eq!(t.to_vec_i64(), vec![0, 1, 2, 3, 4]);

    assert_eq!(Tensor::arange(0).shape(), &[0]);
    assert_eq!(Tensor::arange(-3).shape(), &[0]);
}

#[test]
fn to_dtype_converts_and_detaches() {
    let i = Tensor::arange(4);
    let f = i.to_dtype(DType::F32);
    assert_eq!(f.dtype(), DType::F32);
    assert_eq!(f.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    assert!(!f.requires_grad());

    let d = f.to_dtype(DType::F64);
    assert_eq!(d.dtype(), DType::F64);
    assert_eq!(d.to_vec_f64(), vec![0.0, 1.0, 2.0, 3.0]);

    let back = d.to_dtype(DType::I64);
    assert_eq!(back.dtype(), DType::I64);
    assert_eq!(back.to_vec_i64(), vec![0, 1, 2, 3]);

    // Casting from a grad-tracked f32 tensor drops autograd history.
    let leaf = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap().requires_grad_(true);
    assert!(!leaf.to_dtype(DType::F64).requires_grad());
}

#[test]
fn strided_i64_view_materializes() {
    let t = Tensor::from_vec_i64(vec![1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let tt = t.transpose(0, 1).unwrap();
    assert_eq!(tt.dtype(), DType::I64);
    assert_eq!(tt.shape(), &[3, 2]);
    assert_eq!(tt.to_vec_i64(), vec![1, 4, 2, 5, 3, 6]);
    assert_eq!(tt.to_vec(), vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn reshape_of_strided_view_keeps_dtype() {
    // Materializing a non-contiguous view for reshape must not round-trip
    // through f32: large ids would lose precision before one_hot etc.
    let big = (1i64 << 40) + 1;
    let t = Tensor::from_vec_i64(vec![big, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let r = t.transpose(0, 1).unwrap().reshape(&[6]).unwrap();
    assert_eq!(r.dtype(), DType::I64);
    assert_eq!(r.to_vec_i64(), vec![big, 4, 2, 5, 3, 6]);

    let d = Tensor::ones(&[2, 3]).to_dtype(DType::F64);
    let rd = d.transpose(0, 1).unwrap().reshape(&[6]).unwrap();
    assert_eq!(rd.dtype(), DType::F64);
}

#[test]
#[should_panic(expected = "autograd is f32-only")]
fn requires_grad_on_i64_panics() {
    Tensor::arange(3).requires_grad_(true);
}

#[test]
fn float_ops_reject_non_f32() {
    let i = Tensor::arange(4).reshape(&[2, 2]).unwrap();
    let f = Tensor::ones(&[2, 2]);
    let d = f.to_dtype(DType::F64);

    assert!(matches!(i.add(&i), Err(Error::DtypeMismatch { .. })));
    assert!(matches!(f.add(&i), Err(Error::DtypeMismatch { .. })));
    assert!(matches!(d.add(&d), Err(Error::DtypeMismatch { .. })));
    assert!(matches!(d.mul(&f), Err(Error::DtypeMismatch { .. })));
    assert!(matches!(d.matmul(&f), Err(Error::DtypeMismatch { .. })));

    // Explicit cast is the supported route into float math.
    let sum = i.to_dtype(DType::F32).add(&f).unwrap();
    assert_eq!(sum.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn index_select_t_values_and_errors() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

    let s = a.index_select_t(0, &Tensor::from_vec_i64(vec![2, 0], &[2]).unwrap()).unwrap();
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.to_vec(), vec![5.0, 6.0, 1.0, 2.0]);

    let f32_ids = Tensor::from_vec(vec![0.0], &[1]).unwrap();
    assert!(matches!(a.index_select_t(0, &f32_ids), Err(Error::DtypeMismatch { .. })));

    let ids_2d = Tensor::from_vec_i64(vec![0, 1], &[1, 2]).unwrap();
    assert!(a.index_select_t(0, &ids_2d).is_err());
    assert!(a.index_select_t(0, &Tensor::from_vec_i64(vec![-1], &[1]).unwrap()).is_err());
    assert!(a.index_select_t(0, &Tensor::from_vec_i64(vec![3], &[1]).unwrap()).is_err());
}

#[test]
fn index_select_t_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let ids = Tensor::from_vec_i64(vec![1, 1, 0], &[3]).unwrap();
    let w = Tensor::from_vec(vec![0.5, -1.0, 2.0, 3.0, -0.7, 1.2], &[3, 2]).unwrap();
    grad_check(&[a], |t| t[0].index_select_t(0, &ids).unwrap().mul(&w).unwrap().sum());
}

#[test]
fn embedding_forward_and_grad() {
    let weight =
        Tensor::from_vec(vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1], &[4, 2]).unwrap();
    let ids = Tensor::from_vec_i64(vec![3, 0, 3], &[3]).unwrap();
    let out = embedding(&weight, &ids).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.to_vec(), vec![3.0, 3.1, 0.0, 0.1, 3.0, 3.1]);

    assert!(embedding(&Tensor::ones(&[4]), &ids).is_err());
    assert!(embedding(&weight, &Tensor::from_vec_i64(vec![4], &[1]).unwrap()).is_err());

    let w = Tensor::from_vec(vec![0.5, -1.0, 2.0, 3.0, -0.7, 1.2], &[3, 2]).unwrap();
    grad_check(&[weight], |t| embedding(&t[0], &ids).unwrap().mul(&w).unwrap().sum());
}
