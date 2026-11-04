use ferro_core::{capture, DType, Device, Error, Tensor};
use ferro_core::testkit::grad_check;

#[test]
fn bool_storage_views_casts_and_device_rejection() {
    let x = Tensor::from_vec_bool(vec![true, false, false, true, true, false], &[2, 3]).unwrap();
    assert_eq!(x.dtype(), DType::Bool);
    assert_eq!(x.transpose(0, 1).unwrap().reshape(&[6]).unwrap().to_vec_bool(), [true, true, false, true, false, false]);
    assert_eq!(x.to_dtype(DType::I64).to_vec_i64(), [1, 0, 0, 1, 1, 0]);
    assert_eq!(x.to_dtype(DType::F32).to_vec(), [1., 0., 0., 1., 1., 0.]);
    assert_eq!(x.detach_copy().dtype(), DType::Bool);
    assert!(x.requires_grad_(true).is_err());
    assert!(x.to_device(Device::Cuda(0)).is_err());
    assert!(Tensor::from_vec_bool(vec![true], &[0]).is_err());
    let empty = Tensor::from_vec_bool(vec![], &[2, 0, 3]).unwrap();
    assert_eq!(empty.transpose(0, 2).unwrap().reshape(&[3, 0, 2]).unwrap().dtype(), DType::Bool);
    assert_eq!(Tensor::cat(&[empty.clone(), empty], 0).unwrap().shape(), &[4, 0, 3]);
}

#[test]
fn comparisons_preserve_integer_and_float_precision() {
    let ints = Tensor::from_vec_i64(vec![i64::MIN, (1 << 53) + 1, i64::MAX], &[3]).unwrap();
    let rhs = Tensor::from_vec_i64(vec![i64::MIN + 1, 1 << 53, i64::MAX], &[3]).unwrap();
    assert_eq!(ints.lt_tensor(&rhs).unwrap().to_vec_bool(), [true, false, false]);
    assert_eq!(ints.eq_tensor(&rhs).unwrap().to_vec_bool(), [false, false, true]);
    let x = Tensor::from_vec_f64(vec![1e-300, 0., f64::NAN, f64::INFINITY], &[2, 2]).unwrap();
    let zero = Tensor::from_vec_f64(vec![0.], &[]).unwrap();
    assert_eq!(x.gt_tensor(&zero).unwrap().to_vec_bool(), [true, false, false, true]);
    assert_eq!(x.ne_tensor(&x).unwrap().to_vec_bool(), [false, false, true, false]);
    assert_eq!(x.isnan().unwrap().to_vec_bool(), [false, false, true, false]);
    assert_eq!(x.isfinite().unwrap().to_vec_bool(), [true, true, false, false]);
    let differentiable = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    assert!(!differentiable.eq_tensor(&differentiable).unwrap().requires_grad());
    assert_eq!(ints.eq_tensor(&zero).unwrap().to_vec_bool(), [false, false, false]);
}

#[test]
fn logical_reductions_counts_and_nonzero_are_typed() {
    let x = Tensor::from_vec_bool(vec![true, false, false, true, true, false], &[2, 3]).unwrap();
    let other = Tensor::from_vec_bool(vec![true, false, true], &[1, 3]).unwrap();
    assert_eq!(x.logical_and(&other).unwrap().to_vec_bool(), [true, false, false, true, false, false]);
    assert_eq!(x.logical_xor(&other).unwrap().to_vec_bool(), [false, false, true, false, true, true]);
    assert_eq!(x.bool_reduce(Some(0), false, false).unwrap().to_vec_bool(), [true, true, false]);
    assert_eq!(x.bool_reduce(Some(1), true, true).unwrap().shape(), &[2, 1]);
    assert_eq!(x.sum().dtype(), DType::I64);
    assert_eq!(x.sum().to_vec_i64(), [3]);
    assert_eq!(x.sum_dim(0, false).unwrap().to_vec_i64(), [2, 1, 0]);
    assert_eq!(x.nonzero().unwrap().to_vec_i64(), [0, 0, 1, 0, 1, 1]);
    let empty = Tensor::from_vec_bool(vec![], &[2, 0]).unwrap();
    assert_eq!(empty.bool_reduce(Some(1), false, true).unwrap().to_vec_bool(), [true, true]);
    assert_eq!(empty.bool_reduce(Some(1), false, false).unwrap().to_vec_bool(), [false, false]);
    assert_eq!(empty.sum_dim(1, false).unwrap().to_vec_i64(), [0, 0]);
    for value in [true, false] {
        let scalar = Tensor::from_vec_bool(vec![value], &[]).unwrap();
        assert_eq!(scalar.nonzero().unwrap().shape(), &[usize::from(value), 0]);
    }
    let numeric = Tensor::from_vec_f64(vec![0., -0., 1e-300, f64::NAN], &[4]).unwrap();
    assert_eq!(numeric.to_bool().unwrap().to_vec_bool(), [false, false, true, true]);
}

#[test]
fn bool_mask_selection_preserves_dtype_and_f32_broadcast_adjoint() {
    let mask = Tensor::from_vec_bool(vec![true, true, false, true, false, true], &[2, 3]).unwrap();
    let x = Tensor::from_vec(vec![2., 3., 4.], &[1, 3]).unwrap().requires_grad_(true).unwrap();
    let selected = x.masked_select(&mask).unwrap();
    assert_eq!(selected.to_vec(), [2., 3., 2., 4.]);
    selected.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), [2., 1., 1.]);
    let integer = Tensor::from_vec_i64(vec![i64::MIN, i64::MAX, (1 << 53) + 1], &[1, 3]).unwrap();
    assert_eq!(integer.masked_select(&mask).unwrap().to_vec_i64(), [i64::MIN, i64::MAX, i64::MIN, (1 << 53) + 1]);
    assert_eq!(mask.masked_select(&mask).unwrap().to_vec_bool(), [true; 4]);
    assert_eq!(mask.mask_index(&Tensor::from_vec_bool(vec![false, true], &[2]).unwrap()).unwrap().shape(), &[1, 3]);
    let none = Tensor::from_vec_bool(vec![false], &[]).unwrap();
    assert_eq!(integer.mask_index(&none).unwrap().shape(), &[0, 1, 3]);
    let check = Tensor::from_vec(vec![0.5, -0.4, 0.7], &[1, 3]).unwrap();
    grad_check(&[check], |t| t[0].masked_select(&mask).unwrap().sum());
}

#[test]
fn where_gather_and_masked_fill_keep_boolean_storage() {
    let condition = Tensor::from_vec_bool(vec![true, false], &[2, 1]).unwrap();
    let a = Tensor::from_vec_i64(vec![i64::MAX, i64::MIN], &[1, 2]).unwrap();
    let b = Tensor::from_vec_i64(vec![2, 3], &[1, 2]).unwrap();
    assert_eq!(Tensor::where_cond(&condition, &a, &b).unwrap().to_vec_i64(), [i64::MAX, i64::MIN, 2, 3]);
    let bools = Tensor::from_vec_bool(vec![true, false, false, true], &[2, 2]).unwrap();
    let indices = Tensor::from_vec_i64(vec![1, 0, 0, 0], &[2, 2]).unwrap();
    assert_eq!(bools.gather(1, &indices).unwrap().to_vec_bool(), [false, true, false, false]);
    assert_eq!(bools.masked_fill(&condition, 0.).unwrap().to_vec_bool(), [false, false, false, true]);
    let left = Tensor::from_vec(vec![0.2, 0.4], &[1, 2]).unwrap();
    let right = Tensor::from_vec(vec![0.6, 0.8, -0.3, -0.5], &[2, 2]).unwrap();
    grad_check(&[left, right], |t| Tensor::where_cond(&condition, &t[0], &t[1]).unwrap().sum());
}

#[test]
fn boolean_capture_errors_and_invalid_masks_are_explicit() {
    let mask = Tensor::from_vec_bool(vec![true, false], &[2]).unwrap();
    let x = Tensor::ones(&[2]);
    for result in [capture(|| x.eq_tensor(&x)), capture(|| mask.logical_not()), capture(|| x.masked_select(&mask)),
                   capture(|| mask.bool_sum(None, false)), capture(|| Tensor::where_cond(&mask, &x, &x))] {
        assert!(matches!(result, Err(Error::Unsupported { .. })));
    }
    assert!(x.masked_select(&x).is_err());
    assert!(x.mask_index(&Tensor::from_vec_bool(vec![true; 3], &[3]).unwrap()).is_err());
    assert!(mask.bool_reduce(Some(1), false, true).is_err());
    assert_eq!(x.add(&mask).unwrap().to_vec(), vec![2.0, 1.0]);
}

#[test]
fn unsupported_boolean_numeric_kernels_reject_before_casting() {
    let mask = Tensor::from_vec_bool(vec![true, false, false, true], &[2, 2]).unwrap();
    for result in [mask.softmax(1), mask.log_softmax(1), mask.prod_dim(1, false),
                   mask.argmax(1, false), mask.argmin(1, false), mask.topk(1, 1).map(|(v, _)| v)] {
        assert!(matches!(result, Err(Error::DtypeMismatch { got: DType::Bool, .. })));
    }
    assert_eq!(mask.cumsum(1).unwrap().to_vec_i64(), vec![1, 1, 0, 1]);
    let matrix = mask.reshape(&[1, 2, 2]).unwrap();
    assert!(matches!(matrix.bmm(&matrix), Err(Error::DtypeMismatch { got: DType::Bool, .. })));
    assert!(matches!(Tensor::ones(&[1, 2, 2]).bmm(&matrix), Err(Error::DtypeMismatch { got: DType::Bool, .. })));
}

#[test]
fn empty_boolean_reduction_output_capacity_is_fallible() {
    for shape in [[usize::MAX, 0], [0, usize::MAX]] {
        let x = Tensor::from_vec_bool(vec![], &shape).unwrap();
        let zero_dim = usize::from(shape[0] != 0);
        for result in [x.bool_reduce(Some(zero_dim), false, true), x.bool_reduce(Some(zero_dim), false, false),
                       x.bool_sum(Some(zero_dim), false)] {
            assert!(matches!(result, Err(Error::InvalidShape { .. })));
        }
        assert_eq!(x.bool_reduce(Some(1 - zero_dim), false, true).unwrap().shape(), &[0]);
    }
    let x = Tensor::from_vec_bool(vec![], &[isize::MAX as usize / 8 + 1, 0]).unwrap();
    assert!(matches!(x.bool_sum(Some(1), false), Err(Error::InvalidShape { .. })));
    for value in [true, false] {
        let scalar = Tensor::from_vec_bool(vec![value], &[]).unwrap();
        assert_eq!(scalar.bool_reduce(Some(0), true, true).unwrap().to_vec_bool(), [value]);
        assert_eq!(scalar.bool_sum(Some(0), true).unwrap().shape(), &[]);
    }
}
