use ferro_core::autograd::with_grad_enabled;
use ferro_core::{capture, DType, Error, Tensor};

fn double(values: &[f64], shape: &[usize]) -> Tensor { Tensor::from_vec_f64(values.to_vec(), shape).unwrap() }

#[test]
fn exact_typed_copy_preserves_storage_aliases_and_handles_self_transpose() {
    let dst = double(&[0.0; 4], &[2, 2]);
    let alias = dst.transpose(0, 1).unwrap();
    let id = dst.id();
    let values = [1.0 + 2f64.powi(-40), 1e100, 1e-300, -0.0];
    dst.copy_from(&double(&values, &[2, 2])).unwrap();
    assert_eq!(dst.id(), id);
    assert_eq!(dst.to_vec_f64(), values);
    assert_eq!(alias.to_vec_f64(), vec![values[0], values[2], values[1], values[3]]);
    dst.copy_from(&alias).unwrap();
    assert_eq!(dst.to_vec_f64(), vec![values[0], values[2], values[1], values[3]]);
    assert_eq!(alias.to_vec_f64(), values);
    let ints = Tensor::from_vec_i64(vec![0; 4], &[2, 2]).unwrap();
    let large = Tensor::from_vec_i64(vec![i64::MIN, i64::MAX, (1 << 53) + 1, 7], &[2, 2]).unwrap();
    ints.copy_from(&large.transpose(0, 1).unwrap()).unwrap();
    assert_eq!(ints.to_vec_i64(), vec![i64::MIN, (1 << 53) + 1, i64::MAX, 7]);
    let bools = Tensor::from_vec_bool(vec![false; 4], &[2, 2]).unwrap();
    bools.copy_from(&Tensor::from_vec_bool(vec![true, false, true, true], &[2, 2]).unwrap()).unwrap();
    assert_eq!(bools.to_vec_bool(), vec![true, false, true, true]);
    for dtype in [DType::F64, DType::I64, DType::Bool] {
        let empty = Tensor::zeros(&[2, 0]).to_dtype(dtype);
        empty.copy_from(&Tensor::zeros(&[2, 0]).to_dtype(dtype)).unwrap();
        assert_eq!(empty.dtype(), dtype);
    }
}

#[test]
fn half_copies_preserve_raw_nan_payloads_and_subnormal_bits() {
    let bits = vec![0x7c01, 0x8000, 0x0001, 0xffff];
    let f16 = Tensor::from_vec_f16_bits(vec![0; 4], &[2, 2]).unwrap();
    let bf16 = Tensor::from_vec_bf16_bits(vec![0; 4], &[2, 2]).unwrap();
    f16.copy_from(&Tensor::from_vec_f16_bits(bits.clone(), &[2, 2]).unwrap()).unwrap();
    bf16.copy_from(&Tensor::from_vec_bf16_bits(bits.clone(), &[2, 2]).unwrap()).unwrap();
    assert_eq!(f16.to_vec_f16_bits().unwrap(), bits);
    assert_eq!(bf16.to_vec_bf16_bits().unwrap(), bits);
}

#[test]
fn leaf_copy_retains_id_grad_slot_and_shared_version_semantics() {
    let x = double(&[2.0, 3.0], &[2]).requires_grad_(true).unwrap();
    x.square().unwrap().sum().backward();
    let grad_id = x.grad().unwrap().id();
    let id = x.id();
    let saved = x.square().unwrap().sum();
    let alias = with_grad_enabled(false, || x.reshape(&[1, 2]).unwrap());
    with_grad_enabled(false, || x.copy_leaf_from(&x)).unwrap();
    assert_eq!(saved.grad_wrt(&[&x], false).unwrap()[0].to_vec_f64(), vec![4.0, 6.0]);
    with_grad_enabled(false, || x.copy_leaf_from(&double(&[5.0, 7.0], &[2]))).unwrap();
    assert_eq!(x.id(), id);
    assert_eq!(x.grad().unwrap().id(), grad_id);
    assert_eq!(x.grad().unwrap().to_vec_f64(), vec![4.0, 6.0]);
    assert_eq!(alias.to_vec_f64(), vec![5.0, 7.0]);
    assert!(saved.grad_wrt(&[&x], false).is_err());
    x.zero_grad();
    x.square().unwrap().sum().backward();
    assert_eq!(x.grad().unwrap().to_vec_f64(), vec![10.0, 14.0]);
}

#[test]
fn failed_typed_copies_leave_values_and_saved_versions_unchanged() {
    let base = double(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let leaf = base.requires_grad_(true).unwrap();
    let saved = leaf.square().unwrap().sum();
    assert!(matches!(base.copy_from(&Tensor::ones(&[2, 2])), Err(Error::DtypeMismatch { .. })));
    assert!(matches!(base.copy_from(&double(&[1.0], &[1])), Err(Error::ShapeMismatch { .. })));
    assert!(base.transpose(0, 1).unwrap().copy_from(&double(&[1.0; 4], &[2, 2])).is_err());
    assert!(capture(|| base.copy_from(&double(&[1.0; 4], &[2, 2]))).is_err());
    assert!(leaf.copy_leaf_from(&base).is_err());
    assert!(leaf.copy_from(&base).is_err());
    assert_eq!(base.to_vec_f64(), vec![2.0, 3.0, 4.0, 5.0]);
    assert_eq!(saved.grad_wrt(&[&leaf], false).unwrap()[0].to_vec_f64(), vec![4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn typed_copy_never_silently_loses_source_history() {
    let source = double(&[2.0, 3.0], &[2]).requires_grad_(true).unwrap();
    let dst = double(&[0.0, 0.0], &[2]);
    assert!(dst.copy_from(&source).is_err());
    assert_eq!(dst.to_vec_f64(), vec![0.0, 0.0]);
    with_grad_enabled(false, || dst.copy_from(&source)).unwrap();
    assert_eq!(dst.to_vec_f64(), vec![2.0, 3.0]);
    assert!(!dst.requires_grad());
    let history = source.square().unwrap();
    assert!(with_grad_enabled(false, || history.copy_leaf_from(&dst)).is_err());
}
