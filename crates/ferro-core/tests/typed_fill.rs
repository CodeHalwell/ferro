use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use ferro_core::forward_ad::DualLevel;
use ferro_core::inplace::FillValue;
use ferro_core::{capture, DType, Tensor};

#[test]
fn typed_fills_preserve_exact_values_shared_storage_and_versions() {
    let double = Tensor::from_vec_f64(vec![0.; 3], &[3]).unwrap();
    let alias = double.detach();
    let (id, storage) = (double.id(), double._storage_ptr());
    for (index, value) in [1.0000000000000002, 1e-300, 1e300, -0., f64::from_bits(0x7ff8_0000_0000_1234)].into_iter().enumerate() {
        double.fill_typed_(FillValue::F64(value)).unwrap();
        assert!(alias.to_vec_f64().iter().all(|v| v.to_bits() == value.to_bits()));
        assert_eq!(double._version(), index as u64 + 1);
    }
    assert_eq!((double.id(), double._storage_ptr()), (id, storage));
    let integers = Tensor::from_vec_i64(vec![0; 2], &[2]).unwrap();
    for value in [i64::MIN, i64::MAX, (1i64 << 53) + 1] {
        integers.fill_typed_(FillValue::I64(value)).unwrap();
        assert_eq!(integers.to_vec_i64(), vec![value; 2]);
    }
    let booleans = Tensor::from_vec_bool(vec![false; 2], &[2]).unwrap();
    booleans.fill_typed_(FillValue::Bool(true)).unwrap();
    assert_eq!(booleans.to_vec_bool(), vec![true; 2]);
    for tensor in [&double, &integers, &booleans] {
        let version = tensor._version();
        tensor.zero_().unwrap();
        assert_eq!(tensor.to_vec_f64(), vec![0.; tensor.numel()]);
        assert_eq!(tensor._version(), version + 1);
    }
}

#[test]
fn no_grad_leaf_fill_preserves_handles_slots_and_stale_value_guards() {
    for dtype in [DType::F32, DType::F64] {
        let x = Tensor::ones(&[2]).to_dtype(dtype).requires_grad_(true).unwrap();
        x.sum().backward();
        let previous_grad = x.grad().unwrap();
        let id = x.id();
        let old = x.mul(&x).unwrap().sum();
        let value = if dtype == DType::F64 { FillValue::F64(3.0000000000000004) } else { FillValue::F32(3.) };
        assert!(x.fill_typed_(value).is_err());
        assert_eq!(x._version(), 0);
        with_grad_enabled(false, || x.fill_typed_(value)).unwrap();
        assert!(is_grad_enabled());
        assert_eq!(x.id(), id);
        assert_eq!(x.grad().unwrap().id(), previous_grad.id());
        assert_eq!(x.grad().unwrap().to_vec_f64(), vec![1.; 2]);
        assert!(old.grad_wrt(&[&x], false).is_err());
        let interior = x.mul(&x).unwrap();
        let before = interior.to_vec_f64();
        assert!(with_grad_enabled(false, || interior.fill_typed_(value)).is_err());
        assert_eq!(interior.to_vec_f64(), before);
        with_grad_enabled(false, || x.zero_()).unwrap();
        assert_eq!(x.to_vec_f64(), vec![0.; 2]);
        assert_eq!(x._version(), 2);
    }
}

#[test]
fn invalid_types_layouts_and_recording_modes_reject_before_mutation() {
    let x = Tensor::from_vec_f64(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
    for value in [FillValue::F32(1.), FillValue::I64(1), FillValue::Bool(true)] {
        assert!(x.fill_typed_(value).is_err());
    }
    assert!(x.transpose(0, 1).unwrap().fill_typed_(FillValue::F64(9.)).is_err());
    assert!(x.narrow(0, 1, 1).unwrap().zero_().is_err());
    assert!(capture(|| x.fill_typed_(FillValue::F64(9.))).is_err());
    assert!(!ferro_core::capture::is_recording());
    assert_eq!(x.to_vec_f64(), vec![1., 2., 3., 4.]);
    assert_eq!(x._version(), 0);
    for half in [Tensor::from_vec_f16_bits(vec![0], &[1]).unwrap(), Tensor::from_vec_bf16_bits(vec![0], &[1]).unwrap()] {
        assert!(half.zero_().is_err());
        assert!(half.fill_typed_(FillValue::F32(1.)).is_err());
        assert_eq!(half._version(), 0);
    }
    let primal = Tensor::ones(&[2]);
    let tangent = Tensor::ones(&[2]);
    let level = DualLevel::new(&[&primal], &[&tangent]).unwrap();
    assert!(x.zero_().is_err());
    assert_eq!(x._version(), 0);
    drop(level);
    x.zero_().unwrap();
    assert_eq!(x._version(), 1);
}

#[test]
fn empty_typed_fills_are_valid_versioned_mutations() {
    for (tensor, value) in [
        (Tensor::zeros(&[0, 2]), FillValue::F32(-0.)),
        (Tensor::from_vec_f64(vec![], &[0, 2]).unwrap(), FillValue::F64(1e300)),
        (Tensor::from_vec_i64(vec![], &[0, 2]).unwrap(), FillValue::I64(i64::MAX)),
        (Tensor::from_vec_bool(vec![], &[0, 2]).unwrap(), FillValue::Bool(true)),
    ] {
        tensor.fill_typed_(value).unwrap();
        assert_eq!(tensor.numel(), 0);
        assert_eq!(tensor._version(), 1);
        tensor.zero_().unwrap();
        assert_eq!(tensor._version(), 2);
    }
}
