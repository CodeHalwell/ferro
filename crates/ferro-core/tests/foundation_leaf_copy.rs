use ferro_core::autograd::with_grad_enabled;
use ferro_core::{Device, Error, Tensor};

fn leaf() -> Tensor { Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap().requires_grad_(true).unwrap() }

#[test]
fn checked_leaf_copy_preserves_identity_aliases_and_invalidates_saved_graphs() {
    let x = leaf();
    let alias = x.reshape(&[1, 2]).unwrap();
    let y = x.mul(&x).unwrap().sum();
    y.backward();
    let id = x.id();
    with_grad_enabled(false, || x.copy_leaf_from(&Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap()).unwrap());
    assert_eq!(x.id(), id);
    assert!(x.requires_grad());
    assert_eq!(alias.to_vec(), vec![3.0, 4.0]);
    assert_eq!(x.grad().unwrap().to_vec(), vec![2.0, 4.0]);
    assert!(matches!(y.grad_wrt(&[&x], false), Err(Error::Unsupported { .. })));
    assert_eq!(x.mul(&x).unwrap().sum().grad_wrt(&[&x], false).unwrap()[0].to_vec(), vec![6.0, 8.0]);
}

#[test]
fn leaf_copy_rejects_enabled_mode_history_dtype_shape_and_nonwhole_views() {
    let x = leaf();
    let y = x.mul(&x).unwrap();
    let before = y.sum();
    let zeros = Tensor::zeros(&[2]);
    assert!(x.copy_leaf_from(&zeros).is_err());
    with_grad_enabled(false, || {
        assert!(x.copy_from(&zeros).is_err(), "general mutation gate must not weaken");
        assert!(y.copy_leaf_from(&zeros).is_err());
        assert!(x.copy_leaf_from(&Tensor::zeros(&[1, 2])).is_err());
        assert!(x.copy_leaf_from(&Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap()).is_err());
        let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let view = matrix.transpose(0, 1).unwrap();
        assert!(view.copy_leaf_from(&Tensor::zeros(&[2, 2])).is_err());
        assert!(Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap().copy_leaf_from(&zeros).is_err());
    });
    assert_eq!(x.to_vec(), vec![1.0, 2.0]);
    assert_eq!(before.grad_wrt(&[&x], false).unwrap()[0].to_vec(), vec![2.0, 4.0]);
}

#[test]
fn leaf_copy_device_alias_and_device_mismatch_reject_before_dispatch() {
    use ferro_core::dispatch::{Backend, BinaryKind, DeviceBuffer, UnaryKind};
    use std::sync::{Arc, Mutex};
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const DEV: Device = Device::Cuda(232);
    struct Buffer(usize);
    impl DeviceBuffer for Buffer {
        fn device(&self) -> Device { DEV }
        fn len(&self) -> usize { self.0 }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }
    struct RefuseCompute;
    impl Backend for RefuseCompute {
        fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
        fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
        fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected compute") }
        fn alloc_from_host(&self, data: &[f32]) -> ferro_core::Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buffer(data.len()))) }
        fn copy_to_host(&self, _: &dyn DeviceBuffer) -> ferro_core::Result<Vec<f32>> { panic!("must reject before download") }
    }
    ferro_core::register_backend(DEV, Arc::new(RefuseCompute));
    let dst = Tensor::zeros(&[2]).to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let src = Tensor::zeros(&[2]).to_device(DEV).unwrap();
    let alias = dst.requires_grad_(false).unwrap();
    with_grad_enabled(false, || {
        assert!(matches!(dst.copy_leaf_from(&src), Err(Error::Unsupported { .. })));
        drop(alias);
        assert!(matches!(dst.copy_leaf_from(&Tensor::zeros(&[2])), Err(Error::DeviceMismatch { .. })));
        assert!(matches!(leaf().copy_leaf_from(&src), Err(Error::DeviceMismatch { .. })));
    });
}
