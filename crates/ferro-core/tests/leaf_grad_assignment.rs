use ferro_core::{capture, DType, Device, Error, Tensor};
use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

#[test]
fn direct_assignment_preserves_typed_handles_history_and_storage() {
    for dtype in [DType::F32, DType::F64] {
        let target = Tensor::ones(&[2]).to_dtype(dtype).requires_grad_(true).unwrap();
        let source = Tensor::ones(&[2]).to_dtype(dtype).requires_grad_(true).unwrap();
        let callbacks = Arc::new(AtomicUsize::new(0));
        let seen = callbacks.clone();
        let gradient = Tensor::from_vec_f64(vec![1.0 + 2f64.powi(-40), 1e-30], &[2]).unwrap().to_dtype(dtype)
            .record_fn(vec![source.clone()], move |g| { seen.fetch_add(1, Ordering::Relaxed); vec![g.clone()] });
        let id = target.id();
        target.set_leaf_grad(Some(gradient.clone())).unwrap();
        assert_eq!(callbacks.load(Ordering::Relaxed), 0);
        assert_eq!(target.grad().unwrap().id(), gradient.id());
        assert!(target.grad().unwrap().requires_grad());
        assert_eq!(target.grad().unwrap().to_vec_f64(), gradient.to_vec_f64());
        target.grad().unwrap().sum().backward();
        assert_eq!(callbacks.load(Ordering::Relaxed), 1);
        assert_eq!(source.grad().unwrap().to_vec_f64(), vec![1.0; 2]);
        assert_eq!(target.id(), id);
        assert_eq!(target._version(), 0);
        target.set_leaf_grad(None).unwrap();
        assert!(target.grad().is_none());
        let shared = Tensor::ones(&[2]).to_dtype(dtype).requires_grad_(true).unwrap();
        target.set_leaf_grad(Some(shared.clone())).unwrap();
        with_grad_enabled(false, || shared.copy_leaf_from(&Tensor::full(&[2], 7.).to_dtype(dtype))).unwrap();
        assert_eq!(target.grad().unwrap().to_vec_f64(), vec![7.; 2]);
        assert_eq!(target._version(), 0);
        assert!(is_grad_enabled());
    }
}

#[test]
fn invalid_assignments_and_reference_cycles_leave_the_prior_slot_unchanged() {
    let target = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    let prior = Tensor::full(&[2], 3.);
    target.set_leaf_grad(Some(prior.clone())).unwrap();
    for invalid in [Tensor::ones(&[1, 2]), Tensor::ones(&[2]).to_dtype(DType::F64),
                    Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap(), target.clone(), target.square().unwrap()] {
        assert!(target.set_leaf_grad(Some(invalid)).is_err());
        assert_eq!(target.grad().unwrap().id(), prior.id());
        assert_eq!(target._version(), 0);
    }
    assert!(target.square().unwrap().set_leaf_grad(Some(prior.clone())).is_err());
    assert!(Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap().set_leaf_grad(None).is_err());
    let other = Tensor::full(&[2], 5.).requires_grad_(true).unwrap();
    target.set_leaf_grad(Some(other.clone())).unwrap();
    assert!(other.set_leaf_grad(Some(target.clone())).is_err());
    assert!(other.grad().is_none());
    assert_eq!(target.grad().unwrap().id(), other.id());
    target.set_leaf_grad(None).unwrap();
}

#[test]
fn capture_rejects_assignment_and_clearing_without_changing_mode_or_slots() {
    let target = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    let prior = Tensor::full(&[2], 3.);
    target.set_leaf_grad(Some(prior.clone())).unwrap();
    for replacement in [None, Some(Tensor::zeros(&[2]))] {
        assert!(capture(|| target.set_leaf_grad(replacement)).is_err());
        assert_eq!(target.grad().unwrap().id(), prior.id());
        assert!(is_grad_enabled());
    }
    let input = Tensor::ones(&[2]);
    let result = capture(|| input.mul(&input)).unwrap();
    assert_eq!(ferro_core::graph::CompiledChain::compile(&result).unwrap().replay().unwrap().to_vec(), vec![1.; 2]);
}

#[test]
fn device_validation_and_valid_device_assignment_never_transfer_data() {
    use ferro_core::dispatch::{register_backend, Backend, BinaryKind, DeviceBuffer, UnaryKind};
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|error| error.into_inner());
    const DEV: Device = Device::Cuda(247);
    struct Buffer(usize);
    impl DeviceBuffer for Buffer {
        fn device(&self) -> Device { DEV }
        fn len(&self) -> usize { self.0 }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }
    struct MetadataOnly(Arc<AtomicUsize>);
    impl Backend for MetadataOnly {
        fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
        fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
        fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected compute") }
        fn alloc_from_host(&self, values: &[f32]) -> ferro_core::Result<Box<dyn DeviceBuffer>> {
            self.0.fetch_add(1, Ordering::Relaxed); Ok(Box::new(Buffer(values.len())))
        }
        fn copy_to_host(&self, _: &dyn DeviceBuffer) -> ferro_core::Result<Vec<f32>> { panic!("unexpected download") }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    register_backend(DEV, Arc::new(MetadataOnly(calls.clone())));
    let device_gradient = Tensor::ones(&[2]).to_device(DEV).unwrap();
    let device_target = Tensor::zeros(&[2]).to_device(DEV).unwrap().requires_grad_(true).unwrap();
    calls.store(0, Ordering::Relaxed);
    let host_target = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    assert!(matches!(host_target.set_leaf_grad(Some(device_gradient.clone())), Err(Error::DeviceMismatch { .. })));
    assert!(host_target.grad().is_none());
    device_target.set_leaf_grad(Some(device_gradient.clone())).unwrap();
    assert_eq!(device_target.grad().unwrap().id(), device_gradient.id());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    device_target.set_leaf_grad(None).unwrap();
}
