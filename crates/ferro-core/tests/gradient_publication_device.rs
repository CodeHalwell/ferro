//! Counting-backend ownership proof; no hardware performance claim.
use std::any::Any;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use ferro_core::{Device, Error, Result, Tensor};
use ferro_core::dispatch::{register_backend, Backend, BinaryKind, DeviceBuffer, UnaryKind};

const DEV: Device = Device::Cuda(233);
static LOCK: Mutex<()> = Mutex::new(());
struct Buffer(Mutex<Vec<f32>>);
impl DeviceBuffer for Buffer {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.lock().unwrap().len() }
    fn as_any(&self) -> &dyn Any { self }
}
fn data(buffer: &dyn DeviceBuffer) -> Vec<f32> { buffer.as_any().downcast_ref::<Buffer>().unwrap().0.lock().unwrap().clone() }
fn write(buffer: &dyn DeviceBuffer, values: &[f32]) { buffer.as_any().downcast_ref::<Buffer>().unwrap().0.lock().unwrap().copy_from_slice(values); }
fn buffer(values: Vec<f32>) -> Box<dyn DeviceBuffer> { Box::new(Buffer(Mutex::new(values))) }
#[derive(Default)]
struct Counting {
    copies: AtomicUsize, uploads: AtomicUsize, downloads: AtomicUsize,
    adds: AtomicUsize, writes: AtomicUsize, host_writes: AtomicUsize, fail_copy: AtomicBool,
}
impl Counting {
    fn reset(&self) {
        for count in [&self.copies, &self.uploads, &self.downloads, &self.adds, &self.writes, &self.host_writes] { count.store(0, Ordering::SeqCst); }
    }
    fn counts(&self) -> [usize; 6] {
        [&self.copies, &self.uploads, &self.downloads, &self.adds, &self.writes, &self.host_writes].map(|count| count.load(Ordering::SeqCst))
    }
}
impl Backend for Counting {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected host unary"); }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected host binary"); }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected matmul"); }
    fn alloc_from_host(&self, values: &[f32]) -> Result<Box<dyn DeviceBuffer>> { self.uploads.fetch_add(1, Ordering::SeqCst); Ok(buffer(values.to_vec())) }
    fn copy_to_host(&self, source: &dyn DeviceBuffer) -> Result<Vec<f32>> { self.downloads.fetch_add(1, Ordering::SeqCst); Ok(data(source)) }
    fn copy_dev(&self, source: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        self.copies.fetch_add(1, Ordering::SeqCst);
        if self.fail_copy.load(Ordering::SeqCst) { return Err(Error::Unsupported { op: "copy_dev", msg: "injected ownership-copy failure".into() }); }
        Ok(buffer(data(source)))
    }
    fn fill_dev(&self, value: f32, count: usize) -> Result<Box<dyn DeviceBuffer>> { Ok(buffer(vec![value; count])) }
    fn binary_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        assert!(matches!(kind, BinaryKind::Add)); self.adds.fetch_add(1, Ordering::SeqCst);
        Ok(buffer(data(a).into_iter().zip(data(b)).map(|(a, b)| a + b).collect()))
    }
    fn copy_into_dev(&self, destination: &dyn DeviceBuffer, source: &dyn DeviceBuffer) -> Result<()> {
        self.writes.fetch_add(1, Ordering::SeqCst); write(destination, &data(source)); Ok(())
    }
    fn write_dev_from_host(&self, destination: &dyn DeviceBuffer, values: &[f32]) -> Result<()> {
        self.host_writes.fetch_add(1, Ordering::SeqCst); write(destination, values); Ok(())
    }
}
fn setup() -> Arc<Counting> { let backend = Arc::new(Counting::default()); register_backend(DEV, backend.clone()); backend }
fn full(shape: &[usize], value: f32) -> Tensor { Tensor::full_on(shape, value, DEV).unwrap() }

#[test]
fn externally_held_device_seed_is_copied_once_and_does_not_alias_public_gradient() {
    let _lock = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner); let backend = setup();
    let x = full(&[2], 1.0).requires_grad_(true).unwrap();
    let seed = full(&[2], 3.0).requires_grad_(true).unwrap();
    backend.reset(); x.backward_with(&seed);
    assert_eq!(backend.counts(), [1, 0, 0, 0, 0, 0]);
    let gradient = x.grad().unwrap();
    assert_eq!(gradient.device(), DEV); assert!(!gradient.requires_grad());
    assert_ne!(gradient._storage_ptr(), seed._storage_ptr());
    x.backward_with(&seed);
    assert_eq!(backend.counts(), [1, 0, 0, 1, 1, 0]);
    assert_eq!(gradient.id(), x.grad().unwrap().id());
    assert_eq!(gradient.to_vec(), vec![6.0; 2]); assert_eq!(seed.to_vec(), vec![3.0; 2]);
}

#[test]
fn live_callback_copy_and_unique_detached_callback_steal_have_exact_device_traffic() {
    let _lock = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner); let backend = setup();
    for detached in [false, true] {
        let x = full(&[2], 2.0).requires_grad_(true).unwrap();
        let parent = x.clone();
        let y = full(&[], 1.0).record_fn(vec![x.clone()], move |_| vec![if detached { parent.detach() } else { parent.clone() }]);
        let seed = full(&[], 1.0);
        backend.reset(); y.backward_with(&seed);
        assert_eq!(backend.counts(), [if detached { 1 } else { 2 }, 0, 0, 0, 0, 0]);
        let gradient = x.grad().unwrap();
        assert_eq!(gradient.device(), DEV); assert!(!gradient.requires_grad() && gradient.is_leaf());
        assert_eq!(gradient._storage_ptr() == x._storage_ptr(), detached);
        assert_eq!(gradient.to_vec(), vec![2.0; 2]);
    }
}

#[test]
fn strided_device_manual_slot_uses_explicit_host_staging_and_preserves_buffer() {
    let _lock = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner); let backend = setup();
    let x = full(&[3, 2], 1.0).requires_grad_(true).unwrap();
    let storage = full(&[2, 3], 4.0);
    let gradient = storage.transpose(0, 1).unwrap();
    x.set_leaf_grad(Some(gradient.clone())).unwrap();
    let seed = full(&[3, 2], 2.0);
    backend.reset(); x.backward_with(&seed);
    assert_eq!(backend.counts(), [0, 0, 2, 0, 0, 1]);
    assert_eq!(gradient.id(), x.grad().unwrap().id()); assert_eq!(gradient.device(), DEV);
    assert_eq!(gradient.strides(), &[1, 3]); assert_eq!(storage._version(), 1);
    assert_eq!(storage.to_vec(), vec![6.0; 6]);
}

#[test]
fn failed_late_device_ownership_copy_leaves_all_public_slots_unchanged() {
    let _lock = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner); let backend = setup();
    let a = Tensor::scalar(1.0).requires_grad_(true).unwrap();
    let b = full(&[], 2.0).requires_grad_(true).unwrap();
    let previous = Tensor::scalar(7.0); a.set_leaf_grad(Some(previous.clone())).unwrap();
    let parent = b.clone();
    let output = Tensor::scalar(0.0).record_fn(vec![a.clone(), b.clone()], move |_| vec![Tensor::scalar(1.0), parent.clone()]);
    backend.reset(); backend.fail_copy.store(true, Ordering::SeqCst);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| output.backward())).is_err());
    assert_eq!(backend.counts(), [1, 0, 0, 0, 0, 0]);
    assert_eq!(a.grad().unwrap().id(), previous.id()); assert_eq!(previous.to_vec(), vec![7.0]); assert_eq!(previous._version(), 0);
    assert!(b.grad().is_none() && output.grad().is_none()); assert!(ferro_core::autograd::is_grad_enabled());
    backend.fail_copy.store(false, Ordering::SeqCst); output.backward();
    assert_eq!(previous.to_vec(), vec![8.0]); assert_eq!(b.grad().unwrap().to_vec(), vec![2.0]);
}
