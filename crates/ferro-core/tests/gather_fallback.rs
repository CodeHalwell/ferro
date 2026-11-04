use std::any::Any;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use ferro_core::{Device, Error, Result, Tensor};
use ferro_core::dispatch::{Backend, BinaryKind, DeviceBuffer, UnaryKind, register_backend};

const DEV: Device = Device::Cuda(233);
static SERIAL: Mutex<()> = Mutex::new(());
struct Values(Vec<f32>);
struct Indices(Vec<i64>);
macro_rules! buffer {
    ($t:ty) => {
        impl DeviceBuffer for $t {
            fn device(&self) -> Device { DEV }
            fn len(&self) -> usize { self.0.len() }
            fn as_any(&self) -> &dyn Any { self }
        }
    };
}
buffer!(Values);
buffer!(Indices);

#[derive(Default)]
struct Legacy {
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    index_reads: AtomicUsize,
}
impl Legacy {
    fn counts(&self) -> [usize; 3] {
        [&self.uploads, &self.downloads, &self.index_reads].map(|n| n.load(Ordering::SeqCst))
    }
}
// Inherits the default gather method, as an existing backend without a kernel does.
impl Backend for Legacy {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected host kernel") }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected host kernel") }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected host kernel") }
    fn alloc_from_host(&self, data: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Values(data.to_vec())))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Ok(buf.as_any().downcast_ref::<Values>().unwrap().0.clone())
    }
    fn alloc_i64_from_host(&self, data: &[i64]) -> Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Indices(data.to_vec()))) }
    fn copy_i64_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<i64>> {
        self.index_reads.fetch_add(1, Ordering::SeqCst);
        Ok(buf.as_any().downcast_ref::<Indices>().unwrap().0.clone())
    }
}

struct Gather {
    host: Arc<Legacy>,
    failure: Option<Error>,
    calls: AtomicUsize,
}
impl Backend for Gather {
    fn unary(&self, k: UnaryKind, x: &[f32]) -> Vec<f32> { self.host.unary(k, x) }
    fn binary(&self, k: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> { self.host.binary(k, a, b) }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> { self.host.matmul(a, b, m, k, n) }
    fn alloc_from_host(&self, data: &[f32]) -> Result<Box<dyn DeviceBuffer>> { self.host.alloc_from_host(data) }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> { self.host.copy_to_host(buf) }
    fn alloc_i64_from_host(&self, data: &[i64]) -> Result<Box<dyn DeviceBuffer>> { self.host.alloc_i64_from_host(data) }
    fn copy_i64_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<i64>> { self.host.copy_i64_to_host(buf) }
    fn gather_rows_dev(&self, w: &dyn DeviceBuffer, idx: &dyn DeviceBuffer, _: usize, inner: usize) -> Result<Box<dyn DeviceBuffer>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(error) = &self.failure { return Err(error.clone()); }
        let w = &w.as_any().downcast_ref::<Values>().unwrap().0;
        let idx = &idx.as_any().downcast_ref::<Indices>().unwrap().0;
        Ok(Box::new(Values(idx.iter().flat_map(|&i| w[i as usize * inner..(i as usize + 1) * inner].iter().copied()).collect())))
    }
}

#[test]
fn legacy_gather_fallback_preserves_values_and_duplicate_gradients() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Arc::new(Legacy::default());
    register_backend(DEV, backend.clone());
    let w = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let idx = Tensor::from_vec_i64(vec![1, 0, 1], &[3]).unwrap().to_device(DEV).unwrap();
    let before = backend.counts();
    let y = w.index_select_t(0, &idx).unwrap();
    assert_eq!(y.device(), Device::Cpu);
    assert_eq!(y.to_vec(), [3., 4., 1., 2., 3., 4.]);
    assert_eq!(backend.counts(), [before[0], before[1] + 1, before[2] + 1]);
    y.sum().backward();
    assert_eq!(w.grad().unwrap().device(), DEV);
    assert_eq!(w.grad().unwrap().to_vec(), [1., 1., 2., 2.]);
}

#[test]
fn gather_errors_propagate_without_host_retry_even_for_empty_indices() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for error in [
        Error::Unsupported { op: "gather_rows_dev", msg: "foreign stream owner".into() },
        Error::Io { op: "gather_rows_dev", msg: "injected execution failure".into() },
        Error::InvalidShape { op: "gather_rows_dev", msg: "invalid backing allocation".into() },
    ] {
        let host = Arc::new(Legacy::default());
        let backend = Arc::new(Gather { host: host.clone(), failure: Some(error.clone()), calls: AtomicUsize::new(0) });
        register_backend(DEV, backend.clone());
        let w = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap().to_device(DEV).unwrap();
        for ids in [vec![1, 0], vec![]] {
            let n = ids.len();
            let idx = Tensor::from_vec_i64(ids, &[n]).unwrap().to_device(DEV).unwrap();
            let before = host.counts();
            let calls = backend.calls.load(Ordering::SeqCst);
            assert_eq!(w.index_select_t(0, &idx).err(), Some(error.clone()));
            assert_eq!(host.counts(), [before[0], before[1], before[2] + 1]);
            assert_eq!(backend.calls.load(Ordering::SeqCst), calls + 1);
        }
    }
}

#[test]
fn gather_index_views_never_pass_unvalidated_backing_indices_to_kernel() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let host = Arc::new(Legacy::default());
    let backend = Arc::new(Gather { host: host.clone(), failure: None, calls: AtomicUsize::new(0) });
    register_backend(DEV, backend.clone());
    let w = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap().to_device(DEV).unwrap();
    let base = Tensor::from_vec_i64(vec![-1, 1, 0, -1], &[4]).unwrap().to_device(DEV).unwrap();
    let singleton = Tensor::from_vec_i64(vec![1], &[1]).unwrap().to_device(DEV).unwrap();
    let prefix = Tensor::from_vec_i64(vec![0, 1, -1], &[3]).unwrap().to_device(DEV).unwrap();
    for (idx, expected) in [
        (base.narrow(0, 1, 2).unwrap(), vec![3., 4., 1., 2.]),
        (base.slice(0, 2, 4, 2).unwrap(), vec![1., 2.]),
        (prefix.narrow(0, 0, 2).unwrap(), vec![1., 2., 3., 4.]),
        (singleton.expand(&[3]).unwrap(), vec![3., 4., 3., 4., 3., 4.]),
        (base.narrow(0, 2, 0).unwrap(), vec![]),
    ] {
        let y = w.index_select_t(0, &idx).unwrap();
        assert_eq!(y.device(), Device::Cpu);
        assert_eq!(y.shape(), &[expected.len() / 2, 2]);
        assert_eq!(y.to_vec(), expected);
        assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    }
    let out = w.index_select_t(0, &singleton).unwrap();
    assert_eq!(out.device(), DEV);
    assert_eq!(out.to_vec(), [3., 4.]);
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
}
