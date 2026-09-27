// Fake device memory is ordinary CPU Vec storage. No CUDA dependency/driver.
use std::any::Any;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use ferro_core::{Device, Result, Tensor};
use ferro_core::dispatch::{Backend, BinaryKind, DeviceBuffer, UnaryKind, register_backend};

const DEV: Device = Device::Cuda(89);
static SERIAL: Mutex<()> = Mutex::new(());
struct Buf(Mutex<Vec<f32>>);
struct Idx(Vec<i64>);
impl DeviceBuffer for Idx {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}
impl DeviceBuffer for Buf {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.lock().unwrap().len() }
    fn as_any(&self) -> &dyn Any { self }
}
fn data(b: &dyn DeviceBuffer) -> Vec<f32> { b.as_any().downcast_ref::<Buf>().unwrap().0.lock().unwrap().clone() }
#[derive(Default)]
struct Counting {
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    index_uploads: AtomicUsize,
    layouts: AtomicUsize,
    failure: Option<ferro_core::Error>,
    fallback: Option<ferro_core::dispatch::MaterializeSupport>,
}
impl Counting {
    fn counts(&self) -> [usize; 4] {
        [&self.uploads, &self.downloads, &self.index_uploads, &self.layouts]
            .map(|n| n.load(Ordering::SeqCst))
    }
}
impl Backend for Counting {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected host kernel") }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected host kernel") }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected host kernel") }
    fn binary_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        let values = data(a).iter().zip(data(b)).map(|(&a, b)| match kind {
            BinaryKind::Add => a + b, BinaryKind::Sub => a - b,
            BinaryKind::Mul => a * b, BinaryKind::Div => a / b,
        }).collect();
        Ok(Box::new(Buf(Mutex::new(values))))
    }
    fn fill_inplace_dev(&self, dst: &dyn DeviceBuffer, value: f32) -> Result<()> {
        dst.as_any().downcast_ref::<Buf>().unwrap().0.lock().unwrap().fill(value);
        Ok(())
    }
    fn alloc_from_host(&self, values: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Buf(Mutex::new(values.to_vec()))))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Ok(data(buf).to_vec())
    }
    fn alloc_i64_from_host(&self, values: &[i64]) -> Result<Box<dyn DeviceBuffer>> {
        self.index_uploads.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Idx(values.to_vec())))
    }
    fn copy_i64_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<i64>> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Ok(buf.as_any().downcast_ref::<Idx>().unwrap().0.clone())
    }
    fn materialize_support(&self, _: &[usize], _: &[usize], _: usize) -> ferro_core::dispatch::MaterializeSupport {
        self.fallback.unwrap_or(ferro_core::dispatch::MaterializeSupport::Attempt)
    }
    fn materialize_dev(&self, buf: &dyn DeviceBuffer, shape: &[usize], strides: &[usize], offset: usize) -> Result<Box<dyn DeviceBuffer>> {
        self.layouts.fetch_add(1, Ordering::SeqCst);
        if let Some(error) = &self.failure { return Err(error.clone()); }
        assert_eq!(shape.len(), strides.len());
        let values = (0..shape.iter().product()).map(|mut flat| {
            let mut at = offset;
            for d in (0..shape.len()).rev() {
                at += (flat % shape[d]) * strides[d];
                flat /= shape[d];
            }
            data(buf)[at]
        }).collect();
        Ok(Box::new(Buf(Mutex::new(values))))
    }
}

fn check(mode: &str, transposed: bool) {
    // Lock precedes registration and outlives tensor destruction.
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Arc::new(Counting::default());
    register_backend(DEV, backend.clone());
    let base = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[3, 2]).unwrap()
        .to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let seed = if transposed { base.transpose(0, 1).unwrap() } else { base.clone() };
    let root = Tensor::from_vec(vec![0.; 6], seed.shape()).unwrap()
        .to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let before = backend.counts();
    let result = match mode {
        "detach" => seed.detach_copy(),
        "backward" => { root.backward_with(&seed); root.grad().unwrap() },
        "vjp" => root.vjp_wrt(&[&root], &seed, false).unwrap().remove(0),
        _ => unreachable!(),
    };
    let after = backend.counts();
    let delta: Vec<_> = after.iter().zip(before).map(|(a, b)| a - b).collect();
    // Read values after the measured interval; diagnostic download is not residency work.
    let values = result.to_vec();
    assert_eq!(values, if transposed { vec![1., 3., 5., 2., 4., 6.] } else { vec![1., 2., 3., 4., 5., 6.] });
    assert!(!result.requires_grad(), "seed graph must be detached");
    assert!(base.grad().is_none());
    assert!(seed.grad().is_none());
    assert_eq!(delta, vec![0, 0, 0, usize::from(transposed)], "{mode}: [f32 uploads, downloads, i64 uploads, layouts]");
    assert_eq!(result.device(), DEV);
    if transposed { assert_ne!(result._storage_ptr(), seed._storage_ptr()); }
    else { assert_eq!(result._storage_ptr(), seed._storage_ptr()); }
}

#[test]
fn explicitly_unsupported_layouts_retain_legacy_host_fallback() {
    use ferro_core::dispatch::MaterializeSupport::*;
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for fallback in [UnsupportedBackend, UnsupportedLayout] {
        let backend = Arc::new(Counting { fallback: Some(fallback), ..Counting::default() });
        register_backend(DEV, backend.clone());
        let seed = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[3, 2]).unwrap().to_device(DEV).unwrap().transpose(0, 1).unwrap();
        let before = backend.counts();
        let detached = seed.detach_copy();
        assert_eq!(backend.counts(), [before[0], before[1] + 1, before[2], before[3]]);
        assert_eq!(detached.device(), Device::Cpu);
        assert_eq!(detached.to_vec(), vec![1., 3., 5., 2., 4., 6.]);
    }
}

#[test]
fn unmodified_legacy_backend_retains_host_fallback() {
    struct Legacy(Arc<Counting>);
    impl Backend for Legacy {
        fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { unreachable!() }
        fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { unreachable!() }
        fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { unreachable!() }
        fn alloc_from_host(&self, x: &[f32]) -> Result<Box<dyn DeviceBuffer>> { self.0.alloc_from_host(x) }
        fn copy_to_host(&self, x: &dyn DeviceBuffer) -> Result<Vec<f32>> { self.0.copy_to_host(x) }
    }
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Arc::new(Counting::default());
    register_backend(DEV, Arc::new(Legacy(backend.clone())));
    for mode in ["detach", "backward", "vjp"] {
        let seed = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap().to_device(DEV).unwrap().transpose(0, 1).unwrap();
        let root = Tensor::from_vec(vec![0.; 6], seed.shape()).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
        let before = backend.counts();
        let result = match mode {
            "detach" => seed.detach_copy(),
            "backward" => { root.backward_with(&seed); root.grad().unwrap() },
            _ => root.vjp_wrt(&[&root], &seed, false).unwrap().remove(0),
        };
        let upload = usize::from(mode == "backward");
        assert_eq!(backend.counts(), [before[0] + upload, before[1] + 1, before[2], before[3]], "{mode}");
        assert_eq!(result.device(), if mode == "backward" { DEV } else { Device::Cpu });
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.to_vec(), vec![1., 4., 2., 5., 3., 6.]);
    }
}

#[test]
fn operational_errors_never_download_or_retry() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for error in [
        ferro_core::Error::Io { op: "materialize_dev", msg: "injected OOM/launch failure".into() },
        ferro_core::Error::Unsupported { op: "materialize_dev", msg: "wrong stream/context".into() },
        ferro_core::Error::Unsupported { op: "materialize_dev", msg: "backend does not implement device-resident storage".into() },
    ] {
        let backend = Arc::new(Counting { failure: Some(error), ..Counting::default() });
        register_backend(DEV, backend.clone());
        let seed = Tensor::from_vec(vec![1.; 6], &[2, 3]).unwrap().to_device(DEV).unwrap().transpose(0, 1).unwrap();
        let root = Tensor::from_vec(vec![0.; 6], seed.shape()).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
        for mode in ["detach", "backward", "vjp"] {
            let before = backend.counts();
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match mode {
                "detach" => { seed.detach_copy(); },
                "backward" => root.backward_with(&seed),
                "vjp" => { root.vjp_wrt(&[&root], &seed, false).unwrap(); },
                _ => unreachable!(),
            })).is_err());
            assert_eq!(backend.counts(), [before[0], before[1], before[2], before[3] + 1]);
            assert!(root.grad().is_none());
        }
        assert_eq!(seed.to_vec(), vec![1.; 6]);
    }
}

#[test]
fn fresh_versions_alias_gates_and_seed_lifetime() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Arc::new(Counting::default());
    register_backend(DEV, backend.clone());
    let base = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[3, 2]).unwrap().to_device(DEV).unwrap();
    base.fill_(7.).unwrap();
    let whole = base.detach_copy();
    assert_eq!(base._version(), 1);
    assert_eq!(whole._version(), 1);
    assert!(base.fill_(9.).is_err());
    assert!(whole.fill_(9.).is_err());
    let seed = base.transpose(0, 1).unwrap();
    let detached = seed.detach_copy();
    assert_eq!(detached._version(), 0);
    detached.fill_(11.).unwrap();
    assert_eq!(detached._version(), 1);
    assert_eq!(base._version(), 1);
    assert_eq!(seed.to_vec(), vec![7.; 6]);
    drop(seed); drop(base); drop(whole);
    assert_eq!(detached.to_vec(), vec![11.; 6]);
}

#[test]
fn live_strided_seed_retains_higher_order_dependency() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Arc::new(Counting::default());
    register_backend(DEV, backend.clone());
    let base = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[3, 2]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let seed = base.transpose(0, 1).unwrap();
    let root = Tensor::from_vec(vec![0.; 6], &[2, 3]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let before = backend.counts();
    let live = root.vjp_wrt(&[&root], &seed, true).unwrap().remove(0);
    assert!(live.requires_grad());
    assert_eq!(live._storage_ptr(), seed._storage_ptr());
    assert_eq!(backend.counts(), before);
    let cotangent = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap().to_device(DEV).unwrap();
    let before = backend.counts();
    let higher = live.vjp_wrt(&[&base], &cotangent, false).unwrap().remove(0);
    assert_eq!(backend.counts(), [before[0], before[1], before[2], before[3] + 1]);
    assert_eq!(higher.to_vec(), vec![1., 4., 2., 5., 3., 6.]);
    assert_eq!(higher.device(), DEV);
    assert!(base.grad().is_none());
}

#[test]
fn repeated_backward_preserves_returned_snapshots() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Arc::new(Counting::default());
    register_backend(DEV, backend.clone());
    for op_root in [false, true] {
        let leaf = Tensor::from_vec(vec![0.; 6], &[3, 2]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
        let root = if op_root { leaf.transpose(0, 1).unwrap() } else { leaf.clone() };
        let shape = if op_root { [3, 2] } else { [2, 3] };
        let seed = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &shape).unwrap().to_device(DEV).unwrap().transpose(0, 1).unwrap();
        let before = backend.counts();
        root.backward_with(&seed);
        let first = leaf.grad().unwrap();
        root.backward_with(&seed);
        let second = leaf.grad().unwrap();
        let after = backend.counts();
        if !op_root { assert_eq!(&after[..3], &before[..3]); }
        // Transpose's ordinary adjoint yields a strided contribution; repeated
        // accumulation of that contribution is a separate host-fallback path.
        let expected = if op_root { vec![1., 2., 3., 4., 5., 6.] } else { vec![1., 4., 2., 5., 3., 6.] };
        assert_eq!(first.to_vec(), expected);
        assert_eq!(second.to_vec(), expected.iter().map(|x| 2. * x).collect::<Vec<_>>());
        assert_ne!(first._storage_ptr(), second._storage_ptr());
        assert!(!first.requires_grad());
    }
}

#[test]
fn empty_singleton_and_higher_rank_transposes() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Arc::new(Counting::default());
    register_backend(DEV, backend.clone());
    for (shape, axes) in [(vec![2, 0, 3], (0, 2)), (vec![2, 1, 3], (0, 1)), (vec![2, 3, 4], (0, 2))] {
        let values: Vec<_> = (0..shape.iter().product::<usize>()).map(|x| x as f32).collect();
        let host = Tensor::from_vec(values, &shape).unwrap();
        let seed = host.to_device(DEV).unwrap().transpose(axes.0, axes.1).unwrap();
        let expected = host.transpose(axes.0, axes.1).unwrap().to_vec();
        let before = backend.counts();
        let detached = seed.detach_copy();
        assert_eq!(&backend.counts()[..3], &before[..3]);
        assert_eq!(detached.device(), DEV);
        assert_eq!(detached.to_vec(), expected);
    }
}

#[test]
fn dtype_copies_preserve_bits_including_device_i64() {
    use ferro_core::DType;
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Arc::new(Counting::default());
    register_backend(DEV, backend.clone());
    let ints = vec![i64::MAX, i64::MIN, 16_777_217, -16_777_217];
    let host = Tensor::from_vec_i64(ints, &[2, 2]).unwrap();
    let seed = host.to_device(DEV).unwrap().transpose(0, 1).unwrap();
    let before = backend.counts();
    let copied = seed.detach_copy();
    assert_eq!(copied.dtype(), DType::I64);
    assert_eq!(copied.to_vec_i64(), host.transpose(0, 1).unwrap().to_vec_i64());
    assert_eq!(backend.counts(), [before[0], before[1] + 1, before[2], before[3]]);
    let values = vec![f64::from_bits(0x7ff8000000000001), -0.0, 1.000000000000001, f64::INFINITY];
    let f64s = Tensor::from_vec_f64(values, &[2, 2]).unwrap().transpose(0, 1).unwrap();
    assert_eq!(f64s.detach_copy().to_vec_f64().iter().map(|x| x.to_bits()).collect::<Vec<_>>(), f64s.to_vec_f64().iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    for dtype in [DType::F16, DType::BF16] {
        let bits = vec![0x8000, 0x7e01, 0x0001, 0x3c00];
        let t = if dtype == DType::F16 { Tensor::from_vec_f16_bits(bits, &[2, 2]) } else { Tensor::from_vec_bf16_bits(bits, &[2, 2]) }.unwrap().transpose(0, 1).unwrap();
        let copied = t.detach_copy();
        assert_eq!(copied.dtype(), dtype);
        if dtype == DType::F16 { assert_eq!(copied.to_vec_f16_bits(), t.to_vec_f16_bits()); }
        else { assert_eq!(copied.to_vec_bf16_bits(), t.to_vec_bf16_bits()); }
    }
}

#[test]
fn detached_strided_seed_stays_resident() { check("detach", true); }
#[test]
fn backward_strided_seed_has_no_host_transfers() { check("backward", true); }
#[test]
fn functional_strided_seed_has_no_host_transfers() { check("vjp", true); }
#[test]
fn whole_seed_preserves_shared_storage_fast_path() {
    for mode in ["detach", "backward", "vjp"] { check(mode, false); }
}
