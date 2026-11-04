use std::any::Any;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

use ferro_core::dispatch::{Backend, BinaryKind, DeviceBuffer, UnaryKind};
use ferro_core::nn::scaled_dot_product_attention as attention;
use ferro_core::{DType, Device, Error, Result, Tensor, capture, graph::CompiledChain, register_backend};

#[test]
fn causal_finite_logits_exclude_future_values_and_gradients() {
    let q = Tensor::from_vec(vec![1e5], &[1, 1, 1]).unwrap().requires_grad_(true).unwrap();
    let k = Tensor::from_vec(vec![0., 1e4], &[1, 2, 1]).unwrap().requires_grad_(true).unwrap();
    let v = Tensor::from_vec(vec![3., 7.], &[1, 2, 1]).unwrap().requires_grad_(true).unwrap();
    let y = attention(&q, &k, &v, true).unwrap();
    assert_eq!(y.to_vec(), vec![3.]);
    y.sum().backward();
    assert_eq!(q.grad().unwrap().to_vec(), vec![0.]);
    assert_eq!(k.grad().unwrap().to_vec(), vec![0., 0.]);
    assert_eq!(v.grad().unwrap().to_vec(), vec![1., 0.]);
}

#[test]
fn causal_nonsquare_alignment_and_capture_remain_upper_left() {
    let q = Tensor::zeros(&[1, 4, 1]);
    let k = Tensor::zeros(&[1, 2, 1]);
    let v = Tensor::from_vec(vec![2., 6.], &[1, 2, 1]).unwrap();
    assert_eq!(attention(&q, &k, &v, true).unwrap().to_vec(), vec![2., 4., 4., 4.]);
    let q = Tensor::from_vec(vec![1e5], &[1, 1, 1]).unwrap();
    let k = Tensor::from_vec(vec![0., 1e4], &[1, 2, 1]).unwrap();
    let recorded = capture(|| attention(&q, &k, &v, true).unwrap());
    let compiled = CompiledChain::compile(&recorded).unwrap();
    v.copy_from(&Tensor::from_vec(vec![9., 100.], &[1, 2, 1]).unwrap()).unwrap();
    assert_eq!(compiled.replay().unwrap().to_vec(), vec![9.]);
    q.copy_from(&Tensor::from_vec(vec![2e5], &[1, 1, 1]).unwrap()).unwrap();
    assert_eq!(compiled.replay().unwrap().to_vec(), vec![9.]);
}

#[test]
fn empty_attention_preserves_shapes_and_gradient_connections() {
    for (b, sq, sk, dv) in [(0, 2, 3, 2), (1, 0, 3, 2), (1, 2, 0, 2), (1, 2, 3, 0)] {
        for causal in [false, true] {
            let q = Tensor::zeros(&[b, sq, 3]).requires_grad_(true).unwrap();
            let k = Tensor::zeros(&[b, sk, 3]).requires_grad_(true).unwrap();
            let v = Tensor::zeros(&[b, sk, dv]).requires_grad_(true).unwrap();
            let out = attention(&q, &k, &v, causal).unwrap();
            assert_eq!(out.shape(), &[b, sq, dv]);
            assert!(out.to_vec().iter().all(|&x| x == 0.));
            out.sum().backward();
            for input in [&q, &k, &v] {
                let g = input.grad().unwrap();
                assert_eq!(g.shape(), input.shape());
                assert!(g.to_vec().iter().all(|&x| x == 0.));
            }
        }
    }
}

const DEV: Device = Device::Cuda(247);
static LOCK: Mutex<()> = Mutex::new(());
static CALLS: AtomicUsize = AtomicUsize::new(0);
struct Buffer { owner: usize, values: Vec<f32> }
impl DeviceBuffer for Buffer {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.values.len() }
    fn as_any(&self) -> &dyn Any { self }
}
struct CountingBackend { owner: usize, fail_fill: bool }
impl CountingBackend {
    fn read<'a>(&self, buffer: &'a dyn DeviceBuffer) -> Result<&'a [f32]> {
        let buffer = buffer.as_any().downcast_ref::<Buffer>().unwrap();
        if buffer.owner != self.owner {
            return Err(Error::Unsupported { op: "attention_test", msg: "foreign cached allocation owner".into() });
        }
        Ok(&buffer.values)
    }
    fn make(&self, values: Vec<f32>) -> Result<Box<dyn DeviceBuffer>> {
        CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Buffer { owner: self.owner, values }))
    }
}
impl Backend for CountingBackend {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected host compute") }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected host compute") }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected host compute") }
    fn alloc_from_host(&self, values: &[f32]) -> Result<Box<dyn DeviceBuffer>> { self.make(values.to_vec()) }
    fn copy_to_host(&self, buffer: &dyn DeviceBuffer) -> Result<Vec<f32>> { Ok(self.read(buffer)?.to_vec()) }
    fn fill_dev(&self, value: f32, n: usize) -> Result<Box<dyn DeviceBuffer>> {
        if self.fail_fill {
            CALLS.fetch_add(1, Ordering::SeqCst);
            return Err(Error::Unsupported { op: "attention_fill", msg: "injected allocation failure".into() });
        }
        self.make(vec![value; n])
    }
    fn binary_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        let (a, b) = (self.read(a)?, self.read(b)?);
        assert_eq!((a.len(), b.len()), (1, 1));
        self.make(vec![match kind { BinaryKind::Add => a[0] + b[0], BinaryKind::Sub => a[0] - b[0], BinaryKind::Mul => a[0] * b[0], BinaryKind::Div => a[0] / b[0], BinaryKind::ReluBackward => if b[0] <= 0.0 { 0.0 } else { a[0] } }])
    }
    fn binary_bc_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, _: &[usize], b: &dyn DeviceBuffer, _: &[usize], _: &[usize]) -> Result<Box<dyn DeviceBuffer>> {
        self.binary_dev(kind, a, b)
    }
    fn bmm_dev(&self, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer, _: usize, _: usize, _: usize, _: usize, _: bool, _: bool) -> Result<Box<dyn DeviceBuffer>> {
        self.binary_dev(BinaryKind::Mul, a, b)
    }
    fn softmax_dev(&self, x: &dyn DeviceBuffer, rows: usize, cols: usize) -> Result<Box<dyn DeviceBuffer>> {
        assert_eq!((rows, cols, self.read(x)?.len()), (1, 1, 1));
        self.make(vec![1.])
    }
}

#[test]
fn attention_invalid_metadata_rejects_before_backend_work() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(CountingBackend { owner: 0, fail_fill: true }));
    let device = |shape: &[usize]| Tensor::zeros(shape).to_device(DEV).unwrap();
    let q = device(&[1, 1, 1]);
    let zero_width = device(&[1, 1, 0]);
    let flat = device(&[1]);
    let integer = Tensor::from_vec_i64(vec![1], &[1, 1, 1]).unwrap();
    let host = Tensor::ones(&[1, 1, 1]);
    for (a, b, c) in [(&flat, &q, &q), (&q, &integer, &q), (&q, &host, &q), (&zero_width, &zero_width, &q)] {
        CALLS.store(0, Ordering::SeqCst);
        assert!(attention(a, b, c, true).is_err());
        assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    }
    CALLS.store(0, Ordering::SeqCst);
    assert!(matches!(attention(&q, &integer, &q, true), Err(Error::DtypeMismatch { expected: DType::F32, .. })));
    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    for (sq, sk, dv) in [(usize::MAX, 2, 1), (isize::MAX as usize / 4 + 1, 1, 1), (2, 1, usize::MAX / 2 + 1)] {
        let (a, b, c) = (device(&[0, sq, 1]), device(&[0, sk, 1]), device(&[0, sk, dv]));
        CALLS.store(0, Ordering::SeqCst);
        assert!(matches!(attention(&a, &b, &c, true), Err(Error::InvalidShape { .. })));
        assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    }
    CALLS.store(0, Ordering::SeqCst);
    assert!(matches!(attention(&q, &q, &q, true), Err(Error::Unsupported { op: "attention_fill", .. })));
    assert_eq!(CALLS.load(Ordering::SeqCst), 1, "valid metadata reaches allocation");
}

#[test]
fn cached_causal_mask_respects_replaced_backend_owner() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut keep_alive = Vec::new();
    for owner in [1, 2, 3] {
        let backend = Arc::new(CountingBackend { owner, fail_fill: false });
        register_backend(DEV, backend.clone());
        keep_alive.push(backend);
        let q = Tensor::ones(&[1, 1, 1]).to_device(DEV).unwrap();
        let v = Tensor::full(&[1, 1, 1], 7.).to_device(DEV).unwrap();
        assert_eq!(attention(&q, &q, &v, true).unwrap().to_vec(), vec![7.]);
        // Owner 2 replaces a still-live owner; owner 3 replaces an expired one.
        if owner == 2 { keep_alive.clear(); }
    }
}
