use std::any::Any;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

use ferro_core::dispatch::{Backend, BinaryKind, DeviceBuffer, UnaryKind};
use ferro_core::optim::{AdamW, OptimizerState};
use ferro_core::{Device, Param, Result, Tensor, register_backend};

const DEV: Device = Device::Cuda(247);
static LOCK: Mutex<()> = Mutex::new(());
static UPLOADS: AtomicUsize = AtomicUsize::new(0);
static DOWNLOADS: AtomicUsize = AtomicUsize::new(0);
static FILLS: AtomicUsize = AtomicUsize::new(0);
static INCREMENTS: AtomicUsize = AtomicUsize::new(0);
static STEPS: AtomicUsize = AtomicUsize::new(0);
static COPIES: AtomicUsize = AtomicUsize::new(0);

struct Buffer(Mutex<Vec<f32>>);
impl DeviceBuffer for Buffer {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.lock().unwrap().len() }
    fn as_any(&self) -> &dyn Any { self }
}

fn buffer(t: &dyn DeviceBuffer) -> &Buffer { t.as_any().downcast_ref::<Buffer>().unwrap() }
fn values(t: &dyn DeviceBuffer) -> Vec<f32> { buffer(t).0.lock().unwrap().clone() }

struct FakeDevice;
impl Backend for FakeDevice {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected compute") }
    fn alloc_from_host(&self, values: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        UPLOADS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Buffer(Mutex::new(values.to_vec()))))
    }
    fn copy_to_host(&self, t: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        DOWNLOADS.fetch_add(1, Ordering::SeqCst);
        Ok(values(t))
    }
    fn fill_dev(&self, value: f32, n: usize) -> Result<Box<dyn DeviceBuffer>> {
        FILLS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Buffer(Mutex::new(vec![value; n]))))
    }
    fn copy_into_dev(&self, dst: &dyn DeviceBuffer, src: &dyn DeviceBuffer) -> Result<()> {
        COPIES.fetch_add(1, Ordering::SeqCst);
        *buffer(dst).0.lock().unwrap() = values(src);
        Ok(())
    }
    fn scalar_increment_dev(&self, t: &dyn DeviceBuffer, beta1: f32, beta2: f32) -> Result<()> {
        INCREMENTS.fetch_add(1, Ordering::SeqCst);
        let mut t = buffer(t).0.lock().unwrap();
        t[0] += 1.0;
        t[1] = 1.0 - beta1.powf(t[0]);
        t[2] = 1.0 - beta2.powf(t[0]);
        Ok(())
    }
    fn adamw_step_capturable_dev(&self, p: &dyn DeviceBuffer, m: &dyn DeviceBuffer, v: &dyn DeviceBuffer,
        g: &dyn DeviceBuffer, t: &dyn DeviceBuffer, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32) -> Result<()> {
        STEPS.fetch_add(1, Ordering::SeqCst);
        let (mut pv, mut mv, mut vv, gv, tv) = (values(p), values(m), values(v), values(g), values(t));
        for i in 0..pv.len() {
            mv[i] = beta1 * mv[i] + (1.0 - beta1) * gv[i];
            vv[i] = beta2 * vv[i] + (1.0 - beta2) * gv[i] * gv[i];
            pv[i] -= lr * ((mv[i] / tv[1]) / ((vv[i] / tv[2]).sqrt() + eps) + wd * pv[i]);
        }
        *buffer(p).0.lock().unwrap() = pv;
        *buffer(m).0.lock().unwrap() = mv;
        *buffer(v).0.lock().unwrap() = vv;
        Ok(())
    }
}

fn reset() {
    for counter in [&UPLOADS, &DOWNLOADS, &FILLS, &INCREMENTS, &STEPS, &COPIES] { counter.store(0, Ordering::SeqCst); }
}

fn counts() -> Vec<usize> {
    [&UPLOADS, &DOWNLOADS, &FILLS, &INCREMENTS, &STEPS, &COPIES].iter().map(|c| c.load(Ordering::SeqCst)).collect()
}

fn state(opt: &AdamW) -> Vec<(String, Vec<f32>)> {
    opt.snapshot().into_iter().map(|(name, t)| (name, t.to_vec())).collect()
}

fn parameters() -> Vec<Param> {
    [1.0, -0.5].into_iter().map(|v| Param::new(Tensor::scalar(v).to_device(DEV).unwrap())).collect()
}

#[test]
fn capturable_partial_gradients_reject_before_device_work() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(FakeDevice));
    let params = parameters();
    let seed = Tensor::scalar(1.0).to_device(DEV).unwrap();
    let mut opt = AdamW::new(params.clone(), 0.1).capturable();
    assert!(opt.is_capturable());
    for frozen in [false, true] {
        params[1].set_trainable(!frozen);
        params[0].tensor().backward_with(&seed);
        if frozen { params[1].tensor().backward_with(&seed); }
        let before = state(&opt);
        let versions: Vec<_> = params.iter().map(|p| p.tensor()._version()).chain(opt.state_buffers().unwrap().iter().map(|t| t._version())).collect();
        reset();
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| opt.step()));
        assert!(rejected.is_err());
        assert_eq!(counts(), vec![0; 6]);
        assert!(params[0].grad().is_some());
        assert_eq!(state(&opt), before);
        assert_eq!(params.iter().map(|p| p.tensor()._version()).chain(opt.state_buffers().unwrap().iter().map(|t| t._version())).collect::<Vec<_>>(), versions);
        params[1].set_trainable(true);
        params[1].tensor().backward_with(&seed);
        reset(); opt.step();
        assert_eq!(INCREMENTS.load(Ordering::SeqCst), 1);
        assert_eq!(STEPS.load(Ordering::SeqCst), 2);
        assert_eq!(DOWNLOADS.load(Ordering::SeqCst), 0);
        let before = state(&opt);
        reset(); opt.step();
        assert_eq!(counts(), vec![0; 6]);
        assert_eq!(state(&opt), before);
    }
    let before = state(&opt);
    let pointer = opt.state_buffers().unwrap().last().unwrap()._storage_ptr();
    reset(); opt = opt.capturable();
    assert_eq!(counts(), vec![0; 6]);
    assert_eq!(opt.state_buffers().unwrap().last().unwrap()._storage_ptr(), pointer);
    assert_eq!(state(&opt), before);
}

#[test]
fn capturable_restore_and_enable_require_synchronized_counters() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(FakeDevice));
    let params = parameters();
    let mut captured = AdamW::new(params.clone(), 0.1).capturable();
    let mut incoming = captured.snapshot();
    incoming.iter_mut().find(|(n, _)| n == "t").unwrap().1 = Tensor::from_vec(vec![2.0, 1.0], &[2]).unwrap();
    let before = state(&captured);
    let pointer = captured.state_buffers().unwrap().last().unwrap()._storage_ptr();
    reset();
    assert!(captured.restore(&incoming).is_err());
    assert_eq!(counts(), vec![0; 6]);
    assert_eq!(state(&captured), before);
    assert_eq!(captured.state_buffers().unwrap().last().unwrap()._storage_ptr(), pointer);
    let mut eager = AdamW::new(params.clone(), 0.1);
    eager.restore(&incoming).unwrap();
    reset(); eager = eager.capturable();
    assert!(!eager.is_capturable());
    assert_eq!(counts(), vec![0; 6]);
    incoming.iter_mut().find(|(n, _)| n == "t").unwrap().1 = Tensor::scalar(3.0);
    captured.restore(&incoming).unwrap();
    assert_eq!(captured.state_buffers().unwrap().last().unwrap()._storage_ptr(), pointer);
    assert_eq!(state(&captured)[0].1, vec![3.0, 3.0]);
    let seed = Tensor::scalar(1.0).to_device(DEV).unwrap();
    for p in &params { p.tensor().backward_with(&seed); }
    captured.step();
    assert_eq!(state(&captured)[0].1, vec![4.0, 4.0]);
}

#[test]
fn capturable_optimizer_clone_owns_counter_and_moments() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(FakeDevice));
    let params = parameters();
    let seed = Tensor::scalar(1.0).to_device(DEV).unwrap();
    let mut original = AdamW::new(params.clone(), 0.1).capturable();
    for p in &params { p.tensor().backward_with(&seed); }
    original.step();
    let mut cloned = original.clone();
    assert_eq!(state(&cloned), state(&original));
    for (a, b) in original.state_buffers().unwrap().iter().zip(cloned.state_buffers().unwrap()) {
        assert_ne!(a._storage_ptr(), b._storage_ptr());
    }
    let before = state(&original);
    for p in &params { p.tensor().backward_with(&seed); }
    cloned.step();
    assert_eq!(state(&original), before);
    assert_eq!(state(&cloned)[0].1, vec![2.0, 2.0]);
}
