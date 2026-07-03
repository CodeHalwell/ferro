use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::dispatch::{register_backend, Backend, BinaryKind, DeviceBuffer, UnaryKind};
use ferro_core::{Device, Result, Tensor};

// A fake device backend that stores data in host Vecs but counts every
// transfer and kernel call, so tests can PROVE chained ops stay resident:
// one upload, N device kernels, one download, zero extra host copies.
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static TO_HOST: AtomicUsize = AtomicUsize::new(0);
static UNARY: AtomicUsize = AtomicUsize::new(0);
static BINARY: AtomicUsize = AtomicUsize::new(0);
static MATMUL: AtomicUsize = AtomicUsize::new(0);

const DEV: Device = Device::Cuda(9);

struct FakeBuf(Vec<f32>);

impl DeviceBuffer for FakeBuf {
    fn device(&self) -> Device {
        DEV
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct FakeDevice;

fn data(buf: &dyn DeviceBuffer) -> &[f32] {
    &buf.as_any().downcast_ref::<FakeBuf>().expect("buffer from another backend").0
}

impl Backend for FakeDevice {
    fn unary(&self, _kind: UnaryKind, _x: &[f32]) -> Vec<f32> {
        panic!("host-slice path must not run for device-resident tensors");
    }
    fn binary(&self, _kind: BinaryKind, _a: &[f32], _b: &[f32]) -> Vec<f32> {
        panic!("host-slice path must not run for device-resident tensors");
    }
    fn matmul(&self, _a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        panic!("host-slice path must not run for device-resident tensors");
    }

    fn alloc_from_host(&self, data: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FakeBuf(data.to_vec())))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        TO_HOST.fetch_add(1, Ordering::SeqCst);
        Ok(data(buf).to_vec())
    }
    fn unary_dev(&self, kind: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        UNARY.fetch_add(1, Ordering::SeqCst);
        let out = match kind {
            UnaryKind::Relu => data(x).iter().map(|v| v.max(0.0)).collect(),
            UnaryKind::Exp => data(x).iter().map(|v| v.exp()).collect(),
            UnaryKind::Neg => data(x).iter().map(|v| -v).collect(),
            _ => data(x).to_vec(),
        };
        Ok(Box::new(FakeBuf(out)))
    }
    fn binary_dev(
        &self,
        kind: BinaryKind,
        a: &dyn DeviceBuffer,
        b: &dyn DeviceBuffer,
    ) -> Result<Box<dyn DeviceBuffer>> {
        BINARY.fetch_add(1, Ordering::SeqCst);
        let f = |x: f32, y: f32| match kind {
            BinaryKind::Add => x + y,
            BinaryKind::Sub => x - y,
            BinaryKind::Mul => x * y,
            BinaryKind::Div => x / y,
        };
        let out = data(a).iter().zip(data(b)).map(|(&x, &y)| f(x, y)).collect();
        Ok(Box::new(FakeBuf(out)))
    }
    fn matmul_dev(
        &self,
        a: &dyn DeviceBuffer,
        b: &dyn DeviceBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        MATMUL.fetch_add(1, Ordering::SeqCst);
        let (va, vb) = (data(a), data(b));
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    out[i * n + j] += va[i * k + p] * vb[p * n + j];
                }
            }
        }
        Ok(Box::new(FakeBuf(out)))
    }
}

// The counters are process-global and the test harness runs tests in parallel
// threads, so every test touching the fake device serializes on this lock
// (poison-tolerant: the should_panic test unwinds while holding it).
static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> MutexGuard<'static, ()> {
    register_backend(DEV, Arc::new(FakeDevice));
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn counts() -> (usize, usize, usize, usize, usize) {
    (
        ALLOCS.load(Ordering::SeqCst),
        TO_HOST.load(Ordering::SeqCst),
        UNARY.load(Ordering::SeqCst),
        BINARY.load(Ordering::SeqCst),
        MATMUL.load(Ordering::SeqCst),
    )
}

#[test]
fn chained_ops_stay_resident() {
    let _serial = setup();
    let x = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]).unwrap();
    let w = Tensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[2, 2]).unwrap();

    let before = counts();
    let xd = x.to_device(DEV).unwrap();
    let wd = w.to_device(DEV).unwrap();
    let out = xd.relu().exp().mul(&wd).unwrap().matmul(&wd).unwrap();
    assert_eq!(out.device(), DEV);
    let host = out.to_vec();
    let after = counts();

    // Exactly: 2 uploads, 1 download, 2 unary + 1 binary + 1 matmul on device.
    assert_eq!(after.0 - before.0, 2, "alloc_from_host count");
    assert_eq!(after.1 - before.1, 1, "copy_to_host count");
    assert_eq!(after.2 - before.2, 2, "unary_dev count");
    assert_eq!(after.3 - before.3, 1, "binary_dev count");
    assert_eq!(after.4 - before.4, 1, "matmul_dev count");

    // And the numbers must match the same chain computed on the cpu.
    let cpu = x.relu().exp().mul(&w).unwrap().matmul(&w).unwrap().to_vec();
    for (d, c) in host.iter().zip(cpu.iter()) {
        assert!((d - c).abs() < 1e-5, "device {d} vs cpu {c}");
    }
}

#[test]
fn to_device_round_trip_and_metadata() {
    let _serial = setup();
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let d = x.to_device(DEV).unwrap();
    assert_eq!(d.device(), DEV);
    assert_eq!(d.shape(), &[2, 3]);
    assert_eq!(d.dtype(), ferro_core::DType::F32);
    let back = d.to_device(Device::Cpu).unwrap();
    assert_eq!(back.device(), Device::Cpu);
    assert_eq!(back.to_vec(), x.to_vec());
    // Same-device transfer is a cheap clone.
    assert_eq!(d.to_device(DEV).unwrap().to_vec(), d.to_vec());
}

#[test]
fn unregistered_device_transfer_errors() {
    let x = Tensor::ones(&[2]);
    assert!(x.to_device(Device::Cuda(3)).is_err());
}

#[test]
fn non_f32_transfer_errors() {
    let _serial = setup();
    let ids = Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap();
    assert!(ids.to_device(DEV).is_err());
}

#[test]
#[should_panic(expected = "cpu tensors")]
fn requires_grad_on_device_panics() {
    let _serial = setup();
    let d = Tensor::ones(&[2]).to_device(DEV).unwrap();
    d.requires_grad_(true);
}

#[test]
fn device_broadcast_binary_errors() {
    let _serial = setup();
    let a = Tensor::ones(&[2, 3]).to_device(DEV).unwrap();
    let b = Tensor::ones(&[3]).to_device(DEV).unwrap();
    let Err(err) = a.add(&b) else { panic!("device broadcast add should error") };
    let msg = err.to_string();
    assert!(msg.contains("broadcasting on device tensors"), "got: {msg}");
}

#[test]
fn host_fallback_ops_return_cpu_tensors() {
    let _serial = setup();
    // Ops without device kernels (here: softmax) fall back to host compute and
    // visibly return cpu tensors - the documented phase 3 boundary.
    let d = Tensor::from_vec(vec![0.1, 0.5, 0.4, 0.2, 0.3, 0.5], &[2, 3]).unwrap()
        .to_device(DEV)
        .unwrap();
    let s = d.softmax(1).unwrap();
    assert_eq!(s.device(), Device::Cpu);
    let rows: Vec<f32> = s.to_vec().chunks(3).map(|r| r.iter().sum()).collect();
    for r in rows {
        assert!((r - 1.0).abs() < 1e-5);
    }
}
