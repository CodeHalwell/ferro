use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::dispatch::{register_backend, Backend, BinaryKind, DeviceBuffer, ReduceKind, UnaryKind};
use ferro_core::{Device, Result, Tensor};

// A fake device backend that stores data in host Vecs but counts every
// transfer and kernel call, so tests can PROVE chained ops stay resident:
// one upload, N device kernels, one download, zero extra host copies.
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static TO_HOST: AtomicUsize = AtomicUsize::new(0);
static UNARY: AtomicUsize = AtomicUsize::new(0);
static BINARY: AtomicUsize = AtomicUsize::new(0);
static MATMUL: AtomicUsize = AtomicUsize::new(0);
static ALLOC_ELEMS: AtomicUsize = AtomicUsize::new(0);
static TO_HOST_ELEMS: AtomicUsize = AtomicUsize::new(0);

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
        ALLOC_ELEMS.fetch_add(data.len(), Ordering::SeqCst);
        Ok(Box::new(FakeBuf(data.to_vec())))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        TO_HOST.fetch_add(1, Ordering::SeqCst);
        TO_HOST_ELEMS.fetch_add(buf.len(), Ordering::SeqCst);
        Ok(data(buf).to_vec())
    }
    fn unary_dev(&self, kind: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        UNARY.fetch_add(1, Ordering::SeqCst);
        let out = match kind {
            UnaryKind::Relu => data(x).iter().map(|v| v.max(0.0)).collect(),
            UnaryKind::Exp => data(x).iter().map(|v| v.exp()).collect(),
            UnaryKind::Neg => data(x).iter().map(|v| -v).collect(),
            UnaryKind::Gtz => data(x).iter().map(|v| if *v > 0.0 { 1.0 } else { 0.0 }).collect(),
            other => panic!("fake device kernel not implemented for {other:?}"),
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
        ta: bool,
        tb: bool,
    ) -> Result<Box<dyn DeviceBuffer>> {
        MATMUL.fetch_add(1, Ordering::SeqCst);
        let (va, vb) = (data(a), data(b));
        // Logical A is (m,k): stored (k,m) when ta. Same for B.
        let ai = |i: usize, p: usize| if ta { va[p * m + i] } else { va[i * k + p] };
        let bi = |p: usize, j: usize| if tb { vb[j * k + p] } else { vb[p * n + j] };
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    out[i * n + j] += ai(i, p) * bi(p, j);
                }
            }
        }
        Ok(Box::new(FakeBuf(out)))
    }

    fn binary_bc_dev(
        &self,
        kind: BinaryKind,
        a: &dyn DeviceBuffer,
        sa: &[usize],
        b: &dyn DeviceBuffer,
        sb: &[usize],
        out_shape: &[usize],
    ) -> Result<Box<dyn DeviceBuffer>> {
        BINARY.fetch_add(1, Ordering::SeqCst);
        let f = |x: f32, y: f32| match kind {
            BinaryKind::Add => x + y,
            BinaryKind::Sub => x - y,
            BinaryKind::Mul => x * y,
            BinaryKind::Div => x / y,
        };
        // Right-aligned broadcast indexing over the flat output.
        let n: usize = out_shape.iter().product();
        let idx = |flat: usize, shape: &[usize]| -> usize {
            let pad = out_shape.len() - shape.len();
            let mut off = 0usize;
            let mut stride = 1usize;
            let mut strides = vec![0usize; out_shape.len()];
            for d in (0..out_shape.len()).rev() {
                strides[d] = stride;
                stride *= out_shape[d];
            }
            for d in 0..out_shape.len() {
                let coord = (flat / strides[d]) % out_shape[d];
                if d >= pad && shape[d - pad] != 1 {
                    let mut s = 1usize;
                    for dd in (d - pad + 1)..shape.len() {
                        s *= shape[dd];
                    }
                    off += coord * s;
                }
            }
            off
        };
        let (va, vb) = (data(a), data(b));
        let out = (0..n).map(|i| f(va[idx(i, sa)], vb[idx(i, sb)])).collect();
        Ok(Box::new(FakeBuf(out)))
    }

    fn reduce_dev(&self, kind: ReduceKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        UNARY.fetch_add(1, Ordering::SeqCst);
        let v = data(x);
        let s: f32 = v.iter().sum();
        let out = match kind {
            ReduceKind::Sum => s,
            ReduceKind::Mean => s / v.len().max(1) as f32,
        };
        Ok(Box::new(FakeBuf(vec![out])))
    }

    fn sum_dim_dev(
        &self,
        x: &dyn DeviceBuffer,
        shape: &[usize],
        dim: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        UNARY.fetch_add(1, Ordering::SeqCst);
        let v = data(x);
        let inner: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();
        let mut out = vec![0f32; outer * inner];
        for o in 0..outer {
            for kk in 0..shape[dim] {
                for i in 0..inner {
                    out[o * inner + i] += v[(o * shape[dim] + kk) * inner + i];
                }
            }
        }
        Ok(Box::new(FakeBuf(out)))
    }

    fn fill_dev(&self, value: f32, len: usize) -> Result<Box<dyn DeviceBuffer>> {
        // Device-side fill: no host transfer counted.
        Ok(Box::new(FakeBuf(vec![value; len])))
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
fn device_backward_gradients_match_cpu() {
    let _serial = setup();
    let xs = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.5];
    let ws = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];

    let run = |device: Option<Device>| -> (Vec<f32>, Vec<f32>) {
        let mut x = Tensor::from_vec(xs.clone(), &[2, 3]).unwrap();
        let mut w = Tensor::from_vec(ws.clone(), &[3, 2]).unwrap();
        if let Some(d) = device {
            x = x.to_device(d).unwrap();
            w = w.to_device(d).unwrap();
        }
        let (x, w) = (x.requires_grad_(true), w.requires_grad_(true));
        let loss = x.matmul(&w).unwrap().relu().mean();
        loss.backward();
        (x.grad().unwrap().to_vec(), w.grad().unwrap().to_vec())
    };

    let (gx_cpu, gw_cpu) = run(None);
    let (gx_dev, gw_dev) = run(Some(DEV));
    for (d, c) in gx_dev.iter().zip(gx_cpu.iter()) {
        assert!((d - c).abs() < 1e-5, "x grad: device {d} vs cpu {c}");
    }
    for (d, c) in gw_dev.iter().zip(gw_cpu.iter()) {
        assert!((d - c).abs() < 1e-5, "w grad: device {d} vs cpu {c}");
    }
}

#[test]
fn training_loop_stays_resident() {
    let _serial = setup();
    // Linear regression y = x@w_true + b_true, trained entirely on the fake
    // device: forward (matmul + broadcast bias), MSE loss, backward, manual
    // SGD - asserting the ONLY per-step host traffic is two scalar reads
    // (loss.item() and the mean-backward seed value).
    let x = Tensor::from_vec(vec![1.0, 0.5, -0.3, 1.2, 0.7, -0.8, -1.1, 0.4], &[4, 2]).unwrap();
    let w_true = Tensor::from_vec(vec![2.0, -1.0], &[2, 1]).unwrap();
    let y = x.matmul(&w_true).unwrap().add(&Tensor::from_vec(vec![0.5], &[1]).unwrap()).unwrap();

    let xd = x.to_device(DEV).unwrap();
    let yd = y.to_device(DEV).unwrap();
    let lr = Tensor::scalar(0.1).to_device(DEV).unwrap();
    let mut w = Tensor::from_vec(vec![0.0, 0.0], &[2, 1]).unwrap().to_device(DEV).unwrap().requires_grad_(true);
    let mut b = Tensor::from_vec(vec![0.0], &[1]).unwrap().to_device(DEV).unwrap().requires_grad_(true);

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    let before = (ALLOC_ELEMS.load(Ordering::SeqCst), TO_HOST_ELEMS.load(Ordering::SeqCst));
    for step in 0..40 {
        let pred = xd.matmul(&w).unwrap().add(&b).unwrap();
        let diff = pred.sub(&yd).unwrap();
        let loss = diff.mul(&diff).unwrap().mean();
        loss.backward();
        let (gw, gb) = (w.grad().unwrap(), b.grad().unwrap());
        assert_eq!(gw.device(), DEV);
        assert_eq!(gb.device(), DEV);
        w = w.detach_copy().sub(&gw.mul(&lr).unwrap()).unwrap().requires_grad_(true);
        b = b.detach_copy().sub(&gb.mul(&lr).unwrap()).unwrap().requires_grad_(true);
        assert_eq!(w.device(), DEV);
        let l = loss.item();
        if step == 0 {
            first = l;
        }
        last = l;
    }
    let after = (ALLOC_ELEMS.load(Ordering::SeqCst), TO_HOST_ELEMS.load(Ordering::SeqCst));

    assert!(last < first * 0.05, "loss did not converge on device: {first} -> {last}");
    // Per step the host sees exactly two scalars come back (the mean-backward
    // seed read and loss.item()) and nothing goes up.
    assert_eq!(after.0 - before.0, 0, "no per-step uploads");
    assert_eq!(after.1 - before.1, 2 * 40, "exactly two scalar downloads per step");

    // And the learned parameters match the same loop run on the cpu.
    let mut wc = Tensor::from_vec(vec![0.0, 0.0], &[2, 1]).unwrap().requires_grad_(true);
    let mut bc = Tensor::from_vec(vec![0.0], &[1]).unwrap().requires_grad_(true);
    let lrc = Tensor::scalar(0.1);
    for _ in 0..40 {
        let pred = x.matmul(&wc).unwrap().add(&bc).unwrap();
        let diff = pred.sub(&y).unwrap();
        let loss = diff.mul(&diff).unwrap().mean();
        loss.backward();
        wc = wc.detach_copy().sub(&wc.grad().unwrap().mul(&lrc).unwrap()).unwrap().requires_grad_(true);
        bc = bc.detach_copy().sub(&bc.grad().unwrap().mul(&lrc).unwrap()).unwrap().requires_grad_(true);
    }
    for (d, c) in w.to_vec().iter().zip(wc.to_vec().iter()) {
        assert!((d - c).abs() < 1e-4, "w: device {d} vs cpu {c}");
    }
    for (d, c) in b.to_vec().iter().zip(bc.to_vec().iter()) {
        assert!((d - c).abs() < 1e-4, "b: device {d} vs cpu {c}");
    }
}

#[test]
fn device_broadcast_binary_matches_cpu() {
    let _serial = setup();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let dev = a.to_device(DEV).unwrap().add(&b.to_device(DEV).unwrap()).unwrap();
    assert_eq!(dev.device(), DEV);
    assert_eq!(dev.to_vec(), a.add(&b).unwrap().to_vec());
    // Scalar broadcast too (0-d against 2-D).
    let s = Tensor::scalar(0.5).to_device(DEV).unwrap();
    let scaled = a.to_device(DEV).unwrap().mul(&s).unwrap();
    assert_eq!(scaled.device(), DEV);
    assert_eq!(scaled.to_vec(), a.mul(&Tensor::scalar(0.5)).unwrap().to_vec());
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
