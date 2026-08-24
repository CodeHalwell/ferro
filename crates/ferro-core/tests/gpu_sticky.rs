//! Structural proof that softmax/log_softmax and the Gelu/Silu activations
//! stay device-resident: a fake sticky backend implements the new kernels
//! while counting every transfer, so a host round trip in forward or backward
//! shows up as a copy_to_host / alloc_from_host delta (or a panic from the
//! host-slice methods, which are wired to fail).

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::dispatch::{
    register_backend, Backend, BinaryKind, DeviceBuffer, ReduceKind, UnaryKind,
};
use ferro_core::{Device, Result, Tensor};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static TO_HOST: AtomicUsize = AtomicUsize::new(0);
static UNARY: AtomicUsize = AtomicUsize::new(0);
static BINARY: AtomicUsize = AtomicUsize::new(0);
static SOFTMAX: AtomicUsize = AtomicUsize::new(0);
static LOG_SOFTMAX: AtomicUsize = AtomicUsize::new(0);

const DEV: Device = Device::Cuda(7);

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
    &buf.as_any()
        .downcast_ref::<FakeBuf>()
        .expect("buffer from another backend")
        .0
}

// Row-wise stable softmax over `cols`-element rows of the flat buffer.
fn row_forward(v: &[f32], cols: usize, log: bool) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for (r, row) in v.chunks(cols).enumerate() {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&x| (x - m).exp()).sum();
        for (k, &x) in row.iter().enumerate() {
            out[r * cols + k] =
                if log { x - (m + sum.ln()) } else { (x - m).exp() / sum };
        }
    }
    out
}

impl Backend for FakeDevice {
    // Host-slice paths must never run for device-resident tensors.
    fn unary(&self, _kind: UnaryKind, _x: &[f32]) -> Vec<f32> {
        panic!("host unary must not run for device-resident tensors");
    }
    fn binary(&self, _kind: BinaryKind, _a: &[f32], _b: &[f32]) -> Vec<f32> {
        panic!("host binary must not run for device-resident tensors");
    }
    fn matmul(&self, _a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        panic!("host matmul must not run for device-resident tensors");
    }

    fn alloc_from_host(&self, d: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FakeBuf(d.to_vec())))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        TO_HOST.fetch_add(1, Ordering::SeqCst);
        Ok(data(buf).to_vec())
    }

    fn unary_dev(&self, kind: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        UNARY.fetch_add(1, Ordering::SeqCst);
        let out: Vec<f32> = data(x).iter().map(|&v| match kind {
            UnaryKind::Gelu => {
                let u = 0.797_884_6 * (v + 0.044715 * v * v * v);
                0.5 * v * (1.0 + u.tanh())
            }
            UnaryKind::Silu => v / (1.0 + (-v).exp()),
            UnaryKind::Powf(p) => v.powf(p),
            UnaryKind::Sigmoid => 1.0 / (1.0 + (-v).exp()),
            UnaryKind::Tanh => v.tanh(),
            UnaryKind::Exp => v.exp(),
            other => panic!("fake unary kernel not implemented for {other:?}"),
        }).collect();
        Ok(Box::new(FakeBuf(out)))
    }
    fn binary_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
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
    fn matmul_dev(&self, _a: &dyn DeviceBuffer, _b: &dyn DeviceBuffer, _m: usize, _k: usize, _n: usize, _ta: bool, _tb: bool) -> Result<Box<dyn DeviceBuffer>> {
        Err(ferro_core::Error::Unsupported { op: "matmul_dev", msg: "not needed by these tests".into() })
    }
    fn binary_bc_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, sa: &[usize], b: &dyn DeviceBuffer, sb: &[usize], out_shape: &[usize]) -> Result<Box<dyn DeviceBuffer>> {
        BINARY.fetch_add(1, Ordering::SeqCst);
        let f = |x: f32, y: f32| match kind {
            BinaryKind::Add => x + y,
            BinaryKind::Sub => x - y,
            BinaryKind::Mul => x * y,
            BinaryKind::Div => x / y,
        };
        let n: usize = out_shape.iter().product();
        let idx = |flat: usize, shape: &[usize]| -> usize {
            let pad = out_shape.len() - shape.len();
            // Decompose flat into out_shape coordinates FIRST, then map each
            // coordinate through the source strides (skipping broadcast dims).
            // The previous version folded the coordinate extraction and the
            // stride multiply into one pass using `shape` suffix products,
            // which misindexes whenever a source dim is 1 (broadcast).
            let mut off = 0usize;
            for d in 0..out_shape.len() {
                let coord = (flat / out_shape[d + 1..].iter().product::<usize>()) % out_shape[d];
                if d < pad || shape[d - pad] == 1 {
                    continue;
                }
                let mut s = 1usize;
                for dd in (d - pad + 1)..shape.len() {
                    s *= shape[dd];
                }
                off += coord * s;
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
            ReduceKind::Mean => s / v.len() as f32,
        };
        Ok(Box::new(FakeBuf(vec![out])))
    }
    fn sum_dim_dev(&self, x: &dyn DeviceBuffer, shape: &[usize], dim: usize) -> Result<Box<dyn DeviceBuffer>> {
        UNARY.fetch_add(1, Ordering::SeqCst);
        let v = data(x);
        let inner: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();
        let mut out = vec![0f32; outer * inner];
        for o in 0..outer {
            for k in 0..shape[dim] {
                for i in 0..inner {
                    out[o * inner + i] += v[(o * shape[dim] + k) * inner + i];
                }
            }
        }
        Ok(Box::new(FakeBuf(out)))
    }
    // Device-side fill keeps the autograd gradient seed from uploading.
    fn fill_dev(&self, value: f32, len: usize) -> Result<Box<dyn DeviceBuffer>> {
        Ok(Box::new(FakeBuf(vec![value; len])))
    }

    fn softmax_dev(&self, x: &dyn DeviceBuffer, rows: usize, cols: usize) -> Result<Box<dyn DeviceBuffer>> {        SOFTMAX.fetch_add(1, Ordering::SeqCst);
        assert_eq!(rows * cols, x.len(), "softmax_dev rows*cols mismatch");
        Ok(Box::new(FakeBuf(row_forward(data(x), cols, false))))
    }
    fn log_softmax_dev(&self, x: &dyn DeviceBuffer, rows: usize, cols: usize) -> Result<Box<dyn DeviceBuffer>> {
        LOG_SOFTMAX.fetch_add(1, Ordering::SeqCst);
        assert_eq!(rows * cols, x.len(), "log_softmax_dev rows*cols mismatch");
        Ok(Box::new(FakeBuf(row_forward(data(x), cols, true))))
    }
}

static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> MutexGuard<'static, ()> {
    register_backend(DEV, Arc::new(FakeDevice));
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn softmax_forward_and_backward_never_round_trip() {
    let _serial = setup();
    let (a, b, c) = ([2usize, 5, 6], [3usize, 6], [4usize, 6]);
    let make = |d: &[f32], sh: &[usize]| Tensor::from_vec(d.to_vec(), sh).unwrap();
    let xd = make(&data3(), &a).to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let wd = make(&coef(), &a).to_device(DEV).unwrap();

    let before = counts();
    let y = xd.softmax(2).unwrap();
    assert_eq!(y.device(), DEV);
    assert_eq!(SOFTMAX.load(Ordering::SeqCst) - before.5, 1, "softmax_dev not dispatched");
    assert_eq!(TO_HOST.load(Ordering::SeqCst) - before.1, 0, "forward downloaded");

    y.mul(&wd).unwrap().sum().backward();
    let g = xd.grad().unwrap();
    assert_eq!(g.device(), DEV);
    let after = counts();
    assert_eq!(after.1 - before.1, 0, "backward downloaded to host mid-graph");
    assert_eq!(after.0 - before.0, 0, "backward uploaded from host");

    // Numerically identical to the same graph computed on the cpu.
    let xc = make(&data3(), &a).requires_grad_(true).unwrap();
    let wc = make(&coef(), &a);
    xc.softmax(2).unwrap().mul(&wc).unwrap().sum().backward();
    for (d, c) in g.to_vec().iter().zip(xc.grad().unwrap().to_vec()) {
        assert!((d - c).abs() < 1e-5, "grad device {d} vs cpu {c}");
    }
    let _ = (b, c);
}

#[test]
fn gelu_silu_log_softmax_dispatch_device_kernels() {
    let _serial = setup();
    let xd = Tensor::from_vec(data3(), &[2, 5, 6]).unwrap().to_device(DEV).unwrap();

    let before = counts();
    let g = xd.gelu();
    assert_eq!(g.device(), DEV);
    assert_eq!(UNARY.load(Ordering::SeqCst) - before.2, 1, "gelu did not use unary_dev");
    assert_eq!(g.to_vec(), Tensor::from_vec(data3(), &[2, 5, 6]).unwrap().gelu().to_vec());

    let before = counts();
    let s = xd.silu();
    assert_eq!(s.device(), DEV);
    assert_eq!(UNARY.load(Ordering::SeqCst) - before.2, 1, "silu did not use unary_dev");

    let before = counts();
    let lp = xd.log_softmax(2).unwrap();
    assert_eq!(lp.device(), DEV);
    assert_eq!(LOG_SOFTMAX.load(Ordering::SeqCst) - before.4, 1, "log_softmax_dev not dispatched");
    assert_eq!(
        lp.to_vec(),
        Tensor::from_vec(data3(), &[2, 5, 6]).unwrap().log_softmax(2).unwrap().to_vec()
    );

    // Backward through gelu also stays resident until the final download.
    let xr = Tensor::from_vec(data3(), &[2, 5, 6]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let before = counts();
    xr.gelu().sum().backward();
    assert_eq!(xr.grad().unwrap().device(), DEV);
    assert_eq!(TO_HOST.load(Ordering::SeqCst) - before.1, 0, "gelu backward downloaded");
}

fn data3() -> Vec<f32> {
    (0..60).map(|i| ((i as f32) * 0.41).sin() * 1.7)
        .collect()
}
fn coef() -> Vec<f32> {
    (0..60).map(|i| ((i as f32) * 0.23).cos())
        .collect()
}
fn counts() -> (usize, usize, usize, usize, usize, usize) {
    (
        ALLOCS.load(Ordering::SeqCst),
        TO_HOST.load(Ordering::SeqCst),
        UNARY.load(Ordering::SeqCst),
        BINARY.load(Ordering::SeqCst),
        LOG_SOFTMAX.load(Ordering::SeqCst),
        SOFTMAX.load(Ordering::SeqCst),
    )
}

#[test]
fn debug_dump() {
    let _serial = setup();
    let a = [2usize, 5, 6];
    let make = |d: &[f32], sh: &[usize]| Tensor::from_vec(d.to_vec(), sh).unwrap();
    let xd = make(&data3(), &a).to_device(DEV).unwrap();
    let wd = make(&coef(), &a).to_device(DEV).unwrap();
    let y = xd.softmax(2).unwrap();
    let z = y.mul(&wd).unwrap();
    let s = z.sum();
    println!("softmax dev {:?}", y.to_vec());
    println!("mul dev    {:?}", z.to_vec());
    println!("sum dev    {:?}", s.to_vec());
    let xc = make(&data3(), &a);
    let yc = xc.softmax(2).unwrap();
    println!("softmax cpu {:?}", yc.to_vec());
    let zc = yc.mul(&make(&coef(), &a)).unwrap();
    println!("mul cpu    {:?}", zc.to_vec());
    let xd2 = make(&data3(), &a).to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let wd2 = make(&coef(), &a).to_device(DEV).unwrap();
    let gdev = { let yv = xd2.softmax(2).unwrap(); yv.mul(&wd2).unwrap().sum().backward(); xd2.grad().unwrap().to_vec() };
    let xc2 = make(&data3(), &a).requires_grad_(true).unwrap();
    let wc2 = make(&coef(), &a);
    xc2.softmax(2).unwrap().mul(&wc2).unwrap().sum().backward();
    let gcpu = xc2.grad().unwrap().to_vec();
    for i in 0..gdev.len() {
        if (gdev[i] - gcpu[i]).abs() > 1e-5 {
            println!("DIFF {i}: dev {} cpu {} row {}", gdev[i], gcpu[i], i / 6);
        }
    }
}
