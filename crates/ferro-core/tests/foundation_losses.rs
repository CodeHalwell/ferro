use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::dispatch::{
    register_backend, Backend, BinaryKind, DeviceBuffer, ReduceKind, UnaryKind,
};
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
    &buf.as_any()
        .downcast_ref::<FakeBuf>()
        .expect("buffer from another backend")
        .0
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
            UnaryKind::Gtz => data(x)
                .iter()
                .map(|v| if *v > 0.0 { 1.0 } else { 0.0 })
                .collect(),
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
        let out = data(a)
            .iter()
            .zip(data(b))
            .map(|(&x, &y)| f(x, y))
            .collect();
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
            ReduceKind::Mean => s / v.len() as f32,
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
static SERIAL: Mutex<()> = Mutex::new(());
fn setup() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(FakeDevice));
    guard
}
fn loss(kind: usize, x: &Tensor, t: &Tensor) -> Result<Tensor> {
    match kind {
        0 => x.bce_with_logits_loss(t),
        1 => x.huber_loss(t, 0.7),
        _ => x.smooth_l1_loss(t, 0.7),
    }
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
}

#[test]
fn loss_backward_uses_preplaced_derivatives_without_host_transfers() {
    let _guard = setup();
    for kind in 0..3 {
        let x = Tensor::from_vec(vec![-1., 0.2, 2.], &[3])
            .unwrap()
            .to_device(DEV)
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let t = Tensor::zeros(&[3])
            .to_device(DEV)
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let y = loss(kind, &x, &t).unwrap();
        let before = (
            ALLOCS.load(Ordering::SeqCst),
            TO_HOST.load(Ordering::SeqCst),
        );
        y.backward();
        assert_eq!(
            (
                ALLOCS.load(Ordering::SeqCst),
                TO_HOST.load(Ordering::SeqCst)
            ),
            before
        );
        assert_eq!(x.grad().unwrap().device(), DEV);
        assert_eq!(t.grad().unwrap().device(), DEV);
    }
}

#[test]
fn large_logits_and_zero_logit_have_finite_correct_gradients() {
    let _guard = setup();
    for dev in [Device::Cpu, DEV] {
        let x = Tensor::from_vec(vec![-1000., 0., 1000.], &[3])
            .unwrap()
            .to_device(dev)
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let t = Tensor::from_vec(vec![1., 0.25, 0.], &[3])
            .unwrap()
            .to_device(dev)
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let y = x.bce_with_logits_loss(&t).unwrap();
        assert!(y.item().is_finite());
        y.backward();
        close(&x.grad().unwrap().to_vec(), &[-1. / 3., 0.25 / 3., 1. / 3.]);
        close(&t.grad().unwrap().to_vec(), &[1000. / 3., 0., -1000. / 3.]);
    }
}

#[test]
fn loss_gradchecks_and_saved_input_versions() {
    for kind in 0..3 {
        let x = Tensor::from_vec(vec![-1.2, 0.3, 1.7], &[3]).unwrap();
        let t = Tensor::scalar(0.1);
        ferro_core::testkit::grad_check_strict(&[x.clone(), t.clone()], |v| {
            loss(kind, &v[0], &v[1]).unwrap()
        });
        let x = x.requires_grad_(true).unwrap();
        let y = loss(kind, &x, &t).unwrap();
        t.fill_(0.9).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| y.backward()));
        let payload = panic.expect_err("saved original inputs must retain version checks");
        let msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(msg.contains("modified by an inplace operation"), "{msg}");
    }
}

#[test]
fn loss_invalid_parameters_and_zero_beta_subgradient() {
    let x = Tensor::zeros(&[2]).requires_grad_(true).unwrap();
    let t = Tensor::zeros(&[2]);
    for delta in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(x.huber_loss(&t, delta).is_err());
    }
    for beta in [-1., f32::NAN, f32::INFINITY] {
        assert!(x.smooth_l1_loss(&t, beta).is_err());
    }
    x.smooth_l1_loss(&t, 0.).unwrap().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![0., 0.]);
}

#[test]
fn loss_input_validation_returns_errors() {
    let _guard = setup();
    for kind in 0..3 {
        let x = Tensor::ones(&[2]);
        assert!(loss(kind, &x, &Tensor::ones(&[3])).is_err());
        let integer = Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap();
        assert!(loss(kind, &integer, &x).is_err());
        assert!(loss(kind, &x, &integer).is_err());
        assert!(loss(kind, &x.to_device(DEV).unwrap(), &x).is_err());
        ferro_core::testkit::grad_check_strict(
            &[
                Tensor::scalar(0.2),
                Tensor::from_vec(vec![-0.8, 0.4, 1.7], &[3]).unwrap(),
            ],
            |v| loss(kind, &v[0], &v[1]).unwrap(),
        );
    }
}

#[test]
fn bce_device_backward() {
    device_backward(0);
}
#[test]
fn huber_device_backward() {
    device_backward(1);
}
#[test]
fn smooth_l1_device_backward() {
    device_backward(2);
}

fn device_backward(kind: usize) {
    let _guard = setup();
    {
        for view in [false, true] {
            let run = |dev| {
                let x = Tensor::from_vec(vec![-2., 0., 0.4, 1.1, -0.3, 3.], &[2, 3])
                    .unwrap()
                    .to_device(dev)
                    .unwrap()
                    .requires_grad_(true)
                    .unwrap();
                let t = Tensor::from_vec(vec![0.2, 0.8, 0.4], &[3])
                    .unwrap()
                    .to_device(dev)
                    .unwrap()
                    .requires_grad_(true)
                    .unwrap();
                let (xx, tt) = if view {
                    (x.transpose(0, 1).unwrap(), t.reshape(&[3, 1]).unwrap())
                } else {
                    (x.clone(), t.clone())
                };
                let y = loss(kind, &xx, &tt).unwrap();
                assert_eq!(y.device(), dev);
                assert!(y.requires_grad(), "transfer must not detach loss");
                y.backward();
                assert_eq!(x.grad().unwrap().device(), dev);
                assert_eq!(t.grad().unwrap().device(), dev);
                (
                    y.item(),
                    x.grad().unwrap().to_vec(),
                    t.grad().unwrap().to_vec(),
                )
            };
            let cpu = run(Device::Cpu);
            let device = run(DEV);
            close(&[cpu.0], &[device.0]);
            close(&cpu.1, &device.1);
            close(&cpu.2, &device.2);
        }
    }
}
