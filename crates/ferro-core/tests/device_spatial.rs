//! Resident spatial seams (conv2d via unfold/fold, max_pool2d, batch_norm):
//! a mock device computes each seam from its own reference loops, and
//! counters prove a CNN step never downloads before the loss is read.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::dispatch::{register_backend, Backend, BinaryKind, CpuBackend, DeviceBuffer, ReduceKind, UnaryKind, Window2dDev};
use ferro_core::nn::{cross_entropy, one_hot};
use ferro_core::{Device, Result, Rng, Tensor};

const DEV: Device = Device::Cuda(31);
const BARE: Device = Device::Cuda(32);
static DOWNLOADS: AtomicUsize = AtomicUsize::new(0);
static SPATIAL: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

struct Buf(Vec<f32>, Device);
impl DeviceBuffer for Buf {
    fn device(&self) -> Device { self.1 }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}
fn data(b: &dyn DeviceBuffer) -> &[f32] { &b.as_any().downcast_ref::<Buf>().unwrap().0 }
fn buf(v: Vec<f32>) -> Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buf(v, DEV))) }

/// Input pixel (iy, ix) read by tap (r, q) at output (y, x), if in bounds.
fn source(g: &Window2dDev, r: usize, q: usize, y: usize, x: usize) -> Option<(usize, usize)> {
    let iy = (y * g.stride[0] + r * g.dilation[0]).checked_sub(g.padding[0])?;
    let ix = (x * g.stride[1] + q * g.dilation[1]).checked_sub(g.padding[1])?;
    (iy < g.h && ix < g.w).then_some((iy, ix))
}

struct Mock;
impl Backend for Mock {
    fn unary(&self, k: UnaryKind, x: &[f32]) -> Vec<f32> { CpuBackend.unary(k, x) }
    fn binary(&self, k: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> { CpuBackend.binary(k, a, b) }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> { CpuBackend.matmul(a, b, m, k, n) }
    fn alloc_from_host(&self, x: &[f32]) -> Result<Box<dyn DeviceBuffer>> { buf(x.to_vec()) }
    fn copy_to_host(&self, x: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        DOWNLOADS.fetch_add(1, Ordering::SeqCst);
        Ok(data(x).to_vec())
    }
    fn copy_dev(&self, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> { buf(data(x).to_vec()) }
    fn fill_dev(&self, v: f32, n: usize) -> Result<Box<dyn DeviceBuffer>> { buf(vec![v; n]) }
    fn unary_dev(&self, k: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> { buf(CpuBackend.unary(k, data(x))) }
    fn binary_dev(&self, k: BinaryKind, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> { buf(CpuBackend.binary(k, data(a), data(b))) }
    fn binary_bc_dev(&self, k: BinaryKind, a: &dyn DeviceBuffer, sa: &[usize], b: &dyn DeviceBuffer, sb: &[usize], out: &[usize]) -> Result<Box<dyn DeviceBuffer>> {
        buf(CpuBackend.binary_bc(k, data(a), sa, data(b), sb, out))
    }
    fn matmul_dev(&self, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer, m: usize, k: usize, n: usize, ta: bool, tb: bool) -> Result<Box<dyn DeviceBuffer>> {
        buf(CpuBackend.matmul_t(data(a), data(b), m, k, n, ta, tb))
    }
    fn reduce_dev(&self, kind: ReduceKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        let s: f32 = data(x).iter().sum();
        buf(vec![if matches!(kind, ReduceKind::Sum) { s } else { s / data(x).len() as f32 }])
    }
    fn sum_dim_dev(&self, x: &dyn DeviceBuffer, s: &[usize], d: usize) -> Result<Box<dyn DeviceBuffer>> { buf(CpuBackend.sum_dim(data(x), s, d)) }
    fn softmax_dev(&self, x: &dyn DeviceBuffer, rows: usize, cols: usize) -> Result<Box<dyn DeviceBuffer>> { buf(CpuBackend.softmax(data(x), rows, cols)) }
    fn log_softmax_dev(&self, x: &dyn DeviceBuffer, rows: usize, cols: usize) -> Result<Box<dyn DeviceBuffer>> { buf(CpuBackend.log_softmax(data(x), rows, cols)) }
    fn materialize_dev(&self, x: &dyn DeviceBuffer, shape: &[usize], strides: &[usize], offset: usize) -> Result<Box<dyn DeviceBuffer>> {
        buf((0..shape.iter().product()).map(|mut i| {
            let mut at = offset;
            for d in (0..shape.len()).rev() { at += (i % shape[d]) * strides[d]; i /= shape[d]; }
            data(x)[at]
        }).collect())
    }
    fn unfold2d_dev(&self, x: &dyn DeviceBuffer, g: &Window2dDev) -> Result<Box<dyn DeviceBuffer>> {
        SPATIAL.fetch_add(1, Ordering::SeqCst);
        let (x, [kh, kw], pos) = (data(x), g.kernel, g.oh * g.ow);
        let mut col = vec![0.0; g.taps() * g.positions()];
        for (img, ci, r, q, y, xx) in itertools(g.n, g.c, kh, kw, g.oh, g.ow) {
            if let Some((iy, ix)) = source(g, r, q, y, xx) {
                col[((ci * kh + r) * kw + q) * g.positions() + img * pos + y * g.ow + xx] = x[((img * g.c + ci) * g.h + iy) * g.w + ix];
            }
        }
        buf(col)
    }
    fn fold2d_dev(&self, col: &dyn DeviceBuffer, g: &Window2dDev) -> Result<Box<dyn DeviceBuffer>> {
        SPATIAL.fetch_add(1, Ordering::SeqCst);
        let (col, [kh, kw], pos) = (data(col), g.kernel, g.oh * g.ow);
        let mut x = vec![0.0; g.image_len()];
        for (img, ci, r, q, y, xx) in itertools(g.n, g.c, kh, kw, g.oh, g.ow) {
            if let Some((iy, ix)) = source(g, r, q, y, xx) {
                x[((img * g.c + ci) * g.h + iy) * g.w + ix] += col[((ci * kh + r) * kw + q) * g.positions() + img * pos + y * g.ow + xx];
            }
        }
        buf(x)
    }
    fn max_pool2d_dev(&self, x: &dyn DeviceBuffer, g: &Window2dDev) -> Result<(Box<dyn DeviceBuffer>, Box<dyn DeviceBuffer>)> {
        SPATIAL.fetch_add(1, Ordering::SeqCst);
        let (x, [kh, kw]) = (data(x), g.kernel);
        let (mut out, mut arg) = (vec![0.0; g.output_len()], vec![0.0; g.output_len()]);
        for p in 0..g.n * g.c {
            let plane = &x[p * g.h * g.w..][..g.h * g.w];
            for y in 0..g.oh {
                for xx in 0..g.ow {
                    let best = (0..kh).flat_map(|r| (0..kw).map(move |q| (y * g.stride[0] + r) * g.w + xx * g.stride[1] + q))
                        .fold(None, |b: Option<usize>, i| match b { Some(b) if !(plane[i] > plane[b]) => Some(b), _ => Some(i) }).unwrap();
                    out[(p * g.oh + y) * g.ow + xx] = plane[best];
                    arg[(p * g.oh + y) * g.ow + xx] = best as f32;
                }
            }
        }
        Ok((buf(out)?, buf(arg)?))
    }
    fn max_pool2d_backward_dev(&self, gy: &dyn DeviceBuffer, arg: &dyn DeviceBuffer, g: &Window2dDev) -> Result<Box<dyn DeviceBuffer>> {
        SPATIAL.fetch_add(1, Ordering::SeqCst);
        let mut gx = vec![0.0; g.image_len()];
        for (cell, (&v, &a)) in data(gy).iter().zip(data(arg)).enumerate() {
            gx[cell / (g.oh * g.ow) * g.h * g.w + a as usize] += v;
        }
        buf(gx)
    }
}

fn itertools(n: usize, c: usize, kh: usize, kw: usize, oh: usize, ow: usize) -> impl Iterator<Item = (usize, usize, usize, usize, usize, usize)> {
    (0..n).flat_map(move |i| (0..c).flat_map(move |ci| (0..kh).flat_map(move |r| (0..kw).flat_map(move |q| (0..oh).flat_map(move |y| (0..ow).map(move |x| (i, ci, r, q, y, x)))))))
}

/// A device with storage and transfers only: every spatial seam declines.
struct Bare;
impl Backend for Bare {
    fn unary(&self, k: UnaryKind, x: &[f32]) -> Vec<f32> { CpuBackend.unary(k, x) }
    fn binary(&self, k: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> { CpuBackend.binary(k, a, b) }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> { CpuBackend.matmul(a, b, m, k, n) }
    fn alloc_from_host(&self, x: &[f32]) -> Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buf(x.to_vec(), BARE))) }
    fn copy_to_host(&self, x: &dyn DeviceBuffer) -> Result<Vec<f32>> { Ok(data(x).to_vec()) }
}

fn setup() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(Mock));
    register_backend(BARE, Arc::new(Bare));
    guard
}

fn assert_close(got: &[f32], want: &[f32], rtol: f32, atol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= atol + rtol * w.abs(), "index {i}: {g} vs {w}");
    }
}

fn scaled(t: Tensor, mul: f32, add: f32) -> Tensor { Tensor::from_vec(t.to_vec().iter().map(|v| v * mul + add).collect(), t.shape()).unwrap() }

fn leaf(t: &Tensor, device: Device) -> Tensor { t.to_device(device).unwrap().requires_grad_(true).unwrap() }

fn grads(xs: &[&Tensor]) -> Vec<Vec<f32>> { xs.iter().map(|t| t.grad().expect("grad").to_vec()).collect() }

#[test]
fn conv2d_resident_matches_cpu_values_and_grads() {
    let _g = setup();
    let rng = Rng::new(3);
    for (stride, padding, dilation, k) in [([1, 1], [1, 1], [1, 1], [3, 3]), ([2, 1], [0, 2], [1, 2], [3, 2]), ([1, 1], [0, 0], [1, 1], [1, 1])] {
        let (x, w) = (Tensor::randn(&[2, 3, 7, 6], &rng), Tensor::randn(&[4, 3, k[0], k[1]], &rng));
        let probe = Tensor::randn(&x.conv2d_with_options(&w, stride, padding, dilation, 1).unwrap().shape().to_vec(), &rng);
        let (xc, wc) = (leaf(&x, Device::Cpu), leaf(&w, Device::Cpu));
        let yc = xc.conv2d_with_options(&wc, stride, padding, dilation, 1).unwrap();
        yc.mul(&probe).unwrap().sum().backward();
        let (xd, wd) = (leaf(&x, DEV), leaf(&w, DEV));
        let calls = SPATIAL.load(Ordering::SeqCst);
        let yd = xd.conv2d_with_options(&wd, stride, padding, dilation, 1).unwrap();
        assert_eq!(yd.device(), DEV, "resident conv output");
        yd.mul(&probe.to_device(DEV).unwrap()).unwrap().sum().backward();
        assert_eq!(SPATIAL.load(Ordering::SeqCst) - calls, 3, "unfold, recomputed unfold, fold");
        assert_eq!(xd.grad().unwrap().device(), DEV);
        assert_close(&yd.to_vec(), &yc.to_vec(), 1e-5, 1e-5);
        for (d, c) in grads(&[&xd, &wd]).iter().zip(grads(&[&xc, &wc])) {
            assert_close(d, &c, 1e-4, 1e-4);
        }
    }
}

#[test]
fn max_pool2d_resident_matches_cpu_including_overlap() {
    let _g = setup();
    let rng = Rng::new(5);
    for (k, s) in [(2, 2), (3, 2), (3, 1)] {
        let x = Tensor::randn(&[2, 3, 7, 8], &rng);
        let probe = Tensor::randn(&x.max_pool2d(k, s).unwrap().shape().to_vec(), &rng);
        let xc = leaf(&x, Device::Cpu);
        let yc = xc.max_pool2d(k, s).unwrap();
        yc.mul(&probe).unwrap().sum().backward();
        let xd = leaf(&x, DEV);
        let yd = xd.max_pool2d(k, s).unwrap();
        assert_eq!(yd.device(), DEV);
        yd.mul(&probe.to_device(DEV).unwrap()).unwrap().sum().backward();
        assert_eq!(yd.to_vec(), yc.to_vec());
        assert_close(&xd.grad().unwrap().to_vec(), &xc.grad().unwrap().to_vec(), 1e-6, 1e-6);
    }
}

#[test]
fn batch_norm_resident_matches_cpu_in_train_and_eval() {
    let _g = setup();
    let rng = Rng::new(9);
    let x = scaled(Tensor::randn(&[4, 3, 5, 2], &rng), 2.0, 0.5);
    let (w, b) = (Tensor::randn(&[3], &rng), Tensor::randn(&[3], &rng));
    let (rm, rv) = (Tensor::from_vec(vec![0.1, -0.2, 0.3], &[3]).unwrap(), Tensor::from_vec(vec![1.0, 0.5, 2.0], &[3]).unwrap());
    let probe = Tensor::randn(&[4, 3, 5, 2], &rng);
    for train in [true, false] {
        let run = |device: Device| {
            let (x, w, b) = (leaf(&x, device), leaf(&w, device), leaf(&b, device));
            let out = x.batch_norm(&w, &b, &rm.to_device(device).unwrap(), &rv.to_device(device).unwrap(), 1e-5, train, 0.1).unwrap();
            assert_eq!(out.output.device(), device);
            assert_eq!(out.running_mean.device(), device);
            out.output.mul(&probe.to_device(device).unwrap()).unwrap().sum().backward();
            (out.output.to_vec(), out.running_mean.to_vec(), out.running_var.to_vec(), grads(&[&x, &w, &b]))
        };
        let (c, d) = (run(Device::Cpu), run(DEV));
        assert_close(&d.0, &c.0, 1e-5, 1e-5);
        assert_close(&d.1, &c.1, 1e-6, 1e-6);
        assert_close(&d.2, &c.2, 1e-5, 1e-5);
        for (gd, gc) in d.3.iter().zip(&c.3) {
            assert_close(gd, gc, 1e-4, 1e-4);
        }
    }
}

#[test]
fn cnn_step_downloads_nothing_before_the_loss() {
    let _g = setup();
    let rng = Rng::new(11);
    let x = Tensor::randn(&[4, 2, 8, 8], &rng).to_device(DEV).unwrap();
    let w1 = leaf(&scaled(Tensor::randn(&[3, 2, 3, 3], &rng), 0.3, 0.0), DEV);
    let (bw, bb) = (leaf(&Tensor::ones(&[3]), DEV), leaf(&Tensor::zeros(&[3]), DEV));
    let fc = leaf(&scaled(Tensor::randn(&[48, 5], &rng), 0.1, 0.0), DEV);
    let targets = one_hot(&Tensor::from_vec_i64(vec![0, 3, 1, 4], &[4]).unwrap(), 5).unwrap().to_device(DEV).unwrap();
    let (rm, rv) = (Tensor::zeros(&[3]).to_device(DEV).unwrap(), Tensor::ones(&[3]).to_device(DEV).unwrap());
    let before = DOWNLOADS.load(Ordering::SeqCst);
    let h = x.conv2d(&w1, 1, 1).unwrap();
    let h = h.batch_norm(&bw, &bb, &rm, &rv, 1e-5, true, 0.1).unwrap().output.relu().max_pool2d(2, 2).unwrap();
    let logits = h.reshape(&[4, 48]).unwrap().matmul(&fc).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();
    loss.backward();
    for p in [&w1, &bw, &bb, &fc] {
        assert_eq!(p.grad().unwrap().device(), DEV);
    }
    assert_eq!(DOWNLOADS.load(Ordering::SeqCst), before, "forward and backward stay on the device");
    assert!(loss.item().is_finite());
}

#[test]
fn backends_without_spatial_seams_keep_the_host_fallback() {
    let _g = setup();
    let rng = Rng::new(13);
    let (x, w) = (Tensor::randn(&[1, 2, 5, 5], &rng), Tensor::randn(&[3, 2, 3, 3], &rng));
    let y = x.to_device(BARE).unwrap().conv2d(&w.to_device(BARE).unwrap(), 1, 1).unwrap();
    assert_eq!(y.device(), Device::Cpu);
    assert_eq!(y.to_vec(), x.conv2d(&w, 1, 1).unwrap().to_vec());
    assert_eq!(x.to_device(BARE).unwrap().max_pool2d(2, 2).unwrap().device(), Device::Cpu);
}

#[test]
fn zero_width_matmul_backward_on_cpu() {
    let a = Tensor::zeros(&[2, 0]).requires_grad_(true).unwrap();
    let b = Tensor::zeros(&[0, 3]).requires_grad_(true).unwrap();
    a.matmul(&b).unwrap().sum().backward();
    assert_eq!(a.grad().unwrap().shape(), &[2, 0]);
    assert_eq!(b.grad().unwrap().shape(), &[0, 3]);
}
