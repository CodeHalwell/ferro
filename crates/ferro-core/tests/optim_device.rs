//! Structural proof that optimizer steps are device-resident AND in place:
//! over device params the step math runs as ONE fused *_dev kernel per
//! parameter - zero copy_to_host, zero uploads (scalars ride in as f32
//! kernel arguments, so even Adam's per-step bias corrections cost no
//! traffic), zero allocations after warmup. Follows tests/device.rs's
//! counting backend pattern; buffers are interior-mutable because the
//! in-place kernels overwrite device contents behind `&dyn DeviceBuffer`.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::dispatch::{
    register_backend, AdamWStep, Backend, BinaryKind, DeviceBuffer, UnaryKind,
};
use ferro_core::optim::{Adam, AdamW, Sgd};
use ferro_core::{Device, Param, Result, Tensor};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_ELEMS: AtomicUsize = AtomicUsize::new(0);
static TO_HOST: AtomicUsize = AtomicUsize::new(0);
static BINARY: AtomicUsize = AtomicUsize::new(0);
static UNARY: AtomicUsize = AtomicUsize::new(0);
static FILLS: AtomicUsize = AtomicUsize::new(0);
static FUSED_STEPS: AtomicUsize = AtomicUsize::new(0);

const DEV: Device = Device::Cuda(7);

struct FakeBuf(Mutex<Vec<f32>>);

impl FakeBuf {
    fn new(v: Vec<f32>) -> FakeBuf {
        FakeBuf(Mutex::new(v))
    }
}

impl DeviceBuffer for FakeBuf {
    fn device(&self) -> Device {
        DEV
    }
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn raw(buf: &dyn DeviceBuffer) -> &FakeBuf {
    buf.as_any()
        .downcast_ref::<FakeBuf>()
        .expect("buffer from another backend")
}

fn data(buf: &dyn DeviceBuffer) -> Vec<f32> {
    raw(buf).0.lock().unwrap().clone()
}

struct FakeDevice;

impl Backend for FakeDevice {
    fn unary(&self, _k: UnaryKind, _x: &[f32]) -> Vec<f32> {
        panic!("host-slice path must not run for device-resident tensors");
    }
    fn binary(&self, _k: BinaryKind, _a: &[f32], _b: &[f32]) -> Vec<f32> {
        panic!("host-slice path must not run for device-resident tensors");
    }
    fn matmul(&self, _a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        panic!("host-slice path must not run for device-resident tensors");
    }

    fn alloc_from_host(&self, d: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        ALLOC_ELEMS.fetch_add(d.len(), Ordering::SeqCst);
        Ok(Box::new(FakeBuf::new(d.to_vec())))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        TO_HOST.fetch_add(1, Ordering::SeqCst);
        Ok(data(buf))
    }
    fn unary_dev(&self, kind: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        UNARY.fetch_add(1, Ordering::SeqCst);
        let out = match kind {
            UnaryKind::Sqrt => data(x).iter().map(|v| v.sqrt()).collect(),
            UnaryKind::Neg => data(x).iter().map(|v| -v).collect(),
            // Mean-backward broadcast seed.
            UnaryKind::Gtz => data(x)
                .iter()
                .map(|v| if *v > 0.0 { 1.0 } else { 0.0 })
                .collect(),
            other => panic!("fake device kernel not implemented for {other:?}"),
        };
        Ok(Box::new(FakeBuf::new(out)))
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
        let (va, vb) = (data(a), data(b));
        let out = va.iter().zip(vb.iter()).map(|(&x, &y)| f(x, y)).collect();
        Ok(Box::new(FakeBuf::new(out)))
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
        let (va, vb) = (data(a), data(b));
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
        Ok(Box::new(FakeBuf::new(out)))
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
        let n: usize = out_shape.iter().product();
        // Right-aligned broadcast strides over the flat output.
        let strides = |shape: &[usize]| -> Vec<usize> {
            let pad = out_shape.len() - shape.len();
            let own = {
                let mut st = vec![1usize; shape.len()];
                let mut acc = 1usize;
                for i in (0..shape.len()).rev() {
                    st[i] = acc;
                    acc *= shape[i];
                }
                st
            };
            (0..out_shape.len())
                .map(|i| {
                    if i < pad || shape[i - pad] != out_shape[i] {
                        0
                    } else {
                        own[i - pad]
                    }
                })
                .collect()
        };
        let (sta, stb) = (strides(sa), strides(sb));
        let idx = |flat: usize, st: &[usize]| -> usize {
            let mut off = 0usize;
            let mut stride = 1usize;
            for d in (0..out_shape.len()).rev() {
                off += (flat / stride % out_shape[d]) * st[d];
                stride *= out_shape[d];
            }
            off
        };
        let (va, vb) = (data(a), data(b));
        let out = (0..n)
            .map(|i| f(va[idx(i, &sta)], vb[idx(i, &stb)]))
            .collect();
        Ok(Box::new(FakeBuf::new(out)))
    }

    fn reduce_dev(
        &self,
        _kind: ferro_core::dispatch::ReduceKind,
        x: &dyn DeviceBuffer,
    ) -> Result<Box<dyn DeviceBuffer>> {
        let v = data(x);
        let s: f32 = v.iter().sum();
        Ok(Box::new(FakeBuf::new(vec![s / v.len() as f32])))
    }

    fn sum_dim_dev(
        &self,
        x: &dyn DeviceBuffer,
        shape: &[usize],
        dim: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
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
        Ok(Box::new(FakeBuf::new(out)))
    }

    fn fill_dev(&self, value: f32, len: usize) -> Result<Box<dyn DeviceBuffer>> {
        // Device-side zero state: no host upload counted.
        FILLS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FakeBuf::new(vec![value; len])))
    }

    // --- in-place kernels: mutate buffer contents behind &dyn DeviceBuffer.
    // The math delegates to the trait's host defaults so fused device steps
    // are bit-identical to the cpu path by construction.

    fn write_dev_from_host(&self, dst: &dyn DeviceBuffer, d: &[f32]) -> Result<()> {
        // Host-to-device traffic, same counters as alloc_from_host.
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        ALLOC_ELEMS.fetch_add(d.len(), Ordering::SeqCst);
        raw(dst).0.lock().unwrap().copy_from_slice(d);
        Ok(())
    }

    fn copy_dev(&self, src: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        // Device-to-device: no host traffic counted.
        Ok(Box::new(FakeBuf::new(data(src))))
    }

    fn fill_inplace_dev(&self, dst: &dyn DeviceBuffer, value: f32) -> Result<()> {
        FILLS.fetch_add(1, Ordering::SeqCst);
        raw(dst).0.lock().unwrap().fill(value);
        Ok(())
    }

    fn affine_inplace_dev(&self, dst: &dyn DeviceBuffer, mul: f32, add: f32) -> Result<()> {
        BINARY.fetch_add(1, Ordering::SeqCst);
        self.affine_inplace(&mut raw(dst).0.lock().unwrap(), mul, add);
        Ok(())
    }

    fn binary_inplace_dev(
        &self,
        kind: BinaryKind,
        dst: &dyn DeviceBuffer,
        src: &dyn DeviceBuffer,
    ) -> Result<()> {
        BINARY.fetch_add(1, Ordering::SeqCst);
        let s = data(src); // clone first: dst may be src
        self.binary_inplace(kind, &mut raw(dst).0.lock().unwrap(), &s);
        Ok(())
    }

    fn axpy_inplace_dev(
        &self,
        alpha: f32,
        dst: &dyn DeviceBuffer,
        src: &dyn DeviceBuffer,
    ) -> Result<()> {
        BINARY.fetch_add(1, Ordering::SeqCst);
        let s = data(src);
        self.axpy_inplace(alpha, &mut raw(dst).0.lock().unwrap(), &s);
        Ok(())
    }

    fn sgd_step_dev(
        &self,
        p: &dyn DeviceBuffer,
        v: &dyn DeviceBuffer,
        g: &dyn DeviceBuffer,
        lr: f32,
        momentum: f32,
        nesterov: bool,
    ) -> Result<()> {
        FUSED_STEPS.fetch_add(1, Ordering::SeqCst);
        let gh = data(g);
        let (mut pl, mut vl) = (raw(p).0.lock().unwrap(), raw(v).0.lock().unwrap());
        self.sgd_step(&mut pl, &mut vl, &gh, lr, momentum, nesterov);
        Ok(())
    }

    fn adamw_step_dev(
        &self,
        p: &dyn DeviceBuffer,
        m: &dyn DeviceBuffer,
        v: &dyn DeviceBuffer,
        g: &dyn DeviceBuffer,
        hp: AdamWStep,
    ) -> Result<()> {
        FUSED_STEPS.fetch_add(1, Ordering::SeqCst);
        let gh = data(g);
        let (mut pl, mut ml, mut vl) = (
            raw(p).0.lock().unwrap(),
            raw(m).0.lock().unwrap(),
            raw(v).0.lock().unwrap(),
        );
        self.adamw_step(&mut pl, &mut ml, &mut vl, &gh, hp);
        Ok(())
    }
}

// Counters are process-global and the harness runs tests in parallel.
static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> MutexGuard<'static, ()> {
    register_backend(DEV, Arc::new(FakeDevice));
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn counts() -> (usize, usize, usize, usize, usize, usize, usize) {
    (
        ALLOCS.load(Ordering::SeqCst),
        TO_HOST.load(Ordering::SeqCst),
        BINARY.load(Ordering::SeqCst),
        UNARY.load(Ordering::SeqCst),
        ALLOC_ELEMS.load(Ordering::SeqCst),
        FILLS.load(Ordering::SeqCst),
        FUSED_STEPS.load(Ordering::SeqCst),
    )
}

/// Linear model with two params on DEV; grads come from one backward pass.
fn make_params(rng: &ferro_core::Rng) -> (Param, Param, Tensor, Tensor) {
    let x = Tensor::from_vec(vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.5], &[2, 3])
        .unwrap()
        .to_device(DEV)
        .unwrap();
    let y = Tensor::from_vec(vec![1.0, -1.0], &[2, 1])
        .unwrap()
        .to_device(DEV)
        .unwrap();
    let w = Param::new(Tensor::randn(&[3, 1], rng).to_device(DEV).unwrap());
    let b = Param::new(Tensor::randn(&[1, 1], rng).to_device(DEV).unwrap());
    (w, b, x, y)
}

fn mse_dev(w: &Param, b: &Param, x: &Tensor, y: &Tensor) -> Tensor {
    let diff = x.matmul(&w.tensor()).unwrap().add(&b.tensor()).unwrap();
    let d = diff.sub(y).unwrap();
    d.mul(&d).unwrap().mean()
}

#[test]
fn sgd_step_is_fully_device_resident() {
    let _serial = setup();
    let rng = ferro_core::Rng::new(11);
    let (w, b, x, y) = make_params(&rng);
    let mut opt = Sgd::new(vec![w.clone(), b.clone()], 0.05).with_momentum(0.9);

    // Warmup step: creates velocity buffers via device fills.
    mse_dev(&w, &b, &x, &y).backward();
    opt.step();
    assert_eq!(w.tensor().device(), DEV);
    let (wp, bp) = (w.tensor()._storage_ptr(), b.tensor()._storage_ptr());

    // Measure around step() only: forward/backward traffic is not the
    // optimizer's.
    let mut d = (0, 0, 0, 0, 0, 0, 0);
    for _ in 0..10 {
        opt.zero_grad();
        mse_dev(&w, &b, &x, &y).backward();
        let before = counts();
        opt.step();
        let after = counts();
        d.0 += after.0 - before.0;
        d.1 += after.1 - before.1;
        d.2 += after.2 - before.2;
        d.3 += after.3 - before.3;
        d.4 += after.4 - before.4;
        d.5 += after.5 - before.5;
        d.6 += after.6 - before.6;
    }

    assert_eq!(d.1, 0, "no copy_to_host during sgd steps");
    assert_eq!(d.0, 0, "no uploads during sgd steps");
    assert_eq!(d.5, 0, "state already allocated at warmup");
    assert_eq!(d.6, 20, "one fused kernel per param per step");
    assert_eq!(d.2, 0, "fused steps replace the per-op binary kernels");
    // In place: the parameters kept their storage across every step.
    assert_eq!(w.tensor()._storage_ptr(), wp, "w reallocated");
    assert_eq!(b.tensor()._storage_ptr(), bp, "b reallocated");
}

#[test]
fn adam_step_downloads_nothing_uploads_only_bias_scalars() {
    let _serial = setup();
    let rng = ferro_core::Rng::new(12);
    let (w, b, x, y) = make_params(&rng);
    let mut opt = AdamW::new(vec![w.clone(), b.clone()], 0.05)
        .with_weight_decay(0.01)
        .with_betas(0.9, 0.99);

    mse_dev(&w, &b, &x, &y).backward();
    opt.step();
    let (wp, bp) = (w.tensor()._storage_ptr(), b.tensor()._storage_ptr());

    // Measure around step() only. Scalars (betas, lr, per-step bias
    // corrections) ride into the fused kernel as f32 arguments, so a steady
    // AdamW step moves ZERO bytes in either direction - the old
    // per-step bias-correction uploads are gone.
    const STEPS: usize = 10;
    let mut allocs = 0;
    let mut elems = 0;
    let mut to_host = 0;
    let mut binary = 0;
    let mut unary = 0;
    let mut fused = 0;
    for _ in 0..STEPS {
        opt.zero_grad();
        mse_dev(&w, &b, &x, &y).backward();
        let before = counts();
        opt.step();
        let after = counts();
        allocs += after.0 - before.0;
        to_host += after.1 - before.1;
        binary += after.2 - before.2;
        unary += after.3 - before.3;
        elems += after.4 - before.4;
        fused += after.6 - before.6;
    }

    assert_eq!(to_host, 0, "no copy_to_host during adamw steps");
    assert_eq!(allocs, 0, "no uploads during adamw steps");
    assert_eq!(elems, 0, "zero uploaded elements per step");
    assert_eq!(fused, 2 * STEPS, "one fused kernel per param per step");
    assert_eq!(binary, 0, "fused steps replace the per-op binary kernels");
    assert_eq!(unary, 0, "sqrt runs inside the fused kernel");
    assert_eq!(w.tensor()._storage_ptr(), wp, "w reallocated");
    assert_eq!(b.tensor()._storage_ptr(), bp, "b reallocated");
}

#[test]
fn device_sgd_matches_cpu_bitwise_within_tolerance() {
    let _serial = setup();
    let vals_w = vec![0.3, -0.4, 0.5];
    let vals_b = vec![0.25];

    let run = |dev: Option<Device>| -> (Vec<f32>, Vec<f32>) {
        let x_host = Tensor::from_vec(vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.5], &[2, 3]).unwrap();
        let y_host = Tensor::from_vec(vec![1.0, -1.0], &[2, 1]).unwrap();
        let w_host = Tensor::from_vec(vals_w.clone(), &[3, 1]).unwrap();
        let b_host = Tensor::from_vec(vals_b.clone(), &[1, 1]).unwrap();
        let (x, y) = match dev {
            Some(d) => (x_host.to_device(d).unwrap(), y_host.to_device(d).unwrap()),
            None => (x_host.clone(), y_host.clone()),
        };
        let mut w = Param::new(match dev {
            Some(d) => w_host.to_device(d).unwrap(),
            None => w_host.clone(),
        });
        let mut b = Param::new(match dev {
            Some(d) => b_host.to_device(d).unwrap(),
            None => b_host.clone(),
        });
        let mut opt = Sgd::new(vec![w.clone(), b.clone()], 0.1).with_momentum(0.9);
        for _ in 0..20 {
            let pred = x.matmul(&w.tensor()).unwrap().add(&b.tensor()).unwrap();
            let diff = pred.sub(&y).unwrap();
            let loss = diff.mul(&diff).unwrap().mean();
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
        (w.tensor().to_vec(), b.tensor().to_vec())
    };

    let (wd, bd) = run(Some(DEV));
    let (wc, bc) = run(None);
    for (d, c) in wd.iter().zip(wc.iter()) {
        assert!((d - c).abs() < 1e-6, "w: dev {d} vs cpu {c}");
    }
    for (d, c) in bd.iter().zip(bc.iter()) {
        assert!((d - c).abs() < 1e-6, "b: dev {d} vs cpu {c}");
    }
}

#[test]
fn device_adam_matches_cpu_within_tolerance() {
    let _serial = setup();
    let target_vals = vec![1.5, -2.0];

    let run = |dev: Option<Device>| -> Vec<f32> {
        let target = match dev {
            Some(d) => Tensor::from_vec(target_vals.clone(), &[2])
                .unwrap()
                .to_device(d)
                .unwrap(),
            None => Tensor::from_vec(target_vals.clone(), &[2]).unwrap(),
        };
        let w_host = Tensor::from_vec(vec![0.6, -0.6], &[2]).unwrap();
        let w = Param::new(match dev {
            Some(d) => w_host.to_device(d).unwrap(),
            None => w_host.clone(),
        });
        let mut opt = Adam::new(vec![w.clone()], 0.1);
        for _ in 0..50 {
            let diff = w.tensor().sub(&target).unwrap();
            let loss = diff.mul(&diff).unwrap().mean();
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
        w.tensor().to_vec()
    };

    let wd = run(Some(DEV));
    let wc = run(None);
    for (d, c) in wd.iter().zip(wc.iter()) {
        assert!((d - c).abs() < 1e-5, "w: dev {d} vs cpu {c}");
    }
}
