//! DDP prototype tests: single-process gradient-averaged data parallelism
//! across Cpu + Cuda(0).
//!
//! ferro-core cannot link the real CUDA backend (zero-dependency rule), so
//! Cuda(0) here carries a fake device backend: device-resident host-vec
//! buffers behind the real dispatch seams (`*_dev`, alloc/copy). That still
//! exercises everything v1 DDP owns - replication, per-replica residency,
//! the host gather/scatter round trip - while real-GPU coverage stays with
//! the ferro-cuda crate.

use std::any::Any;
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::ddp::Ddp;
use ferro_core::dispatch::{
    register_backend, Backend, BinaryKind, DeviceBuffer, ReduceKind, UnaryKind,
};
use ferro_core::optim::Sgd;
use ferro_core::params::Param;
use ferro_core::{Device, Result, Tensor};

const DEV: Device = Device::Cuda(0);

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
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        match kind {
            UnaryKind::Neg => x.iter().map(|v| -v).collect(),
            other => panic!("fake host kernel not implemented for {other:?}"),
        }
    }
    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
        let f = |x: f32, y: f32| match kind {
            BinaryKind::Add => x + y,
            BinaryKind::Sub => x - y,
            BinaryKind::Mul => x * y,
            BinaryKind::Div => x / y,
        };
        a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
    }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    out[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        out
    }

    fn alloc_from_host(&self, d: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        Ok(Box::new(FakeBuf(d.to_vec())))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        TO_HOST.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(data(buf).to_vec())
    }
    fn unary_dev(&self, kind: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        Ok(Box::new(FakeBuf(self.unary(kind, data(x)))))
    }
    fn binary_dev(
        &self,
        kind: BinaryKind,
        a: &dyn DeviceBuffer,
        b: &dyn DeviceBuffer,
    ) -> Result<Box<dyn DeviceBuffer>> {
        Ok(Box::new(FakeBuf(self.binary(kind, data(a), data(b)))))
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
        let f = |x: f32, y: f32| match kind {
            BinaryKind::Add => x + y,
            BinaryKind::Sub => x - y,
            BinaryKind::Mul => x * y,
            BinaryKind::Div => x / y,
        };
        let strides = |shape: &[usize]| {
            let mut st = vec![0usize; out_shape.len()];
            let pad = out_shape.len() - shape.len();
            let mut s = 1usize;
            for d in (0..shape.len()).rev() {
                st[pad + d] = if shape[d] == 1 { 0 } else { s };
                s *= shape[d];
            }
            st
        };
        let (sta, stb) = (strides(sa), strides(sb));
        let n: usize = out_shape.iter().product();
        let mut idx = vec![0usize; out_shape.len()];
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let (oa, ob) = idx
                .iter()
                .enumerate()
                .fold((0usize, 0usize), |(a, b), (d, &i)| {
                    (a + i * sta[d], b + i * stb[d])
                });
            out.push(f(data(a)[oa], data(b)[ob]));
            for d in (0..out_shape.len()).rev() {
                idx[d] += 1;
                if idx[d] < out_shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(Box::new(FakeBuf(out)))
    }
    fn reduce_dev(&self, kind: ReduceKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
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
}

static TO_HOST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// register_backend mutates the process-global registry, so serialize on a
// poison-tolerant lock like tests/device.rs does for its fake backend.
static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> MutexGuard<'static, ()> {
    register_backend(DEV, Arc::new(FakeDevice));
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Fixed tiny regression problem: y = x @ W + b, no randomness anywhere so
/// the two trainers below start bit-identical.
fn dataset() -> (Tensor, Tensor, Vec<f32>, Vec<f32>) {
    let x = Tensor::from_vec(
        vec![
            0.5, -1.0, 1.5, 0.25, -0.75, 2.0, 1.0, 1.0, -2.0, 0.5, 0.25, -0.25,
        ],
        &[6, 2],
    )
    .unwrap();
    let y = Tensor::from_vec(
        vec![
            1.0, 0.0, 2.0, 1.0, 0.5, -1.0, -1.5, 0.5, 0.0, 1.0, 2.0, -0.5,
        ],
        &[6, 2],
    )
    .unwrap();
    // W (2,2), b (2,) - identical starting values for both trainers.
    let w0 = vec![0.3, -0.2, 0.15, 0.4];
    let b0 = vec![0.05, -0.1];
    (x, y, w0, b0)
}

fn mse_loss(x: &Tensor, y: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let pred = x.matmul(w)?.add(b)?;
    let diff = pred.sub(y)?;
    Ok(diff.mul(&diff)?.sum())
}

#[test]
fn ddp_convergence_matches_single_device() {
    let _serial = setup();
    let (x, y, w0, b0) = dataset();

    // Reference: plain single-device training on Cpu.
    let rw = Param::new(Tensor::from_vec(w0.clone(), &[2, 2]).unwrap());
    let rb = Param::new(Tensor::from_vec(b0.clone(), &[2]).unwrap());
    let mut ref_opt = Sgd::new(vec![rw.clone(), rb.clone()], 0.05);
    for _ in 0..40 {
        let loss = mse_loss(&x, &y, &rw.tensor(), &rb.tensor()).unwrap();
        loss.backward();
        ref_opt.step();
        ref_opt.zero_grad();
    }

    // Ddp over Cpu + fake-Cuda(0), same data, same optimizer settings.
    let dw = Param::new(Tensor::from_vec(w0.clone(), &[2, 2]).unwrap());
    let db = Param::new(Tensor::from_vec(b0, &[2]).unwrap());
    let ddp = Ddp::new(
        vec![("weight".into(), dw.clone()), ("bias".into(), db.clone())],
        vec![Device::Cpu, DEV],
    )
    .unwrap();
    let mut opt = Sgd::new(vec![dw.clone(), db.clone()], 0.05);
    for _ in 0..40 {
        ddp.step(&x, &mut |xb, ps| {
            mse_loss(
                xb,
                &y.to_device(xb.device())?,
                &ps[0],
                &ps[1].to_device(xb.device())?,
            )
        })
        .unwrap();
        opt.step();
        opt.zero_grad();
    }

    for (name, a, b) in [
        ("weight", rw.tensor(), dw.tensor()),
        ("bias", rb.tensor(), db.tensor()),
    ] {
        let (va, vb) = (a.to_vec(), b.to_vec());
        for (i, (&u, &v)) in va.iter().zip(&vb).enumerate() {
            assert!(
                (u - v).abs() < 1e-5,
                "{name}[{i}] diverged: single-device {u} vs ddp {v}"
            );
        }
    }
}

#[test]
fn averaged_gradient_is_exact_mean_and_lands_on_primary() {
    let _serial = setup();
    let (x, y, w0, _) = dataset();

    let w = Param::new(Tensor::from_vec(w0.clone(), &[2, 2]).unwrap());
    let ddp = Ddp::new(vec![("weight".into(), w.clone())], vec![Device::Cpu, DEV]).unwrap();

    // Replica 0 sees the plain loss, replica 1 three times it: the exact
    // mean of the two gradients is 2x the single-replica gradient.
    let (_, grad_sets) = ddp
        .backward_replicas(&x, &mut |xb, ps| {
            let l = mse_loss(
                xb,
                &y.to_device(xb.device())?,
                &ps[0],
                &Tensor::zeros(&[2]).to_device(xb.device())?,
            )?;
            if xb.device() == DEV {
                Ok(l.mul(&Tensor::full_on(&[1], 3.0, xb.device()).unwrap())
                    .unwrap())
            } else {
                Ok(l)
            }
        })
        .unwrap();

    // Structural claim: every replica's gradient lives on that replica's own
    // device BEFORE averaging, and the canonical leaf has none yet.
    for (i, set) in grad_sets.iter().enumerate() {
        for g in set {
            assert_eq!(g.device(), ddp.replicas()[i], "grad not on its replica");
        }
    }
    assert!(w.grad().is_none());

    let avg = ddp.average(&grad_sets).unwrap();
    assert_eq!(avg[0].device(), ddp.primary());

    // Reference: 2x the loss gradient computed entirely on Cpu.
    let rw = Param::new(Tensor::from_vec(w0.clone(), &[2, 2]).unwrap());
    let l = mse_loss(&x, &y, &rw.tensor(), &Tensor::zeros(&[2]))
        .unwrap()
        .mul(&Tensor::full(&[1], 2.0))
        .unwrap();
    l.backward();
    let expect = rw.grad().unwrap().to_vec();
    let got = avg[0].to_vec();
    for (i, (&g, &e)) in got.iter().zip(&expect).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "averaged grad[{i}] {g} != exact mean {e}"
        );
    }

    // Installing the average puts it on the canonical parameter.
    ddp.step(&x, &mut |xb, ps| {
        mse_loss(
            xb,
            &y.to_device(xb.device())?,
            &ps[0],
            &Tensor::zeros(&[2]).to_device(xb.device())?,
        )
    })
    .unwrap();
    let g = w.grad().expect("step installs grads on canonical params");
    assert_eq!(g.device(), Device::Cpu);
}

#[test]
fn averaging_pulls_each_replica_through_the_host() {
    let _serial = setup();
    let (x, y, w0, b0) = dataset();
    let w = Param::new(Tensor::from_vec(w0.clone(), &[2, 2]).unwrap());
    let b = Param::new(Tensor::from_vec(b0, &[2]).unwrap());
    let ddp = Ddp::new(
        vec![("weight".into(), w), ("bias".into(), b)],
        vec![DEV, Device::Cpu],
    )
    .unwrap();
    let before = TO_HOST.load(std::sync::atomic::Ordering::SeqCst);
    ddp.step(&x, &mut |xb, ps| {
        mse_loss(
            xb,
            &y.to_device(xb.device())?,
            &ps[0],
            &ps[1].to_device(xb.device())?,
        )
    })
    .unwrap();
    let after = TO_HOST.load(std::sync::atomic::Ordering::SeqCst);
    // Two params on the cuda replica are gathered to the host for averaging
    // (plus the loss read); the v1 design is a visible host round trip.
    assert!(
        after - before >= 2,
        "expected host pulls during averaging, saw {}",
        after - before
    );
}

#[test]
fn rejects_bad_device_lists() {
    let w = Param::new(Tensor::ones(&[1]));
    assert!(Ddp::new(vec![("w".into(), w.clone())], vec![]).is_err());
    assert!(Ddp::new(vec![("w".into(), w)], vec![Device::Cpu, Device::Cpu]).is_err());
}
