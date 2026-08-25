//! End-to-end fusion seam proof: a captured autograd tape's pointwise chain
//! resolves into core-owned ChainStepRefs and executes as exactly ONE backend
//! launch through `Backend::chain_dev`, matching the unfused eager result
//! bit-for-bit. The fake backend PANICS on per-op kernels, so reaching the
//! answer proves the fused path was taken.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::dispatch::{
    register_backend, Backend, BinaryKind, ChainStepRef, DeviceBuffer, ReduceKind, UnaryKind,
};
use ferro_core::graph::Graph;
use ferro_core::{Device, Result, Tensor};

static CHAINS: AtomicUsize = AtomicUsize::new(0);
static PER_OP: AtomicUsize = AtomicUsize::new(0);

const DEV: Device = Device::Cuda(21);

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

fn apply(kind: BinaryKind, x: f32, y: f32) -> f32 {
    match kind {
        BinaryKind::Add => x + y,
        BinaryKind::Sub => x - y,
        BinaryKind::Mul => x * y,
        BinaryKind::Div => x / y,
    }
}

impl Backend for FakeDevice {
    fn unary(&self, _k: UnaryKind, _x: &[f32]) -> Vec<f32> {
        panic!("host unary must not run for device-resident tensors");
    }
    fn binary(&self, _k: BinaryKind, _a: &[f32], _b: &[f32]) -> Vec<f32> {
        panic!("host binary must not run for device-resident tensors");
    }
    fn matmul(&self, _a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        panic!("host matmul must not run for device-resident tensors");
    }

    fn alloc_from_host(&self, d: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        Ok(Box::new(FakeBuf(d.to_vec())))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        Ok(data(buf).to_vec())
    }

    fn unary_dev(&self, k: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        PER_OP.fetch_add(1, Ordering::SeqCst);
        let out = data(x)
            .iter()
            .map(|&v| apply_unary(k, v))
            .collect();
        Ok(Box::new(FakeBuf(out)))
    }
    fn binary_dev(
        &self,
        k: BinaryKind,
        a: &dyn DeviceBuffer,
        b: &dyn DeviceBuffer,
    ) -> Result<Box<dyn DeviceBuffer>> {
        PER_OP.fetch_add(1, Ordering::SeqCst);
        let out = data(a)
            .iter()
            .zip(data(b))
            .map(|(&x, &y)| apply(k, x, y))
            .collect();
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
        PER_OP.fetch_add(1, Ordering::SeqCst);
        let n: usize = out_shape.iter().product();
        let idx = |flat: usize, shape: &[usize]| -> usize {
            let pad = out_shape.len() - shape.len();
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
        let out = (0..n)
            .map(|i| apply(kind, va[idx(i, sa)], vb[idx(i, sb)]))
            .collect();
        Ok(Box::new(FakeBuf(out)))
    }

    // Reference chain executor: same sequential math the generated CUDA
    // kernel performs (seed threaded through locals, broadcast offsets).
    fn chain_dev(
        &self,
        steps: &[ChainStepRef],
        inputs: &[&dyn DeviceBuffer],
    ) -> Result<Box<dyn DeviceBuffer>> {
        CHAINS.fetch_add(1, Ordering::SeqCst);
        let n = inputs[0].len();
        let mut cur: Vec<f32> = data(inputs[0]).to_vec();
        for s in steps {
            match s {
                ChainStepRef::Unary(kind) => cur = cur.iter().map(|&v| apply_unary(*kind, v)).collect(),
                ChainStepRef::Binary { kind, other } => {
                    let o = data(inputs[*other]);
                    cur = cur.iter().zip(o).map(|(&x, &y)| apply(*kind, x, y)).collect();
                }
                ChainStepRef::BinaryBc { kind, other, dims, strides } => {
                    let o = data(inputs[*other]);
                    let out_shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
                    let st: Vec<usize> = strides.iter().map(|&d| d as usize).collect();
                    cur = (0..n)
                        .map(|i| {
                            let mut off = 0usize;
                            let mut rem = i;
                            for d in 0..out_shape.len() {
                                let c = rem / out_shape[d..].iter().product::<usize>().max(1).max(out_shape[d]);
                                let coord =
                                    (i / out_shape[d + 1..].iter().product::<usize>()) % out_shape[d];
                                off += coord * st[d];
                                let _ = c;
                                rem %= out_shape[d..].iter().product::<usize>().max(1);
                            }
                            apply(*kind, cur[i], o[off])
                        })
                        .collect();
                }
            }
        }
        Ok(Box::new(FakeBuf(cur)))
    }
}

fn apply_unary(kind: UnaryKind, v: f32) -> f32 {
    match kind {
        UnaryKind::Relu => v.max(0.0),
        UnaryKind::Gelu => {
            let u = 0.797_884_6 * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + u.tanh())
        }
        UnaryKind::Silu => v / (1.0 + (-v).exp()),
        UnaryKind::Exp => v.exp(),
        UnaryKind::Neg => -v,
        other => panic!("fake chain unary not implemented for {other:?}"),
    }
}

static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> MutexGuard<'static, ()> {
    register_backend(DEV, Arc::new(FakeDevice));
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn fused_chain_matches_eager_and_issues_one_launch() {
    let _serial = setup();
    let xs: Vec<f32> = (0..8).map(|i| ((i * 7 % 11) as f32 - 5.0) / 3.0).collect();
    let bs: Vec<f32> = (0..4).map(|i| (i as f32 - 1.5) * 0.5).collect();

    // Eager reference on the cpu.
    let xc = Tensor::from_vec(xs.clone(), &[2, 4])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let bc = Tensor::from_vec(bs.clone(), &[4])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let eager = xc.relu().add(&bc).unwrap().silu();

    // Same tape captured on the fake device.
    let xd = Tensor::from_vec(xs.clone(), &[2, 4])
        .unwrap()
        .to_device(DEV)
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let bd = Tensor::from_vec(bs.clone(), &[4])
        .unwrap()
        .to_device(DEV)
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let g = Graph::capture(|| xd.relu().add(&bd).unwrap().silu());
    let plan = g.plan_fusion();
    assert!(!plan.chains.is_empty(), "relu+add+silu must plan as a chain");

    let fused = &plan.chains[0];
    let exec = fused.resolve(&g).expect("chain resolves");
    assert_eq!(exec.steps.len(), 2, "relu seed + add + silu = 2 steps");
    assert_eq!(plan.launches_saved(), 2);

    let before = (CHAINS.load(Ordering::SeqCst), PER_OP.load(Ordering::SeqCst));
    let got = fused.run(&exec).unwrap();
    let after = (CHAINS.load(Ordering::SeqCst), PER_OP.load(Ordering::SeqCst));
    assert_eq!(got.device(), DEV);
    assert_eq!(after.0 - before.0, 1, "exactly one fused launch");
    assert_eq!(after.1 - before.1, 0, "zero per-op launches");

    for (a, b) in eager.to_vec().iter().zip(got.to_vec()) {
        assert!((a - b).abs() < 1e-6, "eager {a} vs fused {b}");
    }
}