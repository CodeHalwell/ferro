//! Replay executor proof: the same tape replayed from leaves produces values
//! identical to eager, but issues one fused launch per chain instead of one
//! launch per op. The fake backend counts chain_dev calls vs per-op calls.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::dispatch::{
    register_backend, Backend, BinaryKind, ChainStepRef, DeviceBuffer, UnaryKind,
};
use ferro_core::replay::Replay;
use ferro_core::{Device, Result, Tensor};

static CHAINS: AtomicUsize = AtomicUsize::new(0);
static PER_OP: AtomicUsize = AtomicUsize::new(0);

const DEV: Device = Device::Cuda(31);

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

fn apply_unary(kind: UnaryKind, v: f32) -> f32 {
    match kind {
        UnaryKind::Relu => v.max(0.0),
        UnaryKind::Exp => v.exp(),
        UnaryKind::Neg => -v,
        other => panic!("fake unary not implemented for {other:?}"),
    }
}

impl Backend for FakeDevice {
    fn unary(&self, _k: UnaryKind, _x: &[f32]) -> Vec<f32> {
        panic!("host unary must not run");
    }
    fn binary(&self, _k: BinaryKind, _a: &[f32], _b: &[f32]) -> Vec<f32> {
        panic!("host binary must not run");
    }
    fn matmul(&self, _a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        panic!("host matmul must not run");
    }

    fn alloc_from_host(&self, d: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        Ok(Box::new(FakeBuf(d.to_vec())))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        Ok(data(buf).to_vec())
    }
    fn unary_dev(&self, k: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        PER_OP.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FakeBuf(data(x).iter().map(|&v| apply_unary(k, v)).collect())))
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
                ChainStepRef::Unary(kind) => {
                    cur = cur.iter().map(|&v| apply_unary(*kind, v)).collect()
                }
                ChainStepRef::Binary { kind, other } => {
                    let o = data(inputs[*other]);
                    cur = cur.iter().zip(o).map(|(&x, &y)| apply(*kind, x, y)).collect();
                }
                ChainStepRef::BinaryBc { .. } => panic!("bc steps covered in fusion_exec"),
            }
        }
        Ok(Box::new(FakeBuf(cur)))
    }
}

static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> MutexGuard<'static, ()> {
    register_backend(DEV, Arc::new(FakeDevice));
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn replay_matches_eager_with_fewer_launches() {
    let _serial = setup();
    let xs: Vec<f32> = (0..16).map(|i| ((i * 7 % 11) as f32 - 5.0) / 3.0).collect();

    // Capture: runs eagerly (per-op launches happen HERE, during capture).
    let xd0 = Tensor::from_vec(xs.clone(), &[4, 4]).unwrap().to_device(DEV).unwrap();
    let r = Replay::capture(|| {
        let a = xd0.clone().requires_grad_(true).unwrap();
        a.relu().exp().relu().neg().relu()
    });
    let (before_launches, after_launches) = r.plan_launches();
    assert!(
        after_launches < before_launches,
        "plan must save launches: {before_launches} -> {after_launches}"
    );

    // Leaves: walk order puts the single leaf first.
    assert_eq!(r.leaves.len(), 1);

    let before = (CHAINS.load(Ordering::SeqCst), PER_OP.load(Ordering::SeqCst));
    let got = r.replay(&[xd0]).expect("replay succeeds");
    let after = (CHAINS.load(Ordering::SeqCst), PER_OP.load(Ordering::SeqCst));

    // relu->exp->relu->neg->relu is ONE 5-node chain: exactly one chain_dev
    // call and ZERO per-op launches during replay.
    assert_eq!(after.0 - before.0, 1, "one fused launch for the whole chain");
    assert_eq!(after.1 - before.1, 0, "zero per-op launches during replay");

    // Numerically identical to the eager capture-time result.
    let cpu = Tensor::from_vec(xs.clone(), &[4, 4])
        .unwrap()
        .relu()
        .exp()
        .relu()
        .neg()
        .relu();
    for (a, b) in cpu.to_vec().iter().zip(got.to_vec()) {
        assert!((a - b).abs() < 1e-6, "eager {a} vs replayed {b}");
    }
}
