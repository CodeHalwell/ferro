//! Kernel dispatch: named elementwise kernels plus a per-device `Backend`
//! trait and registry (the ATen dispatch idea in miniature). Forward ops name
//! their kernel via `UnaryKind`/`BinaryKind` and route through the backend
//! registered for the input's device; a device crate implements `Backend`
//! once and covers every core forward op. The CPU matmul additionally routes
//! through a swappable function pointer so an optimized CPU kernel (e.g.
//! `ferro-fastcpu`) can be installed without a whole new backend.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use crate::device::Device;
use crate::error::{Error, Result};

/// Named elementwise unary kernels. Parametrized kinds carry their scalar
/// arguments so a backend sees the full op, not a pre-baked closure.
#[derive(Clone, Copy, Debug)]
pub enum UnaryKind {
    Neg,
    Relu,
    Exp,
    Sigmoid,
    Tanh,
    Sqrt,
    Abs,
    Log,
    Powf(f32),
    Clamp { min: f32, max: f32 },
}

/// Named elementwise binary kernels.
#[derive(Clone, Copy, Debug)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

/// Per-device compute kernels. Buffers are contiguous row-major f32 host
/// slices for now (storage is host-resident until phase 3); a backend only
/// supplies math, while broadcasting/materialization stay in core.
pub trait Backend: Send + Sync {
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32>;
    /// a and b are same-length, already-broadcast contiguous buffers.
    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32>;
    /// Row-major (m,k) @ (k,n) -> (m,n).
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;
}

/// Reference CPU backend; pre-registered for `Device::Cpu`.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        let f = move |v: f32| match kind {
            UnaryKind::Neg => -v,
            UnaryKind::Relu => v.max(0.0),
            UnaryKind::Exp => v.exp(),
            UnaryKind::Sigmoid => 1.0 / (1.0 + (-v).exp()),
            UnaryKind::Tanh => v.tanh(),
            UnaryKind::Sqrt => v.sqrt(),
            UnaryKind::Abs => v.abs(),
            UnaryKind::Log => v.ln(),
            UnaryKind::Powf(p) => v.powf(p),
            // max/min chain, not f32::clamp (which panics on min > max);
            // matches torch: min > max yields max everywhere.
            UnaryKind::Clamp { min, max } => v.max(min).min(max),
        };
        x.iter().map(|&v| f(v)).collect()
    }

    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
        let f = move |x: f32, y: f32| match kind {
            BinaryKind::Add => x + y,
            BinaryKind::Sub => x - y,
            BinaryKind::Mul => x * y,
            BinaryKind::Div => x / y,
        };
        a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        // Consult the swappable pointer so set_matmul_kernel keeps working.
        (MATMUL.read().unwrap())(a, b, m, k, n)
    }
}

static BACKENDS: LazyLock<RwLock<HashMap<Device, Arc<dyn Backend>>>> = LazyLock::new(|| {
    let mut map: HashMap<Device, Arc<dyn Backend>> = HashMap::new();
    map.insert(Device::Cpu, Arc::new(CpuBackend));
    RwLock::new(map)
});

/// Register (or replace) the backend for a device, process-wide.
pub fn register_backend(device: Device, backend: Arc<dyn Backend>) {
    BACKENDS.write().unwrap().insert(device, backend);
}

/// Look up the backend serving `device`. Cpu is always registered.
pub fn backend_for(device: Device) -> Result<Arc<dyn Backend>> {
    BACKENDS.read().unwrap().get(&device).cloned().ok_or_else(|| Error::Unsupported {
        op: "backend_for",
        msg: format!("no backend registered for device {device}"),
    })
}

/// Row-major (m,k) @ (k,n) -> (m,n).
pub type MatmulKernel = fn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;

static MATMUL: RwLock<MatmulKernel> = RwLock::new(naive_matmul);

/// Install a replacement matmul kernel (e.g. a blocked/threaded backend).
/// Applies process-wide to all CPU tensors from the next call onward.
pub fn set_matmul_kernel(kernel: MatmulKernel) {
    *MATMUL.write().unwrap() = kernel;
}

/// Reference implementation: ikj loop order with a zero-skip that helps the
/// ReLU-sparse gradients common in backward passes.
pub fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            let brow = p * n;
            let orow = i * n;
            for j in 0..n {
                out[orow + j] += aip * b[brow + j];
            }
        }
    }
    out
}
