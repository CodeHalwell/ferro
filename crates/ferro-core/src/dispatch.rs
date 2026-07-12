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
    /// Heaviside step (1.0 where x > 0, else 0.0); the relu gradient mask.
    Gtz,
}

/// Named elementwise binary kernels.
#[derive(Clone, Copy, Debug)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

/// Named full-tensor reductions (device kernels produce a 1-element buffer).
#[derive(Clone, Copy, Debug)]
pub enum ReduceKind {
    Sum,
    Mean,
}

/// Opaque device-resident buffer owned by a backend: contiguous f32 elements
/// living wherever the backend keeps them (GPU memory, etc.). Core never sees
/// the bytes; backends downcast via `as_any` to their concrete buffer type.
pub trait DeviceBuffer: Send + Sync {
    fn device(&self) -> Device;
    fn len(&self) -> usize;
    fn as_any(&self) -> &dyn std::any::Any;
}

fn not_resident<T>(op: &'static str) -> Result<T> {
    Err(Error::Unsupported { op, msg: "backend does not implement device-resident storage".into() })
}

/// Per-device compute kernels. The slice methods take contiguous row-major
/// f32 host buffers (the CPU path; broadcasting/materialization stay in core).
/// The `*_dev` methods operate on backend-owned `DeviceBuffer`s so chained ops
/// stay resident on the device; they have host-rejecting defaults so a
/// host-only backend is still a valid `Backend`.
pub trait Backend: Send + Sync {
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32>;
    /// a and b are same-length, already-broadcast contiguous buffers.
    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32>;
    /// Row-major (m,k) @ (k,n) -> (m,n).
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;

    /// Broadcasting binary over unmaterialized whole-buffer operands: `a`/`b`
    /// are contiguous row-major buffers of shapes `sa`/`sb`, which broadcast
    /// (numpy rules) to `out_shape`. The default walks the output index space
    /// and addresses each input through broadcast strides (stride 0 on
    /// expanded dims), so a much smaller operand (e.g. a bias row) is read
    /// repeatedly instead of being materialized to `out_shape` size first -
    /// every existing `Backend` impl keeps working with no code change.
    /// Override for a fused/threaded kernel.
    fn binary_bc(&self, kind: BinaryKind, a: &[f32], sa: &[usize], b: &[f32], sb: &[usize], out_shape: &[usize]) -> Vec<f32> {
        binary_bc_odometer(kind, a, sa, b, sb, out_shape)
    }

    fn alloc_from_host(&self, _data: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("alloc_from_host")
    }
    fn copy_to_host(&self, _buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        not_resident("copy_to_host")
    }
    fn unary_dev(&self, _kind: UnaryKind, _x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("unary_dev")
    }
    /// a and b are same-length device buffers (no broadcasting on device yet).
    fn binary_dev(
        &self,
        _kind: BinaryKind,
        _a: &dyn DeviceBuffer,
        _b: &dyn DeviceBuffer,
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("binary_dev")
    }
    /// Logical (m,k) @ (k,n) -> (m,n); `ta`/`tb` mark an operand as stored
    /// transposed (so its buffer is (k,m) / (n,k) row-major). Backward passes
    /// need these flags to avoid materializing transposes.
    fn matmul_dev(
        &self,
        _a: &dyn DeviceBuffer,
        _b: &dyn DeviceBuffer,
        _m: usize,
        _k: usize,
        _n: usize,
        _ta: bool,
        _tb: bool,
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("matmul_dev")
    }

    /// Broadcasting binary: `sa`/`sb` broadcast (numpy rules) to `out_shape`;
    /// both inputs are whole contiguous buffers of their shapes.
    fn binary_bc_dev(
        &self,
        _kind: BinaryKind,
        _a: &dyn DeviceBuffer,
        _sa: &[usize],
        _b: &dyn DeviceBuffer,
        _sb: &[usize],
        _out_shape: &[usize],
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("binary_bc_dev")
    }

    /// Reduce the whole buffer to a single element.
    fn reduce_dev(&self, _kind: ReduceKind, _x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("reduce_dev")
    }

    /// Sum over one dim of a contiguous row-major buffer of `shape`; output is
    /// the keepdim=true layout (shape with dim set to 1).
    fn sum_dim_dev(
        &self,
        _x: &dyn DeviceBuffer,
        _shape: &[usize],
        _dim: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("sum_dim_dev")
    }

    /// Constant-filled buffer. Defaulted via a host upload so every resident
    /// backend gets it for free; override with a device-side fill for speed.
    fn fill_dev(&self, value: f32, len: usize) -> Result<Box<dyn DeviceBuffer>> {
        self.alloc_from_host(&vec![value; len])
    }
}

/// Reference CPU backend; pre-registered for `Device::Cpu`.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        let f = move |v: f32| match kind {
            UnaryKind::Neg => -v,
            // Not v.max(0.0): f32::max drops NaN, torch's relu propagates it.
            UnaryKind::Relu => if v > 0.0 || v.is_nan() { v } else { 0.0 },
            UnaryKind::Exp => v.exp(),
            UnaryKind::Sigmoid => 1.0 / (1.0 + (-v).exp()),
            UnaryKind::Tanh => v.tanh(),
            UnaryKind::Sqrt => v.sqrt(),
            UnaryKind::Abs => v.abs(),
            UnaryKind::Log => v.ln(),
            UnaryKind::Powf(p) => v.powf(p),
            // max/min chain, not f32::clamp (which panics on min > max);
            // matches torch: min > max yields max everywhere. NaN passes
            // through explicitly since f32::max/min would drop it.
            UnaryKind::Clamp { min, max } => {
                if v.is_nan() { v } else { v.max(min).min(max) }
            }
            UnaryKind::Gtz => {
                if v > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
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

/// Row-major strides for `shape`, computed locally since dispatch.rs sits
/// below tensor.rs in the module graph and cannot borrow its helpers.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        stride[i] = acc;
        acc *= shape[i];
    }
    stride
}

/// `shape`'s strides for reading as if broadcast (numpy rules) to
/// `out_shape`: a padded or size-1-expanded dim reads with stride 0.
fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let own = contiguous_strides(shape);
    let pad = out_shape.len() - shape.len();
    (0..out_shape.len())
        .map(|i| if i < pad || shape[i - pad] != out_shape[i] { 0 } else { own[i - pad] })
        .collect()
}

/// Default `Backend::binary_bc`: the naive odometer over the output index
/// space (the same shape as `Tensor::gather`'s), addressing each input
/// through broadcast strides. When the last dim is contiguous (stride 1) in
/// both inputs - the common bias-add case - the inner loop is a plain slice
/// walk instead of per-element index arithmetic.
fn binary_bc_odometer(kind: BinaryKind, a: &[f32], sa: &[usize], b: &[f32], sb: &[usize], out_shape: &[usize]) -> Vec<f32> {
    let f = move |x: f32, y: f32| match kind {
        BinaryKind::Add => x + y,
        BinaryKind::Sub => x - y,
        BinaryKind::Mul => x * y,
        BinaryKind::Div => x / y,
    };
    if out_shape.is_empty() {
        return vec![f(a[0], b[0])];
    }
    let n: usize = out_shape.iter().product();
    let mut out = vec![0f32; n];
    if n == 0 {
        return out;
    }
    let sta = broadcast_strides(sa, out_shape);
    let stb = broadcast_strides(sb, out_shape);
    let ndim = out_shape.len();
    let inner = out_shape[ndim - 1];
    let outer = n / inner;
    let (ia, ib) = (sta[ndim - 1], stb[ndim - 1]);
    let mut idx = vec![0usize; ndim - 1];
    for o in 0..outer {
        let base_a: usize = idx.iter().zip(&sta).map(|(&i, &s)| i * s).sum();
        let base_b: usize = idx.iter().zip(&stb).map(|(&i, &s)| i * s).sum();
        let obase = o * inner;
        if ia == 1 && ib == 1 {
            for j in 0..inner {
                out[obase + j] = f(a[base_a + j], b[base_b + j]);
            }
        } else {
            for j in 0..inner {
                out[obase + j] = f(a[base_a + j * ia], b[base_b + j * ib]);
            }
        }
        for d in (0..ndim - 1).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}
