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
    Clamp {
        min: f32,
        max: f32,
    },
    /// Heaviside step (1.0 where x > 0, else 0.0); the relu gradient mask.
    Gtz,
    /// GELU, tanh approximation: 0.5*v*(1 + tanh(sqrt(2/pi)*(v + 0.044715 v^3))).
    Gelu,
    /// SiLU (swish): v * sigmoid(v).
    Silu,
}

/// Named elementwise binary kernels.
#[derive(Clone, Copy, Debug)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

/// The scalar math behind a `BinaryKind`, shared by every host loop here.
fn binary_scalar_fn(kind: BinaryKind) -> impl Fn(f32, f32) -> f32 {
    move |x: f32, y: f32| match kind {
        BinaryKind::Add => x + y,
        BinaryKind::Sub => x - y,
        BinaryKind::Mul => x * y,
        BinaryKind::Div => x / y,
    }
}

/// Hyperparameters of one fused Adam/AdamW step (`Backend::adamw_step`).
/// bc1/bc2 are the step's bias corrections 1 - beta^t, computed host-side.
#[derive(Clone, Copy, Debug)]
pub struct AdamWStep {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub bc1: f32,
    pub bc2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

/// Core-owned description of one step in a fused pointwise chain: the tag
/// recorded on an autograd op, resolved by the fusion planner. Mirrors the
/// backend-side chain-kernel step so core never depends on a backend crate.
/// `other` indexes the chain's trailing operand buffers (0 is the seed).
#[derive(Clone, Debug)]
pub enum ChainStepRef {
    Unary(UnaryKind),
    Binary {
        kind: BinaryKind,
        other: usize,
    },
    BinaryBc {
        kind: BinaryKind,
        other: usize,
        /// Output decomposition dims and the operand's padded strides, so one
        /// compiled kernel serves every shape of that rank.
        dims: Vec<u32>,
        strides: Vec<u32>,
    },
}

/// Which named kernel an autograd-recorded op ran, captured at record time so
/// the graph compiler can re-derive the math of a recorded node (fusion
/// planning, lazy re-execution). Only kind-routed ops carry a tag; composite
/// ops record None and stay fusion barriers.
#[derive(Clone, Copy, Debug)]
pub enum OpTag {
    Unary(UnaryKind),
    Binary(BinaryKind),
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
    Err(Error::Unsupported {
        op,
        msg: "backend does not implement device-resident storage".into(),
    })
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
    fn binary_bc(
        &self,
        kind: BinaryKind,
        a: &[f32],
        sa: &[usize],
        b: &[f32],
        sb: &[usize],
        out_shape: &[usize],
    ) -> Vec<f32> {
        binary_bc_odometer(kind, a, sa, b, sb, out_shape)
    }

    /// Row-major (batch,m,k) @ (batch,k,n) -> (batch,m,n): batch independent
    /// GEMMs. Default loops over batches calling `matmul`, so a backend that
    /// threads each `matmul` call internally pays one thread-pool spin-up
    /// per batch element; a backend should override this to parallelize
    /// across the whole batch under a single thread::scope instead.
    fn matmul_batch(
        &self,
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; batch * m * n];
        for bi in 0..batch {
            let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
            let c = self.matmul(&a[ao..ao + m * k], &b[bo..bo + k * n], m, k, n);
            out[co..co + m * n].copy_from_slice(&c);
        }
        out
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

    /// Batched (batch,m,k) @ (batch,k,n) -> (batch,m,n) over whole contiguous
    /// device buffers, one launch (strided-batched GEMM). `ta`/`tb` mark an
    /// operand as stored transposed within each batch slab.
    fn bmm_dev(
        &self,
        _a: &dyn DeviceBuffer,
        _b: &dyn DeviceBuffer,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
        _ta: bool,
        _tb: bool,
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("bmm_dev")
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

    /// Evaluate a fused pointwise chain in ONE launch: `steps` thread the
    /// seed buffer (inputs[0]) through locals, reading trailing operand
    /// buffers by index per `ChainStepRef::other`. Backends without a chain
    /// generator return the default error; callers fall back to per-op
    /// execution.
    fn chain_dev(
        &self,
        _steps: &[ChainStepRef],
        _inputs: &[&dyn DeviceBuffer],
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("chain_dev")
    }

    /// Reduce the whole buffer to a single element.
    fn reduce_dev(
        &self,
        _kind: ReduceKind,
        _x: &dyn DeviceBuffer,
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("reduce_dev")
    }

    /// Row-wise softmax over the last dim of a whole contiguous device buffer
    /// of `rows` x `cols` elements; output has the same layout.
    fn softmax_dev(
        &self,
        _x: &dyn DeviceBuffer,
        _rows: usize,
        _cols: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("softmax_dev")
    }

    /// Row-wise log_softmax over the last dim; same contract as `softmax_dev`.
    fn log_softmax_dev(
        &self,
        _x: &dyn DeviceBuffer,
        _rows: usize,
        _cols: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("log_softmax_dev")
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

    // --- in-place kernels ---------------------------------------------------
    // Mutating counterparts of the compute kernels above, used by the in-place
    // tensor ops and the optimizers. Host variants take `&mut [f32]` (core
    // holds the storage write lock) and carry loop defaults every backend
    // inherits; an override may vectorize/thread but must keep the exact
    // per-element operation order - optimizer results are asserted bitwise
    // against the unfused reference math. Device variants mutate contents
    // behind `&dyn DeviceBuffer` (device memory is not Rust-owned; the same
    // buffer may legally appear as both dst and src, so kernels must only
    // ever combine same-index elements). Their defaults FAIL FAST with no
    // side effects - no downloads, no partial writes - so a backend without
    // them keeps its exact legacy behavior (callers fall back to allocating
    // paths) and transfer-counting tests stay byte-honest. On any Err a
    // kernel must have mutated nothing: callers treat Err as "unchanged" and
    // rerun the math through the allocating fallback.

    fn fill_inplace(&self, dst: &mut [f32], value: f32) {
        dst.fill(value);
    }

    /// dst[i] = dst[i] * mul + add.
    fn affine_inplace(&self, dst: &mut [f32], mul: f32, add: f32) {
        for d in dst.iter_mut() {
            *d = *d * mul + add;
        }
    }

    /// dst[i] = dst[i] kind src[i]; same-length contract as `binary`.
    fn binary_inplace(&self, kind: BinaryKind, dst: &mut [f32], src: &[f32]) {
        let f = binary_scalar_fn(kind);
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = f(*d, s);
        }
    }

    /// dst[i] += alpha * src[i].
    fn axpy_inplace(&self, alpha: f32, dst: &mut [f32], src: &[f32]) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d += alpha * s;
        }
    }

    /// One fused SGD step with heavy-ball momentum (callers handle the
    /// momentum == 0 case via `axpy_inplace`): v = momentum*v + g, then
    /// p -= lr * (nesterov ? momentum*v + g : v). Buffers are same-length.
    fn sgd_step(&self, p: &mut [f32], v: &mut [f32], g: &[f32], lr: f32, momentum: f32, nesterov: bool) {
        for i in 0..p.len() {
            v[i] = v[i] * momentum + g[i];
            let d = if nesterov { v[i] * momentum + g[i] } else { v[i] };
            p[i] -= d * lr;
        }
    }

    /// One fused Adam/AdamW step (weight_decay == 0 is exactly Adam):
    /// m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*g*g;
    /// p -= lr * (m/bc1 / (sqrt(v/bc2) + eps) [+ weight_decay*p]).
    /// The decay term reads the pre-update parameter (decoupled decay).
    fn adamw_step(&self, p: &mut [f32], m: &mut [f32], v: &mut [f32], g: &[f32], hp: AdamWStep) {
        let (nb1, nb2) = (1.0 - hp.beta1, 1.0 - hp.beta2);
        for i in 0..p.len() {
            let gi = g[i];
            m[i] = m[i] * hp.beta1 + gi * nb1;
            v[i] = v[i] * hp.beta2 + (gi * gi) * nb2;
            let m_hat = m[i] / hp.bc1;
            let denom = (v[i] / hp.bc2).sqrt();
            let mut upd = m_hat / (denom + hp.eps);
            if hp.weight_decay != 0.0 {
                upd += hp.weight_decay * p[i];
            }
            p[i] -= upd * hp.lr;
        }
    }

    /// Overwrite a device buffer's contents from host data, preserving the
    /// buffer's address (the stable-address contract in-place ops establish).
    fn write_dev_from_host(&self, _dst: &dyn DeviceBuffer, _data: &[f32]) -> Result<()> {
        not_resident("write_dev_from_host")
    }

    fn fill_inplace_dev(&self, dst: &dyn DeviceBuffer, value: f32) -> Result<()> {
        self.write_dev_from_host(dst, &vec![value; dst.len()])
    }

    fn affine_inplace_dev(&self, _dst: &dyn DeviceBuffer, _mul: f32, _add: f32) -> Result<()> {
        not_resident("affine_inplace_dev")
    }

    /// dst and src are same-length device buffers; dst may be src.
    fn binary_inplace_dev(
        &self,
        _kind: BinaryKind,
        _dst: &dyn DeviceBuffer,
        _src: &dyn DeviceBuffer,
    ) -> Result<()> {
        not_resident("binary_inplace_dev")
    }

    fn axpy_inplace_dev(
        &self,
        _alpha: f32,
        _dst: &dyn DeviceBuffer,
        _src: &dyn DeviceBuffer,
    ) -> Result<()> {
        not_resident("axpy_inplace_dev")
    }

    /// Copy src's contents into dst (same length, same backend), preserving
    /// dst's address.
    fn copy_into_dev(&self, _dst: &dyn DeviceBuffer, _src: &dyn DeviceBuffer) -> Result<()> {
        not_resident("copy_into_dev")
    }

    /// Fresh device buffer with src's contents. Defaulted via a host round
    /// trip so every resident backend gets it (cold paths only: parameter
    /// construction); override with a device-side copy.
    fn copy_dev(&self, src: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        self.alloc_from_host(&self.copy_to_host(src)?)
    }

    fn sgd_step_dev(
        &self,
        _p: &dyn DeviceBuffer,
        _v: &dyn DeviceBuffer,
        _g: &dyn DeviceBuffer,
        _lr: f32,
        _momentum: f32,
        _nesterov: bool,
    ) -> Result<()> {
        not_resident("sgd_step_dev")
    }

    fn adamw_step_dev(
        &self,
        _p: &dyn DeviceBuffer,
        _m: &dyn DeviceBuffer,
        _v: &dyn DeviceBuffer,
        _g: &dyn DeviceBuffer,
        _hp: AdamWStep,
    ) -> Result<()> {
        not_resident("adamw_step_dev")
    }

    // --- i64 index buffers --------------------------------------------------
    // DeviceBuffer stays opaque: i64 device buffers are produced only by
    // `alloc_i64_from_host` and consumed only by `copy_i64_to_host` /
    // `gather_rows_dev`, so a backend may tag its concrete buffer type however
    // it likes without core ever inspecting bytes.

    /// Upload host i64 data (e.g. embedding indices) as a device-resident
    /// buffer. The result is an opaque DeviceBuffer carrying i64 elements.
    fn alloc_i64_from_host(&self, _data: &[i64]) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("alloc_i64_from_host")
    }

    /// Download an i64 buffer previously produced by `alloc_i64_from_host`
    /// (or a backend-internal equivalent). Passing an f32 buffer is an error.
    fn copy_i64_to_host(&self, _buf: &dyn DeviceBuffer) -> Result<Vec<i64>> {
        not_resident("copy_i64_to_host")
    }

    /// Row-gather over device-resident operands: `w` holds a contiguous
    /// row-major table of rows `dim_size` x `inner` f32 elements; `idx` holds
    /// `n` i64 indices; the output is n rows of `inner` f32 elements where
    /// row o copies w[idx[o]]. Requires the weight to be a single "outer"
    /// block (outer == 1, which covers embedding); callers needing an outer
    /// loop fall back to the host path.
    fn gather_rows_dev(
        &self,
        _w: &dyn DeviceBuffer,
        _idx: &dyn DeviceBuffer,
        _dim_size: usize,
        _inner: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        not_resident("gather_rows_dev")
    }
}

/// Reference CPU backend; pre-registered for `Device::Cpu`.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        let f = move |v: f32| match kind {
            UnaryKind::Neg => -v,
            // Not v.max(0.0): f32::max drops NaN, torch's relu propagates it.
            UnaryKind::Relu => {
                if v > 0.0 || v.is_nan() {
                    v
                } else {
                    0.0
                }
            }
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
                if v.is_nan() {
                    v
                } else {
                    v.max(min).min(max)
                }
            }
            UnaryKind::Gtz => {
                if v > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            UnaryKind::Gelu => {
                let u = 0.797_884_6 * (v + 0.044715 * v * v * v);
                0.5 * v * (1.0 + u.tanh())
            }
            UnaryKind::Silu => v / (1.0 + (-v).exp()),
        };
        x.iter().map(|&v| f(v)).collect()
    }

    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
        let f = binary_scalar_fn(kind);
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
    BACKENDS
        .read()
        .unwrap()
        .get(&device)
        .cloned()
        .ok_or_else(|| Error::Unsupported {
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
        .map(|i| {
            if i < pad || shape[i - pad] != out_shape[i] {
                0
            } else {
                own[i - pad]
            }
        })
        .collect()
}

/// Default `Backend::binary_bc`: the naive odometer over the output index
/// space (the same shape as `Tensor::gather`'s), addressing each input
/// through broadcast strides. When the last dim is contiguous (stride 1) in
/// both inputs - the common bias-add case - the inner loop is a plain slice
/// walk instead of per-element index arithmetic.
fn binary_bc_odometer(
    kind: BinaryKind,
    a: &[f32],
    sa: &[usize],
    b: &[f32],
    sb: &[usize],
    out_shape: &[usize],
) -> Vec<f32> {
    let f = binary_scalar_fn(kind);
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
