use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::autograd::Op;
use crate::device::Device;
use crate::dispatch::{backend_for, BinaryKind, DeviceBuffer, ReduceKind, UnaryKind};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::reduce::pairwise_sum_strided;
use crate::rng::Rng;
use crate::shape::{broadcast_shapes, default_strides, numel};

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

fn fresh_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Typed element storage. Compute ops and autograd are f32-only; F64/I64
/// storage carries data (indices, targets) through views and materialization.
/// `Device` holds a backend-owned buffer (f32 elements, always contiguous) so
/// chained ops stay resident on the device between explicit transfers.
pub enum Storage {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    Device(Box<dyn DeviceBuffer>),
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::F32(v) => write!(f, "F32({} elems)", v.len()),
            Storage::F64(v) => write!(f, "F64({} elems)", v.len()),
            Storage::I64(v) => write!(f, "I64({} elems)", v.len()),
            Storage::Device(b) => write!(f, "Device({} elems on {})", b.len(), b.device()),
        }
    }
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Storage::F32(_) | Storage::Device(_) => DType::F32,
            Storage::F64(_) => DType::F64,
            Storage::I64(_) => DType::I64,
        }
    }

    pub fn as_f32(&self) -> &[f32] {
        match self {
            Storage::F32(v) => v,
            other => panic!("expected f32 host storage, got {other:?}"),
        }
    }
}

/// Shared storage plus its version counter. The counter attaches here, not to
/// `Storage` itself, so every view sharing the `Arc` sees the same count:
/// bumping through one view is visible through the base and every sibling
/// view. Starts at 0; `Tensor::version`/`Tensor::bump_version` are the only
/// readers/writers.
pub(crate) struct StorageCell {
    pub(crate) data: Storage,
    version: AtomicU64,
}

impl StorageCell {
    fn new(data: Storage) -> StorageCell {
        StorageCell { data, version: AtomicU64::new(0) }
    }
}

pub(crate) struct TensorInner {
    pub(crate) id: usize,
    pub(crate) storage: Arc<StorageCell>,
    pub(crate) shape: Vec<usize>,
    pub(crate) stride: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) device: Device,
    pub(crate) requires_grad: bool,
    /// How this tensor was produced, for reverse-mode autodiff. `None` for leaves.
    pub(crate) op: Option<Op>,
    pub(crate) grad: Mutex<Option<Tensor>>,
}

/// Dropping a graph naively recurses Tensor -> Op -> input Tensor, which
/// overflows the stack on deep op chains; unlink iteratively instead.
impl Drop for TensorInner {
    fn drop(&mut self) {
        let mut ops: Vec<Op> = self.op.take().into_iter().collect();
        while let Some(op) = ops.pop() {
            for t in op.into_inputs() {
                if let Ok(mut inner) = Arc::try_unwrap(t.0) {
                    if let Some(next) = inner.op.take() {
                        ops.push(next);
                    }
                }
            }
        }
    }
}

/// Reference-counted, autograd-aware tensor. Cloning is cheap (bumps an `Arc`)
/// and shares identity, so a value used in several ops accumulates grad once.
#[derive(Clone)]
pub struct Tensor(pub(crate) Arc<TensorInner>);

impl Tensor {
    pub(crate) fn from_parts(
        storage: Arc<StorageCell>,
        shape: Vec<usize>,
        stride: Vec<usize>,
        offset: usize,
        device: Device,
        requires_grad: bool,
        op: Option<Op>,
    ) -> Tensor {
        Tensor(Arc::new(TensorInner {
            id: fresh_id(),
            storage,
            shape,
            stride,
            offset,
            device,
            requires_grad,
            op,
            grad: Mutex::new(None),
        }))
    }

    fn contiguous_leaf(op: &'static str, len: usize, storage: Storage, shape: &[usize]) -> Result<Tensor> {
        if len != numel(shape) {
            return Err(Error::InvalidShape { op, msg: format!("{len} elements do not fit shape {shape:?}") });
        }
        Ok(Tensor::from_parts(
            Arc::new(StorageCell::new(storage)),
            shape.to_vec(),
            default_strides(shape),
            0,
            Device::Cpu,
            false,
            None,
        ))
    }

    /// Build a contiguous leaf tensor from row-major data.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        Tensor::contiguous_leaf("from_vec", data.len(), Storage::F32(data), shape)
    }

    pub fn from_vec_f64(data: Vec<f64>, shape: &[usize]) -> Result<Tensor> {
        Tensor::contiguous_leaf("from_vec_f64", data.len(), Storage::F64(data), shape)
    }

    pub fn from_vec_i64(data: Vec<i64>, shape: &[usize]) -> Result<Tensor> {
        Tensor::contiguous_leaf("from_vec_i64", data.len(), Storage::I64(data), shape)
    }

    /// I64 tensor of `[0, end)` with shape `[end]` (empty when `end <= 0`).
    pub fn arange(end: i64) -> Tensor {
        let n = end.max(0);
        Tensor::from_vec_i64((0..n).collect(), &[n as usize]).unwrap()
    }

    pub fn full(shape: &[usize], value: f32) -> Tensor {
        Tensor::from_vec(vec![value; numel(shape)], shape).unwrap()
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    pub fn scalar(value: f32) -> Tensor {
        Tensor::from_vec(vec![value], &[]).unwrap()
    }

    pub fn randn(shape: &[usize], rng: &Rng) -> Tensor {
        let data = (0..numel(shape)).map(|_| rng.normal()).collect();
        Tensor::from_vec(data, shape).unwrap()
    }

    // --- metadata ---------------------------------------------------------

    pub fn id(&self) -> usize {
        self.0.id
    }
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }
    pub fn ndim(&self) -> usize {
        self.0.shape.len()
    }
    pub fn numel(&self) -> usize {
        numel(&self.0.shape)
    }
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad
    }
    pub fn device(&self) -> Device {
        self.0.device
    }
    pub fn dtype(&self) -> DType {
        self.0.storage.data.dtype()
    }

    /// Mark a leaf as requiring gradients (like `tensor.requires_grad_(True)`).
    /// Returns a fresh leaf sharing storage; only meaningful on leaves.
    /// Panics on non-f32 tensors: autograd is f32-only.
    pub fn requires_grad_(&self, req: bool) -> Tensor {
        assert!(
            !req || self.dtype() == DType::F32,
            "requires_grad_ supports only f32 tensors (autograd is f32-only), got {}",
            self.dtype()
        );
        Tensor::from_parts(
            self.0.storage.clone(),
            self.0.shape.clone(),
            self.0.stride.clone(),
            self.0.offset,
            self.0.device,
            req,
            None,
        )
    }

    // --- materialization --------------------------------------------------

    /// Gather a (possibly strided/broadcast) view of `data` into a contiguous
    /// row-major Vec, generic over the element type so every dtype shares the
    /// one strided-odometer implementation.
    fn gather<T: Copy>(&self, data: &[T]) -> Vec<T> {
        let inner = &self.0;
        let n = self.numel();
        if self.is_contiguous() {
            return data[inner.offset..inner.offset + n].to_vec();
        }
        let ndim = inner.shape.len();
        if ndim == 0 {
            return vec![data[inner.offset]];
        }
        let mut out = Vec::with_capacity(n);
        let mut idx = vec![0usize; ndim];
        for _ in 0..n {
            let mut off = inner.offset;
            for d in 0..ndim {
                off += idx[d] * inner.stride[d];
            }
            out.push(data[off]);
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < inner.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    }

    /// Materialize as row-major f32, casting from f64/i64 storage (lossy for
    /// |i64| > 2^24). Every compute kernel reads through this, so strided
    /// views (transpose, broadcast) work transparently.
    pub fn to_vec(&self) -> Vec<f32> {
        match &self.0.storage.data {
            Storage::F32(v) => self.gather(v),
            Storage::F64(v) => self.gather(v).into_iter().map(|x| x as f32).collect(),
            Storage::I64(v) => self.gather(v).into_iter().map(|x| x as f32).collect(),
            Storage::Device(b) => {
                let host = device_to_host(self.0.device, b.as_ref());
                self.gather(&host)
            }
        }
    }

    /// Materialize as row-major f64 (exact from f32/i64 up to 2^53).
    pub fn to_vec_f64(&self) -> Vec<f64> {
        match &self.0.storage.data {
            Storage::F32(v) => self.gather(v).into_iter().map(|x| x as f64).collect(),
            Storage::F64(v) => self.gather(v),
            Storage::I64(v) => self.gather(v).into_iter().map(|x| x as f64).collect(),
            Storage::Device(_) => self.to_vec().into_iter().map(|x| x as f64).collect(),
        }
    }

    /// Materialize as row-major i64; floats truncate toward zero.
    pub fn to_vec_i64(&self) -> Vec<i64> {
        match &self.0.storage.data {
            Storage::F32(v) => self.gather(v).into_iter().map(|x| x as i64).collect(),
            Storage::F64(v) => self.gather(v).into_iter().map(|x| x as i64).collect(),
            Storage::I64(v) => self.gather(v),
            Storage::Device(_) => self.to_vec().into_iter().map(|x| x as i64).collect(),
        }
    }

    /// Cast to `dtype`, returning a detached contiguous leaf on the same
    /// device. This is the only route from F64/I64 data into float math.
    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        // Device storage is f32-only, so a same-dtype cast of a whole device
        // buffer stays resident; every other cast materializes host storage
        // and must report Cpu (a stale device tag would make a later
        // to_device a no-op on host data).
        if dtype == self.dtype() && self.device_resident_whole() {
            return self.detach_copy();
        }
        let storage = match dtype {
            DType::F32 => Storage::F32(self.to_vec()),
            DType::F64 => Storage::F64(self.to_vec_f64()),
            DType::I64 => Storage::I64(self.to_vec_i64()),
        };
        Tensor::from_parts(
            Arc::new(StorageCell::new(storage)),
            self.0.shape.clone(),
            default_strides(&self.0.shape),
            0,
            Device::Cpu,
            false,
            None,
        )
    }

    /// Scalar value of a 0-d (or single-element) tensor.
    pub fn item(&self) -> f32 {
        self.to_vec()[0]
    }

    // --- device transfer ---------------------------------------------------

    /// Move this tensor's data to `device`, returning a detached contiguous
    /// leaf there (like `to_dtype`, transfers never carry autograd history).
    /// Only f32 tensors can move off the host. Cross-device goes via the host.
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        if self.0.device == device {
            return Ok(self.clone());
        }
        if self.dtype() != DType::F32 {
            return Err(Error::DtypeMismatch {
                op: "to_device",
                expected: DType::F32,
                got: self.dtype(),
            });
        }
        // Materialize on the host first (a copy back for device sources, a
        // contiguous gather for host views), then upload if the target is not
        // the cpu.
        let host = self.to_vec();
        let shape = self.0.shape.clone();
        if device == Device::Cpu {
            return Tensor::from_vec(host, &shape);
        }
        let buf = backend_for(device)?.alloc_from_host(&host)?;
        Ok(Tensor::from_parts(
            Arc::new(StorageCell::new(Storage::Device(buf))),
            shape.clone(),
            default_strides(&shape),
            0,
            device,
            false,
            None,
        ))
    }

    /// True when storage is a backend-owned device buffer usable by the
    /// device kernel paths: those pass the raw buffer, so the view must be
    /// the whole buffer in natural order.
    fn device_resident_whole(&self) -> bool {
        matches!(&self.0.storage.data, Storage::Device(_))
            && self.0.offset == 0
            && self.is_contiguous()
    }

    /// Row-major contiguous check without allocating (size-1 dims may carry any
    /// stride since they contribute no offset).
    pub(crate) fn is_contiguous(&self) -> bool {
        let mut acc = 1usize;
        for i in (0..self.0.shape.len()).rev() {
            let dim = self.0.shape[i];
            if dim != 1 {
                if self.0.stride[i] != acc {
                    return false;
                }
                acc *= dim;
            }
        }
        true
    }

    // --- grad storage -----------------------------------------------------

    pub fn grad(&self) -> Option<Tensor> {
        self.0.grad.lock().unwrap().clone()
    }

    pub fn zero_grad(&self) {
        *self.0.grad.lock().unwrap() = None;
    }

    pub(crate) fn accumulate_grad(&self, g: Tensor) {
        assert!(
            g.shape() == self.shape(),
            "gradient shape {:?} does not match tensor shape {:?}; op backwards must \
             reduce broadcasted gradients back to the input shape",
            g.shape(),
            self.shape()
        );
        // Invariant: a tensor's grad lives on the tensor's device. Host-based
        // backwards (ops_ext closures) produce cpu grads for device tensors;
        // align here so mixed contributions accumulate instead of erroring.
        let g = if g.device() == self.0.device {
            g
        } else {
            g.to_device(self.0.device).expect("grad transfer to the tensor's device")
        };
        let mut slot = self.0.grad.lock().unwrap();
        *slot = Some(match slot.take() {
            None => g,
            Some(existing) => raw_binary_k("grad_acc", &existing, &g, BinaryKind::Add).unwrap(),
        });
    }

    // --- views (share storage) -------------------------------------------

    /// Broadcast to `shape` without copying (inserts zero strides). Detached:
    /// broadcasting's gradient is handled by reducing in backward.
    pub(crate) fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        let cur = &self.0.shape;
        if shape.len() < cur.len() {
            return Err(Error::ShapeMismatch { op: "broadcast_to", lhs: cur.clone(), rhs: shape.to_vec() });
        }
        let pad = shape.len() - cur.len();
        let mut new_stride = vec![0usize; shape.len()];
        for i in 0..shape.len() {
            if i < pad {
                new_stride[i] = 0;
            } else {
                let ci = i - pad;
                if cur[ci] == shape[i] {
                    new_stride[i] = self.0.stride[ci];
                } else if cur[ci] == 1 {
                    new_stride[i] = 0;
                } else {
                    return Err(Error::ShapeMismatch { op: "broadcast_to", lhs: cur.clone(), rhs: shape.to_vec() });
                }
            }
        }
        Ok(Tensor::from_parts(
            self.0.storage.clone(),
            shape.to_vec(),
            new_stride,
            self.0.offset,
            self.0.device,
            false,
            None,
        ))
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        if numel(shape) != self.numel() {
            return Err(Error::InvalidShape {
                op: "reshape",
                msg: format!("cannot reshape {:?} into {shape:?}", self.0.shape),
            });
        }
        // reshape needs contiguous data; materialize if this is a strided view
        // (through the host, preserving dtype, then back to the source device
        // so the result's device tag always matches its storage; non-F32
        // views are always host tensors, so their to_device is a no-op).
        let base = if self.is_contiguous() {
            self.clone()
        } else {
            let host = match self.dtype() {
                DType::F32 => Tensor::from_vec(self.to_vec(), &self.0.shape)?,
                DType::F64 => Tensor::from_vec_f64(self.to_vec_f64(), &self.0.shape)?,
                DType::I64 => Tensor::from_vec_i64(self.to_vec_i64(), &self.0.shape)?,
            };
            host.to_device(self.0.device)?
        };
        let out = Tensor::from_parts(
            base.0.storage.clone(),
            shape.to_vec(),
            default_strides(shape),
            base.0.offset,
            base.0.device,
            false,
            None,
        );
        let in_shape = self.0.shape.clone();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![Tensor::from_vec(g.to_vec(), &in_shape).unwrap()]
        }))
    }

    /// Detached transpose view (swaps two dims' shape/stride, shares storage).
    pub(crate) fn transpose_view(&self, d0: usize, d1: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if d0 >= ndim || d1 >= ndim {
            return Err(Error::InvalidShape {
                op: "transpose",
                msg: format!("dims ({d0},{d1}) out of range for rank {ndim}"),
            });
        }
        let mut shape = self.0.shape.clone();
        let mut stride = self.0.stride.clone();
        shape.swap(d0, d1);
        stride.swap(d0, d1);
        Ok(Tensor::from_parts(self.0.storage.clone(), shape, stride, self.0.offset, self.0.device, false, None))
    }

    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Tensor> {
        let out = self.transpose_view(d0, d1)?;
        Ok(out.record_fn(vec![self.clone()], move |g| vec![g.transpose_view(d0, d1).unwrap()]))
    }

    /// A detached, contiguous copy that shares no autograd history or storage.
    pub fn detach_copy(&self) -> Tensor {
        if self.device_resident_whole() {
            // Device buffers are immutable and whole; share instead of copying
            // through the host (which would also move the tensor to cpu).
            return Tensor::from_parts(
                self.0.storage.clone(),
                self.0.shape.clone(),
                self.0.stride.clone(),
                0,
                self.0.device,
                false,
                None,
            );
        }
        Tensor::from_vec(self.to_vec(), &self.0.shape).unwrap()
    }

    /// Constant tensor on the given device (cpu allocation or backend fill).
    pub fn full_on(shape: &[usize], value: f32, device: Device) -> Result<Tensor> {
        if device == Device::Cpu {
            return Ok(Tensor::full(shape, value));
        }
        let buf = backend_for(device)?.fill_dev(value, numel(shape))?;
        Ok(device_leaf(buf, shape, device))
    }

    /// The single autograd recording hook: `inputs` are the differentiable
    /// operands; `backward` maps the output gradient to one gradient per input
    /// (same order; the engine asserts arity and shapes). Recorded only when
    /// some input requires grad. `self` must be a freshly-created, uniquely-
    /// owned output (as returned by the raw kernels).
    pub fn record_fn<F>(mut self, inputs: Vec<Tensor>, backward: F) -> Tensor
    where
        F: Fn(&Tensor) -> Vec<Tensor> + Send + Sync + 'static,
    {
        if inputs.iter().any(|t| t.requires_grad()) {
            let inner = Arc::get_mut(&mut self.0).expect("fresh output is uniquely owned");
            inner.requires_grad = true;
            inner.op = Some(Op::new(inputs, Box::new(backward)));
        }
        self
    }

    // --- version counters ---------------------------------------------------
    // Every shared storage carries a version, bumped when it is mutated. Views
    // created via from_parts clone the same Arc<StorageCell>, so they and their
    // base always report the same version; a fresh Vec (from_vec, detach_copy,
    // to_dtype, to_device) gets a fresh StorageCell starting at 0. No in-place
    // op exists yet to call bump_version outside tests - it is the seam the
    // first one will use.

    pub(crate) fn version(&self) -> u64 {
        self.0.storage.version.load(Ordering::Relaxed)
    }

    pub(crate) fn bump_version(&self) {
        self.0.storage.version.fetch_add(1, Ordering::Relaxed);
    }

    /// Test-only seam to simulate an in-place mutation before real in-place ops
    /// exist, so the version-mismatch assertion in `backward` is exercisable.
    #[doc(hidden)]
    pub fn _bump_version_for_test(&self) {
        self.bump_version();
    }

    /// Test-only accessor for the storage version counter.
    #[doc(hidden)]
    pub fn _version(&self) -> u64 {
        self.version()
    }
}

/// Copy a device buffer back to host. Infallible by construction: a device
/// tensor can only exist if its backend was registered at creation time.
fn device_to_host(device: Device, buf: &dyn DeviceBuffer) -> Vec<f32> {
    backend_for(device)
        .expect("device tensor exists, so its backend must be registered")
        .copy_to_host(buf)
        .expect("device-to-host copy failed")
}

// --- raw (detached) compute kernels --------------------------------------
// These never record autograd; forward wrappers and backward both call them.
// Two flavors:
// - `raw_unary_k`/`raw_binary_k` take a named kind and route the math through
//   the backend registered for the input's device. Forward ops use these.
// - `raw_binary` takes an inline closure that runs on the host: the CPU-only
//   escape hatch used by ops_ext backward closures and composite forwards
//   without a named kernel. Core op backwards are fully kind-routed, so they
//   run on whatever device the gradients live on.
//
// Device residency: whole-buffer device tensors run unary/binary (equal
// shapes or broadcast)/matmul on their backend and stay resident. For those
// three kinds, device VIEWS (transposes, strided slices) fall back to host
// math on the cpu backend's kernels and the result is uploaded back to the
// source device: forward outputs must stay on the operand device or backward
// would mix cpu gradients with saved device operands. Reductions and composite
// ops without a named device kernel (`raw_sum_dim`, `raw_binary`, ops_ext
// forwards) still return cpu tensors for non-whole operands - a visible
// (result.device() == Cpu), documented fallback.

/// Float math is f32-only: F64/I64 operands must be cast explicitly via
/// `to_dtype(DType::F32)` rather than silently at kernel entry.
fn check_f32(op: &'static str, t: &Tensor) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::DtypeMismatch { op, expected: DType::F32, got: t.dtype() });
    }
    Ok(())
}

pub(crate) fn raw_unary_k(a: &Tensor, kind: UnaryKind) -> Result<Tensor> {
    check_f32("unary", a)?;
    if a.device_resident_whole() {
        let backend = backend_for(a.0.device)?;
        let Storage::Device(buf) = &a.0.storage.data else { unreachable!() };
        let out = backend.unary_dev(kind, buf.as_ref())?;
        return Ok(device_leaf(out, &a.0.shape, a.0.device));
    }
    // Contiguous host f32 (any offset): slice straight into the backend, no
    // materializing copy - the read side of the ~3x-bytes-moved bug this
    // function used to have via `a.to_vec()`.
    if let Storage::F32(v) = &a.0.storage.data {
        if a.0.device == Device::Cpu && a.is_contiguous() {
            let n = a.numel();
            let cpu = backend_for(Device::Cpu)?;
            return Tensor::from_vec(cpu.unary(kind, &v[a.0.offset..a.0.offset + n]), &a.0.shape);
        }
    }
    let cpu = backend_for(Device::Cpu)?;
    Tensor::from_vec(cpu.unary(kind, &a.to_vec()), &a.0.shape)?.to_device(a.0.device)
}

/// Wrap a backend-produced buffer as a contiguous detached device tensor.
fn device_leaf(buf: Box<dyn DeviceBuffer>, shape: &[usize], device: Device) -> Tensor {
    Tensor::from_parts(
        Arc::new(StorageCell::new(Storage::Device(buf))),
        shape.to_vec(),
        default_strides(shape),
        0,
        device,
        false,
        None,
    )
}

pub(crate) fn raw_binary_k(op: &'static str, a: &Tensor, b: &Tensor, kind: BinaryKind) -> Result<Tensor> {
    check_f32(op, a)?;
    check_f32(op, b)?;
    if a.0.device != b.0.device {
        return Err(Error::DeviceMismatch { op, lhs: a.0.device, rhs: b.0.device });
    }
    if a.device_resident_whole() && b.device_resident_whole() {
        let backend = backend_for(a.0.device)?;
        let Storage::Device(ba) = &a.0.storage.data else { unreachable!() };
        let Storage::Device(bb) = &b.0.storage.data else { unreachable!() };
        if a.0.shape == b.0.shape {
            let out = backend.binary_dev(kind, ba.as_ref(), bb.as_ref())?;
            return Ok(device_leaf(out, &a.0.shape, a.0.device));
        }
        let out_shape = broadcast_shapes(op, &a.0.shape, &b.0.shape)?;
        let out =
            backend.binary_bc_dev(kind, ba.as_ref(), &a.0.shape, bb.as_ref(), &b.0.shape, &out_shape)?;
        return Ok(device_leaf(out, &out_shape, a.0.device));
    }
    // Contiguous host f32 operands: skip the copy-then-compute path.
    if let (Storage::F32(va), Storage::F32(vb)) = (&a.0.storage.data, &b.0.storage.data) {
        if a.0.device == Device::Cpu && a.is_contiguous() && b.is_contiguous() {
            let cpu = backend_for(Device::Cpu)?;
            // Equal shapes: slice both operands directly (any offset).
            if a.0.shape == b.0.shape {
                let n = a.numel();
                let sa = &va[a.0.offset..a.0.offset + n];
                let sb = &vb[b.0.offset..b.0.offset + n];
                return Tensor::from_vec(cpu.binary(kind, sa, sb), &a.0.shape);
            }
            // Different shapes: hand the backend the whole unmaterialized
            // buffers (e.g. a [4096] bias against a [4096,4096] output) so a
            // smaller operand is read repeatedly instead of expanded first.
            // Restricted to zero-offset whole buffers - an offset view still
            // takes the materializing fallback below.
            if a.0.offset == 0 && b.0.offset == 0 && va.len() == a.numel() && vb.len() == b.numel() {
                let out_shape = broadcast_shapes(op, &a.0.shape, &b.0.shape)?;
                let out = cpu.binary_bc(kind, va, &a.0.shape, vb, &b.0.shape, &out_shape);
                return Tensor::from_vec(out, &out_shape);
            }
        }
    }
    let out_shape = broadcast_shapes(op, &a.0.shape, &b.0.shape)?;
    let va = a.broadcast_to(&out_shape)?.to_vec();
    let vb = b.broadcast_to(&out_shape)?.to_vec();
    let cpu = backend_for(Device::Cpu)?;
    Tensor::from_vec(cpu.binary(kind, &va, &vb), &out_shape)?.to_device(a.0.device)
}

pub(crate) fn raw_binary(
    op: &'static str,
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f32, f32) -> f32,
) -> Result<Tensor> {
    check_f32(op, a)?;
    check_f32(op, b)?;
    if a.0.device != b.0.device {
        return Err(Error::DeviceMismatch { op, lhs: a.0.device, rhs: b.0.device });
    }
    let out_shape = broadcast_shapes(op, &a.0.shape, &b.0.shape)?;
    let va = a.broadcast_to(&out_shape)?.to_vec();
    let vb = b.broadcast_to(&out_shape)?.to_vec();
    let data = va.iter().zip(vb.iter()).map(|(&x, &y)| f(x, y)).collect();
    Tensor::from_vec(data, &out_shape)
}

/// 2-D matmul: (m,k) @ (k,n) -> (m,n), routed through the device's backend
/// (the CPU backend consults the swappable kernel pointer, so a backend crate
/// can still swap in a faster kernel). Higher ranks are a follow-up.
pub(crate) fn raw_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_f32("matmul", a)?;
    check_f32("matmul", b)?;
    if a.0.device != b.0.device {
        return Err(Error::DeviceMismatch { op: "matmul", lhs: a.0.device, rhs: b.0.device });
    }
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(Error::Unsupported {
            op: "matmul",
            msg: format!("only 2-D supported in MVP, got {:?} and {:?}", a.0.shape, b.0.shape),
        });
    }
    let (m, k) = (a.0.shape[0], a.0.shape[1]);
    let (k2, n) = (b.0.shape[0], b.0.shape[1]);
    if k != k2 {
        return Err(Error::ShapeMismatch { op: "matmul", lhs: a.0.shape.clone(), rhs: b.0.shape.clone() });
    }
    if a.device_resident_whole() && b.device_resident_whole() {
        let backend = backend_for(a.0.device)?;
        let Storage::Device(ba) = &a.0.storage.data else { unreachable!() };
        let Storage::Device(bb) = &b.0.storage.data else { unreachable!() };
        let out = backend.matmul_dev(ba.as_ref(), bb.as_ref(), m, k, n, false, false)?;
        return Ok(device_leaf(out, &[m, n], a.0.device));
    }
    let va = a.to_vec();
    let vb = b.to_vec();
    let cpu = backend_for(Device::Cpu)?;
    Tensor::from_vec(cpu.matmul(&va, &vb, m, k, n), &[m, n])?.to_device(a.0.device)
}

/// Full-tensor reduction on a resident device tensor, if it is one.
pub(crate) fn raw_reduce_dev(t: &Tensor, kind: ReduceKind) -> Option<Tensor> {
    if !t.device_resident_whole() {
        return None;
    }
    let backend = backend_for(t.0.device).expect("device tensor implies registered backend");
    let Storage::Device(buf) = &t.0.storage.data else { unreachable!() };
    let out = backend.reduce_dev(kind, buf.as_ref()).expect("device backend lacks reduce_dev");
    Some(device_leaf(out, &[], t.0.device))
}

/// Matmul with logical transpose flags: computes op(a) @ op(b) where op is
/// transpose when the flag is set. Device operands stay resident (backends
/// take the flags natively, e.g. cuBLAS); host operands go through transpose
/// views. Used by matmul's backward so gradients never materialize transposes.
pub(crate) fn raw_matmul_t(a: &Tensor, b: &Tensor, ta: bool, tb: bool) -> Result<Tensor> {
    if a.device_resident_whole() && b.device_resident_whole() && a.ndim() == 2 && b.ndim() == 2 {
        let (m, k) = if ta { (a.0.shape[1], a.0.shape[0]) } else { (a.0.shape[0], a.0.shape[1]) };
        let (k2, n) = if tb { (b.0.shape[1], b.0.shape[0]) } else { (b.0.shape[0], b.0.shape[1]) };
        if k != k2 {
            return Err(Error::ShapeMismatch { op: "matmul", lhs: a.0.shape.clone(), rhs: b.0.shape.clone() });
        }
        let backend = backend_for(a.0.device)?;
        let Storage::Device(ba) = &a.0.storage.data else { unreachable!() };
        let Storage::Device(bb) = &b.0.storage.data else { unreachable!() };
        let out = backend.matmul_dev(ba.as_ref(), bb.as_ref(), m, k, n, ta, tb)?;
        return Ok(device_leaf(out, &[m, n], a.0.device));
    }
    let av = if ta { a.transpose_view(0, 1)? } else { a.clone() };
    let bv = if tb { b.transpose_view(0, 1)? } else { b.clone() };
    raw_matmul(&av, &bv)
}

/// Sum over one dim, matching PyTorch's keepdim semantics.
pub(crate) fn raw_sum_dim(t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    let in_shape = t.0.shape.clone();
    let ndim = in_shape.len();
    if t.device_resident_whole() {
        let backend = backend_for(t.0.device).expect("device tensor implies registered backend");
        let Storage::Device(buf) = &t.0.storage.data else { unreachable!() };
        let out = backend
            .sum_dim_dev(buf.as_ref(), &in_shape, dim)
            .expect("device backend lacks sum_dim_dev");
        let mut keep_shape = in_shape.clone();
        keep_shape[dim] = 1;
        let out_shape: Vec<usize> = if keepdim {
            keep_shape
        } else {
            in_shape.iter().enumerate().filter(|(d, _)| *d != dim).map(|(_, &s)| s).collect()
        };
        return device_leaf(out, &out_shape, t.0.device);
    }
    let v = t.to_vec();
    let strides = default_strides(&in_shape);
    let (n, stride) = (in_shape[dim], strides[dim]);
    let mut keep_shape = in_shape.clone();
    keep_shape[dim] = 1;
    let mut out = vec![0f32; numel(&keep_shape)];
    let mut idx = vec![0usize; ndim];
    for slot in out.iter_mut() {
        let off: usize = (0..ndim).map(|d| idx[d] * strides[d]).sum();
        *slot = pairwise_sum_strided(&v, off, n, stride);
        for d in (0..ndim).rev() {
            if d == dim {
                continue;
            }
            idx[d] += 1;
            if idx[d] < in_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    let out_shape: Vec<usize> = if keepdim {
        keep_shape
    } else {
        in_shape.iter().enumerate().filter(|(d, _)| *d != dim).map(|(_, &s)| s).collect()
    };
    Tensor::from_vec(out, &out_shape).unwrap()
}

/// Reduce a (broadcasted) gradient back down to `target` shape by summing over
/// the dims that were expanded during the forward broadcast.
pub(crate) fn unbroadcast(g: &Tensor, target: &[usize]) -> Tensor {
    let mut g = g.clone();
    while g.ndim() > target.len() {
        g = raw_sum_dim(&g, 0, false);
    }
    for d in 0..target.len() {
        if target[d] == 1 && g.0.shape[d] != 1 {
            g = raw_sum_dim(&g, d, true);
        }
    }
    g
}

#[cfg(test)]
mod tests {
    // raw_unary_k/raw_binary_k's fast paths slice storage at `tensor.offset`,
    // but no public op (narrow/as_strided; see docs/CAPABILITY.md 2.2) yet
    // produces a nonzero-offset tensor, so this exercises the offset
    // arithmetic directly via the pub(crate) constructor an external
    // integration test cannot reach.
    use super::*;

    fn offset_view(v: Vec<f32>, offset: usize, shape: &[usize]) -> Tensor {
        Tensor::from_parts(Arc::new(StorageCell::new(Storage::F32(v))), shape.to_vec(), default_strides(shape), offset, Device::Cpu, false, None)
    }

    #[test]
    fn raw_unary_k_reads_from_the_view_offset_not_index_zero() {
        let a = offset_view(vec![9.0, -1.0, -2.0, 3.0, 9.0], 1, &[3]);
        assert_eq!(raw_unary_k(&a, UnaryKind::Relu).unwrap().to_vec(), vec![0.0, 0.0, 3.0]);
    }

    #[test]
    fn raw_binary_k_reads_both_operands_from_their_view_offset() {
        let a = offset_view(vec![100.0, 1.0, 2.0, 3.0], 1, &[3]);
        let b = offset_view(vec![100.0, 10.0, 20.0, 30.0], 1, &[3]);
        assert_eq!(raw_binary_k("add", &a, &b, BinaryKind::Add).unwrap().to_vec(), vec![11.0, 22.0, 33.0]);
    }
}
