use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::autograd::Op;
use crate::device::Device;
use crate::dispatch::{backend_for, BinaryKind, DeviceBuffer, ReduceKind, UnaryKind};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::shape::{broadcast_shapes, checked_numel, default_strides, numel};

#[path = "grad_publication.rs"]
mod grad_publication;

#[cfg(test)]
#[path = "empty_cotangent_tests.rs"]
mod empty_cotangent_tests;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

fn fresh_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Allocation-free node watermark for functional-transform bookkeeping.
#[doc(hidden)]
pub fn tensor_id_watermark() -> usize { NEXT_ID.load(Ordering::Relaxed) }

/// Typed storage. F32 uses device backends; CPU F64 supports native basic math
/// and autograd, while I64/Bool preserve typed arithmetic, masks and indices.
/// `Device` holds a backend-owned buffer (f32 elements, always contiguous) so
/// chained ops stay resident on the device between explicit transfers.
pub enum Storage {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    /// IEEE binary16, raw bit patterns (see the `half` module). Compute is
    /// f32-only, so these enter math via explicit `to_dtype(DType::F32)`;
    /// they exist to carry checkpoint weights at half the memory.
    F16(Vec<u16>),
    /// bfloat16, raw bit patterns (f32's top half).
    BF16(Vec<u16>),
    Device(Arc<dyn DeviceBuffer>),
    /// Device-resident i64 indices. Kept as its own variant so the f32
    /// compute paths can never pick one up by mistake; produced/consumed
    /// only through `Backend::alloc_i64_from_host`/`copy_i64_to_host`/
    /// `gather_rows_dev`.
    DeviceI64(Box<dyn DeviceBuffer>),
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::F32(v) => write!(f, "F32({} elems)", v.len()),
            Storage::F64(v) => write!(f, "F64({} elems)", v.len()),
            Storage::I64(v) => write!(f, "I64({} elems)", v.len()),
            Storage::Bool(v) => write!(f, "Bool({} elems)", v.len()),
            Storage::F16(v) => write!(f, "F16({} elems)", v.len()),
            Storage::BF16(v) => write!(f, "BF16({} elems)", v.len()),
            Storage::Device(b) => write!(f, "Device({} elems on {})", b.len(), b.device()),
            Storage::DeviceI64(b) => write!(f, "DeviceI64({} elems on {})", b.len(), b.device()),
        }
    }
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Storage::F32(_) | Storage::Device(_) => DType::F32,
            Storage::F64(_) => DType::F64,
            Storage::I64(_) | Storage::DeviceI64(_) => DType::I64,
            Storage::Bool(_) => DType::Bool,
            Storage::F16(_) => DType::F16,
            Storage::BF16(_) => DType::BF16,
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
///
/// The lock guards element CONTENTS only. No code path ever changes a cell's
/// Storage variant, a host Vec's length or allocation, or a device buffer's
/// identity after construction: in-place ops overwrite values through
/// `write()` (host) or through the backend's `*_inplace_dev` kernels under
/// `read()` (device memory is not Rust-owned, so no `&mut` is needed and
/// concurrent device-side access follows the backend's stream ordering).
/// Raw pointers exported while unlocked (DLPack views) therefore stay valid
/// across mutation - values change, addresses never do.
pub(crate) struct StorageCell {
    data: RwLock<Storage>,
    version: AtomicU64,
    structural_zero: bool,
    publication: Mutex<()>,
}

/// Dying f32 storage feeds the buffer pool; the next tensor of the same
/// size reuses the allocation instead of paying malloc + page faults
/// (docs/CAPABILITY.md 4.2). Other dtypes free normally.
impl Drop for StorageCell {
    fn drop(&mut self) {
        let data = match self.data.get_mut() {
            Ok(d) => d,
            Err(p) => p.into_inner(),
        };
        if let Storage::F32(v) = data {
            crate::pool::give(std::mem::take(v));
        }
    }
}

#[cfg(test)]
#[path = "concurrent_copy_tests.rs"]
mod concurrent_copy_tests;

#[cfg(test)]
#[path = "structural_zero_tests.rs"]
mod structural_zero_tests;

impl StorageCell {
    pub(crate) fn version(&self) -> u64 { self.version.load(Ordering::Relaxed) }

    fn new(data: Storage) -> StorageCell {
        StorageCell {
            data: RwLock::new(data),
            version: AtomicU64::new(0),
            structural_zero: false,
            publication: Mutex::new(()),
        }
    }

    /// Poison-tolerant read guard: a panicked writer can leave torn VALUES,
    /// but never a torn structure (the variant/allocation invariant above),
    /// and the panicking test conventions here require poison tolerance.
    pub(crate) fn read(&self) -> RwLockReadGuard<'_, Storage> {
        #[cfg(test)]
        crate::tensor::concurrent_copy_tests::before_read(self);
        self.data.read().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, Storage> {
        assert!(!self.structural_zero, "structural zero storage is immutable; clone it before mutation");
        self.data.write().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Read guards over two tensors' storages, collapsing to ONE guard when they
/// share a `StorageCell` (a second same-thread read of one `RwLock` can
/// deadlock behind a queued writer, so aliased operands must never take
/// two), and otherwise ALWAYS acquiring the lower-address storage first
/// regardless of argument order. Without that ordering, `PairGuard::new(a,
/// b)` on one thread and `PairGuard::new(b, a)` on another lock the same
/// pair of RwLocks in opposite orders; two readers alone can't deadlock, but
/// once any writer (an in-place op) can queue on either lock, a
/// writer-preferring RwLock blocks new readers behind it and the two
/// PairGuards complete the classic AB-BA cycle. Fixed acquisition order
/// (independent of which tensor is logically "first") removes the cycle for
/// any number of concurrent callers, mirroring the address-order discipline
/// `inplace.rs`'s multi-buffer locks already use.
pub(crate) struct PairGuard<'a> {
    g0: RwLockReadGuard<'a, Storage>,
    g1: Option<RwLockReadGuard<'a, Storage>>,
    /// True when `g0` is `a`'s guard (and `g1`, if any, is `b`'s); false
    /// when address order put `b` first, swapping the mapping.
    a_first: bool,
}

impl<'a> PairGuard<'a> {
    pub(crate) fn new(a: &'a Tensor, b: &'a Tensor) -> PairGuard<'a> {
        if Arc::ptr_eq(&a.0.storage, &b.0.storage) {
            return PairGuard {
                g0: a.0.storage.read(),
                g1: None,
                a_first: true,
            };
        }
        if Arc::as_ptr(&a.0.storage) < Arc::as_ptr(&b.0.storage) {
            PairGuard {
                g0: a.0.storage.read(),
                g1: Some(b.0.storage.read()),
                a_first: true,
            }
        } else {
            PairGuard {
                g0: b.0.storage.read(),
                g1: Some(a.0.storage.read()),
                a_first: false,
            }
        }
    }

    /// (a's storage, b's storage), regardless of which was locked first.
    pub(crate) fn get(&self) -> (&Storage, &Storage) {
        match (&self.g1, self.a_first) {
            (None, _) => (&self.g0, &self.g0),
            (Some(g1), true) => (&self.g0, g1),
            (Some(g1), false) => (g1, &self.g0),
        }
    }
}

pub(crate) struct TensorInner {
    pub(crate) id: usize,
    pub(crate) storage: Arc<StorageCell>,
    pub(crate) shape: Vec<usize>,
    pub(crate) stride: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) device: Device,
    // The own flag enables a leaf edge; a view may separately inherit its
    // ultimate base's public flag without acquiring a gradient edge.
    pub(crate) requires_grad: Arc<AtomicBool>,
    view_base: Option<Arc<AtomicBool>>,
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
    pub(crate) fn structural_zero(shape: &[usize], dtype: DType, device: Device) -> Result<Tensor> {
        if device != Device::Cpu {
            return Err(Error::Unsupported { op: "structural_zero", msg: "structural zeros currently require CPU storage".into() });
        }
        let storage = match dtype {
            DType::F32 => Storage::F32(crate::typed_math::filled("structural_zero", shape, 0.0f32)?),
            DType::F64 => Storage::F64(crate::typed_math::filled("structural_zero", shape, 0.0f64)?),
            _ => return Err(Error::DtypeMismatch { op: "structural_zero", expected: DType::F32, got: dtype }),
        };
        let mut cell = StorageCell::new(storage);
        cell.structural_zero = true;
        Ok(Tensor::from_parts(Arc::new(cell), shape.to_vec(), default_strides(shape), 0, device, false, None))
    }

    /// A symbolic zero has immutable storage shared by all its aliases.
    #[doc(hidden)]
    pub fn is_structural_zero(&self) -> bool { self.0.storage.structural_zero }

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
            requires_grad: Arc::new(AtomicBool::new(requires_grad)),
            view_base: None,
            op,
            grad: Mutex::new(None),
        }))
    }

    fn contiguous_leaf(
        op: &'static str,
        len: usize,
        storage: Storage,
        shape: &[usize],
    ) -> Result<Tensor> {
        if len != checked_numel(op, shape)? {
            return Err(Error::InvalidShape {
                op,
                msg: format!("{len} elements do not fit shape {shape:?}"),
            });
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

    pub fn from_vec_bool(data: Vec<bool>, shape: &[usize]) -> Result<Tensor> {
        Tensor::contiguous_leaf("from_vec_bool", data.len(), Storage::Bool(data), shape)
    }

    /// Build an f16 tensor from raw IEEE binary16 bit patterns.
    pub fn from_vec_f16_bits(data: Vec<u16>, shape: &[usize]) -> Result<Tensor> {
        Tensor::contiguous_leaf("from_vec_f16_bits", data.len(), Storage::F16(data), shape)
    }

    /// Build a bf16 tensor from raw bfloat16 bit patterns.
    pub fn from_vec_bf16_bits(data: Vec<u16>, shape: &[usize]) -> Result<Tensor> {
        Tensor::contiguous_leaf("from_vec_bf16_bits", data.len(), Storage::BF16(data), shape)
    }

    /// I64 tensor of `[0, end)` with shape `[end]` (empty when `end <= 0`).
    pub fn arange(end: i64) -> Tensor {
        let n = end.max(0);
        Tensor::from_vec_i64((0..n).collect(), &[n as usize]).unwrap()
    }

    pub fn full(shape: &[usize], value: f32) -> Tensor {
        Tensor::from_vec(crate::pool::take_filled(numel(shape), value), shape).unwrap()
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    pub fn scalar(value: f32) -> Tensor {
        Tensor::from_vec(crate::pool::take_filled(1, value), &[]).unwrap()
    }

    pub fn randn(shape: &[usize], rng: &Rng) -> Tensor {
        crate::forward_ad::check_random_operation().expect("random operation in a forward AD error scope");
        let mut data = crate::pool::take_uninit(numel(shape));
        for x in data.iter_mut() {
            *x = rng.normal();
        }
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
        self.0.requires_grad.load(Ordering::Relaxed) || self.0.view_base.as_ref().is_some_and(|base| base.load(Ordering::Relaxed))
    }
    pub fn is_leaf(&self) -> bool { !self.requires_grad() || self.0.op.is_none() }
    pub(crate) fn has_gradient_edge(&self) -> bool { self.0.requires_grad.load(Ordering::Relaxed) }
    pub(crate) fn is_inherited_grad_view(&self) -> bool { self.0.view_base.is_some() && !self.has_gradient_edge() && self.requires_grad() }
    pub(crate) fn grad_flag_handles(&self) -> (Arc<AtomicBool>, Option<Arc<AtomicBool>>) {
        (self.0.requires_grad.clone(), self.0.view_base.clone())
    }

    pub(crate) fn rebase_alias(&self, own: Arc<AtomicBool>, base: Option<Arc<AtomicBool>>, op: Option<Op>) -> Tensor {
        Tensor(Arc::new(TensorInner { id: fresh_id(), storage: self.0.storage.clone(), shape: self.0.shape.clone(),
            stride: self.0.stride.clone(), offset: self.0.offset, device: self.0.device,
            requires_grad: own, view_base: base, op, grad: Mutex::new(None) }))
    }

    pub fn device(&self) -> Device {
        self.0.device
    }
    pub fn dtype(&self) -> DType {
        self.0.storage.read().dtype()
    }

    /// Mark a leaf as requiring gradients (like `tensor.requires_grad_(True)`).
    /// Unchanged flags preserve identity and gradient slots. Changing a leaf's
    /// flag returns a fresh leaf sharing storage; changing an interior node's
    /// flag is rejected instead of silently cutting its history.
    /// Only float32 and float64 tensors support autograd.
    pub fn requires_grad_(&self, req: bool) -> Result<Tensor> {
        if req == self.requires_grad() && !self.is_inherited_grad_view() { return Ok(self.clone()); }
        if self.0.op.is_some() {
            return Err(Error::Unsupported {
                op: "requires_grad_",
                msg: "only leaf tensors can change requires_grad; detach first to create a fresh leaf".into(),
            });
        }
        if req {
            if !matches!(self.dtype(), DType::F32 | DType::F64) {
                return Err(Error::DtypeMismatch {
                    op: "requires_grad_",
                    expected: DType::F32,
                    got: self.dtype(),
                });
            }
        }
        Ok(Tensor::from_parts(
            self.0.storage.clone(),
            self.0.shape.clone(),
            self.0.stride.clone(),
            self.0.offset,
            self.0.device,
            req,
            None,
        ))
    }

    /// Statement-style leaf flag update. Handles, existing graph operands and
    /// gradient slots retain their identity; recorded edges retain eligibility
    /// from their forward pass independently of this mutable flag.
    #[doc(hidden)]
    pub fn set_requires_grad(&self, req: bool) -> Result<()> {
        if req == self.has_gradient_edge() { return Ok(()); }
        if self.0.op.is_some() {
            return Err(Error::Unsupported { op: "requires_grad_", msg: "only leaf tensors can change requires_grad; detach first to create a fresh leaf".into() });
        }
        if req && !matches!(self.dtype(), DType::F32 | DType::F64) {
            return Err(Error::DtypeMismatch { op: "requires_grad_", expected: DType::F32, got: self.dtype() });
        }
        self.0.requires_grad.store(req, Ordering::Relaxed);
        Ok(())
    }

    // --- materialization --------------------------------------------------

    /// Gather a (possibly strided/broadcast) view of `data` into a contiguous
    /// row-major Vec, generic over the element type so every dtype shares the
    /// one strided-odometer implementation.
    fn gather_view<T: Copy>(&self, data: &[T]) -> Vec<T> {
        let inner = &self.0;
        let n = self.numel();
        if n == 0 { return Vec::new(); }
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

    /// `gather_view` writing into a caller-provided buffer instead of
    /// allocating: the pooled-temporary path of `to_vec_pooled`.
    fn gather_into(&self, data: &[f32], out: &mut [f32]) {
        let inner = &self.0;
        let n = self.numel();
        if n == 0 { return; }
        if self.is_contiguous() {
            out.copy_from_slice(&data[inner.offset..inner.offset + n]);
            return;
        }
        let ndim = inner.shape.len();
        if ndim == 0 {
            out[0] = data[inner.offset];
            return;
        }
        let mut idx = vec![0usize; ndim];
        for slot in out.iter_mut().take(n) {
            let mut off = inner.offset;
            for d in 0..ndim {
                off += idx[d] * inner.stride[d];
            }
            *slot = data[off];
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < inner.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    /// Raw IEEE binary16 bit patterns of an f16 tensor, row-major (strided
    /// views gather). Errors on any other dtype: bit access never converts.
    pub fn to_vec_f16_bits(&self) -> Result<Vec<u16>> {
        let g = self.0.storage.read();
        match &*g {
            Storage::F16(v) => Ok(self.gather_view(v)),
            _ => Err(Error::DtypeMismatch {
                op: "to_vec_f16_bits",
                expected: DType::F16,
                got: g.dtype(),
            }),
        }
    }

    /// Raw bfloat16 bit patterns of a bf16 tensor, row-major.
    pub fn to_vec_bf16_bits(&self) -> Result<Vec<u16>> {
        let g = self.0.storage.read();
        match &*g {
            Storage::BF16(v) => Ok(self.gather_view(v)),
            _ => Err(Error::DtypeMismatch {
                op: "to_vec_bf16_bits",
                expected: DType::BF16,
                got: g.dtype(),
            }),
        }
    }

    /// `to_vec` for INTERNAL temporaries: host f32 sources materialize into a
    /// pool buffer, and the caller must `pool::give` it back when done (the
    /// public `to_vec` hands ownership to the caller, so it never draws from
    /// the pool - user code would drain it). Non-f32/device sources fall
    /// through to the ordinary allocation, which give() then recycles.
    pub(crate) fn to_vec_pooled(&self) -> Vec<f32> {
        {
            let g = self.0.storage.read();
            if let Storage::F32(v) = &*g {
                let mut out = crate::pool::take_uninit(self.numel());
                self.gather_into(v, &mut out);
                return out;
            }
        }
        self.to_vec()
    }

    /// Runs `f` on the tensor's row-major f32 data without materializing a
    /// copy when it can: contiguous f32 storage is borrowed in place (under
    /// the storage read lock), anything else goes through a pooled temporary.
    pub(crate) fn with_host_f32<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        if self.numel() == 0 { return f(&[]); }
        {
            let g = self.0.storage.read();
            if let Storage::F32(v) = &*g {
                if self.is_contiguous() {
                    return f(&v[self.0.offset..self.0.offset + self.numel()]);
                }
            }
        }
        let tmp = self.to_vec_pooled();
        let r = f(&tmp);
        crate::pool::give(tmp);
        r
    }

    /// Host-side full-tensor sum via the cpu backend's fixed-order tree.
    pub(crate) fn raw_host_sum(&self) -> f32 {
        let cpu = backend_for(Device::Cpu).expect("cpu backend is always registered");
        self.with_host_f32(|v| cpu.sum(v))
    }

    /// Materialize row-major DATA IN self's OWN dtype (no cast): the general
    /// worker `detach_copy` calls so a non-f32 host tensor's copy stays that
    /// dtype instead of silently downcasting through `to_vec`'s f32 path.
    /// Reads `dtype()` (which takes and releases its own guard) rather than
    /// matching under a held guard - each `to_vec_*` call below takes its
    /// own read guard too, and nesting two same-thread reads of one RwLock
    /// can deadlock behind a queued writer (the exact hazard fixed in
    /// `PairGuard`). Device views reach this host path only for non-F32
    /// storage or an explicitly unsupported detached materialization.
    fn detach_copy_same_dtype(&self) -> Tensor {
        match self.dtype() {
            DType::F32 => Tensor::from_vec(self.to_vec(), &self.0.shape).unwrap(),
            DType::F64 => Tensor::from_vec_f64(self.to_vec_f64(), &self.0.shape).unwrap(),
            DType::I64 => Tensor::from_vec_i64(self.to_vec_i64(), &self.0.shape).unwrap(),
            DType::Bool => Tensor::from_vec_bool(self.to_vec_bool(), &self.0.shape).unwrap(),
            DType::F16 => {
                Tensor::from_vec_f16_bits(self.to_vec_f16_bits().unwrap(), &self.0.shape).unwrap()
            }
            DType::BF16 => {
                Tensor::from_vec_bf16_bits(self.to_vec_bf16_bits().unwrap(), &self.0.shape).unwrap()
            }
        }
    }

    /// Materialize as row-major f32, casting from f64/i64 storage (lossy for
    /// |i64| > 2^24). F32 kernels use this view-aware materializer; typed kernels
    /// use to_vec_f64/to_vec_i64 to retain their precision.
    pub fn to_vec(&self) -> Vec<f32> {
        match &*self.0.storage.read() {
            Storage::F32(v) => self.gather_view(v),
            Storage::F64(v) => self.gather_view(v).into_iter().map(|x| x as f32).collect(),
            Storage::I64(v) => self.gather_view(v).into_iter().map(|x| x as f32).collect(),
            Storage::Bool(v) => self.gather_view(v).into_iter().map(|x| u8::from(x) as f32).collect(),
            Storage::F16(v) => self
                .gather_view(v)
                .into_iter()
                .map(crate::half::f32_from_f16_bits)
                .collect(),
            Storage::BF16(v) => self
                .gather_view(v)
                .into_iter()
                .map(crate::half::f32_from_bf16_bits)
                .collect(),
            Storage::DeviceI64(b) => {
                let host = backend_for(self.0.device)
                    .expect("device tensor exists, so its backend must be registered")
                    .copy_i64_to_host(b.as_ref())
                    .expect("device i64-to-host copy failed");
                self.gather_view(&host).into_iter().map(|x| x as f32).collect()
            }
            Storage::Device(b) => {
                let host = device_to_host(self.0.device, b.as_ref());
                self.gather_view(&host)
            }
        }
    }

    /// Materialize as row-major f64 (exact from f32/i64 up to 2^53). Host
    /// storage reads under one guard; device storage falls through to the
    /// download paths after releasing it (nested reads of one storage lock
    /// on a single thread can deadlock behind a queued writer).
    pub fn to_vec_f64(&self) -> Vec<f64> {
        {
            let g = self.0.storage.read();
            match &*g {
                Storage::F32(v) => {
                    return self.gather_view(v).into_iter().map(|x| x as f64).collect()
                }
                Storage::F64(v) => return self.gather_view(v),
                Storage::I64(v) => {
                    return self.gather_view(v).into_iter().map(|x| x as f64).collect()
                }
                Storage::Bool(v) => return self.gather_view(v).into_iter().map(|x| u8::from(x) as f64).collect(),
                Storage::F16(v) => {
                    return self
                        .gather_view(v)
                        .into_iter()
                        .map(|b| crate::half::f32_from_f16_bits(b) as f64)
                        .collect()
                }
                Storage::BF16(v) => {
                    return self
                        .gather_view(v)
                        .into_iter()
                        .map(|b| crate::half::f32_from_bf16_bits(b) as f64)
                        .collect()
                }
                Storage::Device(_) | Storage::DeviceI64(_) => {}
            }
        }
        if self.dtype() == DType::I64 {
            self.to_vec_i64().into_iter().map(|x| x as f64).collect()
        } else {
            self.to_vec().into_iter().map(|x| x as f64).collect()
        }
    }

    /// Materialize as row-major i64; floats truncate toward zero. Device-i64
    /// storage downloads through the backend's i64 channel.
    pub fn to_vec_i64(&self) -> Vec<i64> {
        {
            let g = self.0.storage.read();
            match &*g {
                Storage::F32(v) => {
                    return self.gather_view(v).into_iter().map(|x| x as i64).collect()
                }
                Storage::F64(v) => {
                    return self.gather_view(v).into_iter().map(|x| x as i64).collect()
                }
                Storage::I64(v) => return self.gather_view(v),
                Storage::Bool(v) => return self.gather_view(v).into_iter().map(i64::from).collect(),
                Storage::F16(v) => {
                    return self
                        .gather_view(v)
                        .into_iter()
                        .map(|b| crate::half::f32_from_f16_bits(b) as i64)
                        .collect()
                }
                Storage::BF16(v) => {
                    return self
                        .gather_view(v)
                        .into_iter()
                        .map(|b| crate::half::f32_from_bf16_bits(b) as i64)
                        .collect()
                }
                Storage::DeviceI64(b) => {
                    let host = backend_for(self.0.device)
                        .expect("device tensor exists, so its backend must be registered")
                        .copy_i64_to_host(b.as_ref())
                        .expect("device i64-to-host copy failed");
                    return self.gather_view(&host);
                }
                Storage::Device(_) => {}
            }
        }
        self.to_vec().into_iter().map(|x| x as i64).collect()
    }

    /// Materialize boolean values; numeric zero is false and every other
    /// value (including NaN) is true. Boolean storage is never read as floats.
    pub fn to_vec_bool(&self) -> Vec<bool> {
        {
            let storage = self.0.storage.read();
            if let Storage::Bool(values) = &*storage { return self.gather_view(values); }
        }
        if self.dtype() == DType::I64 {
            self.to_vec_i64().into_iter().map(|x| x != 0).collect()
        } else {
            self.to_vec_f64().into_iter().map(|x| x != 0.0).collect()
        }
    }

    /// Cast to `dtype`, returning a detached contiguous leaf. Host conversions
    /// report CPU; use to_dtype_autograd for differentiable floating CPU casts.
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
            DType::Bool => Storage::Bool(self.to_vec_bool()),
            // Half targets round (RNE) from the f32 materialization; a
            // same-dtype half cast round-trips bit-exactly since every f16/
            // bf16 value is exactly representable in f32.
            DType::F16 => Storage::F16(
                self.to_vec().into_iter().map(crate::half::f16_bits_from_f32).collect(),
            ),
            DType::BF16 => Storage::BF16(
                self.to_vec().into_iter().map(crate::half::bf16_bits_from_f32).collect(),
            ),
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

    /// Scalar value of a 0-d (or single-element) tensor. Host f32 reads the
    /// element straight from storage - no per-item Vec (offset addresses the
    /// first element of any layout, matching to_vec()[0]).
    pub fn item(&self) -> f32 {
        {
            let g = self.0.storage.read();
            if let Storage::F32(v) = &*g {
                return v[self.0.offset];
            }
        }
        self.to_vec()[0]
    }

    // --- device transfer ---------------------------------------------------

    /// Move this tensor's data to `device`, returning a detached contiguous
    /// leaf there (like `to_dtype`, transfers never carry autograd history).
    /// F32 tensors move via `alloc_from_host`; I64 tensors (indices) move via
    /// `alloc_i64_from_host`. Cross-device goes via the host.
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        if self.0.device == device {
            return Ok(self.clone());
        }
        // Materialize on the host first (a copy back for device sources, a
        // contiguous gather for host views), then upload if the target is not
        // the cpu.
        let shape = self.0.shape.clone();
        match self.dtype() {
            DType::F32 => {
                let host = self.to_vec();
                if device == Device::Cpu {
                    return Tensor::from_vec(host, &shape);
                }
                let buf = backend_for(device)?.alloc_from_host(&host)?;
                Ok(Tensor::from_parts(
                    Arc::new(StorageCell::new(Storage::Device(Arc::from(buf)))),
                    shape.clone(),
                    default_strides(&shape),
                    0,
                    device,
                    false,
                    None,
                ))
            }
            DType::I64 => {
                let host = self.to_vec_i64();
                if device == Device::Cpu {
                    return Tensor::from_vec_i64(host, &shape);
                }
                let buf = backend_for(device)?.alloc_i64_from_host(&host)?;
                Ok(Tensor::from_parts(
                    Arc::new(StorageCell::new(Storage::DeviceI64(buf))),
                    shape.clone(),
                    default_strides(&shape),
                    0,
                    device,
                    false,
                    None,
                ))
            }
            dt => Err(Error::DtypeMismatch {
                op: "to_device",
                expected: DType::F32,
                got: dt,
            }),
        }
    }

    /// True when storage is a backend-owned device buffer usable by the
    /// device kernel paths: those pass the raw buffer, so the view must be
    /// the whole buffer in natural order. Only the f32 flavor counts - i64
    /// index buffers have their own gather seam.
    pub(crate) fn device_resident_whole(&self) -> bool {
        matches!(&*self.0.storage.read(), Storage::Device(b) if b.len() == self.numel())
            && self.0.offset == 0
            && self.is_contiguous()
    }

    /// Row-major contiguous check without allocating (size-1 dims may carry any
    /// stride since they contribute no offset).
    pub fn is_contiguous(&self) -> bool {
        if self.numel() == 0 { return true; }
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

    /// Replace a floating leaf's gradient handle after metadata validation.
    /// This does not run backward, copy storage, or change storage versions.
    #[doc(hidden)]
    pub fn set_leaf_grad(&self, gradient: Option<Tensor>) -> Result<()> {
        const OP: &str = "set_leaf_grad";
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op: OP, msg: "gradient assignment is not supported during inference capture".into() });
        }
        if self.0.op.is_some() {
            return Err(Error::Unsupported { op: OP, msg: "gradient assignment requires a leaf without autograd history".into() });
        }
        if !matches!(self.dtype(), DType::F32 | DType::F64) {
            return Err(Error::DtypeMismatch { op: OP, expected: DType::F32, got: self.dtype() });
        }
        if let Some(value) = &gradient {
            if value.dtype() != self.dtype() {
                return Err(Error::DtypeMismatch { op: OP, expected: self.dtype(), got: value.dtype() });
            }
            if value.shape() != self.shape() {
                return Err(Error::ShapeMismatch { op: OP, lhs: self.shape().to_vec(), rhs: value.shape().to_vec() });
            }
            if value.device() != self.device() {
                return Err(Error::DeviceMismatch { op: OP, lhs: self.device(), rhs: value.device() });
            }
        }
        // Serialize direct assignments so two concurrent writes cannot create
        // a cycle through one another's gradient slots after validation.
        static ASSIGNMENTS: Mutex<()> = Mutex::new(());
        let _assignment = ASSIGNMENTS.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(value) = &gradient {
            let mut pending = vec![value.clone()];
            let mut visited = std::collections::HashSet::new();
            while let Some(node) = pending.pop() {
                if node.id() == self.id() {
                    return Err(Error::Unsupported { op: OP, msg: "a gradient must not reference its destination through history or gradient slots".into() });
                }
                if !visited.insert(node.id()) { continue; }
                if let Some(op) = &node.0.op { pending.extend(op.inputs().iter().cloned()); }
                if let Some(grad) = node.grad() { pending.push(grad); }
            }
        }
        *self.0.grad.lock().unwrap() = gradient;
        Ok(())
    }

    pub fn zero_grad(&self) {
        *self.0.grad.lock().unwrap() = None;
    }

    pub(crate) fn accumulate_grad(&self, g: Tensor) {
        assert_eq!(g.dtype(), self.dtype(), "gradient dtype must match its tensor");
        assert!(matches!(self.dtype(), DType::F32 | DType::F64), "only floating tensors have gradients");
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
            g.to_device(self.0.device)
                .expect("grad transfer to the tensor's device")
        };
        let mut slot = self.0.grad.lock().unwrap();
        let stride = if self.device() == Device::Cpu && self.0.op.is_none() {
            match slot.as_ref() {
                Some(previous) if previous.has_dense_layout() => Some(previous.0.stride.clone()),
                Some(_) => None,
                None => Some(if self.has_dense_layout() { self.0.stride.clone() } else { default_strides(self.shape()) }),
            }
        } else { None };
        let accumulated = Self::merge_grad(slot.take(), g);
        *slot = Some(match stride {
            Some(stride) if accumulated.strides() != stride || accumulated.storage_offset() != 0 => accumulated.copy_grad_layout(stride),
            _ => accumulated,
        });
    }

    pub(crate) fn has_dense_layout(&self) -> bool {
        if self.numel() == 0 { return true; }
        let mut axes = (0..self.ndim()).filter(|&axis| self.shape()[axis] > 1).collect::<Vec<_>>();
        axes.sort_unstable_by_key(|&axis| self.0.stride[axis]);
        let mut expected = 1usize;
        axes.into_iter().all(|axis| {
            if self.0.stride[axis] != expected { return false; }
            let Some(next) = expected.checked_mul(self.shape()[axis]) else { return false; };
            expected = next;
            true
        })
    }

    // Reorder values into a dense leaf's physical layout without recording an
    // operation. Functional cotangents and device accumulation keep their own
    // layouts; only published CPU gradient slots use this default contract.
    pub(crate) fn copy_grad_layout(&self, stride: Vec<usize>) -> Tensor {
        let width = if self.dtype() == DType::F64 { 8 } else { 4 };
        assert!(self.numel() <= isize::MAX as usize / width, "gradient storage exceeds typed allocation capacity");
        let mut axes = (0..self.ndim()).collect::<Vec<_>>();
        axes.sort_unstable_by_key(|&axis| std::cmp::Reverse(stride[axis]));
        let physical = Tensor::from_parts(self.0.storage.clone(), axes.iter().map(|&axis| self.shape()[axis]).collect(),
            axes.iter().map(|&axis| self.0.stride[axis]).collect(), self.0.offset, Device::Cpu, false, None);
        let owned = physical.detach_copy_same_dtype();
        Tensor::from_parts(owned.0.storage.clone(), self.0.shape.clone(), stride, 0, Device::Cpu, false, None)
    }

    /// Functional traversal input-buffer accumulation. Graph recording and
    /// active forward levels keep the genuine Add operation. With recording
    /// disabled, a uniquely owned empty ordinary cotangent can be reused by a
    /// real zero-element add-in-place, retaining its layout, flags and version.
    pub(crate) fn accumulate_vjp(existing: Tensor, contribution: Tensor) -> Result<Tensor> {
        if !crate::autograd::is_grad_enabled() && !crate::forward_ad::is_active()
            && existing.numel() == 0 && existing.device() == Device::Cpu
            && matches!(existing.dtype(), DType::F32 | DType::F64)
            && !existing.is_structural_zero()
            // Torch's views retain a base Tensor/storage owner. Our views
            // retain only its flag owner, so exclude that alias explicitly
            // instead of treating the smaller Arc count as sole ownership.
            && existing.0.view_base.is_none()
            && Arc::strong_count(&existing.0) == 1
            && Arc::strong_count(&existing.0.storage) == 1
        {
            crate::inplace::raw_empty_add_(&existing, &contribution)?;
            Ok(existing)
        } else {
            existing.add(&contribution)
        }
    }

    pub(crate) fn merge_grad(existing: Option<Tensor>, g: Tensor) -> Tensor {
        match existing {
            None => g,
            Some(existing) => {
                // In-place accumulation is a pure optimization, applied only
                // when `existing` is provably unshared (sole Tensor handle AND
                // sole storage reference; holding the grad mutex means no new
                // handle can appear): backward can hand several inputs the
                // same gradient tensor (y = a + b seeds both with one), and
                // user-held grad() clones must keep their values, so anything
                // shared falls back to the allocating sum. A failed in-place
                // attempt (exotic layout, backend without in-place kernels)
                // falls back too - in-place kernels never partially mutate on
                // Err.
                let unshared = Arc::strong_count(&existing.0) == 1
                    && Arc::strong_count(&existing.0.storage) == 1;
                if unshared && existing.dtype() == DType::F32
                    && crate::inplace::raw_binary_("grad_acc", BinaryKind::Add, &existing, &g)
                        .is_ok()
                {
                    existing
                } else {
                    raw_binary_k("grad_acc", &existing, &g, BinaryKind::Add).unwrap()
                }
            }
        }
    }

    // --- views (share storage) -------------------------------------------
    /// Broadcast to `shape` without copying (inserts zero strides). Detached:
    /// broadcasting's gradient is handled by reducing in backward.
    pub(crate) fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        checked_numel("broadcast_to", shape)?;
        let cur = &self.0.shape;
        if shape.len() < cur.len() {
            return Err(Error::ShapeMismatch {
                op: "broadcast_to",
                lhs: cur.clone(),
                rhs: shape.to_vec(),
            });
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
                    return Err(Error::ShapeMismatch {
                        op: "broadcast_to",
                        lhs: cur.clone(),
                        rhs: shape.to_vec(),
                    });
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

    fn materialize_device(&self) -> Result<Option<Tensor>> {
        if self.device() == Device::Cpu || self.dtype() != DType::F32 { return Ok(None); }
        let backend = backend_for(self.device())?;
        {
            let storage = self.0.storage.read();
            let Storage::Device(buf) = &*storage else { return Ok(None); };
            if let Ok(out) = backend.materialize_dev(buf.as_ref(), self.shape(), &self.0.stride, self.0.offset) {
                return Ok(Some(device_leaf(out, self.shape(), self.device())));
            }
        }
        // Layout and buffer identity are immutable. Build/upload gather indices
        // unlocked, then reacquire only for dispatch; device values follow the
        // backend's stream ordering, just as in the direct materialization seam.
        let mut indices = Vec::with_capacity(self.numel());
        for i in 0..self.numel() {
            let mut rem = i;
            let mut offset = self.0.offset;
            for d in (0..self.ndim()).rev() {
                offset += (rem % self.shape()[d]) * self.0.stride[d];
                rem /= self.shape()[d];
            }
            indices.push(i64::try_from(offset).map_err(|_| Error::Unsupported { op: "reshape", msg: "layout exceeds i64 indexing".into() })?);
        }
        let Ok(index) = backend.alloc_i64_from_host(&indices) else { return Ok(None); };
        let storage = self.0.storage.read();
        let Storage::Device(buf) = &*storage else { unreachable!("storage identity is immutable"); };
        match backend.gather_rows_dev(buf.as_ref(), index.as_ref(), buf.len(), 1) {
            Ok(out) => Ok(Some(device_leaf(out, self.shape(), self.device()))),
            Err(_) => Ok(None),
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        if checked_numel("reshape", shape)? != self.numel() {
            return Err(Error::InvalidShape {
                op: "reshape",
                msg: format!("cannot reshape {:?} into {shape:?}", self.0.shape),
            });
        }
        let same_shape = shape == self.shape();
        if self.dtype() == DType::F64 && !self.is_contiguous() && !same_shape {
            crate::typed_math::check_shape("reshape", shape)?;
        }
        // Materialize strided f32 device views directly, or via gather fallback.
        // Other backends/dtypes retain the dtype-preserving host fallback.
        let base = if same_shape || self.is_contiguous() {
            self.clone()
        } else if let Some(resident) = self.materialize_device()? {
            resident
        } else {
            let host = match self.dtype() {
                DType::F32 => Tensor::from_vec(self.to_vec(), &self.0.shape)?,
                DType::F64 => Tensor::from_vec_f64(self.to_vec_f64(), &self.0.shape)?,
                DType::I64 => Tensor::from_vec_i64(self.to_vec_i64(), &self.0.shape)?,
                DType::Bool => Tensor::from_vec_bool(self.to_vec_bool(), &self.0.shape)?,
                DType::F16 => {
                    Tensor::from_vec_f16_bits(self.to_vec_f16_bits().unwrap(), &self.0.shape)?
                }
                DType::BF16 => {
                    Tensor::from_vec_bf16_bits(self.to_vec_bf16_bits().unwrap(), &self.0.shape)?
                }
            };
            host.to_device(self.0.device)?
        };
        let out = Tensor::from_parts(
            base.0.storage.clone(),
            shape.to_vec(),
            if same_shape { self.0.stride.clone() } else { default_strides(shape) },
            base.0.offset,
            base.0.device,
            false,
            None,
        );
        let out = self.capture_layout(out)?;
        let in_shape = self.0.shape.clone();
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Reshape(shape.to_vec()), move |g| {
            vec![g.reshape(&in_shape).unwrap()]
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
        Ok(Tensor::from_parts(
            self.0.storage.clone(),
            shape,
            stride,
            self.0.offset,
            self.0.device,
            false,
            None,
        ))
    }

    // Isolate inference views of graph leaves, the public update seam. Computed
    // inputs already forbid public mutation through their op history. Copy into
    // a NEW cell, preserving layout; never relax ordinary alias/snapshot gates.
    // Replay disables capture and pays no copy.
    pub(crate) fn capture_layout(&self, out: Tensor) -> Result<Tensor> {
        let out = out.inherit_no_grad_view(self);
        if !crate::capture::is_recording() || self.requires_grad() || self.0.op.is_some()
            || !Arc::ptr_eq(&self.0.storage, &out.0.storage) { return Ok(out); }
        let storage = out.0.storage.read();
        let Storage::Device(buf) = &*storage else { return Ok(out.clone()); };
        let copy = backend_for(out.device())?.copy_dev(buf.as_ref())?;
        Ok(Tensor::from_parts(
            Arc::new(StorageCell::new(Storage::Device(Arc::from(copy)))),
            out.0.shape.clone(), out.0.stride.clone(), out.0.offset, out.device(), false, None,
        ))
    }

    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Tensor> {
        let out = self.capture_layout(self.transpose_view(d0, d1)?)?;
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Transpose(d0, d1), move |g| {
            vec![g.transpose(d0, d1).unwrap()]
        }))
    }

    /// Element strides and offset into the shared allocation, without reads.
    pub fn strides(&self) -> &[usize] { &self.0.stride }
    pub fn storage_offset(&self) -> usize { self.0.offset }

    // A view tracks its ultimate base's flag without retaining a Tensor/Op.
    // Recording may establish its own edge; no-grad views only inherit flags.
    pub(crate) fn inherit_no_grad_view(mut self, input: &Tensor) -> Tensor {
        if Arc::ptr_eq(&self.0.storage, &input.0.storage) {
            let inner = Arc::get_mut(&mut self.0).expect("fresh view is uniquely owned");
            debug_assert!(inner.op.is_none());
            inner.view_base = Some(input.0.view_base.as_ref().unwrap_or(&input.0.requires_grad).clone());
        }
        self
    }

    /// Public detached alias. Internal backward snapshots continue using
    /// detach_copy or owned_detach_copy so they cannot be overwritten by users.
    pub fn detach(&self) -> Tensor { self.detached_view() }

    /// Ordinary CPU zeros for an absent transform tangent, without value reads.
    #[doc(hidden)]
    pub fn zeros_like_layout(&self) -> Result<Tensor> {
        const OP: &str = "absent tangent zeros";
        if self.device() != Device::Cpu {
            return Err(Error::Unsupported { op: OP, msg: "only CPU transform outputs are supported".into() });
        }
        let raw = match self.dtype() {
            DType::F32 => Tensor::from_vec(crate::typed_math::filled(OP, self.shape(), 0.0f32)?, self.shape())?,
            DType::F64 => Tensor::from_vec_f64(crate::typed_math::filled(OP, self.shape(), 0.0f64)?, self.shape())?,
            _ => return Err(Error::Unsupported { op: OP, msg: "transform outputs require float32 or float64".into() }),
        };
        let stride = crate::typed_math::copy_pointwise_layout(self);
        Ok(Tensor::from_parts(raw.0.storage.clone(), self.shape().to_vec(), stride, 0, Device::Cpu, false, None))
    }

    /// Differentiable owning copy. Preserve dense nonoverlapping strides when
    /// requested; gapped and broadcast views materialize in logical order.
    pub fn clone_tensor(&self, preserve_layout: bool) -> Result<Tensor> {
        let count = checked_numel("clone", self.shape())?;
        let width = match self.dtype() { DType::Bool => 1, DType::F16 | DType::BF16 => 2, DType::F32 => 4, DType::F64 | DType::I64 => 8 };
        if count > isize::MAX as usize / width {
            return Err(Error::InvalidShape { op: "clone", msg: "copy exceeds typed allocation capacity".into() });
        }
        let dense = if count == 0 { true } else {
            let mut axes = (0..self.ndim()).filter(|&i| self.shape()[i] > 1).collect::<Vec<_>>();
            axes.sort_unstable_by_key(|&i| self.0.stride[i]);
            let mut expected = 1usize;
            axes.iter().all(|&i| {
                if self.0.stride[i] != expected { return false; }
                expected *= self.shape()[i];
                true
            })
        };
        let (owned, stride) = if preserve_layout && dense {
            let physical = Tensor::from_parts(self.0.storage.clone(), vec![count], vec![1], self.0.offset, self.device(), false, None);
            (physical.try_owned_detach_copy()?, self.0.stride.clone())
        } else { (self.try_owned_detach_copy()?, default_strides(self.shape())) };
        let output = Tensor::from_parts(owned.0.storage.clone(), self.shape().to_vec(), stride, 0, self.device(), false, None);
        Ok(output.record_identity_copy(self, preserve_layout))
    }

    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() { Ok(self.clone()) } else { self.clone_tensor(false) }
    }

    fn record_identity_copy(self, input: &Tensor, preserve_layout: bool) -> Tensor {
        let mut output = self.record_fn_forward(vec![input.clone()], crate::autograd::ForwardOp::Clone { preserve_layout }, |g| vec![g.clone()]);
        if let Some(inner) = Arc::get_mut(&mut output.0) {
            if let Some(op) = &mut inner.op { op.omit_saved_input_values(); }
        }
        output
    }

    /// Declare output data read by this backward rule. Only storage is retained,
    /// never the output Tensor/Op, so this does not create an autograd cycle.
    pub(crate) fn save_output_version(self) -> Tensor {
        let storage = self.0.storage.clone();
        let version = self.version();
        if let Some(op) = &self.0.op { op.save_storage(storage, version); }
        self
    }

    /// Version dependencies on sibling outputs, such as Q and R from QR.
    pub(crate) fn save_tensor_versions(self, values: &[&Tensor]) -> Tensor {
        if let Some(op) = &self.0.op {
            for value in values { op.save_storage(value.0.storage.clone(), value.version()); }
        }
        self
    }

    /// Share storage and layout, stripping recording and gradient state.
    pub(crate) fn detached_view(&self) -> Tensor {
        Tensor::from_parts(self.0.storage.clone(), self.0.shape.clone(), self.0.stride.clone(), self.0.offset, self.0.device, false, None)
    }

    /// A detached, contiguous copy sharing no autograd history. Host copies
    /// own fresh storage; a whole device buffer is SHARED rather than copied
    /// through the host (which would also move the tensor to cpu) - which is
    /// why public in-place ops refuse device tensors with shared storage,
    /// and why `Param` construction goes through `owned_detach_copy`.
    /// Non-whole F32 device views materialize directly without recording an op.
    /// Only explicit unsupported capability/layout permits a host copy; any
    /// operational error panics rather than silently downloading or retrying.
    pub fn detach_copy(&self) -> Tensor {
        if self.is_structural_zero() {
            return Self::structural_zero(self.shape(), self.dtype(), self.device()).expect("structural zero snapshot allocation failed");
        }
        if self.device_resident_whole() && self.storage_len() == self.numel() {
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
        if self.device() != Device::Cpu && self.dtype() == DType::F32 {
            let backend = backend_for(self.device()).expect("detached materialization backend missing");
            let storage = self.0.storage.read();
            if let Storage::Device(buf) = &*storage {
                match backend.materialize_support_for(buf.as_ref(), self.shape(), &self.0.stride, self.0.offset)
                    .expect("detached device materialization preflight failed") {
                    crate::dispatch::MaterializeSupport::Attempt => {
                        match backend.materialize_dev(buf.as_ref(), self.shape(), &self.0.stride, self.0.offset) {
                            Ok(out) => return device_leaf(out, self.shape(), self.device()),
                            Err(Error::MaterializeUnavailable) => {},
                            Err(error) => panic!("detached device materialization failed: {error:?}"),
                        }
                    }
                    crate::dispatch::MaterializeSupport::UnsupportedBackend
                    | crate::dispatch::MaterializeSupport::UnsupportedLayout => {}
                }
            }
        }
        self.detach_copy_same_dtype()
    }

    /// Owning CPU snapshot for dense saved outputs, retaining physical order.
    /// Device snapshots keep the existing sharing/materialization policy.
    pub(crate) fn detach_copy_layout(&self) -> Result<Tensor> {
        if self.device() != Device::Cpu || self.is_structural_zero() { return Ok(self.detach_copy()); }
        let count = checked_numel("saved_copy", self.shape())?;
        let width = match self.dtype() { DType::Bool => 1, DType::F16 | DType::BF16 => 2, DType::F32 => 4, DType::F64 | DType::I64 => 8 };
        if count > isize::MAX as usize / width {
            return Err(Error::InvalidShape { op: "saved_copy", msg: "snapshot exceeds typed allocation capacity".into() });
        }
        let stride = crate::typed_math::copy_pointwise_layout(self);
        let physical = crate::typed_math::pointwise_source(self, self.shape(), &stride)?;
        let owned = physical.detach_copy_same_dtype();
        Ok(Self::from_parts(owned.0.storage.clone(), self.shape().to_vec(), stride, 0, self.device(), false, None))
    }

    /// A detached contiguous copy that always OWNS its storage, device
    /// included (one `copy_dev` for whole device buffers; strided device
    /// views materialize through the host and re-upload). Parameters are
    /// built from this so an in-place optimizer step can never scribble
    /// over the caller's tensor. Cold paths only.
    pub(crate) fn owned_detach_copy(&self) -> Tensor {
        self.try_owned_detach_copy().expect("owning tensor copy failed")
    }

    /// Preserve dtype bits and placement; propagate device transfer failures.
    pub(crate) fn try_owned_detach_copy(&self) -> Result<Tensor> {
        if self.device_resident_whole() {
            let backend = backend_for(self.0.device)?;
            let g = self.0.storage.read();
            let Storage::Device(buf) = &*g else { unreachable!() };
            let copy = backend.copy_dev(buf.as_ref())?;
            return Ok(device_leaf(copy, &self.0.shape, self.0.device));
        }
        let host = {
            let g = self.0.storage.read();
            match &*g {
                Storage::Device(buf) => Tensor::from_vec(
                    self.gather_view(&backend_for(self.device())?.copy_to_host(buf.as_ref())?), self.shape())?,
                Storage::DeviceI64(buf) => Tensor::from_vec_i64(
                    self.gather_view(&backend_for(self.device())?.copy_i64_to_host(buf.as_ref())?), self.shape())?,
                _ => { drop(g); self.detach_copy_same_dtype() }
            }
        };
        host.to_device(self.0.device)
    }

    /// Constant tensor on the given device (cpu allocation or backend fill).
    pub fn full_on(shape: &[usize], value: f32, device: Device) -> Result<Tensor> {
        let n = checked_numel("full_on", shape)?;
        if device == Device::Cpu {
            return Ok(Tensor::full(shape, value));
        }
        let buf = backend_for(device)?.fill_dev(value, n)?;
        Ok(device_leaf(buf, shape, device))
    }

    /// The single autograd recording hook: `inputs` are the differentiable
    /// operands; `backward` maps the output gradient to one gradient per input
    /// (same order; the engine asserts arity and shapes). Recorded only when
    /// grad mode is enabled and some input requires grad, or inference capture
    /// is active independently of grad mode. Capture-only
    /// nodes discard the backward closure. `self` must be freshly-created, uniquely-
    /// owned output (as returned by the raw kernels).
    pub fn record_fn<F>(self, inputs: Vec<Tensor>, backward: F) -> Tensor
    where
        F: Fn(&Tensor) -> Vec<Tensor> + Send + Sync + 'static,
    {
        self.record_operation(inputs, None, None, move |inputs| Op::new(inputs, Box::new(backward)))
    }

    /// Record exact data dependencies while retaining every differentiable
    /// operand. Storage/version snapshots precede forward-mode dispatch.
    pub(crate) fn record_fn_saved<F>(self, inputs: Vec<Tensor>, saved_inputs: &[usize], save_output: bool, backward: F) -> Tensor
    where F: Fn(&Tensor) -> Vec<Tensor> + Send + Sync + 'static {
        assert!(saved_inputs.iter().all(|&index| index < inputs.len()), "saved input index is out of range");
        let output = save_output.then(|| (self.0.storage.clone(), self.version()));
        self.record_operation(inputs, None, None, move |inputs| {
            let saved: Vec<_> = saved_inputs.iter().map(|&index| (inputs[index].0.storage.clone(), inputs[index].version())).collect();
            let mut op = Op::new(inputs, Box::new(backward));
            op.omit_saved_input_values();
            for (storage, version) in saved { op.save_storage(storage, version); }
            if let Some((storage, version)) = output { op.save_storage(storage, version); }
            op
        })
    }

    /// Record optional adjoints in the same autograd engine. `None` means the
    /// corresponding input is unused, so ordinary backward leaves its grad
    /// slot untouched. Present adjoints must match their input shape/dtype.
    /// Functional VJPs retain their explicit-zero convention for unused inputs.
    pub fn record_fn_optional<F>(self, inputs: Vec<Tensor>, backward: F) -> Tensor
    where F: Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync + 'static {
        self.record_operation(inputs, None, None, move |inputs| Op::new_optional(inputs, Box::new(backward)))
    }

    /// Same contract as `record_fn`, but the recorded op also names which
    /// kind-routed kernel produced this output, so the graph compiler can
    /// re-derive the math (fusion planning). `self` must be the raw kernel's
    /// freshly-created output; the tag must match the kernel actually run.
    pub fn record_fn_tagged<F>(
        self,
        inputs: Vec<Tensor>,
        tag: crate::dispatch::OpTag,
        backward: F,
    ) -> Tensor
    where
        F: Fn(&Tensor) -> Vec<Tensor> + Send + Sync + 'static,
    {
        self.record_operation(inputs, Some(tag), None, move |inputs| Op::new_tagged(inputs, tag, Box::new(backward)))
    }

    pub(crate) fn record_fn_forward<F>(self, inputs: Vec<Tensor>, forward: crate::autograd::ForwardOp, backward: F) -> Tensor
    where F: Fn(&Tensor) -> Vec<Tensor> + Send + Sync + 'static {
        self.record_operation(inputs, None, Some(forward), move |inputs| Op::new(inputs, Box::new(backward)))
    }

    pub(crate) fn record_fn_forward_saved_jvp<F>(self, inputs: Vec<Tensor>, forward: crate::autograd::ForwardOp, saved: Vec<Tensor>, backward: F) -> Tensor
    where F: Fn(&Tensor) -> Vec<Tensor> + Send + Sync + 'static {
        self.record_operation(inputs, None, Some(forward), move |inputs| {
            let mut op = Op::new(inputs, Box::new(backward));
            op.set_saved_jvp(saved);
            op
        })
    }

    pub(crate) fn record_fn_forward_optional<F>(self, inputs: Vec<Tensor>, forward: crate::autograd::ForwardOp, backward: F) -> Tensor
    where F: Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync + 'static {
        self.record_operation(inputs, None, Some(forward), move |inputs| Op::new_optional(inputs, Box::new(backward)))
    }

    /// Record one shared first-order operation with distinct optional cotangent slots.
    #[doc(hidden)]
    pub fn record_many_optional<F>(outputs: Vec<Tensor>, inputs: Vec<Tensor>, differentiable: &[bool], backward: F) -> Result<Vec<Tensor>>
    where F: Fn(&[Option<Tensor>]) -> Vec<Option<Tensor>> + Send + Sync + 'static {
        Self::record_many_operation(outputs, inputs, differentiable, None, backward)
    }

    pub(crate) fn record_many_forward_optional<F>(outputs: Vec<Tensor>, inputs: Vec<Tensor>, differentiable: &[bool], forward: crate::autograd::MultiForwardOp, backward: F) -> Result<Vec<Tensor>>
    where F: Fn(&[Option<Tensor>]) -> Vec<Option<Tensor>> + Send + Sync + 'static {
        Self::record_many_operation(outputs, inputs, differentiable, Some(forward), backward)
    }

    fn record_many_operation<F>(mut outputs: Vec<Tensor>, inputs: Vec<Tensor>, differentiable: &[bool], forward: Option<crate::autograd::MultiForwardOp>, backward: F) -> Result<Vec<Tensor>>
    where F: Fn(&[Option<Tensor>]) -> Vec<Option<Tensor>> + Send + Sync + 'static {
        let invalid = |message: &str| Error::Unsupported { op: "record_many_optional", msg: message.into() };
        if crate::capture::is_recording() || (forward.is_none() && crate::forward_ad::is_active()) {
            return Err(invalid("generic multi-output recording requires eager reverse mode"));
        }
        if outputs.is_empty() || outputs.len() != differentiable.len() { return Err(invalid("nonempty output and differentiability lists must agree")); }
        let mut identities = std::collections::HashSet::new();
        for (output, differentiable) in outputs.iter().zip(differentiable) {
            if !identities.insert(output.id()) || Arc::strong_count(&output.0) != 1 || Arc::weak_count(&output.0) != 0 || output.0.op.is_some() || output.requires_grad() || output.grad().is_some() {
                return Err(invalid("outputs must have distinct fresh Tensor identities"));
            }
            if *differentiable && !matches!(output.dtype(), DType::F32 | DType::F64) { return Err(invalid("differentiable output slots must have floating dtype")); }
        }
        let inputs = crate::forward_ad::recording_inputs(inputs);
        let reverse = crate::autograd::is_grad_enabled() && inputs.iter().any(crate::forward_ad::records_grad);
        let dual = crate::forward_ad::needs(&inputs);
        if !reverse && !dual {
            for output in &outputs { crate::forward_ad::record_actual_flags(output, &inputs); }
            return Ok(outputs);
        }
        let mut operations = Op::new_many(inputs, &outputs, differentiable, forward, Box::new(backward)).into_iter().map(Some).collect::<Vec<_>>();
        for (index, (output, differentiable)) in outputs.iter_mut().zip(differentiable).enumerate() {
            if reverse && *differentiable {
                let inner = Arc::get_mut(&mut output.0).expect("validated fresh output");
                inner.requires_grad.store(true, Ordering::Relaxed);
                inner.op = operations[index].take();
            }
        }
        let operation = outputs.iter().find_map(|output| output.0.op.as_ref()).or_else(|| operations.iter().find_map(Option::as_ref)).expect("recorded shared operation");
        // Capture every sibling's actual owner before dispatch can create local derivatives.
        for output in &outputs { crate::forward_ad::record_actual_flags(output, operation.inputs()); }
        if dual { crate::forward_ad::record_many(&outputs, operation); }
        Ok(outputs)
    }

    fn record_operation(mut self, inputs: Vec<Tensor>, tag: Option<crate::dispatch::OpTag>, forward: Option<crate::autograd::ForwardOp>, make: impl FnOnce(Vec<Tensor>) -> Op) -> Tensor {
        let inputs = crate::forward_ad::recording_inputs(inputs);
        let flags = inputs.iter().fold(false, |any, input| crate::forward_ad::records_grad(input) || any);
        let reverse = crate::autograd::is_grad_enabled() && flags;
        let dual = crate::forward_ad::needs(&inputs);
        let capture = crate::capture::is_recording();
        if !reverse && !dual && !capture { crate::forward_ad::record_actual_flags(&self, &inputs); return self; }
        let mut op = if reverse || dual { make(inputs) } else { Op::inference(inputs, tag) };
        op.forward = forward;
        if matches!(op.forward, Some(crate::autograd::ForwardOp::IndexedGather { .. } | crate::autograd::ForwardOp::IndexedScatter { .. })) {
            let save_input = matches!(op.forward, Some(crate::autograd::ForwardOp::IndexedGather { save_input: true, .. } | crate::autograd::ForwardOp::IndexedScatter { save_input: true, .. }));
            op.omit_saved_input_values();
            if save_input { op.save_storage(op.inputs()[0].0.storage.clone(), op.inputs()[0].version()); }
            op.save_storage(op.inputs()[1].0.storage.clone(), op.inputs()[1].version());
        }
        if matches!(op.forward, Some(crate::autograd::ForwardOp::WhereConditional)) {
            op.omit_saved_input_values();
            op.save_storage(op.inputs()[0].0.storage.clone(), op.inputs()[0].version());
        }
        if matches!(op.forward, Some(crate::autograd::ForwardOp::RoundedDiv(_))) {
            op.omit_saved_input_values();
            if op.input_edges()[1] { op.save_storage(op.inputs()[0].0.storage.clone(), op.inputs()[0].version()); }
            op.save_storage(op.inputs()[1].0.storage.clone(), op.inputs()[1].version());
        }
        if matches!(op.forward, Some(crate::autograd::ForwardOp::FloorDivide)) { op.omit_saved_input_values(); }
        if matches!(op.forward, Some(crate::autograd::ForwardOp::CumProd { .. } | crate::autograd::ForwardOp::Prod { .. })) { op.save_storage(self.0.storage.clone(), self.version()); }

        if matches!(op.forward, Some(crate::autograd::ForwardOp::CopyDualLayout { .. } | crate::autograd::ForwardOp::Alias | crate::autograd::ForwardOp::Clone { .. }
            | crate::autograd::ForwardOp::AddAlpha { .. } | crate::autograd::ForwardOp::Cast(_) | crate::autograd::ForwardOp::CumSum { .. }
            | crate::autograd::ForwardOp::Rounding(_) | crate::autograd::ForwardOp::Frac | crate::autograd::ForwardOp::Rsqrt
            | crate::autograd::ForwardOp::Reshape(_) | crate::autograd::ForwardOp::View(_) | crate::autograd::ForwardOp::Squeeze(_) | crate::autograd::ForwardOp::Unsqueeze(_)
            | crate::autograd::ForwardOp::Transpose(_, _) | crate::autograd::ForwardOp::Permute(_) | crate::autograd::ForwardOp::Expand(_) | crate::autograd::ForwardOp::Cat(_)
            | crate::autograd::ForwardOp::Slice { .. } | crate::autograd::ForwardOp::SliceAdjoint { .. }
            | crate::autograd::ForwardOp::Sum | crate::autograd::ForwardOp::Mean | crate::autograd::ForwardOp::SumDim(_, _)))
            || matches!(op.tag, Some(crate::dispatch::OpTag::Unary(crate::dispatch::UnaryKind::Sqrt | crate::dispatch::UnaryKind::Exp | crate::dispatch::UnaryKind::Tanh | crate::dispatch::UnaryKind::Sigmoid))) {
            op.omit_saved_input_values();
        }
        if reverse || capture {
            let inner = Arc::get_mut(&mut self.0).expect("fresh output is uniquely owned");
            if reverse { inner.requires_grad.store(true, Ordering::Relaxed); }
            inner.op = Some(op);
            crate::forward_ad::record_actual_flags(&self, self.0.op.as_ref().unwrap().inputs());
            if dual { crate::forward_ad::record(&self, self.0.op.as_ref().unwrap()); }
        } else if dual { crate::forward_ad::record_actual_flags(&self, op.inputs()); crate::forward_ad::record(&self, &op); }
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

    /// Element count of the underlying storage buffer (not the view). A
    /// tensor is a "whole" view of its storage exactly when `offset == 0`,
    /// it is contiguous, and this equals `numel()` - the shape in-place ops
    /// and device kernels require.
    pub(crate) fn storage_len(&self) -> usize {
        match &*self.0.storage.read() {
            Storage::F32(v) => v.len(),
            Storage::F64(v) => v.len(),
            Storage::I64(v) => v.len(),
            Storage::Bool(v) => v.len(),
            Storage::F16(v) | Storage::BF16(v) => v.len(),
            Storage::Device(b) => b.len(),
            Storage::DeviceI64(b) => b.len(),
        }
    }

    /// Test-only seam to simulate an in-place mutation directly, so the
    /// version-mismatch assertion in `backward` is exercisable in isolation
    /// from the real in-place ops.
    #[doc(hidden)]
    pub fn _bump_version_for_test(&self) {
        self.bump_version();
    }

    /// Test-only accessor for the storage version counter.
    #[doc(hidden)]
    pub fn _version(&self) -> u64 {
        self.version()
    }

    /// Test-only identity of the storage allocation: equal values mean two
    /// tensors share one `StorageCell` (and an unchanged value across an
    /// optimizer step proves the step mutated in place instead of
    /// reallocating).
    #[doc(hidden)]
    pub fn _storage_ptr(&self) -> usize {
        Arc::as_ptr(&self.0.storage) as *const () as usize
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

/// Guard for kernels that only implement F32. Typed operations dispatch before
/// this guard; unsupported dtypes must never be silently narrowed at entry.
pub(crate) fn check_f32(op: &'static str, t: &Tensor) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::DtypeMismatch {
            op,
            expected: DType::F32,
            got: t.dtype(),
        });
    }
    Ok(())
}

pub(crate) fn raw_unary_k(a: &Tensor, kind: UnaryKind) -> Result<Tensor> {
    if matches!(a.dtype(), DType::F64 | DType::I64) { return crate::typed_math::unary(a, kind); }
    check_f32("unary", a)?;
    if a.device_resident_whole() {
        let backend = backend_for(a.0.device)?;
        let g = a.0.storage.read();
        let Storage::Device(buf) = &*g else {
            unreachable!()
        };
        let out = backend.unary_dev(kind, buf.as_ref())?;
        return Ok(device_leaf(out, &a.0.shape, a.0.device));
    }
    let host_stride = if a.device() == Device::Cpu {
        let count = checked_numel("unary", a.shape())?;
        if count > isize::MAX as usize / 4 { return Err(Error::InvalidShape { op: "unary", msg: "pointwise output exceeds allocation capacity".into() }); }
        Some(crate::typed_math::unary_pointwise_layout(a))
    } else { None };
    // Contiguous host f32 (any offset): slice straight into the backend, no
    // materializing copy - the read side of the ~3x-bytes-moved bug this
    // function used to have via `a.to_vec()`.
    {
        let g = a.0.storage.read();
        if let Storage::F32(v) = &*g {
            if a.0.device == Device::Cpu && a.is_contiguous()
                && host_stride.as_ref().is_some_and(|stride| *stride == default_strides(a.shape())) {
                let n = a.numel();
                let cpu = backend_for(Device::Cpu)?;
                return Tensor::from_vec(
                    cpu.unary(kind, if n == 0 { &[] } else { &v[a.0.offset..a.0.offset + n] }),
                    &a.0.shape,
                );
            }
        }
    }
    let cpu = backend_for(Device::Cpu)?;
    let physical = if let Some(stride) = &host_stride { crate::typed_math::pointwise_source(a, a.shape(), stride)? } else { a.clone() };
    let va = physical.to_vec_pooled();
    let out = cpu.unary(kind, &va);
    crate::pool::give(va);
    let raw = Tensor::from_vec(out, a.shape())?;
    if let Some(stride) = host_stride {
        Ok(Tensor::from_parts(raw.0.storage.clone(), a.shape().to_vec(), stride, 0, Device::Cpu, false, None))
    } else { raw.to_device(a.0.device) }
}

/// Wrap a backend-produced buffer as a contiguous detached device tensor.
pub(crate) fn device_leaf(buf: Box<dyn DeviceBuffer>, shape: &[usize], device: Device) -> Tensor {
    Tensor::from_parts(
        Arc::new(StorageCell::new(Storage::Device(Arc::from(buf)))),
        shape.to_vec(),
        default_strides(shape),
        0,
        device,
        false,
        None,
    )
}

pub(crate) fn raw_structural_binary(op: &'static str, a: &Tensor, b: &Tensor, kind: BinaryKind) -> Result<Option<Tensor>> {
    let (az, bz) = (a.is_structural_zero(), b.is_structural_zero());
    if (!az && !bz) || matches!(kind, BinaryKind::ReluBackward) { return Ok(None); }
    if a.dtype() != b.dtype() { return Err(Error::DtypeMismatch { op, expected: a.dtype(), got: b.dtype() }); }
    if a.device() != b.device() { return Err(Error::DeviceMismatch { op, lhs: a.device(), rhs: b.device() }); }
    if a.device() != Device::Cpu || crate::capture::is_recording() {
        return Err(Error::Unsupported { op, msg: "structural-zero arithmetic requires eager CPU execution".into() });
    }
    let shape = broadcast_shapes(op, a.shape(), b.shape())?;
    let width = match a.dtype() {
        DType::F32 => 4, DType::F64 => 8,
        dtype => return Err(Error::DtypeMismatch { op, expected: DType::F32, got: dtype }),
    };
    if checked_numel(op, &shape)? > isize::MAX as usize / width {
        return Err(Error::InvalidShape { op, msg: "structural-zero result exceeds typed allocation capacity".into() });
    }
    let zero = matches!(kind, BinaryKind::Mul) || az && bz && matches!(kind, BinaryKind::Add | BinaryKind::Sub)
        || az && !bz && matches!(kind, BinaryKind::Div);
    if zero { return Tensor::structural_zero(&shape, a.dtype(), a.device()).map(Some); }
    let output = match kind {
        BinaryKind::Add | BinaryKind::Sub => {
            let ordinary = if az { b } else { a }.broadcast_to(&shape)?;
            let stride = if az && matches!(kind, BinaryKind::Sub) {
                crate::typed_math::binary_pointwise_layout(&ordinary, &Tensor::scalar(-1.).to_dtype(a.dtype()), &shape, a.dtype())
            } else { crate::typed_math::copy_pointwise_layout(&ordinary) };
            let physical = crate::typed_math::pointwise_source(&ordinary, &shape, &stride)?;
            let owned = if az && matches!(kind, BinaryKind::Sub) { raw_unary_k(&physical, UnaryKind::Neg)? }
                else { physical.detach_copy_same_dtype() };
            Tensor::from_parts(owned.0.storage.clone(), shape.clone(), stride, 0, Device::Cpu, false, None)
        }
        BinaryKind::Div => {
            let value = if az { f64::NAN } else { f64::INFINITY };
            match a.dtype() {
                DType::F32 => Tensor::from_vec(crate::typed_math::filled(op, &shape, value as f32)?, &shape)?,
                DType::F64 => Tensor::from_vec_f64(crate::typed_math::filled(op, &shape, value)?, &shape)?,
                dtype => return Err(Error::DtypeMismatch { op, expected: DType::F32, got: dtype }),
            }
        }
        BinaryKind::Mul | BinaryKind::ReluBackward => unreachable!(),
    };
    Ok(Some(output))
}

pub(crate) fn raw_binary_k(
    op: &'static str,
    a: &Tensor,
    b: &Tensor,
    kind: BinaryKind,
) -> Result<Tensor> {
    if let Some(zero_result) = raw_structural_binary(op, a, b, kind)? { return Ok(zero_result); }
    if matches!(a.dtype(), DType::F64 | DType::I64) { return crate::typed_math::binary(op, a, b, kind); }
    check_f32(op, a)?;
    check_f32(op, b)?;
    if a.0.device != b.0.device {
        return Err(Error::DeviceMismatch {
            op,
            lhs: a.0.device,
            rhs: b.0.device,
        });
    }
    if a.device_resident_whole() && b.device_resident_whole() {
        let backend = backend_for(a.0.device)?;
        let pair = PairGuard::new(a, b);
        let (sa, sb) = pair.get();
        let (Storage::Device(ba), Storage::Device(bb)) = (sa, sb) else {
            unreachable!()
        };
        if a.0.shape == b.0.shape {
            let out = backend.binary_dev(kind, ba.as_ref(), bb.as_ref())?;
            return Ok(device_leaf(out, &a.0.shape, a.0.device));
        }
        let out_shape = broadcast_shapes(op, &a.0.shape, &b.0.shape)?;
        let out = backend.binary_bc_dev(
            kind,
            ba.as_ref(),
            &a.0.shape,
            bb.as_ref(),
            &b.0.shape,
            &out_shape,
        )?;
        return Ok(device_leaf(out, &out_shape, a.0.device));
    }
    let out_shape = broadcast_shapes(op, &a.0.shape, &b.0.shape)?;
    let host_stride = if a.device() == Device::Cpu {
        let count = checked_numel(op, &out_shape)?;
        if count > isize::MAX as usize / 4 { return Err(Error::InvalidShape { op, msg: "pointwise output exceeds allocation capacity".into() }); }
        Some(crate::typed_math::binary_pointwise_layout(a, b, &out_shape, DType::F32))
    } else { None };
    // Contiguous host f32 operands: skip the copy-then-compute path.
    {
        let pair = PairGuard::new(a, b);
        let (sa_st, sb_st) = pair.get();
        if let (Storage::F32(va), Storage::F32(vb)) = (sa_st, sb_st) {
            if a.0.device == Device::Cpu && a.is_contiguous() && b.is_contiguous()
                && host_stride.as_ref().is_some_and(|stride| *stride == default_strides(&out_shape)) {
                let cpu = backend_for(Device::Cpu)?;
                // Equal shapes: slice both operands directly (any offset).
                if a.0.shape == b.0.shape {
                    let n = a.numel();
                    let sa = if n == 0 { &[] } else { &va[a.0.offset..a.0.offset + n] };
                    let sb = if n == 0 { &[] } else { &vb[b.0.offset..b.0.offset + n] };
                    return Tensor::from_vec(cpu.binary(kind, sa, sb), &a.0.shape);
                }
                // Different shapes: hand the backend the whole unmaterialized
                // buffers (e.g. a [4096] bias against a [4096,4096] output) so a
                // smaller operand is read repeatedly instead of expanded first.
                // Restricted to zero-offset whole buffers - an offset view still
                // takes the materializing fallback below.
                if a.0.offset == 0
                    && b.0.offset == 0
                    && va.len() == a.numel()
                    && vb.len() == b.numel()
                {
                    let out_shape = broadcast_shapes(op, &a.0.shape, &b.0.shape)?;
                    let out = cpu.binary_bc(kind, va, &a.0.shape, vb, &b.0.shape, &out_shape);
                    return Tensor::from_vec(out, &out_shape);
                }
            }
        }
    }
    let (left, right) = if let Some(stride) = &host_stride {
        (crate::typed_math::pointwise_source(a, &out_shape, stride)?, crate::typed_math::pointwise_source(b, &out_shape, stride)?)
    } else { (a.broadcast_to(&out_shape)?, b.broadcast_to(&out_shape)?) };
    let va = left.to_vec_pooled();
    let vb = right.to_vec_pooled();
    let cpu = backend_for(Device::Cpu)?;
    let out = cpu.binary(kind, &va, &vb);
    crate::pool::give(va);
    crate::pool::give(vb);
    let raw = Tensor::from_vec(out, &out_shape)?;
    if let Some(stride) = host_stride {
        Ok(Tensor::from_parts(raw.0.storage.clone(), out_shape, stride, 0, Device::Cpu, false, None))
    } else { raw.to_device(a.0.device) }
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
        return Err(Error::DeviceMismatch {
            op,
            lhs: a.0.device,
            rhs: b.0.device,
        });
    }
    let out_shape = broadcast_shapes(op, &a.0.shape, &b.0.shape)?;
    let va = a.broadcast_to(&out_shape)?.to_vec_pooled();
    let vb = b.broadcast_to(&out_shape)?.to_vec_pooled();
    let mut data = crate::pool::take_uninit(va.len());
    for ((o, &x), &y) in data.iter_mut().zip(&va).zip(&vb) {
        *o = f(x, y);
    }
    crate::pool::give(va);
    crate::pool::give(vb);
    Tensor::from_vec(data, &out_shape)
}

/// 2-D matmul: (m,k) @ (k,n) -> (m,n), routed through the device's backend
/// (the CPU backend consults the swappable kernel pointer, so a backend crate
/// can still swap in a faster kernel). Higher ranks are a follow-up.
pub(crate) fn raw_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() == DType::F64 { return crate::typed_math::matmul(a, b, false); }
    check_f32("matmul", a)?;
    check_f32("matmul", b)?;
    if a.0.device != b.0.device {
        return Err(Error::DeviceMismatch {
            op: "matmul",
            lhs: a.0.device,
            rhs: b.0.device,
        });
    }
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(Error::Unsupported {
            op: "matmul",
            msg: format!(
                "only 2-D supported in MVP, got {:?} and {:?}",
                a.0.shape, b.0.shape
            ),
        });
    }
    let (m, k) = (a.0.shape[0], a.0.shape[1]);
    let (k2, n) = (b.0.shape[0], b.0.shape[1]);
    if k != k2 {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            lhs: a.0.shape.clone(),
            rhs: b.0.shape.clone(),
        });
    }
    if a.device_resident_whole() && b.device_resident_whole() {
        let backend = backend_for(a.0.device)?;
        let pair = PairGuard::new(a, b);
        let (sa, sb) = pair.get();
        let (Storage::Device(ba), Storage::Device(bb)) = (sa, sb) else {
            unreachable!()
        };
        let out = backend.matmul_dev(ba.as_ref(), bb.as_ref(), m, k, n, false, false)?;
        return Ok(device_leaf(out, &[m, n], a.0.device));
    }
    let va = a.to_vec_pooled();
    let vb = b.to_vec_pooled();
    let cpu = backend_for(Device::Cpu)?;
    let out = cpu.matmul(&va, &vb, m, k, n);
    crate::pool::give(va);
    crate::pool::give(vb);
    Tensor::from_vec(out, &[m, n])?.to_device(a.0.device)
}

/// Full-tensor reduction on a resident device tensor, if it is one.
/// Only the default missing-seam sentinel permits fallback: backends may also
/// encode operational failures as Unsupported. Infallible callers expose those.
pub(crate) fn raw_reduce_dev(t: &Tensor, kind: ReduceKind) -> Option<Tensor> {
    if !t.device_resident_whole() {
        return None;
    }
    let backend = backend_for(t.0.device).expect("device tensor implies registered backend");
    let g = t.0.storage.read();
    let Storage::Device(buf) = &*g else {
        unreachable!()
    };
    match backend.reduce_dev(kind, buf.as_ref()) {
        Ok(out) => Some(device_leaf(out, &[], t.0.device)),
        Err(Error::Unsupported { op: "reduce_dev", ref msg })
            if msg == "backend does not implement device-resident storage" => None,
        Err(err) => panic!("device reduction failed: {err}"),
    }
}

/// Matmul with logical transpose flags: computes op(a) @ op(b) where op is
/// transpose when the flag is set. Device operands stay resident (backends
/// take the flags natively, e.g. cuBLAS); host operands go through transpose
/// views. Used by matmul's backward so gradients never materialize transposes.
pub(crate) fn raw_matmul_t(a: &Tensor, b: &Tensor, ta: bool, tb: bool) -> Result<Tensor> {
    if a.device_resident_whole() && b.device_resident_whole() && a.ndim() == 2 && b.ndim() == 2 {
        let (m, k) = if ta {
            (a.0.shape[1], a.0.shape[0])
        } else {
            (a.0.shape[0], a.0.shape[1])
        };
        let (k2, n) = if tb {
            (b.0.shape[1], b.0.shape[0])
        } else {
            (b.0.shape[0], b.0.shape[1])
        };
        if k != k2 {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                lhs: a.0.shape.clone(),
                rhs: b.0.shape.clone(),
            });
        }
        let backend = backend_for(a.0.device)?;
        let pair = PairGuard::new(a, b);
        let (sa, sb) = pair.get();
        let (Storage::Device(ba), Storage::Device(bb)) = (sa, sb) else {
            unreachable!()
        };
        let out = backend.matmul_dev(ba.as_ref(), bb.as_ref(), m, k, n, ta, tb)?;
        return Ok(device_leaf(out, &[m, n], a.0.device));
    }
    let av = if ta {
        a.transpose_view(0, 1)?
    } else {
        a.clone()
    };
    let bv = if tb {
        b.transpose_view(0, 1)?
    } else {
        b.clone()
    };
    raw_matmul(&av, &bv)
}

/// Sum over one dim, matching PyTorch's keepdim semantics.
pub(crate) fn raw_sum_dim(t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    try_raw_sum_dim(t, dim, keepdim).expect("sum_dim failed")
}

/// Fallible seam for forward callers; legacy backward closures cannot return errors.
pub(crate) fn try_raw_sum_dim(t: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if matches!(t.dtype(), DType::F64 | DType::I64) { return crate::typed_math::sum(t, Some(dim), keepdim); }
    check_f32("sum_dim", t)?;
    let in_shape = t.0.shape.clone();
    let ndim = in_shape.len();
    if dim >= ndim {
        return Err(Error::InvalidShape { op: "sum_dim", msg: format!("dim {dim} out of range for rank {ndim}") });
    }
    if t.device_resident_whole() {
        let backend = backend_for(t.0.device)?;
        let g = t.0.storage.read();
        let Storage::Device(buf) = &*g else {
            unreachable!()
        };
        let out = match backend.sum_dim_dev(buf.as_ref(), &in_shape, dim) {
            Ok(out) => Some(out),
            Err(Error::Unsupported { op: "sum_dim_dev", ref msg })
                if msg == "backend does not implement device-resident storage" => None,
            Err(err) => return Err(err),
        };
        if let Some(out) = out {
            let mut keep_shape = in_shape.clone();
            keep_shape[dim] = 1;
            let out_shape: Vec<usize> = if keepdim {
                keep_shape
            } else {
                in_shape
                    .iter()
                    .enumerate()
                    .filter(|(d, _)| *d != dim)
                    .map(|(_, &s)| s)
                    .collect()
            };
            return Ok(device_leaf(out, &out_shape, t.0.device));
        }
    }
    let cpu = backend_for(Device::Cpu)?;
    let out = {
        let storage = t.0.storage.read();
        match &*storage {
            Storage::Device(buf) => cpu.sum_dim(&t.gather_view(&backend_for(t.device())?.copy_to_host(buf.as_ref())?), &in_shape, dim),
            Storage::DeviceI64(buf) => {
                let v: Vec<f32> = t.gather_view(&backend_for(t.device())?.copy_i64_to_host(buf.as_ref())?).into_iter().map(|x| x as f32).collect();
                cpu.sum_dim(&v, &in_shape, dim)
            }
            _ => { drop(storage); t.with_host_f32(|v| cpu.sum_dim(v, &in_shape, dim)) }
        }
    };
    let mut keep_shape = in_shape.clone();
    keep_shape[dim] = 1;
    let out_shape: Vec<usize> = if keepdim {
        keep_shape
    } else {
        in_shape
            .iter()
            .enumerate()
            .filter(|(d, _)| *d != dim)
            .map(|(_, &s)| s)
            .collect()
    };
    Tensor::from_vec(out, &out_shape)
}

/// Row-wise softmax/log_softmax over the last dim of a whole contiguous device
/// tensor, via the backend's `softmax_dev`/`log_softmax_dev` kernels. Returns
/// None when the tensor is not resident or the reduced dim is not last (those
/// keep the host fallback in ops_ext).
pub(crate) fn raw_row_softmax(t: &Tensor, dim: usize, log: bool) -> Option<Tensor> {
    if !t.device_resident_whole() || dim + 1 != t.ndim() || !t.is_contiguous() {
        return None;
    }
    let backend = backend_for(t.0.device).expect("device tensor implies registered backend");
    let g = t.0.storage.read();
    let Storage::Device(buf) = &*g else {
        unreachable!()
    };
    let cols = t.0.shape[dim];
    let rows = t.0.shape[..dim].iter().product();
    // A backend without the row kernels (or a launch failure) keeps the
    // documented fallback: ops_ext computes on the host and re-uploads.
    let out = if log {
        backend.log_softmax_dev(buf.as_ref(), rows, cols)
    } else {
        backend.softmax_dev(buf.as_ref(), rows, cols)
    }
    .ok()?;
    Some(device_leaf(out, &t.0.shape, t.0.device))
}

/// Reduce empty adjoints without allocating nonempty intermediate dimensions.
pub(crate) fn sum_empty_to_shape(g: &Tensor, target: &[usize]) -> Result<Tensor> {
    debug_assert_eq!(g.numel(), 0);
    if g.shape() == target { return Ok(g.clone()); }
    // Match ordinary reduction layouts without removing a protective zero axis
    // before huge nonzero extents. Above the existing 2^20-element work budget,
    // retain the virtual-zero extension: values/AD stay exact, strides may differ.
    let count = crate::shape::checked_numel("empty broadcast adjoint", target)?;
    if count <= 1 << 20 {
        let leading = g.ndim() - target.len();
        let axes = (0..g.ndim()).filter(|&axis| axis < leading || target[axis - leading] == 1 && g.shape()[axis] != 1).collect::<Vec<_>>();
        let mut value = g.clone();
        for zero in [false, true] {
            for &axis in &axes {
                if (g.shape()[axis] == 0) == zero { value = value.sum_dim(axis, true)?; }
            }
        }
        return value.reshape(target);
    }
    g.sum().reshape(&vec![1; target.len()])?.expand(target)
}

#[cfg(test)]
#[path = "empty_reduction_layout_tests.rs"]
mod empty_reduction_layout_tests;

/// Sum dimensions expanded during a forward broadcast back to `target`.
pub(crate) fn unbroadcast(g: &Tensor, target: &[usize]) -> Tensor {
    if g.numel() == 0 { return sum_empty_to_shape(g, target).expect("valid empty broadcast adjoint"); }
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
#[path = "layout_lock_tests.rs"]
mod layout_lock_tests;

#[cfg(test)]
mod detached_layout_tests {
    use super::*;
    use crate::dispatch::{Backend, register_backend};
    use std::sync::Mutex;
    const DEV: Device = Device::Cuda(91);
    static SERIAL: Mutex<()> = Mutex::new(());
    struct Buf(Vec<f32>);
    impl DeviceBuffer for Buf {
        fn device(&self) -> Device { DEV }
        fn len(&self) -> usize { self.0.len() }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }
    struct Layout(AtomicUsize);
    impl Backend for Layout {
        fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { unreachable!() }
        fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { unreachable!() }
        fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { unreachable!() }
        fn copy_to_host(&self, _: &dyn DeviceBuffer) -> Result<Vec<f32>> { panic!("unexpected download") }
        fn materialize_dev(&self, buf: &dyn DeviceBuffer, shape: &[usize], strides: &[usize], offset: usize) -> Result<Box<dyn DeviceBuffer>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            let data = &buf.as_any().downcast_ref::<Buf>().unwrap().0;
            let values = (0..shape.iter().product()).map(|mut flat| {
                let mut at = offset;
                for d in (0..shape.len()).rev() { at += flat % shape[d] * strides[d]; flat /= shape[d]; }
                data[at]
            }).collect();
            Ok(Box::new(Buf(values)))
        }
    }
    #[test]
    fn detach_internal_offsets_broadcasts_and_prefixes() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let backend = Arc::new(Layout(AtomicUsize::new(0)));
        register_backend(DEV, backend.clone());
        let base = device_leaf(Box::new(Buf(vec![1., 2., 3., 4., 5., 6.])), &[2, 3], DEV);
        base.bump_version();
        for (shape, stride, offset, expected) in [
            (vec![2], vec![1], 0, vec![1., 2.]),
            (vec![2], vec![2], 1, vec![2., 4.]),
            (vec![0], vec![1], 6, vec![]),
            (vec![], vec![], 2, vec![3.]),
            (vec![2, 3], vec![0, 1], 0, vec![1., 2., 3., 1., 2., 3.]),
        ] {
            let view = Tensor::from_parts(base.0.storage.clone(), shape, stride, offset, DEV, false, None);
            let before = backend.0.load(Ordering::SeqCst);
            let result = view.detach_copy();
            assert_eq!(backend.0.load(Ordering::SeqCst), before + 1);
            assert!(result.device_resident_whole());
            assert_eq!(result.storage_len(), result.numel());
            assert!(!result.requires_grad());
            assert!(result.0.op.is_none());
            assert_eq!(result.version(), 0);
            assert_eq!(view.version(), 1);
            assert!(!Arc::ptr_eq(&result.0.storage, &view.0.storage));
            let storage = result.0.storage.read();
            let Storage::Device(buf) = &*storage else { panic!("not resident") };
            assert_eq!(buf.as_any().downcast_ref::<Buf>().unwrap().0, expected);
        }
        let row = device_leaf(Box::new(Buf(vec![1., 2., 3.])), &[1, 3], DEV);
        let broadcast = row.broadcast_to(&[2, 3]).unwrap().detach_copy();
        assert_eq!(broadcast.storage_len(), 6);
    }
}

#[cfg(test)]
mod tests {
    // raw_unary_k/raw_binary_k's fast paths slice storage at `tensor.offset`,
    // but no public op (narrow/as_strided; see docs/CAPABILITY.md 2.2) yet
    // produces a nonzero-offset tensor, so this exercises the offset
    // arithmetic directly via the pub(crate) constructor an external
    // integration test cannot reach.
    use super::*;

    #[test]
    fn fallible_sum_dim_preserves_backend_error() {
        use crate::dispatch::{Backend, register_backend};
        struct Failure;
        impl Backend for Failure {
            fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { unreachable!() }
            fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { unreachable!() }
            fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { unreachable!() }
            fn sum_dim_dev(&self, _: &dyn DeviceBuffer, _: &[usize], _: usize) -> Result<Box<dyn DeviceBuffer>> {
                Err(Error::Io { op: "sum_dim_dev", msg: "injected launch failure".into() })
            }
        }
        struct Buf;
        impl DeviceBuffer for Buf {
            fn device(&self) -> Device { Device::Cuda(88) }
            fn len(&self) -> usize { 6 }
            fn as_any(&self) -> &dyn std::any::Any { self }
        }
        register_backend(Device::Cuda(88), Arc::new(Failure));
        let t = device_leaf(Box::new(Buf), &[2, 3], Device::Cuda(88));
        assert_eq!(try_raw_sum_dim(&t, 1, false).err(), Some(Error::Io {
            op: "sum_dim_dev", msg: "injected launch failure".into(),
        }));
        assert!(matches!(try_raw_sum_dim(&t, 2, false), Err(Error::InvalidShape { .. })));
    }

    fn offset_view(v: Vec<f32>, offset: usize, shape: &[usize]) -> Tensor {
        Tensor::from_parts(
            Arc::new(StorageCell::new(Storage::F32(v))),
            shape.to_vec(),
            default_strides(shape),
            offset,
            Device::Cpu,
            false,
            None,
        )
    }

    #[test]
    fn raw_unary_k_reads_from_the_view_offset_not_index_zero() {
        let a = offset_view(vec![9.0, -1.0, -2.0, 3.0, 9.0], 1, &[3]);
        assert_eq!(
            raw_unary_k(&a, UnaryKind::Relu).unwrap().to_vec(),
            vec![0.0, 0.0, 3.0]
        );
    }

    #[test]
    fn raw_binary_k_reads_both_operands_from_their_view_offset() {
        let a = offset_view(vec![100.0, 1.0, 2.0, 3.0], 1, &[3]);
        let b = offset_view(vec![100.0, 10.0, 20.0, 30.0], 1, &[3]);
        assert_eq!(
            raw_binary_k("add", &a, &b, BinaryKind::Add)
                .unwrap()
                .to_vec(),
            vec![11.0, 22.0, 33.0]
        );
    }
}
