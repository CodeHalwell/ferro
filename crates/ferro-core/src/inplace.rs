//! In-place tensor operations: the first mutation seam in an otherwise
//! immutable engine, built on the storage version counters (see
//! docs/CAPABILITY.md 4.1-4.2). Two layers:
//!
//! - `raw_*` seams mutate a whole contiguous f32 buffer through the device's
//!   backend and bump the storage version, with NO autograd gates: the
//!   optimizers call these on grad-requiring leaf parameters, exactly like
//!   torch's `with no_grad()` step. A graph that saved the mutated storage
//!   errors loudly on its next backward (the version check), never silently.
//! - The public `Tensor` methods add the safety gates: the target must not
//!   require grad and must carry no op history (rebinding a live graph node
//!   is a later capability), and a DEVICE target's storage must be uniquely
//!   referenced - device `detach_copy` shares storage with backward-closure
//!   snapshots, so mutating shared device storage could silently poison a
//!   saved output that no version check covers. Host snapshots always own
//!   fresh allocations, so host targets may be freely aliased by views:
//!   mutation is visible through every view (torch semantics) and any graph
//!   input among them is version-protected.
//! - `copy_leaf_from` and scalar fills permit a grad-requiring leaf under explicit
//!   no-grad, retaining storage/layout/history gates and shared version bumps.
//!   Other public mutation methods retain their original grad rejection.
//! - CPU copies preserve all storage types; scalar fills additionally support
//!   F64/I64/Bool. Arithmetic mutation and device kernels retain f32 storage.
//!
//! Locking protocol: layout metadata (shape/stride/offset) is immutable, so
//! preconditions are checked lock-free; then every needed storage lock is
//! taken in one deterministic sweep - multi-buffer ops acquire in global
//! address order and never lock one cell twice (aliased operands collapse to
//! one guard or pre-materialize) - so concurrent in-place ops cannot
//! deadlock, only interleave.

use std::sync::Arc;

use crate::device::Device;
use crate::dispatch::{backend_for, AdamWStep, BinaryKind};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{PairGuard, Storage, Tensor};

/// Exact scalar types accepted by checked fill; no implicit narrowing occurs.
#[derive(Clone, Copy, Debug)]
pub enum FillValue { F32(f32), F64(f64), I64(i64), Bool(bool) }

/// Arithmetic mutation requires f32 elements in a whole contiguous buffer.
fn check_dst(op: &'static str, t: &Tensor) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::DtypeMismatch {
            op,
            expected: DType::F32,
            got: t.dtype(),
        });
    }
    check_dst_layout(op, t)
}

fn check_dst_layout(op: &'static str, t: &Tensor) -> Result<()> {
    if crate::forward_ad::is_active() {
        return Err(Error::Unsupported { op, msg: "in-place mutation during forward AD is unsupported".into() });
    }
    if t.is_structural_zero() {
        return Err(Error::Unsupported { op, msg: "structural zero storage is immutable; clone it before mutation".into() });
    }
    if t.0.offset != 0 || !t.is_contiguous() || t.storage_len() != t.numel() {
        return Err(Error::Unsupported {
            op,
            msg: "in-place destination must be a whole contiguous tensor (offset 0, \
                  row-major, covering its entire storage); materialize the view first"
                .to_string(),
        });
    }
    Ok(())
}

/// Whether `t`'s storage is a device buffer (as opposed to a host Vec).
/// The dst contract was already checked, so f32-ness is settled.
fn is_device(t: &Tensor) -> bool {
    t.0.device != Device::Cpu
}

/// dst = value.
pub(crate) fn raw_fill_(op: &'static str, dst: &Tensor, value: f32) -> Result<()> {
    check_dst(op, dst)?;
    let backend = backend_for(dst.0.device)?;
    if is_device(dst) {
        let g = dst.0.storage.read();
        let Storage::Device(b) = &*g else { unreachable!() };
        backend.fill_inplace_dev(b.as_ref(), value)?;
    } else {
        let mut g = dst.0.storage.write();
        let Storage::F32(v) = &mut *g else { unreachable!() };
        backend.fill_inplace(v, value);
    }
    dst.bump_version();
    Ok(())
}

fn raw_fill_value_(op: &'static str, dst: &Tensor, value: FillValue) -> Result<()> {
    let dtype = match value { FillValue::F32(_) => DType::F32, FillValue::F64(_) => DType::F64, FillValue::I64(_) => DType::I64, FillValue::Bool(_) => DType::Bool };
    if dst.dtype() != dtype { return Err(Error::DtypeMismatch { op, expected: dst.dtype(), got: dtype }); }
    if let FillValue::F32(value) = value { return raw_fill_(op, dst, value); }
    if dst.device() != Device::Cpu {
        return Err(Error::Unsupported { op, msg: "typed scalar fills require CPU storage; device fills require float32".into() });
    }
    check_dst_layout(op, dst)?;
    {
        let mut storage = dst.0.storage.write();
        match (&mut *storage, value) {
            (Storage::F64(values), FillValue::F64(value)) => values.fill(value),
            (Storage::I64(values), FillValue::I64(value)) => values.fill(value),
            (Storage::Bool(values), FillValue::Bool(value)) => values.fill(value),
            _ => unreachable!("validated scalar and storage dtype"),
        }
    }
    dst.bump_version();
    Ok(())
}

/// dst = dst * mul + add. `mul_scalar_` passes add = -0.0 and `add_scalar_`
/// passes mul = 1.0: both identities are exact in IEEE f32 (x + -0.0 == x
/// and 1.0 * x == x for every x), so the shared kernel changes no results.
pub(crate) fn raw_affine_(op: &'static str, dst: &Tensor, mul: f32, add: f32) -> Result<()> {
    check_dst(op, dst)?;
    let backend = backend_for(dst.0.device)?;
    if is_device(dst) {
        let g = dst.0.storage.read();
        let Storage::Device(b) = &*g else { unreachable!() };
        backend.affine_inplace_dev(b.as_ref(), mul, add)?;
    } else {
        let mut g = dst.0.storage.write();
        let Storage::F32(v) = &mut *g else { unreachable!() };
        backend.affine_inplace(v, mul, add);
    }
    dst.bump_version();
    Ok(())
}

/// Shared shape/device/dtype validation for the two-operand mutations.
fn check_src(op: &'static str, dst: &Tensor, src: &Tensor) -> Result<()> {
    if src.dtype() != dst.dtype() {
        return Err(Error::DtypeMismatch {
            op,
            expected: dst.dtype(),
            got: src.dtype(),
        });
    }
    if dst.0.device != src.0.device {
        return Err(Error::DeviceMismatch {
            op,
            lhs: dst.0.device,
            rhs: src.0.device,
        });
    }
    if dst.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            op,
            lhs: dst.0.shape.clone(),
            rhs: src.0.shape.clone(),
        });
    }
    Ok(())
}

/// The zero-element case of CPU floating addition in place. No values need
/// reading or writing, but a successful mutation still advances the shared
/// version. The autograd input buffer calls this only for an exclusively owned
/// ordinary cotangent; this is not a public arbitrary-view mutation API.
pub(crate) fn raw_empty_add_(dst: &Tensor, src: &Tensor) -> Result<()> {
    let op = "empty cotangent accumulation";
    check_src(op, dst, src)?;
    if crate::autograd::is_grad_enabled() || crate::forward_ad::is_active()
        || crate::capture::is_recording() || dst.device() != Device::Cpu
        || !matches!(dst.dtype(), DType::F32 | DType::F64)
        || dst.numel() != 0 || dst.is_structural_zero()
    {
        return Err(Error::Unsupported { op, msg: "requires an ordinary empty CPU floating destination outside gradient recording".into() });
    }
    // Empty strided tensors have no addressed elements, including an empty
    // view with a nonzero offset. Do not materialize either underlying storage.
    let guard = dst.0.storage.write();
    debug_assert!(matches!(&*guard, Storage::F32(_) | Storage::F64(_)));
    dst.bump_version();
    drop(guard);
    Ok(())
}

/// Run `host` / `dev` against dst's buffer and src's data, handling the
/// three src cases: aliased with dst (pre-materialize, then single write
/// guard), direct (whole/contiguous, guards in address order), or a strided
/// host view (materialize). Device dst requires a whole-resident device src.
fn with_dst_src(
    op: &'static str,
    dst: &Tensor,
    src: &Tensor,
    host: impl Fn(&mut [f32], &[f32]),
    dev: impl Fn(&dyn crate::dispatch::DeviceBuffer, &dyn crate::dispatch::DeviceBuffer) -> Result<()>,
) -> Result<()> {
    check_dst(op, dst)?;
    check_src(op, dst, src)?;
    if is_device(dst) {
        if !src.device_resident_whole() {
            return Err(Error::Unsupported {
                op,
                msg: "in-place device ops need a whole contiguous device-resident \
                      source; materialize the view first"
                    .to_string(),
            });
        }
        // One guard when aliased (same cell may not be read-locked twice on
        // one thread); the kernels only combine same-index elements, so an
        // aliased dst/src pair is well-defined.
        let guards = PairGuard::new(dst, src);
        let (Storage::Device(bd), Storage::Device(bs)) = guards.get() else {
            unreachable!()
        };
        dev(bd.as_ref(), bs.as_ref())?;
    } else if Arc::ptr_eq(&dst.0.storage, &src.0.storage) {
        // Aliased host operands: materialize src first (read guard released
        // before the write guard is taken).
        let data = src.to_vec();
        let mut g = dst.0.storage.write();
        let Storage::F32(v) = &mut *g else { unreachable!() };
        host(v, &data);
    } else {
        let direct = {
            let g = src.0.storage.read();
            matches!(&*g, Storage::F32(_)) && src.is_contiguous()
        };
        if direct {
            // Write and read guards on distinct cells, acquired in global
            // address order so a concurrent b.add_(&a) cannot deadlock this
            // a.add_(&b).
            let dst_first = Arc::as_ptr(&dst.0.storage) < Arc::as_ptr(&src.0.storage);
            let (mut gd, gs);
            if dst_first {
                gd = dst.0.storage.write();
                gs = src.0.storage.read();
            } else {
                gs = src.0.storage.read();
                gd = dst.0.storage.write();
            }
            let Storage::F32(v) = &mut *gd else { unreachable!() };
            let Storage::F32(s) = &*gs else { unreachable!() };
            let n = src.numel();
            host(v, if n == 0 { &[] } else { &s[src.0.offset..src.0.offset + n] });
        } else {
            let data = src.to_vec();
            let mut g = dst.0.storage.write();
            let Storage::F32(v) = &mut *g else { unreachable!() };
            host(v, &data);
        }
    }
    dst.bump_version();
    Ok(())
}

/// dst = dst kind src (same shape, same device).
pub(crate) fn raw_binary_(
    op: &'static str,
    kind: BinaryKind,
    dst: &Tensor,
    src: &Tensor,
) -> Result<()> {
    let backend = backend_for(dst.0.device)?;
    let b2 = backend.clone();
    with_dst_src(
        op,
        dst,
        src,
        move |d, s| backend.binary_inplace(kind, d, s),
        move |d, s| b2.binary_inplace_dev(kind, d, s),
    )
}

/// dst += alpha * src (same shape, same device).
pub(crate) fn raw_axpy_(op: &'static str, alpha: f32, dst: &Tensor, src: &Tensor) -> Result<()> {
    let backend = backend_for(dst.0.device)?;
    let b2 = backend.clone();
    with_dst_src(
        op,
        dst,
        src,
        move |d, s| backend.axpy_inplace(alpha, d, s),
        move |d, s| b2.axpy_inplace_dev(alpha, d, s),
    )
}

/// Exact CPU typed copy keeps the allocation and shared version cell. Reading
/// the source before locking the destination also handles aliased strided views.
fn raw_typed_copy_(op: &'static str, dst: &Tensor, src: &Tensor) -> Result<()> {
    if crate::capture::is_recording() {
        return Err(Error::Unsupported { op, msg: "typed copy does not support inference capture".into() });
    }
    check_dst_layout(op, dst)?;
    check_src(op, dst, src)?;
    if dst.device() != Device::Cpu {
        return Err(Error::Unsupported { op, msg: "typed copy currently requires CPU tensors".into() });
    }
    if Arc::ptr_eq(&dst.0.storage, &src.0.storage) && src.is_contiguous() && src.0.offset == 0 {
        return Ok(());
    }
    macro_rules! copy {
        ($data:expr, $variant:ident) => {{
            let data = $data;
            let mut guard = dst.0.storage.write();
            let Storage::$variant(values) = &mut *guard else { unreachable!() };
            values.copy_from_slice(&data);
        }};
    }
    match dst.dtype() {
        DType::F64 => copy!(src.to_vec_f64(), F64),
        DType::I64 => copy!(src.to_vec_i64(), I64),
        DType::Bool => copy!(src.to_vec_bool(), Bool),
        DType::F16 => copy!(src.to_vec_f16_bits()?, F16),
        DType::BF16 => copy!(src.to_vec_bf16_bits()?, BF16),
        DType::F32 => unreachable!(),
    }
    dst.bump_version();
    Ok(())
}

/// dst's contents become src's values. CPU copies preserve matching dtypes;
/// f32 copies also support cross-device transfer through the existing backend.
pub(crate) fn raw_copy_(op: &'static str, dst: &Tensor, src: &Tensor) -> Result<()> {
    if dst.dtype() != DType::F32 {
        return raw_typed_copy_(op, dst, src);
    }
    check_dst(op, dst)?;
    if src.dtype() != DType::F32 {
        return Err(Error::DtypeMismatch {
            op,
            expected: DType::F32,
            got: src.dtype(),
        });
    }
    if dst.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            op,
            lhs: dst.0.shape.clone(),
            rhs: src.0.shape.clone(),
        });
    }
    if Arc::ptr_eq(&dst.0.storage, &src.0.storage) && src.is_contiguous() && src.0.offset == 0 {
        return Ok(()); // whole self-copy: nothing to do, and no version noise
    }
    let backend = backend_for(dst.0.device)?;
    if is_device(dst) {
        if src.0.device == dst.0.device && src.device_resident_whole() {
            let guards = PairGuard::new(dst, src);
            let (Storage::Device(bd), Storage::Device(bs)) = guards.get() else {
                unreachable!()
            };
            backend.copy_into_dev(bd.as_ref(), bs.as_ref())?;
        } else {
            let data = src.to_vec();
            let g = dst.0.storage.read();
            let Storage::Device(bd) = &*g else { unreachable!() };
            backend.write_dev_from_host(bd.as_ref(), &data)?;
        }
    } else {
        let data = src.to_vec();
        let mut g = dst.0.storage.write();
        let Storage::F32(v) = &mut *g else { unreachable!() };
        v.copy_from_slice(&data);
    }
    dst.bump_version();
    Ok(())
}

// Match PreparedChain's exclusive sweep even though device mutations use
// read guards. Deduplication also prevents recursive reads behind a writer.
fn with_device_buffers<const N: usize>(
    ts: [&Tensor; N],
    f: impl FnOnce([&dyn crate::dispatch::DeviceBuffer; N]) -> Result<()>,
) -> Result<()> {
    let mut cells: Vec<_> = ts.iter().map(|t| &t.0.storage).collect();
    cells.sort_unstable_by_key(|s| Arc::as_ptr(s));
    cells.dedup_by_key(|s| Arc::as_ptr(s));
    let guards: Vec<_> = cells.iter().map(|s| s.read()).collect();
    let buffers = std::array::from_fn(|i| {
        let index = cells.iter().position(|s| Arc::ptr_eq(s, &ts[i].0.storage)).unwrap();
        let Storage::Device(b) = &*guards[index] else { unreachable!() };
        b.as_ref()
    });
    f(buffers)
}

/// One fused SGD-with-momentum step over a parameter, its velocity buffer,
/// and its gradient: v = momentum*v + g; p -= lr*(nesterov ? momentum*v + g
/// : v). All three must be whole contiguous f32 on one device with matching
/// element counts, in three distinct storages.
pub(crate) fn raw_sgd_step_(
    p: &Tensor,
    v: &Tensor,
    g: &Tensor,
    lr: f32,
    momentum: f32,
    nesterov: bool,
) -> Result<()> {
    const OP: &str = "sgd_step";
    for t in [p, v, g] {
        check_dst(OP, t)?;
    }
    check_step_operands(OP, &[p, v, g])?;
    let backend = backend_for(p.0.device)?;
    if is_device(p) {
        with_device_buffers([p, v, g], |[bp, bv, bg]| {
            backend.sgd_step_dev(bp, bv, bg, lr, momentum, nesterov)
        })?;
    } else {
        // Write guards in address order (see module docs); the grad is
        // read-only and locked in the same sweep.
        let mut cells: Vec<(usize, &Tensor)> = vec![(0, p), (1, v), (2, g)];
        cells.sort_by_key(|(_, t)| Arc::as_ptr(&t.0.storage) as usize);
        let (mut wp, mut wv, mut rg) = (None, None, None);
        for (role, t) in cells {
            match role {
                0 => wp = Some(t.0.storage.write()),
                1 => wv = Some(t.0.storage.write()),
                _ => rg = Some(t.0.storage.read()),
            }
        }
        let (mut wp, mut wv, rg) = (wp.unwrap(), wv.unwrap(), rg.unwrap());
        let Storage::F32(vp) = &mut *wp else { unreachable!() };
        let Storage::F32(vv) = &mut *wv else { unreachable!() };
        let Storage::F32(vg) = &*rg else { unreachable!() };
        backend.sgd_step(vp, vv, vg, lr, momentum, nesterov);
    }
    p.bump_version();
    v.bump_version();
    Ok(())
}

/// One fused Adam/AdamW step (see `Backend::adamw_step` for the update
/// rule); weight_decay == 0 is exactly Adam. Same operand contract as
/// `raw_sgd_step_`, over four distinct storages.
pub(crate) fn raw_adamw_step_(
    p: &Tensor,
    m: &Tensor,
    v: &Tensor,
    g: &Tensor,
    hp: AdamWStep,
) -> Result<()> {
    const OP: &str = "adamw_step";
    for t in [p, m, v, g] {
        check_dst(OP, t)?;
    }
    check_step_operands(OP, &[p, m, v, g])?;
    let backend = backend_for(p.0.device)?;
    if is_device(p) {
        with_device_buffers([p, m, v, g], |[bp, bm, bv, bg]| {
            backend.adamw_step_dev(bp, bm, bv, bg, hp)
        })?;
    } else {
        let mut cells: Vec<(usize, &Tensor)> = vec![(0, p), (1, m), (2, v), (3, g)];
        cells.sort_by_key(|(_, t)| Arc::as_ptr(&t.0.storage) as usize);
        let (mut wp, mut wm, mut wv, mut rg) = (None, None, None, None);
        for (role, t) in cells {
            match role {
                0 => wp = Some(t.0.storage.write()),
                1 => wm = Some(t.0.storage.write()),
                2 => wv = Some(t.0.storage.write()),
                _ => rg = Some(t.0.storage.read()),
            }
        }
        let (mut wp, mut wm, mut wv, rg) = (wp.unwrap(), wm.unwrap(), wv.unwrap(), rg.unwrap());
        let Storage::F32(vp) = &mut *wp else { unreachable!() };
        let Storage::F32(vm) = &mut *wm else { unreachable!() };
        let Storage::F32(vv) = &mut *wv else { unreachable!() };
        let Storage::F32(vg) = &*rg else { unreachable!() };
        backend.adamw_step(vp, vm, vv, vg, hp);
    }
    p.bump_version();
    m.bump_version();
    v.bump_version();
    Ok(())
}

/// Capturable AdamW step: like `raw_adamw_step_` but reads this step's bias
/// correction from a device-resident timestep tensor `t = [step, bc1, bc2]`
/// (advanced by `raw_scalar_increment_`) instead of host-computed `bc1`/`bc2`,
/// so the step is safe to record in a CUDA graph and replay with an advancing
/// correction. Device-only: `t` and all four operands must be device-resident.
pub(crate) fn raw_adamw_step_capturable_(
    p: &Tensor,
    m: &Tensor,
    v: &Tensor,
    g: &Tensor,
    t: &Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) -> Result<()> {
    const OP: &str = "adamw_step_capturable";
    for x in [p, m, v, g] {
        check_dst(OP, x)?;
    }
    check_step_operands(OP, &[p, m, v, g])?;
    if !is_device(p) || !is_device(t) {
        return Err(Error::Unsupported {
            op: OP,
            msg: "capturable AdamW step is device-only (params and timestep must be resident)"
                .to_string(),
        });
    }
    let backend = backend_for(p.0.device)?;
    with_device_buffers([p, m, v, g, t], |[bp, bm, bv, bg, bt]| {
        backend.adamw_step_capturable_dev(bp, bm, bv, bg, bt, lr, beta1, beta2, eps, weight_decay)
    })?;
    p.bump_version();
    m.bump_version();
    v.bump_version();
    Ok(())
}

/// Advance a device-resident AdamW timestep `t = [step, bc1, bc2]` by one step
/// (recomputing bias correction in-kernel). Device-only; `t` must be resident.
pub(crate) fn raw_scalar_increment_(t: &Tensor, beta1: f32, beta2: f32) -> Result<()> {
    const OP: &str = "scalar_increment";
    check_dst(OP, t)?;
    if !is_device(t) {
        return Err(Error::Unsupported {
            op: OP,
            msg: "scalar_increment is device-only".to_string(),
        });
    }
    let backend = backend_for(t.0.device)?;
    let gt = t.0.storage.read();
    let Storage::Device(bt) = &*gt else {
        unreachable!()
    };
    backend.scalar_increment_dev(bt.as_ref(), beta1, beta2)?;
    t.bump_version();
    Ok(())
}
/// be distinct (they are by construction: state buffers are allocated by the
/// optimizer, gradients by backward) and equally sized.
fn check_step_operands(op: &'static str, ts: &[&Tensor]) -> Result<()> {
    let n = ts[0].numel();
    for t in &ts[1..] {
        if t.numel() != n {
            return Err(Error::ShapeMismatch {
                op,
                lhs: ts[0].0.shape.clone(),
                rhs: t.0.shape.clone(),
            });
        }
    }
    for i in 0..ts.len() {
        for j in i + 1..ts.len() {
            if Arc::ptr_eq(&ts[i].0.storage, &ts[j].0.storage) {
                return Err(Error::Unsupported {
                    op,
                    msg: "fused step operands must live in distinct storages".to_string(),
                });
            }
        }
    }
    Ok(())
}

impl Tensor {
    /// The public-API gates on top of the raw seams (see module docs).
    fn check_inplace_allowed(&self, op: &'static str) -> Result<()> {
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op, msg: "in-place mutation is not supported during inference capture".into() });
        }
        if self.requires_grad() || self.0.op.is_some() {
            return Err(Error::Unsupported {
                op,
                msg: "in-place ops are not allowed on tensors that require grad or \
                      carry autograd history; optimizers use the internal no-grad \
                      step path, and everything else should detach_copy first"
                    .to_string(),
            });
        }
        self.check_inplace_storage(op)
    }

    fn check_inplace_source(&self, op: &'static str, source: &Tensor) -> Result<()> {
        self.check_inplace_allowed(op)?;
        if source.requires_grad() && crate::autograd::is_grad_enabled() {
            return Err(Error::Unsupported { op, msg: "mutation with a differentiable source requires disabled grad recording".into() });
        }
        Ok(())
    }

    fn check_inplace_storage(&self, op: &'static str) -> Result<()> {
        if self.0.device != Device::Cpu && Arc::strong_count(&self.0.storage) > 1 {
            return Err(Error::Unsupported {
                op,
                msg: "in-place ops on device tensors require uniquely-referenced \
                      storage (device snapshots share buffers with autograd \
                      closures); detach_copy through the host first"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Set every element to zero, in place.
    pub fn zero_(&self) -> Result<()> {
        let value = match self.dtype() {
            DType::F32 => FillValue::F32(0.), DType::F64 => FillValue::F64(0.),
            DType::I64 => FillValue::I64(0), DType::Bool => FillValue::Bool(false),
            _ => return Err(Error::Unsupported { op: "zero_", msg: "scalar fills support float32, float64, int64 and bool storage".into() }),
        };
        self.fill_value_("zero_", value)
    }

    /// Set every element to `value`, in place.
    pub fn fill_(&self, value: f32) -> Result<()> {
        self.fill_typed_(FillValue::F32(value))
    }

    /// Fill whole storage with an exact matching scalar. CPU supports F32/F64/
    /// I64/Bool; devices retain their F32 kernel. Requiring leaves need no-grad,
    /// and tensors with history remain unsupported even under no-grad.
    pub fn fill_typed_(&self, value: FillValue) -> Result<()> {
        self.fill_value_("fill_", value)
    }

    fn fill_value_(&self, op: &'static str, value: FillValue) -> Result<()> {
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op, msg: "in-place mutation is not supported during inference capture".into() });
        }
        if self.0.op.is_some() || self.requires_grad() && crate::autograd::is_grad_enabled() {
            return Err(Error::Unsupported { op, msg: "fill requires a tensor without history; requiring leaves need disabled grad recording".into() });
        }
        self.check_inplace_storage(op)?;
        raw_fill_value_(op, self, value)
    }

    /// self += other, in place (same shape, same device).
    pub fn add_(&self, other: &Tensor) -> Result<()> {
        self.check_inplace_source("add_", other)?;
        raw_binary_("add_", BinaryKind::Add, self, other)
    }

    /// self -= other, in place.
    pub fn sub_(&self, other: &Tensor) -> Result<()> {
        self.check_inplace_source("sub_", other)?;
        raw_binary_("sub_", BinaryKind::Sub, self, other)
    }

    /// self *= other, in place.
    pub fn mul_(&self, other: &Tensor) -> Result<()> {
        self.check_inplace_source("mul_", other)?;
        raw_binary_("mul_", BinaryKind::Mul, self, other)
    }

    /// self /= other, in place.
    pub fn div_(&self, other: &Tensor) -> Result<()> {
        self.check_inplace_source("div_", other)?;
        raw_binary_("div_", BinaryKind::Div, self, other)
    }

    /// self += value, elementwise in place.
    pub fn add_scalar_(&self, value: f32) -> Result<()> {
        self.check_inplace_allowed("add_scalar_")?;
        raw_affine_("add_scalar_", self, 1.0, value)
    }

    /// self *= value, elementwise in place.
    pub fn mul_scalar_(&self, value: f32) -> Result<()> {
        self.check_inplace_allowed("mul_scalar_")?;
        raw_affine_("mul_scalar_", self, value, -0.0)
    }

    /// Copy values into an existing leaf only while grad recording is disabled.
    /// Requires matching dtype/shape/device, a whole contiguous destination, no
    /// history, and uniquely referenced device storage. Identity, grad slots and
    /// storage are retained; writes bump the shared version (self-copy is a no-op).
    /// Unlike copy_from, this is not a cross-device transfer escape.
    pub fn copy_leaf_from(&self, src: &Tensor) -> Result<()> {
        const OP: &str = "copy_leaf_from";
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op: OP, msg: "in-place mutation is not supported during inference capture".into() });
        }
        if crate::autograd::is_grad_enabled() || self.0.op.is_some() {
            return Err(Error::Unsupported { op: OP, msg: "leaf copy requires disabled grad recording and a destination without history".into() });
        }
        self.check_inplace_storage(OP)?;
        check_src(OP, self, src)?;
        raw_copy_(OP, self, src)
    }

    /// Overwrite self's contents with matching dtype/shape values. Typed CPU
    /// copies are exact; f32 also permits explicit cross-device transfers.
    /// Copying a differentiable source into a new typed destination requires
    /// no-grad because mutation does not currently record a copy adjoint.
    pub fn copy_from(&self, src: &Tensor) -> Result<()> {
        self.check_inplace_source("copy_from", src)?;
        raw_copy_("copy_from", self, src)
    }
}
