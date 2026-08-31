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
use crate::tensor::{Storage, Tensor};

/// The destination contract every in-place mutation requires: f32 elements
/// in a whole contiguous buffer (offset 0, row-major, view covers the whole
/// allocation), so host slices and device kernels see the entire storage.
fn check_dst(op: &'static str, t: &Tensor) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::DtypeMismatch {
            op,
            expected: DType::F32,
            got: t.dtype(),
        });
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
    if src.dtype() != DType::F32 {
        return Err(Error::DtypeMismatch {
            op,
            expected: DType::F32,
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
        let gd = dst.0.storage.read();
        let gs = if Arc::ptr_eq(&dst.0.storage, &src.0.storage) {
            None
        } else {
            Some(src.0.storage.read())
        };
        let Storage::Device(bd) = &*gd else { unreachable!() };
        let Storage::Device(bs) = gs.as_deref().unwrap_or(&gd) else {
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
            host(v, &s[src.0.offset..src.0.offset + n]);
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

/// dst's contents become src's values (same shape; src may live on any
/// device and be any f32 view - copy_from is the explicit transfer escape).
pub(crate) fn raw_copy_(op: &'static str, dst: &Tensor, src: &Tensor) -> Result<()> {
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
            let gd = dst.0.storage.read();
            let gs = if Arc::ptr_eq(&dst.0.storage, &src.0.storage) {
                None
            } else {
                Some(src.0.storage.read())
            };
            let Storage::Device(bd) = &*gd else { unreachable!() };
            let Storage::Device(bs) = gs.as_deref().unwrap_or(&gd) else {
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
        let (gp, gv, gg) = (p.0.storage.read(), v.0.storage.read(), g.0.storage.read());
        let (Storage::Device(bp), Storage::Device(bv), Storage::Device(bg)) =
            (&*gp, &*gv, &*gg)
        else {
            unreachable!()
        };
        backend.sgd_step_dev(bp.as_ref(), bv.as_ref(), bg.as_ref(), lr, momentum, nesterov)?;
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
        let (gp, gm, gv, gg) = (
            p.0.storage.read(),
            m.0.storage.read(),
            v.0.storage.read(),
            g.0.storage.read(),
        );
        let (Storage::Device(bp), Storage::Device(bm), Storage::Device(bv), Storage::Device(bg)) =
            (&*gp, &*gm, &*gv, &*gg)
        else {
            unreachable!()
        };
        backend.adamw_step_dev(bp.as_ref(), bm.as_ref(), bv.as_ref(), bg.as_ref(), hp)?;
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
    let (gp, gm, gv, gg, gt) = (
        p.0.storage.read(),
        m.0.storage.read(),
        v.0.storage.read(),
        g.0.storage.read(),
        t.0.storage.read(),
    );
    let (
        Storage::Device(bp),
        Storage::Device(bm),
        Storage::Device(bv),
        Storage::Device(bg),
        Storage::Device(bt),
    ) = (&*gp, &*gm, &*gv, &*gg, &*gt)
    else {
        unreachable!()
    };
    backend.adamw_step_capturable_dev(
        bp.as_ref(),
        bm.as_ref(),
        bv.as_ref(),
        bg.as_ref(),
        bt.as_ref(),
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
    )?;
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
        if self.0.requires_grad || self.0.op.is_some() {
            return Err(Error::Unsupported {
                op,
                msg: "in-place ops are not allowed on tensors that require grad or \
                      carry autograd history; optimizers use the internal no-grad \
                      step path, and everything else should detach_copy first"
                    .to_string(),
            });
        }
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
        self.check_inplace_allowed("zero_")?;
        raw_fill_("zero_", self, 0.0)
    }

    /// Set every element to `value`, in place.
    pub fn fill_(&self, value: f32) -> Result<()> {
        self.check_inplace_allowed("fill_")?;
        raw_fill_("fill_", self, value)
    }

    /// self += other, in place (same shape, same device).
    pub fn add_(&self, other: &Tensor) -> Result<()> {
        self.check_inplace_allowed("add_")?;
        raw_binary_("add_", BinaryKind::Add, self, other)
    }

    /// self -= other, in place.
    pub fn sub_(&self, other: &Tensor) -> Result<()> {
        self.check_inplace_allowed("sub_")?;
        raw_binary_("sub_", BinaryKind::Sub, self, other)
    }

    /// self *= other, in place.
    pub fn mul_(&self, other: &Tensor) -> Result<()> {
        self.check_inplace_allowed("mul_")?;
        raw_binary_("mul_", BinaryKind::Mul, self, other)
    }

    /// self /= other, in place.
    pub fn div_(&self, other: &Tensor) -> Result<()> {
        self.check_inplace_allowed("div_")?;
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

    /// Overwrite self's contents with src's values (same shape; src may be
    /// any f32 tensor on any device - this is the explicit transfer escape,
    /// e.g. weight surgery on a loaded model).
    pub fn copy_from(&self, src: &Tensor) -> Result<()> {
        self.check_inplace_allowed("copy_from")?;
        raw_copy_("copy_from", self, src)
    }
}
