use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::tensor::Tensor;

/// A shared parameter slot with independent optimizer and leaf AD flags.
/// Ordinary construction enables both; explicit shared construction may freeze
/// differentiation. Forward passes read the current leaf; optimizer
/// steps mutate its storage in place (through the no-grad seams in
/// `inplace`), so the leaf's identity and address are stable across steps.
/// The constructor takes an OWNING copy of its argument - the caller's
/// tensor is the initial value, never aliased state the step would scribble
/// over. Single-threaded for the MVP (Rc/RefCell); a threaded runtime would
/// swap these for Arc/Mutex.
#[derive(Clone)]
pub struct Param(Rc<RefCell<Tensor>>, Rc<Cell<bool>>);

impl Param {
    /// Optimizer slot over the exact existing leaf and gradient slot. Unlike
    /// owning construction or shared-data construction, no tensor is detached.
    pub fn from_leaf(t: &Tensor) -> crate::Result<Param> {
        const OP: &str = "optimizer leaf";
        if !matches!(t.dtype(), crate::DType::F32 | crate::DType::F64) || t.device() != crate::Device::Cpu {
            return Err(crate::Error::Unsupported { op: OP, msg: "plain optimizer leaves require CPU float32 or float64 storage".into() });
        }
        if t.0.op.is_some() {
            return Err(crate::Error::Unsupported { op: OP, msg: "plain optimizer tensors must be leaves without autograd history".into() });
        }
        if t.is_structural_zero() {
            return Err(crate::Error::Unsupported { op: OP, msg: "structural-zero storage is immutable; clone before optimization".into() });
        }
        if t.storage_offset() != 0 || !t.is_contiguous() || t.storage_len() != t.numel() {
            return Err(crate::Error::Unsupported { op: OP, msg: "plain optimizer leaves require whole contiguous storage".into() });
        }
        Ok(Param(Rc::new(RefCell::new(t.clone())), Rc::new(Cell::new(true))))
    }

    /// Explicit shared-data leaf for pretrained tables. Ordinary `new` remains
    /// owning; shared destinations require whole CPU floating storage.
    pub fn from_shared(t: &Tensor, requires_grad: bool) -> crate::Result<Param> {
        const OP: &str = "shared parameter";
        if !matches!(t.dtype(), crate::DType::F32 | crate::DType::F64) || t.device() != crate::Device::Cpu {
            return Err(crate::Error::Unsupported { op: OP, msg: "shared parameters require CPU float32 or float64 storage".into() });
        }
        if t.is_structural_zero() {
            return Err(crate::Error::Unsupported { op: OP, msg: "structural-zero storage is immutable; clone before sharing".into() });
        }
        if t.storage_offset() != 0 || !t.is_contiguous() || t.storage_len() != t.numel() {
            return Err(crate::Error::Unsupported { op: OP, msg: "shared parameters require whole contiguous storage".into() });
        }
        let leaf = t.detach().requires_grad_(requires_grad)?;
        Ok(Param(Rc::new(RefCell::new(leaf)), Rc::new(Cell::new(true))))
    }

    pub fn new(t: Tensor) -> Param {
        Self::try_new(t).expect("Param::new requires a supported floating leaf")
    }

    /// Fallible owning construction. CPU f64 values are copied exactly; f32
    /// retains its existing device-copy policy. Source flags are independent.
    pub fn try_new(t: Tensor) -> crate::Result<Param> {
        Ok(Param(Rc::new(RefCell::new(owning_leaf(t, "parameter")?)), Rc::new(Cell::new(true))))
    }

    /// The current leaf tensor (cheap clone; shares autograd identity).
    pub fn tensor(&self) -> Tensor {
        self.0.borrow().clone()
    }

    /// Install a new value (an owning copy, like `new`), re-flagging it as a
    /// grad-requiring leaf.
    pub fn set(&self, t: Tensor) {
        self.try_set(t).expect("Param::set requires a supported floating leaf");
    }

    /// Validate and materialize before replacing the current leaf.
    pub fn try_set(&self, t: Tensor) -> crate::Result<()> {
        let leaf = owning_leaf(t, "parameter set")?;
        *self.0.borrow_mut() = leaf;
        Ok(())
    }

    /// Stable identity of the shared parameter slot, including tied names.
    pub fn identity(&self) -> usize { Rc::as_ptr(&self.0) as usize }

    /// Optimizer freeze, shared across tied slots, without replacing tensor or
    /// autograd identity. Forward differentiation remains enabled; optimizers
    /// ignore this slot's gradient. Toggling clears stale accumulated gradients.
    pub fn set_trainable(&self, trainable: bool) {
        if self.1.replace(trainable) != trainable { self.zero_grad(); }
    }
    pub fn is_trainable(&self) -> bool { self.1.get() }

    pub fn grad(&self) -> Option<Tensor> {
        if self.is_trainable() { self.0.borrow().grad() } else { None }
    }

    pub fn zero_grad(&self) {
        self.0.borrow().zero_grad();
    }
}

fn owning_leaf(t: Tensor, op: &'static str) -> crate::Result<Tensor> {
    let width = match t.dtype() {
        crate::DType::F32 => 4,
        crate::DType::F64 if t.device() == crate::Device::Cpu => 8,
        _ => return Err(crate::Error::Unsupported { op, msg: "parameters support float32 and CPU float64 storage".into() }),
    };
    if t.0.op.is_some() {
        return Err(crate::Error::Unsupported { op, msg: "parameter construction takes a leaf without autograd history".into() });
    }
    if crate::capture::is_recording() {
        return Err(crate::Error::Unsupported { op, msg: "parameter construction or replacement during capture is unsupported".into() });
    }
    if crate::shape::checked_numel(op, t.shape())? > isize::MAX as usize / width {
        return Err(crate::Error::InvalidShape { op, msg: "parameter materialization exceeds allocation capacity".into() });
    }
    let leaf = t.try_owned_detach_copy()?;
    leaf.set_requires_grad(true)?;
    Ok(leaf)
}
