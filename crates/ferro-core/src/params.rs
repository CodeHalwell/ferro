use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::tensor::Tensor;

/// A trainable parameter: a shared, mutable slot holding a leaf tensor with
/// `requires_grad = true`. Forward passes read the current leaf; optimizer
/// steps mutate its storage in place (through the no-grad seams in
/// `inplace`), so the leaf's identity and address are stable across steps.
/// The constructor takes an OWNING copy of its argument - the caller's
/// tensor is the initial value, never aliased state the step would scribble
/// over. Single-threaded for the MVP (Rc/RefCell); a threaded runtime would
/// swap these for Arc/Mutex.
#[derive(Clone)]
pub struct Param(Rc<RefCell<Tensor>>, Rc<Cell<bool>>);

impl Param {
    pub fn new(t: Tensor) -> Param {
        // Leaf-only contract checked on the original (silently detaching an
        // interior node would cut its graph), then an owning copy becomes
        // the parameter; the copy is a fresh leaf, so re-flagging cannot
        // fail.
        let leaf = t.requires_grad_(true).expect("Param::new takes a leaf");
        Param(Rc::new(RefCell::new(
            leaf.owned_detach_copy().requires_grad_(true).unwrap(),
        )), Rc::new(Cell::new(true)))
    }

    /// The current leaf tensor (cheap clone; shares autograd identity).
    pub fn tensor(&self) -> Tensor {
        self.0.borrow().clone()
    }

    /// Install a new value (an owning copy, like `new`), re-flagging it as a
    /// grad-requiring leaf.
    pub fn set(&self, t: Tensor) {
        let leaf = t.requires_grad_(true).expect("Param::set takes a leaf");
        *self.0.borrow_mut() = leaf.owned_detach_copy().requires_grad_(true).unwrap();
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
