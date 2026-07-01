use std::cell::RefCell;
use std::rc::Rc;

use crate::tensor::Tensor;

/// A trainable parameter: a shared, mutable slot holding a leaf tensor with
/// `requires_grad = true`. Forward passes read the current leaf; optimizers
/// overwrite the slot with the updated leaf after a step. Single-threaded for
/// the MVP (Rc/RefCell); a threaded runtime would swap these for Arc/Mutex.
#[derive(Clone)]
pub struct Param(Rc<RefCell<Tensor>>);

impl Param {
    pub fn new(t: Tensor) -> Param {
        Param(Rc::new(RefCell::new(t.requires_grad_(true))))
    }

    /// The current leaf tensor (cheap clone; shares autograd identity).
    pub fn tensor(&self) -> Tensor {
        self.0.borrow().clone()
    }

    /// Install a new value, re-flagging it as a grad-requiring leaf.
    pub fn set(&self, t: Tensor) {
        *self.0.borrow_mut() = t.requires_grad_(true);
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.0.borrow().grad()
    }

    pub fn zero_grad(&self) {
        self.0.borrow().zero_grad();
    }
}
