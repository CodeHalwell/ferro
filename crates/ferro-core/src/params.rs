use std::cell::RefCell;
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
pub struct Param(Rc<RefCell<Tensor>>);

impl Param {
    pub fn new(t: Tensor) -> Param {
        // Leaf-only contract checked on the original (silently detaching an
        // interior node would cut its graph), then an owning copy becomes
        // the parameter; the copy is a fresh leaf, so re-flagging cannot
        // fail.
        let leaf = t.requires_grad_(true).expect("Param::new takes a leaf");
        Param(Rc::new(RefCell::new(
            leaf.owned_detach_copy().requires_grad_(true).unwrap(),
        )))
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

    pub fn grad(&self) -> Option<Tensor> {
        self.0.borrow().grad()
    }

    pub fn zero_grad(&self) {
        self.0.borrow().zero_grad();
    }
}
