//! Optimizers (SGD, Adam) operating on parameter tensors and their `.grad()`.
//!
//! Stub owned by the `optim` workstream. A parameter is a leaf `Tensor` created
//! with `requires_grad_(true)`; after `loss.backward()` its `.grad()` is set.
//! Apply updates by reading `param.to_vec()` / `grad.to_vec()` and rebuilding
//! the leaf, then re-installing it (the training loop holds the params).
