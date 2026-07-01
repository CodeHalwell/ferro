//! Extended operators, one per file. Each adds methods to `Tensor` via its own
//! `impl Tensor` block and records autograd through `Tensor::record_fn`, so ops
//! are independent and never touch a shared enum. `log` is the worked reference.

pub mod abs;
pub mod bmm;
pub mod clamp;
pub mod log;
pub mod log_softmax;
pub mod max;
pub mod mean_dim;
pub mod powf;
pub mod softmax;
pub mod sqrt;
pub mod sum_dim;
pub mod tanh;
