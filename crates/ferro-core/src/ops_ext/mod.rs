//! Extended operators, one per file. Each adds methods to `Tensor` via its own
//! `impl Tensor` block and records autograd through `Tensor::record_fn`, so ops
//! are independent and never touch a shared enum. `log` is the worked reference.

pub mod abs;
pub mod argmax;
pub mod avg_pool2d;
pub mod batch_norm;
pub mod bmm;
pub mod cat;
pub mod clamp;
pub mod conv2d;
pub mod cumsum;
pub mod dropout;
pub mod embedding;
pub mod gather;
pub mod gelu;
pub mod group_norm;
pub mod index_select;
pub mod layer_norm;
pub mod log;
pub mod log_softmax;
pub mod logsumexp;
pub mod max;
pub mod max_pool2d;
pub mod mean_dim;
pub mod pad;
pub mod powf;
pub mod rope;
pub mod scatter;
pub mod scatter_add;
pub mod silu;
pub mod softmax;
pub mod sqrt;
pub mod squeeze;
pub mod sum_dim;
pub mod tanh;
pub mod topk;
pub mod tri;
pub mod where_cond;

// `embedding` is a free function over (weight, ids) rather than a method.
pub use embedding::embedding;
