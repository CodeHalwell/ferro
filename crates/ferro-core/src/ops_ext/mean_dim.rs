//! `mean_dim` operator: mean reduction over one dim. Follows the log.rs pattern:
//! compute the value with raw kernels, then record a self-contained backward.
//! Backward: mean = sum / n, so each input contributes g / n; expand the grad
//! back to the input shape and scale by 1/n.

use crate::tensor::{raw_sum_dim, raw_unary, Tensor};

impl Tensor {
    pub fn mean_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        let n = self.shape()[dim] as f32;
        let s = raw_sum_dim(self, dim, keepdim);
        let out = raw_unary(&s, |v| v / n);
        let in_shape = self.shape().to_vec();
        out.record_fn(vec![self.clone()], move |g| {
            let mut keep_shape = in_shape.clone();
            keep_shape[dim] = 1;
            let expanded = g
                .reshape(&keep_shape)
                .unwrap()
                .broadcast_to(&in_shape)
                .unwrap()
                .detach_copy();
            vec![raw_unary(&expanded, |v| v / n)]
        })
    }
}
