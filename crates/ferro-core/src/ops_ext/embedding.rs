//! `embedding`: row lookup into a `[num_embeddings, dim]` f32 weight by a 1-D
//! I64 id tensor, returning `[n, dim]`. Implemented as `index_select_t` along
//! dim 0, so the gradient scatter-adds into `weight` via the recorded
//! `index_select` backward (duplicate ids accumulate).

use crate::error::{Error, Result};
use crate::tensor::Tensor;

pub fn embedding(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    if weight.ndim() != 2 {
        return Err(Error::InvalidShape {
            op: "embedding",
            msg: format!(
                "weight must be 2-D [num_embeddings, dim], got {:?}",
                weight.shape()
            ),
        });
    }
    weight.index_select_t(0, ids)
}
