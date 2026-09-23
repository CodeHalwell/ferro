//! cosine_embedding_loss(other, target, margin, eps): self is x1, other is
//! x2 and target is +1 or -1 per row. With cos the cosine_similarity over
//! the last dim, the per-row loss is 1 - cos when target == 1, else
//! max(0, cos - margin); mean-reduced. The +/-1 label becomes a 0/1 mask via
//! (target + 1) / 2 so the two branches blend arithmetically without a
//! select. Composed from existing ops, so autograd flows through the
//! composition and no backward closure is needed.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn cosine_embedding_loss(&self, other: &Tensor, target: &Tensor, margin: f32, eps: f32) -> Result<Tensor> {
        let ndim = self.ndim();
        if ndim == 0 {
            return Err(Error::InvalidShape { op: "cosine_embedding_loss", msg: "cannot reduce a scalar".into() });
        }
        let dev = self.device();
        let one = Tensor::full_on(&[], 1.0, dev)?;
        let half = Tensor::full_on(&[], 0.5, dev)?;
        let margin = Tensor::full_on(&[], margin, dev)?;
        let cos = self.cosine_similarity(other, ndim - 1, eps)?;
        let mask = target.add(&one)?.mul(&half)?;
        let pos = cos.neg().add(&one)?;
        let neg = cos.sub(&margin)?.relu();
        Ok(mask.mul(&pos)?.add(&mask.neg().add(&one)?.mul(&neg)?)?.mean())
    }
}
