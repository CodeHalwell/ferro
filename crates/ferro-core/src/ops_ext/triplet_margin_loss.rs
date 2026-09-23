//! triplet_margin_loss(positive, negative, margin, p, eps): self is the
//! anchor. Per-row loss is max(0, d(a, pos) - d(a, neg) + margin) with d the
//! `pairwise_distance` p-norm over the last dim, mean-reduced over rows.
//! Composed from existing ops, so autograd flows through the composition and
//! no backward closure is needed; margin is built on the anchor's device.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn triplet_margin_loss(&self, positive: &Tensor, negative: &Tensor, margin: f32, p: f32, eps: f32) -> Result<Tensor> {
        let margin = Tensor::full_on(&[], margin, self.device())?;
        let d_pos = self.pairwise_distance(positive, p, eps)?;
        let d_neg = self.pairwise_distance(negative, p, eps)?;
        Ok(d_pos.sub(&d_neg)?.add(&margin)?.relu().mean())
    }
}
