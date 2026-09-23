//! margin_ranking_loss(other, target, margin): self is x1, other is x2 and
//! target is +1 (x1 should rank higher) or -1. Per-element loss is
//! max(0, -target * (x1 - x2) + margin), mean-reduced. Composed from
//! existing ops, so autograd flows through the composition and no backward
//! closure is needed; margin is built on the input's device with `full_on`.

use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn margin_ranking_loss(&self, other: &Tensor, target: &Tensor, margin: f32) -> Result<Tensor> {
        let margin = Tensor::full_on(&[], margin, self.device())?;
        Ok(self.sub(other)?.mul(target)?.neg().add(&margin)?.relu().mean())
    }
}
