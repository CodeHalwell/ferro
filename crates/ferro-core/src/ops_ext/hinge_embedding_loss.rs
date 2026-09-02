//! hinge_embedding_loss(target, margin): self is x (a distance), target is
//! +1 or -1. Per-element loss is x when target == 1, else max(0, margin - x);
//! mean-reduced. Backward (pre-reduction): dx = 1 when target == 1, -1 when
//! target != 1 and margin - x > 0, else 0; dtarget = 0 (target is a label,
//! not a differentiable input). No device kernel exists for this op, so the
//! forward goes through the host `raw_binary` path; chaining the existing
//! `mean()` folds in the 1/numel scaling and its own backward.

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn hinge_embedding_loss(&self, target: &Tensor, margin: f32) -> Result<Tensor> {
        let out = raw_binary("hinge_embedding_loss", self, target, move |x, t| {
            if t == 1.0 {
                x
            } else {
                (margin - x).max(0.0)
            }
        })?;
        let (x, t) = (self.detach_copy(), target.detach_copy());
        let (sx, st) = (self.shape().to_vec(), target.shape().to_vec());
        let elem = out.record_fn(vec![self.clone(), target.clone()], move |g| {
            let dx = raw_binary("hinge_embedding_loss_dx", &x, &t, move |xx, tt| {
                if tt == 1.0 {
                    1.0
                } else if margin - xx > 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .unwrap();
            let ga = raw_binary("hinge_embedding_loss_bwa", g, &dx, |gg, p| gg * p).unwrap();
            let gb = raw_binary("hinge_embedding_loss_bwb", g, g, |_, _| 0.0).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &st)]
        });
        Ok(elem.mean())
    }
}
