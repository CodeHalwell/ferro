//! Elementwise select with torch.where semantics: out = if cond != 0 { a }
//! else { b }, broadcasting across all three operands. `cond` is an f32 mask
//! and receives no gradient; only `a` and `b` are differentiable inputs.

use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn where_cond(cond: &Tensor, a: &Tensor, b: &Tensor) -> crate::Result<Tensor> {
        let pa = raw_binary("where_cond", cond, a, |c, x| if c != 0.0 { x } else { 0.0 })?;
        let pb = raw_binary("where_cond", cond, b, |c, x| if c != 0.0 { 0.0 } else { x })?;
        let out = raw_binary("where_cond", &pa, &pb, |x, y| x + y)?;
        let cond = cond.detach_copy();
        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        Ok(out.record_fn(vec![a.clone(), b.clone()], move |g| {
            let ga = raw_binary("where_bw", g, &cond, |gg, c| if c != 0.0 { gg } else { 0.0 });
            let gb = raw_binary("where_bw", g, &cond, |gg, c| if c != 0.0 { 0.0 } else { gg });
            vec![unbroadcast(&ga.unwrap(), &a_shape), unbroadcast(&gb.unwrap(), &b_shape)]
        }))
    }
}
