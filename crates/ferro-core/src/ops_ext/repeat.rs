//! `repeat(reps)`: tile the tensor `reps[d]` times along each dim (torch
//! semantics; extra leading reps prepend dims). Composed as reshape to
//! [1, s0, 1, s1, ...], expand to [r0, s0, r1, s1, ...], reshape to
//! [r0*s0, r1*s1, ...]: one materializing copy, and the gradient (sum over
//! the tiles) comes from expand's and reshape's recorded backwards.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn repeat(&self, reps: &[usize]) -> Result<Tensor> {
        let ndim = self.ndim();
        if reps.len() < ndim {
            return Err(Error::InvalidShape { op: "repeat", msg: format!("{} reps given for rank {ndim}", reps.len()) });
        }
        let mut shape = vec![1usize; reps.len() - ndim];
        shape.extend_from_slice(self.shape());
        let out: Vec<usize> = reps.iter().zip(&shape).map(|(&r, &s)| r.checked_mul(s)).collect::<Option<_>>()
            .ok_or_else(|| Error::InvalidShape { op: "repeat", msg: "output size overflow".into() })?;
        crate::shape::checked_numel("repeat", &out)?;
        let paired: Vec<usize> = shape.iter().flat_map(|&s| [1, s]).collect();
        let tiled: Vec<usize> = reps.iter().zip(&shape).flat_map(|(&r, &s)| [r, s]).collect();
        self.reshape(&paired)?.expand(&tiled)?.reshape(&out)
    }
}
