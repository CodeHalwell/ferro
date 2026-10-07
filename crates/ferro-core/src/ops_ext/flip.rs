//! `flip(dims)` and `roll(shifts, dims)`: reverse or cyclically shift the
//! order along dims. Strides are unsigned, so neither can be a view; each dim
//! is one `index_select` with a permuted index list, which supplies dtype
//! coverage, the host fallback and the gradient (scatter back through the
//! same permutation).

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn flip(&self, dims: &[usize]) -> Result<Tensor> {
        let ndim = self.ndim();
        let mut seen = vec![false; ndim];
        if let Some(&d) = dims.iter().find(|&&d| d >= ndim || std::mem::replace(&mut seen[d], true)) {
            return Err(Error::InvalidShape { op: "flip", msg: format!("dim {d} out of range or repeated for rank {ndim}") });
        }
        let mut out = self.reshape(self.shape())?;
        for &d in dims {
            let idx: Vec<usize> = (0..self.shape()[d]).rev().collect();
            out = out.index_select(d, &idx)?;
        }
        Ok(out)
    }

    pub fn roll(&self, shifts: &[isize], dims: &[usize]) -> Result<Tensor> {
        if dims.is_empty() {
            let [shift] = shifts else {
                return Err(Error::InvalidShape { op: "roll", msg: format!("{} shifts given without dims; expected 1", shifts.len()) });
            };
            return self.reshape(&[self.numel()])?.roll(&[*shift], &[0])?.reshape(self.shape());
        }
        let ndim = self.ndim();
        if shifts.len() != dims.len() {
            return Err(Error::InvalidShape { op: "roll", msg: format!("{} shifts for {} dims", shifts.len(), dims.len()) });
        }
        if let Some(&d) = dims.iter().find(|&&d| d >= ndim) {
            return Err(Error::InvalidShape { op: "roll", msg: format!("dim {d} out of range for rank {ndim}") });
        }
        let mut out = self.reshape(self.shape())?;
        for (&shift, &d) in shifts.iter().zip(dims) {
            let n = self.shape()[d];
            if n == 0 {
                continue;
            }
            let s = shift.rem_euclid(n as isize) as usize;
            let idx: Vec<usize> = (0..n).map(|i| (i + n - s) % n).collect();
            out = out.index_select(d, &idx)?;
        }
        Ok(out)
    }
}
