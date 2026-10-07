//! `split`, `split_sizes` and `chunk`: partition one dim into consecutive
//! pieces, each a zero-copy `narrow` view with its own recorded backward.
//! `split(size)` leaves a smaller last piece; `chunk(n)` uses pieces of
//! ceil(len/n) and may return fewer than n (torch semantics).

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn split_sizes(&self, sizes: &[usize], dim: usize) -> Result<Vec<Tensor>> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op: "split", msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        let len = self.shape()[dim];
        if sizes.iter().try_fold(0usize, |a, &s| a.checked_add(s)) != Some(len) {
            return Err(Error::InvalidShape { op: "split", msg: format!("sizes {sizes:?} do not sum to dim {dim} of size {len}") });
        }
        let mut start = 0;
        sizes.iter().map(|&s| {
            start += s;
            self.narrow(dim, start - s, s)
        }).collect()
    }

    pub fn split(&self, size: usize, dim: usize) -> Result<Vec<Tensor>> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op: "split", msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        if size == 0 {
            return Err(Error::InvalidShape { op: "split", msg: "split size must be positive".into() });
        }
        let len = self.shape()[dim];
        if len == 0 {
            return Ok(vec![self.narrow(dim, 0, 0)?]);
        }
        let sizes: Vec<usize> = (0..len).step_by(size).map(|s| size.min(len - s)).collect();
        self.split_sizes(&sizes, dim)
    }

    pub fn chunk(&self, n: usize, dim: usize) -> Result<Vec<Tensor>> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op: "chunk", msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        if n == 0 {
            return Err(Error::InvalidShape { op: "chunk", msg: "number of chunks must be positive".into() });
        }
        self.split(self.shape()[dim].div_ceil(n).max(1), dim)
    }
}
