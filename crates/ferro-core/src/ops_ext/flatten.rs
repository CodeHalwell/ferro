//! flatten(start, end): merge the inclusive dim range [start, end] into one
//! dim, keeping the dims outside it. A scalar flattens to shape [1], as in
//! torch. Delegates to `reshape`, so autograd and layout handling are its.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn flatten(&self, start: usize, end: usize) -> Result<Tensor> {
        let shape = self.shape();
        if shape.is_empty() {
            return self.reshape(&[1]);
        }
        if start > end || end >= shape.len() {
            return Err(Error::InvalidShape { op: "flatten", msg: format!("dims {start}..={end} out of range for rank {}", shape.len()) });
        }
        let mut out = shape[..start].to_vec();
        out.push(shape[start..=end].iter().product());
        out.extend_from_slice(&shape[end + 1..]);
        self.reshape(&out)
    }
}
