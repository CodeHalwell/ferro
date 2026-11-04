//! Legacy one-axis entry point shares the typed product implementation.
use crate::{Result, Tensor};
impl Tensor {
    pub fn prod_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        crate::tensor::check_f32("prod_dim", self)?;
        self.product_impl(Some(dim), keepdim, None, true)
    }
}
