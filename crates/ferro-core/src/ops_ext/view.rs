//! Shape-only aliases. Adjacent logical axes coalesce only within one stride chunk.
use crate::{Error, Result, Tensor};
use crate::shape::{checked_numel, default_strides};

fn view_strides(input: &Tensor, shape: &[usize]) -> Result<Vec<usize>> {
    let invalid = || Error::InvalidShape { op: "view", msg: "requested shape cannot share this layout; use reshape for a copy".into() };
    if input.numel() == 0 {
        return Ok(if shape == input.shape() { input.strides().to_vec() } else { default_strides(shape) });
    }
    if input.ndim() == 0 { return Ok(default_strides(shape)); }
    let mut strides = vec![0; shape.len()];
    let mut output_axis = shape.len();
    let mut chunk_stride = input.strides()[input.ndim() - 1];
    let (mut input_count, mut output_count) = (1usize, 1usize);
    for axis in (0..input.ndim()).rev() {
        input_count = input_count.checked_mul(input.shape()[axis]).ok_or_else(invalid)?;
        let boundary = axis == 0 || (input.shape()[axis - 1] != 1
            && input.strides()[axis - 1] != input_count.checked_mul(chunk_stride).ok_or_else(invalid)?);
        if boundary {
            while output_axis > 0 && (output_count < input_count || shape[output_axis - 1] == 1) {
                output_axis -= 1;
                strides[output_axis] = output_count.checked_mul(chunk_stride).ok_or_else(invalid)?;
                output_count = output_count.checked_mul(shape[output_axis]).ok_or_else(invalid)?;
            }
            if output_count != input_count { return Err(invalid()); }
            if axis > 0 { chunk_stride = input.strides()[axis - 1]; input_count = 1; output_count = 1; }
        }
    }
    if output_axis != 0 { return Err(invalid()); }
    Ok(strides)
}

impl Tensor {
    /// A fresh storage alias with an identity pullback, including live seeds.
    pub fn alias_view(&self) -> Result<Tensor> {
        let output = Tensor::from_parts(self.0.storage.clone(), self.shape().to_vec(), self.strides().to_vec(), self.storage_offset(), self.device(), false, None);
        let output = self.capture_layout(output)?;
        Ok(output.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Alias, |g| vec![g.clone()]))
    }

    /// Change shape without materializing layout. Incompatible stride chunks reject.
    /// Eager views and CPU capture alias storage. Inference recording snapshots
    /// device leaves through the existing capture isolation seam; replay aliases.
    pub fn view(&self, shape: &[usize]) -> Result<Tensor> {
        if checked_numel("view", shape)? != self.numel() {
            return Err(Error::InvalidShape { op: "view", msg: "requested shape has a different number of elements".into() });
        }
        let strides = view_strides(self, shape)?;
        let output = Tensor::from_parts(self.0.storage.clone(), shape.to_vec(), strides,
            self.storage_offset(), self.device(), false, None);
        let output = self.capture_layout(output)?;
        let original = self.shape().to_vec();
        Ok(output.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::View(shape.to_vec()), move |gradient| {
            vec![gradient.reshape(&original).expect("view backward shape is validated")]
        }))
    }

    pub fn view_as(&self, other: &Tensor) -> Result<Tensor> { self.view(other.shape()) }
}
