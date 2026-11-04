//! Unflatten one logical axis into a shared-storage view, including strided inputs.
use crate::{Error, Result, Tensor};
use crate::shape::checked_numel;

impl Tensor {
    pub fn unflatten(&self, dim: isize, sizes: &[isize]) -> Result<Tensor> {
        let invalid = |message: &str| Error::InvalidShape { op: "unflatten", msg: message.into() };
        let rank = self.ndim();
        if rank == 0 || dim < -(rank as isize) || dim >= rank as isize {
            return Err(invalid("dimension is out of range"));
        }
        if sizes.is_empty() || sizes.iter().any(|&n| n < -1) || sizes.iter().filter(|&&n| n == -1).count() > 1 {
            return Err(invalid("sizes require a nonempty sequence of nonnegative dimensions and at most one -1"));
        }
        let axis = if dim < 0 { (dim + rank as isize) as usize } else { dim as usize };
        let mut split = sizes.iter().map(|&n| n.max(0) as usize).collect::<Vec<_>>();
        let known = if sizes.contains(&0) { 0 } else {
            sizes.iter().filter(|&&n| n != -1).try_fold(1usize, |product, &n| product.checked_mul(n as usize))
                .ok_or_else(|| invalid("size product overflow"))?
        };
        if let Some(inferred) = sizes.iter().position(|&n| n == -1) {
            if known == 0 || self.shape()[axis] % known != 0 { return Err(invalid("cannot infer the selected dimension")); }
            split[inferred] = self.shape()[axis] / known;
        } else if known != self.shape()[axis] { return Err(invalid("sizes must multiply to the selected dimension")); }
        let mut shape = self.shape()[..axis].to_vec();
        shape.extend(&split);
        shape.extend_from_slice(&self.shape()[axis + 1..]);
        let numel = checked_numel("unflatten", &shape)?;
        let layout = shape.iter().try_fold(1usize, |product, &n| product.checked_mul(n.max(1)))
            .ok_or_else(|| invalid("layout capacity overflow"))?;
        if layout > isize::MAX as usize { return Err(invalid("layout exceeds addressable capacity")); }
        let strides = if shape == self.shape() {
            self.0.stride.clone()
        } else if numel == 0 {
            let mut strides = vec![0; shape.len()];
            let mut stride = 1;
            for index in (0..shape.len()).rev() {
                strides[index] = stride;
                stride *= shape[index].max(1);
            }
            strides
        } else {
            let mut split_strides = vec![0; split.len()];
            let mut stride = self.0.stride[axis];
            for index in (0..split.len()).rev() {
                split_strides[index] = stride;
                if index > 0 { stride = stride.checked_mul(split[index]).ok_or_else(|| invalid("stride overflow"))?; }
            }
            let mut strides = self.0.stride[..axis].to_vec();
            strides.extend(split_strides);
            strides.extend_from_slice(&self.0.stride[axis + 1..]);
            strides
        };
        let view = Tensor::from_parts(self.0.storage.clone(), shape.clone(), strides, self.0.offset, self.device(), false, None);
        let output = self.capture_layout(view)?;
        let input_shape = self.shape().to_vec();
        Ok(output.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Reshape(shape), move |gradient| {
            vec![gradient.reshape(&input_shape).expect("unflatten backward shape is validated")]
        }))
    }
}
