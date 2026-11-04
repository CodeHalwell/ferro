//! Window argmax for N-d max pooling over [N,C,*S]. A window is the product of
//! one `(start, end, step)` range per spatial axis, so regular, padded, dilated,
//! ceil-mode and adaptive pooling all reduce to per-axis range lists chosen by
//! the caller. Selection is discrete: callers gather values through an
//! ordinary differentiable op. Windows are scanned in row-major order; a value
//! replaces the running maximum when it is greater or NaN, matching Torch CPU.

use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::DType;

impl Tensor {
    /// Returns `(flat, local)`, both I64 [N,C,*O]: `flat` indexes the input
    /// flattened to one axis, `local` the position within its [*S] plane.
    pub fn max_pool_windows(&self, axes: &[Vec<(usize, usize, usize)>]) -> Result<(Tensor, Tensor)> {
        let invalid = |msg: String| Error::InvalidShape { op: "max_pool_windows", msg };
        if self.dtype() != DType::F32 { return Err(Error::Unsupported { op: "max_pool_windows", msg: format!("{:?} input; expected float32", self.dtype()) }); }
        let shape = self.shape();
        if shape.len() != axes.len() + 2 || axes.is_empty() { return Err(invalid(format!("rank {} input for {} spatial axes", shape.len(), axes.len()))); }
        let spatial = &shape[2..];
        for (ranges, &length) in axes.iter().zip(spatial) {
            if ranges.iter().any(|&(start, end, step)| step == 0 || start >= end || start >= length || end > length) {
                return Err(invalid("every window must select at least one in-bounds element".into()));
            }
        }
        let planes = shape[0] * shape[1];
        let volume: usize = spatial.iter().product();
        let mut out_shape = shape[..2].to_vec();
        out_shape.extend(axes.iter().map(Vec::len));
        let count: usize = axes.iter().map(Vec::len).product();
        let strides: Vec<usize> = (0..spatial.len()).map(|d| spatial[d + 1..].iter().product()).collect();
        let steps: Vec<Vec<Vec<usize>>> = axes.iter().zip(&strides).map(|(ranges, &stride)| {
            ranges.iter().map(|&(s, e, k)| (s..e).step_by(k).map(|p| p * stride).collect()).collect()
        }).collect();
        let mut point = vec![0usize; axes.len()];
        let mut windows = Vec::with_capacity(count);
        for _ in 0..count {
            let mut offsets = vec![0usize];
            for (axis, &p) in steps.iter().zip(&point) {
                offsets = offsets.iter().flat_map(|&o| axis[p].iter().map(move |&s| o + s)).collect();
            }
            windows.push(offsets);
            for d in (0..point.len()).rev() { point[d] += 1; if point[d] < axes[d].len() { break; } point[d] = 0; }
        }
        let data = self.to_vec();
        let (mut flat, mut local) = (Vec::with_capacity(planes * count), Vec::with_capacity(planes * count));
        for plane in 0..planes {
            let values = &data[plane * volume..(plane + 1) * volume];
            for window in &windows {
                let mut best = window[0];
                for &o in &window[1..] { if values[o].is_nan() || values[o] > values[best] { best = o; } }
                flat.push((plane * volume + best) as i64);
                local.push(best as i64);
            }
        }
        Ok((Tensor::from_vec_i64(flat, &out_shape)?, Tensor::from_vec_i64(local, &out_shape)?))
    }
}
