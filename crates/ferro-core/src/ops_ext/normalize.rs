//! normalize: x / max(||x||_2 over dim, eps). The direct backward rule avoids
//! sqrt's undefined derivative at zero and squaring a tiny epsilon.
//! Numerical parity covers finite inputs and finite positive eps. NaN eps is
//! rejected because clamp bounds would otherwise silently discard it.

use crate::error::{Error, Result};
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn normalize(&self, dim: usize, eps: f32) -> Result<Tensor> {
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op: "normalize", msg: "inference capture is not supported".into() });
        }
        if eps.is_nan() {
            return Err(Error::InvalidShape { op: "normalize", msg: "eps must not be NaN".into() });
        }
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "normalize",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        if self.dtype() != crate::DType::F32 {
            return Err(Error::DtypeMismatch { op: "normalize", expected: crate::DType::F32, got: self.dtype() });
        }
        if self.numel() > isize::MAX as usize / std::mem::size_of::<f32>() {
            return Err(Error::InvalidShape { op: "normalize", msg: "input exceeds allocation capacity".into() });
        }
        let x = self.detach_copy();
        if x.device() == crate::Device::Cpu && x.numel() == 0 {
            return Ok(x.record_fn(vec![self.clone()], |g| vec![g.clone()]));
        }
        let norm = x.square()?.sum_dim(dim, true)?.sqrt();
        let denom = if norm.device() == crate::Device::Cpu {
            raw_binary("normalize_denom", &norm, &norm, |value, _| {
                if value.is_nan() { value } else { value.max(eps) }
            })?
        } else {
            norm.clamp(eps, f32::INFINITY)
        };
        let out = x.div(&denom)?;
        let y = out.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let dot = g.mul(&y).unwrap().sum_dim(dim, true).unwrap();
            let projection = raw_binary("normalize_bw", &dot, &norm, |gg, rr| {
                if rr >= eps { gg } else { 0.0 }
            }).unwrap().to_device(dot.device()).unwrap();
            vec![g.sub(&y.mul(&projection).unwrap()).unwrap().div(&denom).unwrap()]
        }))
    }
}
