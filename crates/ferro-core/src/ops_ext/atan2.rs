//! Typed CPU atan2 and the existing F32 device fallback. Ordinary reverse
//! uses Torch's staged products and its zero squared-radius convention.
use crate::tensor::{raw_binary, unbroadcast, Tensor};
use crate::{DType, Device, Error, Result};

fn binary(
    op: &'static str, a: &Tensor, b: &Tensor,
    f32_fn: impl Fn(f32, f32) -> f32, f64_fn: impl Fn(f64, f64) -> f64,
) -> Result<Tensor> {
    if a.device() == Device::Cpu && b.device() == Device::Cpu {
        crate::typed_math::pointwise_binary_real(op, a, b, f32_fn, f64_fn)
    } else {
        raw_binary(op, a, b, f32_fn)
    }
}

fn cast(input: &Tensor, dtype: DType) -> Tensor {
    // A detached source guarantees this layout-preserving conversion stays
    // raw even when the caller supplies a live cotangent.
    input.detach().to_dtype_autograd(dtype, false).expect("validated atan2 cast")
}

impl Tensor {
    pub fn atan2(&self, other: &Tensor) -> Result<Tensor> {
        for input in [self, other] {
            if !matches!(input.dtype(), DType::F32 | DType::F64) {
                return Err(Error::DtypeMismatch { op: "atan2", expected: DType::F32, got: input.dtype() });
            }
        }
        if self.dtype() == DType::F64 || other.dtype() == DType::F64 {
            if self.device() != Device::Cpu || other.device() != Device::Cpu || crate::capture::is_recording() {
                return Err(Error::Unsupported { op: "atan2", msg: "float64 atan2 requires eager CPU inputs".into() });
            }
        }
        let out = binary("atan2", self, other, f32::atan2, f64::atan2)?.to_device(self.device())?;
        let output_dtype = out.dtype();
        let (x, y) = (self.detach(), other.detach());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            // Convert while empty, before virtual broadcast reductions: never
            // materialize an irrelevant huge nonempty primal or zero adjoint.
            if g.numel() == 0 {
                return vec![unbroadcast(&cast(g, x.dtype()), &sx), unbroadcast(&cast(g, y.dtype()), &sy)];
            }
            let g = cast(g, output_dtype);
            let xx = binary("atan2_square_a", &x, &x, |a, b| a * b, |a, b| a * b).unwrap();
            let yy = binary("atan2_square_b", &y, &y, |a, b| a * b, |a, b| a * b).unwrap();
            let reciprocal = binary("atan2_reciprocal", &xx, &yy,
                |a, b| { let sum = a + b; if sum == 0.0 { 0.0 } else { 1.0 / sum } },
                |a, b| { let sum = a + b; if sum == 0.0 { 0.0 } else { 1.0 / sum } }).unwrap();
            let a = binary("atan2_seed_a", &g, &y, |g, y| g * y, |g, y| g * y).unwrap();
            let b = binary("atan2_seed_b", &g, &x, |g, x| (-g) * x, |g, x| (-g) * x).unwrap();
            let ga = binary("atan2_backward_a", &a, &reciprocal, |a, r| a * r, |a, r| a * r).unwrap();
            let gb = binary("atan2_backward_b", &b, &reciprocal, |b, r| b * r, |b, r| b * r).unwrap();
            vec![cast(&unbroadcast(&ga, &sx), x.dtype()), cast(&unbroadcast(&gb, &sy), y.dtype())]
        }))
    }
}
