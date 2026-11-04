//! Real cumulative products with zero-aware, staged ordinary adjoints.
use crate::{DType, Error, Result, Tensor};
use crate::autograd::ForwardOp;
use super::cumsum::{prepare, real_tensor, real_values, rounded, Geometry};

fn backward(values: &[f64], output: &[f64], seed: &Tensor, geometry: Geometry, dtype: DType, input_dtype: DType) -> Result<Tensor> {
    let Geometry { outer, width, inner, count } = geometry;
    if count == 0 || width == 1 { return seed.to_dtype_autograd(input_dtype, false); }
    // The generated Torch rule casts the seed to the original input dtype
    // before invoking the cumulative adjoint, even with an explicit dtype.
    let gradient = real_values(seed, input_dtype, "cumprod backward")?;
    let arithmetic = if dtype == DType::F64 || input_dtype == DType::F64 { DType::F64 } else { DType::F32 };
    let mut result = crate::typed_math::filled("cumprod backward", seed.shape(), 0.0f64)?;
    for group in 0..outer { for lane in 0..inner {
        let at = |position| (group * width + position) * inner + lane;
        let first = (0..width).find(|&position| values[at(position)] == 0.0);
        let end = first.unwrap_or(width);
        let mut total = 0.0f64;
        for position in (0..end).rev() {
            let index = at(position);
            total += rounded(gradient[index] * output[index], arithmetic);
            result[index] = rounded(rounded(total, arithmetic) / values[index], arithmetic);
        }
        if let Some(first) = first {
            let second = (first + 1..width).find(|&position| values[at(position)] == 0.0);
            let stop = second.map_or(width, |position| position + 1);
            let mut product = 1.0f64;
            let mut terms = crate::typed_math::filled("cumprod backward", &[stop - first], 0.0f64)?;
            for position in first..stop {
                if position != first { product *= values[at(position)]; }
                let g = if second == Some(position) { 0.0 } else { gradient[at(position)] };
                terms[position - first] = rounded(g * rounded(product, input_dtype), input_dtype);
            }
            // This is a reduction in the original input dtype, not a widened
            // cumulative sum. Casting only its final result loses cancellation.
            let total = terms.into_iter().fold(0.0, |total, value| rounded(total + value, input_dtype));
            let before = if first == 0 { 1.0 } else { output[at(first - 1)] };
            result[at(first)] = rounded(rounded(total, input_dtype) * before, input_dtype);
        }
    } }
    real_tensor(result, seed.shape(), input_dtype, "cumprod backward")
}

pub(crate) fn graph_vjp(input: &Tensor, output: &Tensor, seed: &Tensor, dim: usize) -> Result<Tensor> {
    if input.numel() == 0 || input.shape().get(dim).copied().unwrap_or(1) == 1 {
        return seed.to_dtype_autograd(input.dtype(), false);
    }
    let seed = seed.to_dtype_autograd(input.dtype(), false)?;
    let values = input;
    if real_values(values, input.dtype(), "cumprod backward")?.iter().any(|&value| value == 0.0) {
        let width = input.shape()[dim];
        // The zero-safe higher rule retains a quadratic product-exclusion
        // graph. Bound that explicit work before creating any of its nodes.
        if input.numel().checked_mul(width).is_none_or(|work| work > 1 << 20) {
            return Err(Error::Unsupported { op: "cumprod higher backward", msg: "zero-containing higher cumulative products exceed the 1048576-element graph-work limit".into() });
        }
        let mut shape = input.shape().to_vec(); shape[dim] = 1;
        let one = real_tensor(crate::typed_math::filled("cumprod higher backward", &shape, 1.0f64)?, &shape, input.dtype(), "cumprod higher backward")?;
        let mut parts = Vec::new();
        parts.try_reserve_exact(width).map_err(|_| Error::InvalidShape { op: "cumprod higher backward", msg: "cannot allocate cumulative graph handles".into() })?;
        let mut before = one.clone();
        for position in 0..width {
            let after = if position + 1 == width { one.clone() }
                else { Tensor::cat(&[one.clone(), values.narrow(dim, position + 1, width - position - 1)?.cumprod(dim)?], dim)? };
            let coefficients = before.mul(&after)?;
            parts.push(coefficients.mul(&seed.narrow(dim, position, width - position)?)?.sum_dim(dim, true)?);
            if position + 1 != width { before = before.mul(&values.narrow(dim, position, 1)?)?; }
        }
        return Tensor::cat(&parts, dim)?.to_dtype_autograd(input.dtype(), false);
    }
    seed.mul(output)?.cumulative_sum(dim, None, true)?.div(values)?.to_dtype_autograd(input.dtype(), false)
}

pub(crate) fn jvp(input: &Tensor, output: &Tensor, seed: &Tensor, dim: usize) -> Result<Tensor> {
    if input.numel() == 0 {
        return Ok(seed.to_dtype_autograd(output.dtype(), false)?.clone_tensor(false)?.save_tensor_versions(&[input, output]));
    }
    let snapshot = real_values(input, input.dtype(), "cumprod forward_ad")?;
    let regular = seed.div(input)?.cumulative_sum(dim, None, false)?.mul(output)?;
    let tangent = if snapshot.contains(&0.0) {
        let (Geometry { outer, width, inner, .. }, _) = prepare(input, dim, None, "cumprod forward_ad", false)?;
        let mut before = crate::typed_math::filled("cumprod forward_ad", input.shape(), false)?;
        let mut first = crate::typed_math::filled("cumprod forward_ad", input.shape(), false)?;
        for group in 0..outer { for lane in 0..inner {
            let at = |position| (group * width + position) * inner + lane;
            let zero = (0..width).find(|&position| snapshot[at(position)] == 0.0);
            for position in 0..width {
                before[at(position)] = zero.is_none_or(|first| position < first);
                first[at(position)] = zero == Some(position);
            }
        } }
        let before = Tensor::from_vec_bool(before, input.shape())?;
        let first = Tensor::from_vec_bool(first, input.shape())?;
        let promoted = crate::typed_math::result_type(seed, input)?;
        let alternate = Tensor::where_cond(&first, &seed.to_dtype_autograd(promoted, false)?, &input.to_dtype_autograd(promoted, false)?)?.cumprod(dim)?;
        let promoted = crate::typed_math::result_type(&regular, &alternate)?;
        Tensor::where_cond(&before, &regular.to_dtype_autograd(promoted, false)?, &alternate.to_dtype_autograd(promoted, false)?)?
    } else { regular };
    Ok(tangent.to_dtype_autograd(output.dtype(), false)?.save_tensor_versions(&[input, output]))
}

impl Tensor {
    pub fn cumprod(&self, dim: usize) -> Result<Tensor> { self.cumprod_dtype(dim, None) }

    pub fn cumprod_dtype(&self, dim: usize, dtype: Option<DType>) -> Result<Tensor> {
        let (geometry, output_dtype) = prepare(self, dim, dtype, "cumprod", false)?;
        let Geometry { outer, width, inner, count } = geometry;
        if output_dtype == DType::I64 {
            let values = self.to_dtype(DType::I64).to_vec_i64();
            let mut output = crate::typed_math::filled("cumprod", self.shape(), 0i64)?;
            if count != 0 { for group in 0..outer { for lane in 0..inner {
                let mut product = 1i64;
                for position in 0..width {
                    let index = (group * width + position) * inner + lane;
                    product = product.wrapping_mul(values[index]); output[index] = product;
                }
            } } }
            return Tensor::from_vec_i64(output, self.shape());
        }
        let values = real_values(self, output_dtype, "cumprod")?;
        let mut result = crate::typed_math::filled("cumprod", self.shape(), 0.0f64)?;
        if count != 0 { for group in 0..outer { for lane in 0..inner {
            let mut product = 1.0f64;
            for position in 0..width {
                let index = (group * width + position) * inner + lane;
                product *= values[index]; result[index] = rounded(product, output_dtype);
            }
        } } }
        let output = real_tensor(result.clone(), self.shape(), output_dtype, "cumprod")?;
        let input_dtype = self.dtype();
        let original = if matches!(input_dtype, DType::F32 | DType::F64) { real_values(self, input_dtype, "cumprod")? } else { values };
        Ok(output.record_fn_forward(vec![self.clone()], ForwardOp::CumProd { dim, dtype }, move |g| {
            vec![backward(&original, &result, g, geometry, output_dtype, input_dtype).expect("validated cumulative product adjoint")]
        }))
    }
}
