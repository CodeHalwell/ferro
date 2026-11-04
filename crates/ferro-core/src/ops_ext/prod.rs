//! Typed eager CPU product reductions on the existing Op graph.
use crate::{DType, Device, Error, Result, Tensor};
use crate::autograd::ForwardOp;
use super::cumsum::{real_tensor, real_values, rounded};

#[derive(Clone)]
struct Shape { output: Vec<usize>, outer: usize, width: usize, inner: usize, count: usize }

fn prepare(input: &Tensor, dim: Option<usize>, keepdim: bool, dtype: Option<DType>, legacy: bool) -> Result<(Shape, DType)> {
    if !legacy && (crate::capture::is_recording() || input.device() != Device::Cpu) {
        return Err(Error::Unsupported { op: "prod", msg: "typed products require eager CPU inputs".into() });
    }
    for actual in [input.dtype(), dtype.unwrap_or(input.dtype())] {
        if !matches!(actual, DType::Bool | DType::I64 | DType::F32 | DType::F64) {
            return Err(Error::DtypeMismatch { op: "prod", expected: DType::F32, got: actual });
        }
    }
    if dim.is_some_and(|dim| dim >= input.ndim().max(1)) || dim.is_none() && keepdim {
        return Err(Error::InvalidShape { op: "prod", msg: "invalid product dimension or keepdim without dimension".into() });
    }
    let count = crate::typed_math::check_shape("prod", input.shape())?;
    let output_dtype = dtype.unwrap_or(if input.dtype() == DType::Bool { DType::I64 } else { input.dtype() });
    let (output, outer, width, inner) = if let Some(dim) = dim.filter(|_| input.ndim() != 0) {
        let mut output = input.shape().to_vec();
        if keepdim { output[dim] = 1; } else { output.remove(dim); }
        (output, input.shape()[..dim].iter().product(), input.shape()[dim], input.shape()[dim+1..].iter().product())
    } else { (vec![], 1, count, 1) };
    crate::typed_math::check_shape("prod", &output)?;
    Ok((Shape { output, outer, width, inner, count }, output_dtype))
}

fn backward(input: &[f64], output: &[f64], seed: &Tensor, shape: &Shape, original: &[usize], input_dtype: DType, dtype: DType, all: bool) -> Result<Tensor> {
    if original.is_empty() { return seed.to_dtype_autograd(input_dtype, false); }
    let mut result = crate::typed_math::filled("prod backward", original, 0.0f64)?;
    if shape.count == 0 { return real_tensor(result, original, input_dtype, "prod backward"); }
    let zero_count = input.iter().filter(|&&x| x == 0.0).count();
    if all && zero_count > 1 { return real_tensor(result, original, input_dtype, "prod backward"); }
    let seed = real_values(seed, dtype, "prod backward")?;
    for outer in 0..shape.outer { for inner in 0..shape.inner {
        let at = |position| (outer * shape.width + position) * shape.inner + inner;
        let gradient = seed[outer * shape.inner + inner];
        if zero_count == 0 {
            for position in 0..shape.width {
                let index = at(position);
                result[index] = rounded(gradient * rounded(output[outer * shape.inner + inner] / input[index], dtype), dtype);
            }
        } else if shape.width == 1 { result[at(0)] = gradient; }
        else {
            let mut before = crate::typed_math::filled("prod backward", &[shape.width], 1.0f64)?;
            let mut product = 1.0;
            for position in 0..shape.width { before[position] = rounded(product, dtype); product *= input[at(position)]; }
            product = 1.0;
            for position in (0..shape.width).rev() {
                let coefficient = rounded(before[position] * rounded(product, dtype), dtype);
                result[at(position)] = rounded(gradient * coefficient, dtype);
                product *= input[at(position)];
            }
        }
    } }
    real_tensor(result, original, input_dtype, "prod backward")
}

fn reverse(input: &Tensor, dim: usize) -> Result<Tensor> {
    let width = input.shape()[dim];
    let mut index_shape = vec![1; input.ndim()]; index_shape[dim] = width;
    let mut values = crate::typed_math::filled("prod reverse", &[width], 0i64)?;
    for (position, value) in values.iter_mut().enumerate() { *value = (width - 1 - position) as i64; }
    let index = Tensor::from_vec_i64(values, &index_shape)?.expand(input.shape())?;
    input.indexed_gather(&index, super::indexed::IndexMap::Axis(dim), false)
}

pub(crate) fn graph_vjp(input: &Tensor, output: Option<&Tensor>, seed: &Tensor, dim: Option<usize>, keepdim: bool, safe: bool) -> Result<Tensor> {
    if input.ndim() == 0 { return seed.to_dtype_autograd(input.dtype(), false); }
    let values = input.to_dtype_autograd(seed.dtype(), false)?;
    let (values, seed, axis) = if let Some(dim) = dim {
        (values, if keepdim { seed.clone() } else { seed.unsqueeze(dim)? }, dim)
    } else { (values.reshape(&[input.numel()])?, seed.clone(), 0) };
    let any_zero = real_values(&values, values.dtype(), "prod backward")?.contains(&0.0);
    let gradient = if !safe && !any_zero {
        let output = output.ok_or_else(|| Error::Unsupported { op: "prod backward", msg: "product adjoint requires its saved output".into() })?;
        let output = if let Some(dim) = dim.filter(|_| !keepdim) { output.unsqueeze(dim)? } else { output.clone() };
        seed.mul(&output.div(&values)?)?
    } else if values.numel() == 0 { seed.expand(values.shape())? }
    else if values.shape()[axis] == 1 { seed }
    else {
        let width = values.shape()[axis];
        let mut ones_shape = values.shape().to_vec(); ones_shape[axis] = 1;
        let ones = real_tensor(crate::typed_math::filled("prod backward", &ones_shape, 1.0f64)?, &ones_shape, values.dtype(), "prod backward")?;
        let before = Tensor::cat(&[ones.clone(), values.narrow(axis, 0, width - 1)?], axis)?.cumprod(axis)?;
        let reversed = reverse(&values.narrow(axis, 1, width - 1)?, axis)?;
        let after = reverse(&Tensor::cat(&[ones, reversed], axis)?.cumprod(axis)?, axis)?;
        seed.mul(&before.mul(&after)?)?
    };
    gradient.reshape(input.shape())?.to_dtype_autograd(input.dtype(), false)
}

pub(crate) fn jvp(input: &Tensor, output: &Tensor, seed: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    let one = real_tensor(crate::typed_math::filled("prod forward_ad", output.shape(), 1.0f64)?, output.shape(), output.dtype(), "prod forward_ad")?;
    let consumed = input.to_dtype_autograd(output.dtype(), false)?;
    let coefficient = graph_vjp(&consumed, Some(output), &one, dim, keepdim, crate::forward_ad::is_functional())?;
    let weighted = coefficient.mul(seed)?;
    if let Some(dim) = dim.filter(|_| input.ndim() != 0) { weighted.sum_dim(dim, keepdim) } else { Ok(weighted.sum()) }
}

impl Tensor {
    pub fn product(&self, dim: Option<usize>, keepdim: bool, dtype: Option<DType>) -> Result<Tensor> {
        self.product_impl(dim, keepdim, dtype, false)
    }

    pub(crate) fn product_impl(&self, dim: Option<usize>, keepdim: bool, dtype: Option<DType>, legacy: bool) -> Result<Tensor> {
        let (shape, output_dtype) = prepare(self, dim, keepdim, dtype, legacy)?;
        if output_dtype == DType::Bool {
            let values = self.to_dtype_autograd(DType::Bool, false)?.to_vec_bool();
            let mut output = crate::typed_math::filled("prod", &shape.output, true)?;
            if shape.count != 0 { for outer in 0..shape.outer { for inner in 0..shape.inner {
                for position in 0..shape.width { output[outer*shape.inner+inner] &= values[(outer*shape.width+position)*shape.inner+inner]; }
            } } }
            return Tensor::from_vec_bool(output, &shape.output);
        }
        if output_dtype == DType::I64 {
            let values = self.to_dtype_autograd(DType::I64, false)?.to_vec_i64();
            let mut output = crate::typed_math::filled("prod", &shape.output, 1i64)?;
            if shape.count != 0 { for outer in 0..shape.outer { for inner in 0..shape.inner {
                for position in 0..shape.width { let index=outer*shape.inner+inner; output[index] = output[index].wrapping_mul(values[(outer*shape.width+position)*shape.inner+inner]); }
            } } }
            return Tensor::from_vec_i64(output, &shape.output);
        }
        let input = real_values(self, output_dtype, "prod")?;
        let mut result = crate::typed_math::filled("prod", &shape.output, 1.0f64)?;
        if shape.count != 0 { for outer in 0..shape.outer { for inner in 0..shape.inner {
            let index = outer*shape.inner+inner;
            for position in 0..shape.width { result[index] = rounded(result[index] * input[(outer*shape.width+position)*shape.inner+inner], output_dtype); }
        } } }
        let output = real_tensor(result.clone(), &shape.output, output_dtype, "prod")?;
        let input_dtype = self.dtype(); let original = self.shape().to_vec();
        let backward = move |g: &Tensor| {
            vec![backward(&input, &result, g, &shape, &original, input_dtype, output_dtype, dim.is_none()).expect("validated product adjoint")]
        };
        // Legacy F32 prod_dim remains an opaque inference barrier.
        if crate::capture::is_recording() { return Ok(output.record_fn(vec![self.clone()], backward).save_output_version()); }
        Ok(output.record_fn_forward(vec![self.clone()], ForwardOp::Prod { dim, keepdim, dtype }, backward))
    }
}
