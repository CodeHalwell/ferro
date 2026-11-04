//! Dense CPU embedding bags without an intermediate [tokens, width] tensor.
use std::collections::HashMap;
use std::sync::Arc;
use crate::{DType, Device, Error, Result, Tensor};
use crate::tensor::Storage;

fn invalid(message: &str) -> Error { Error::InvalidShape { op: "embedding_bag", msg: message.into() } }

/// Sum, mean or first-winner max over fixed-size or offset-delimited bags.
/// Explicit offsets are monotone, start at zero and cover every input index.
#[allow(clippy::too_many_arguments)]
pub fn embedding_bag(weight: &Tensor, ids: &Tensor, offsets: Option<&Tensor>, mode: &str,
    per_sample_weights: Option<&Tensor>, include_last_offset: bool, padding: Option<usize>,
    max_norm: Option<f64>, norm_type: f64, scale_grad_by_freq: bool) -> Result<Tensor> {
    const OP: &str = "embedding_bag";
    if crate::capture::is_recording() { return Err(Error::Unsupported { op: OP, msg: "embedding bags require eager execution".into() }); }
    let mode = match mode { "sum" => 0, "mean" => 1, "max" => 2, _ => return Err(invalid("mode must be sum, mean or max")) };
    if mode == 2 && scale_grad_by_freq { return Err(Error::Unsupported { op: OP, msg: "max mode does not support frequency scaling".into() }); }
    if mode != 0 && per_sample_weights.is_some() { return Err(Error::Unsupported { op: OP, msg: "per-sample weights require sum mode".into() }); }
    if !matches!(weight.dtype(), DType::F32 | DType::F64) { return Err(Error::DtypeMismatch { op: OP, expected: DType::F32, got: weight.dtype() }); }
    for value in [Some(weight), Some(ids), offsets, per_sample_weights].into_iter().flatten() {
        if value.device() != Device::Cpu { return Err(Error::Unsupported { op: OP, msg: "embedding bags require CPU tensors".into() }); }
    }
    if ids.dtype() != DType::I64 { return Err(Error::DtypeMismatch { op: OP, expected: DType::I64, got: ids.dtype() }); }
    if weight.ndim() != 2 || !matches!(ids.ndim(), 1 | 2) { return Err(invalid("weight must be rank two and input rank one or two")); }
    let (rows, width) = (weight.shape()[0], weight.shape()[1]);
    if padding.is_some_and(|row| row >= rows) { return Err(invalid("padding_idx is outside the embedding table")); }
    crate::typed_math::check_shape(OP, weight.shape())?;
    if let Some(samples) = per_sample_weights {
        if samples.dtype() != weight.dtype() { return Err(Error::DtypeMismatch { op: OP, expected: weight.dtype(), got: samples.dtype() }); }
        if samples.shape() != ids.shape() { return Err(invalid("per-sample weights must match the input shape")); }
        crate::typed_math::check_shape(OP, samples.shape())?;
    }
    let (bags, boundaries) = if ids.ndim() == 2 {
        if offsets.is_some() { return Err(invalid("two-dimensional input must not supply offsets")); }
        if ids.shape()[1] == 0 { return Err(invalid("two-dimensional input needs a positive bag width")); }
        (ids.shape()[0], Vec::new())
    } else {
        let offsets = offsets.ok_or_else(|| invalid("one-dimensional input requires offsets"))?;
        if offsets.dtype() != DType::I64 { return Err(Error::DtypeMismatch { op: OP, expected: DType::I64, got: offsets.dtype() }); }
        if offsets.ndim() != 1 { return Err(invalid("offsets must be one-dimensional")); }
        crate::typed_math::check_shape(OP, offsets.shape())?;
        let values = offsets.to_vec_i64();
        if (include_last_offset && values.is_empty()) || (values.is_empty() && ids.numel() != 0) { return Err(invalid("offsets must delimit every input index")); }
        if values.first().is_some_and(|&v| v != 0) { return Err(invalid("the first offset must be zero")); }
        if values.iter().any(|&v| v < 0 || v as u128 > ids.numel() as u128) || values.windows(2).any(|v| v[0] > v[1]) { return Err(invalid("offsets must be monotone and within the input length")); }
        if include_last_offset && values.last().copied() != Some(ids.numel() as i64) { return Err(invalid("the final offset must equal the input length")); }
        (values.len() - usize::from(include_last_offset), values)
    };
    let output_shape = [bags, width];
    crate::typed_math::check_shape(OP, &output_shape)?;
    let compact = if width == 0 {
        let shape = ids.shape().iter().zip(ids.strides()).map(|(&n, &s)| if s == 0 && n != 0 { 1 } else { n }).collect();
        Tensor::from_parts(ids.0.storage.clone(), shape, ids.strides().to_vec(), ids.storage_offset(), Device::Cpu, false, None)
    } else { ids.clone() };
    crate::typed_math::check_shape(OP, compact.shape())?;
    let indices = compact.to_vec_i64().into_iter().map(|value| usize::try_from(value).ok().filter(|&row| row < rows).ok_or_else(|| invalid("input index is outside the embedding table"))).collect::<Result<Vec<_>>>()?;
    if let Some(max_norm) = max_norm { super::embedding_options::renorm(weight, &indices, max_norm, norm_type)?; }
    let token_count = ids.numel();
    let fixed_width = if ids.ndim() == 2 { ids.shape()[1] } else { 0 };
    let range = move |bag: usize| if fixed_width != 0 { bag * fixed_width..(bag + 1) * fixed_width } else { boundaries[bag] as usize..boundaries.get(bag + 1).map_or(token_count, |&v| v as usize) };
    let mut divisors = HashMap::new();
    if scale_grad_by_freq && width != 0 {
        let mut counts = HashMap::new();
        for &row in &indices { *counts.entry(row).or_insert(0usize) += 1; }
        let mut sorted = indices.clone(); sorted.sort_unstable();
        let mut unique: Vec<_> = counts.keys().copied().collect(); unique.sort_unstable();
        // Torch CPU uses the sorted unique ordinal to select the count's ID.
        for (ordinal, row) in unique.into_iter().enumerate() { divisors.insert(row, counts[&sorted[ordinal]]); }
    }
    let weight_shape = weight.shape().to_vec();
    let samples = per_sample_weights.cloned();
    let input_shape = ids.shape().to_vec();
    let saved_weight = weight.clone();
    macro_rules! compute {
        ($ty:ty, $variant:ident, $values:ident, $from:ident) => {{
            let mut output = crate::typed_math::filled(OP, &output_shape, 0.0 as $ty)?;
            let mut winners = if mode == 2 { crate::typed_math::filled(OP, &output_shape, usize::MAX)? } else { Vec::new() };
            let mut counts = if width != 0 { crate::typed_math::filled(OP, &[bags], 0usize)? } else { Vec::new() };
            let sample_values = if width != 0 { samples.as_ref().map(Tensor::$values) } else { None };
            if width != 0 {
                let guard = weight.0.storage.read();
                let Storage::$variant(values) = &*guard else { unreachable!() };
                for bag in 0..bags {
                    for position in range(bag) {
                        let row = indices[position];
                        if Some(row) == padding { continue; }
                        counts[bag] += 1;
                        for column in 0..width {
                            let value = values[weight.0.offset + row * weight.0.stride[0] + column * weight.0.stride[1]];
                            let target = bag * width + column;
                            if mode == 2 {
                                if winners[target] == usize::MAX || value > output[target] { output[target] = value; winners[target] = row; }
                            } else { output[target] += value * sample_values.as_ref().map_or(1.0, |v| v[position]); }
                        }
                    }
                    if mode == 1 && counts[bag] != 0 { for value in &mut output[bag * width..(bag + 1) * width] { *value /= counts[bag] as $ty; } }
                }
            }
            let output = Tensor::$from(output, &output_shape)?;
            let mut inputs = vec![weight.clone()];
            if let Some(samples) = &samples { inputs.push(samples.clone()); }
            output.record_fn_optional(inputs, move |g| {
                if width == 0 {
                    let mut result = vec![Some(Tensor::$from(Vec::new(), &weight_shape).unwrap())];
                    if samples.is_some() { result.push(Some(Tensor::$from(vec![0.0 as $ty], &[]).unwrap().expand(&input_shape).unwrap())); }
                    return result;
                }
                if g.is_structural_zero() {
                    let mut result = vec![Some(Tensor::structural_zero(&weight_shape, g.dtype(), Device::Cpu).unwrap())];
                    if samples.is_some() { result.push(Some(Tensor::structural_zero(&input_shape, g.dtype(), Device::Cpu).unwrap())); }
                    return result;
                }
                let mut gradient = crate::typed_math::filled(OP, &weight_shape, 0.0 as $ty).unwrap();
                let mut sample_gradient = if samples.is_some() { crate::typed_math::filled(OP, &input_shape, 0.0 as $ty).unwrap() } else { Vec::new() };
                if width != 0 {
                    let incoming = g.$values();
                    let values = if samples.is_some() { saved_weight.$values() } else { Vec::new() };
                    for bag in 0..bags {
                        if mode == 2 {
                            for column in 0..width { let row = winners[bag * width + column]; if row != usize::MAX { gradient[row * width + column] += incoming[bag * width + column]; } }
                            continue;
                        }
                        for position in range(bag) {
                            let row = indices[position];
                            if Some(row) == padding { continue; }
                            let divisor = (if mode == 1 { counts[bag] } else { 1 }) as $ty * divisors.get(&row).copied().unwrap_or(1) as $ty;
                            for column in 0..width {
                                let g = incoming[bag * width + column];
                                gradient[row * width + column] += g * sample_values.as_ref().map_or(1.0, |v| v[position]) / divisor;
                                if samples.is_some() { sample_gradient[position] += g * values[row * width + column]; }
                            }
                        }
                    }
                }
                let mut result = vec![Some(Tensor::$from(gradient, &weight_shape).unwrap())];
                if samples.is_some() { result.push(Some(Tensor::$from(sample_gradient, &input_shape).unwrap())); }
                result
            })
        }};
    }
    let mut output = match weight.dtype() { DType::F32 => compute!(f32, F32, to_vec, from_vec), DType::F64 => compute!(f64, F64, to_vec_f64, from_vec_f64), _ => unreachable!() };
    if per_sample_weights.is_none() {
        if let Some(inner) = Arc::get_mut(&mut output.0) { if let Some(op) = &mut inner.op { op.omit_saved_input_values(); } }
    }
    let mut metadata = vec![ids]; if let Some(offsets) = offsets { metadata.push(offsets); }
    Ok(output.save_tensor_versions(&metadata))
}
