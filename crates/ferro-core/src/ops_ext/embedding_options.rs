//! Eager CPU row lookup with padding, frequency scaling and no-grad renorm.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{DType, Device, Error, Result, Tensor};
use crate::tensor::Storage;

pub fn embedding_with_options(weight: &Tensor, ids: &Tensor, padding: Option<usize>, max_norm: Option<f64>, norm_type: f64, scale_grad_by_freq: bool) -> Result<Tensor> {
    const OP: &str = "embedding";
    if crate::capture::is_recording() {
        return Err(Error::Unsupported { op: OP, msg: "embedding options require eager execution".into() });
    }
    if weight.device() != Device::Cpu || ids.device() != Device::Cpu {
        return Err(Error::Unsupported { op: OP, msg: "embedding options require CPU tensors".into() });
    }
    if !matches!(weight.dtype(), DType::F32 | DType::F64) {
        return Err(Error::DtypeMismatch { op: OP, expected: DType::F32, got: weight.dtype() });
    }
    if ids.dtype() != DType::I64 {
        return Err(Error::DtypeMismatch { op: OP, expected: DType::I64, got: ids.dtype() });
    }
    if weight.ndim() != 2 {
        return Err(Error::InvalidShape { op: OP, msg: "weight must have shape [num_embeddings, embedding_dim]".into() });
    }
    let (rows, width) = (weight.shape()[0], weight.shape()[1]);
    if padding.is_some_and(|index| index >= rows) {
        return Err(Error::InvalidShape { op: OP, msg: "padding_idx is outside the embedding table".into() });
    }
    crate::typed_math::check_shape(OP, weight.shape())?;
    let mut shape = ids.shape().to_vec();
    shape.push(width);
    crate::typed_math::check_shape(OP, &shape)?;
    let compact = if width == 0 {
        let shape = ids.shape().iter().zip(ids.strides()).map(|(&n, &s)| if s == 0 && n != 0 { 1 } else { n }).collect();
        Tensor::from_parts(ids.0.storage.clone(), shape, ids.strides().to_vec(), ids.storage_offset(), Device::Cpu, false, None)
    } else { ids.clone() };
    let indices = compact.to_vec_i64().into_iter().map(|value| {
        usize::try_from(value).ok().filter(|&value| value < rows).ok_or_else(|| Error::InvalidShape { op: OP, msg: format!("index {value} is outside the embedding table") })
    }).collect::<Result<Vec<_>>>()?;
    if let Some(max_norm) = max_norm { renorm(weight, &indices, max_norm, norm_type)?; }
    macro_rules! gather {
        ($ty:ty, $variant:ident, $from:ident) => {{
            let mut output = crate::typed_math::filled(OP, &shape, 0.0 as $ty)?;
            if width != 0 {
                let guard = weight.0.storage.read();
                let Storage::$variant(values) = &*guard else { unreachable!() };
                for (position, &row) in indices.iter().enumerate() {
                    for column in 0..width {
                        output[position * width + column] = values[weight.0.offset + row * weight.0.stride[0] + column * weight.0.stride[1]];
                    }
                }
            }
            Tensor::$from(output, &shape)?
        }};
    }
    let output = match weight.dtype() {
        DType::F32 => gather!(f32, F32, from_vec),
        DType::F64 => gather!(f64, F64, from_vec_f64),
        _ => unreachable!(),
    };
    let mut counts = HashMap::new();
    if scale_grad_by_freq && width != 0 {
        for &index in &indices { *counts.entry(index).or_insert(0usize) += 1; }
    }
    let weight_shape = weight.shape().to_vec();
    let dtype = weight.dtype();
    let mut output = output.record_fn_optional(vec![weight.clone()], move |g| {
        if g.is_structural_zero() {
            return vec![Some(Tensor::structural_zero(&weight_shape, dtype, Device::Cpu).unwrap())];
        }
        macro_rules! scatter {
            ($ty:ty, $values:ident, $from:ident) => {{
                let mut data = crate::typed_math::filled(OP, &weight_shape, 0.0 as $ty).unwrap();
                if width != 0 {
                    let gradient = g.$values();
                    for (position, &row) in indices.iter().enumerate() {
                        if Some(row) == padding { continue; }
                        let divisor = if scale_grad_by_freq { counts[&row] as $ty } else { 1.0 };
                        for column in 0..width {
                            data[row * width + column] += gradient[position * width + column] / divisor;
                        }
                    }
                }
                Tensor::$from(data, &weight_shape).unwrap()
            }};
        }
        let gradient = match dtype {
            DType::F32 => scatter!(f32, to_vec, from_vec),
            DType::F64 => scatter!(f64, to_vec_f64, from_vec_f64),
            _ => unreachable!(),
        };
        vec![Some(gradient)]
    });
    if let Some(inner) = Arc::get_mut(&mut output.0) {
        if let Some(op) = &mut inner.op { op.omit_saved_input_values(); }
    }
    Ok(output.save_tensor_versions(&[ids]))
}

// Callers validate table dtype/layout metadata and every index before renorm.
pub(crate) fn renorm(weight: &Tensor, indices: &[usize], max_norm: f64, norm_type: f64) -> Result<()> {
    const OP: &str = "embedding";
    let width = weight.shape()[1];
    // Validate the destination before constructing or writing any updates.
    crate::inplace::raw_copy_(OP, weight, weight)?;
    let unique: HashSet<_> = indices.iter().copied().collect();
    macro_rules! renorm {
        ($ty:ty, $values:ident, $from:ident) => {{
            let mut values = weight.$values();
            let p = norm_type as $ty;
            for &row in &unique {
                let row = &mut values[row * width..(row + 1) * width];
                let norm = if p == 0.0 {
                    row.iter().filter(|&&value| value != 0.0).count() as $ty
                } else if row.iter().any(|value| value.is_nan()) {
                    <$ty>::NAN
                } else if p == <$ty>::INFINITY {
                    row.iter().fold(0.0 as $ty, |value, next| value.max(next.abs()))
                } else if p == <$ty>::NEG_INFINITY {
                    row.iter().fold(<$ty>::INFINITY, |value, next| value.min(next.abs()))
                } else if p == 2.0 {
                    row.iter().map(|value| value * value).sum::<$ty>().sqrt()
                } else {
                    row.iter().map(|value| value.abs().powf(p)).sum::<$ty>().powf(1.0 / p)
                };
                if norm > max_norm as $ty {
                    let scale = max_norm as $ty / (norm + 1e-7 as $ty);
                    for value in row { *value *= scale; }
                }
            }
            let updated = Tensor::$from(values, weight.shape())?;
            crate::inplace::raw_copy_(OP, weight, &updated)?;
        }};
    }
    match weight.dtype() {
        DType::F32 => renorm!(f32, to_vec, from_vec),
        DType::F64 => renorm!(f64, to_vec_f64, from_vec_f64),
        _ => unreachable!(),
    }
    Ok(())
}
