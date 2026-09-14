//! Unsorted, integer-indexed segment algebra. CPU f32 only; no capture/replay support.
use crate::{DType, Device, Error, Result, Tensor};

pub(crate) fn cpu_f32(x: &Tensor, op: &'static str) -> Result<()> {
    if x.device() != Device::Cpu {
        return Err(Error::Unsupported { op, msg: "CPU tensors required; move inputs explicitly".into() });
    }
    if x.dtype() != DType::F32 {
        return Err(Error::DtypeMismatch { op, expected: DType::F32, got: x.dtype() });
    }
    Ok(())
}

fn layout(x: &Tensor, ids: &[usize], n: usize, op: &'static str) -> Result<(usize, Vec<usize>)> {
    cpu_f32(x, op)?;
    if x.shape().is_empty() || x.shape()[0] != ids.len() || ids.iter().any(|&i| i >= n) {
        return Err(Error::InvalidShape { op, msg: "expected [edges, ...] and one in-bounds segment id per edge".into() });
    }
    let width = x.shape()[1..].iter().try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| Error::InvalidShape { op, msg: "feature size overflow".into() })?;
    n.checked_mul(width).ok_or_else(|| Error::InvalidShape { op, msg: "output size overflow".into() })?;
    let mut shape = x.shape().to_vec();
    shape[0] = n;
    Ok((width, shape))
}

/// Sum along axis zero; missing segments are zero. Repeated ids accumulate.
pub fn sum(x: &Tensor, ids: &[usize], num_segments: usize) -> Result<Tensor> {
    let (width, shape) = layout(x, ids, num_segments, "segment_sum")?;
    let data = x.to_vec();
    let mut out = vec![0.; num_segments * width];
    for (e, &s) in ids.iter().enumerate() {
        for f in 0..width { out[s * width + f] += data[e * width + f]; }
    }
    let ids = ids.to_vec();
    let input_shape = x.shape().to_vec();
    Ok(Tensor::from_vec(out, &shape)?.record_fn(vec![x.clone()], move |g| {
        let g = g.to_vec();
        let mut dx = vec![0.; ids.len() * width];
        for (e, &s) in ids.iter().enumerate() {
            for f in 0..width { dx[e * width + f] = g[s * width + f]; }
        }
        vec![Tensor::from_vec(dx, &input_shape).unwrap()]
    }))
}


/// Mean along axis zero; empty segments are zero (and have no input adjoint).
pub fn mean(x: &Tensor, ids: &[usize], num_segments: usize) -> Result<Tensor> {
    let (width, shape) = layout(x, ids, num_segments, "segment_mean")?;
    let data = x.to_vec();
    let mut counts = vec![0usize; num_segments];
    let mut out = vec![0.; num_segments * width];
    for (e, &s) in ids.iter().enumerate() {
        counts[s] += 1;
        for f in 0..width { out[s * width + f] += data[e * width + f]; }
    }
    for s in 0..num_segments {
        if counts[s] != 0 { for f in 0..width { out[s * width + f] /= counts[s] as f32; } }
    }
    let ids = ids.to_vec();
    let input_shape = x.shape().to_vec();
    Ok(Tensor::from_vec(out, &shape)?.record_fn(vec![x.clone()], move |g| {
        let g = g.to_vec();
        let mut dx = vec![0.; ids.len() * width];
        for (e, &s) in ids.iter().enumerate() {
            for f in 0..width { dx[e * width + f] = g[s * width + f] / counts[s] as f32; }
        }
        vec![Tensor::from_vec(dx, &input_shape).unwrap()]
    }))
}

/// Coordinatewise maximum; empty segments are -infinity. Tied maxima share
/// the adjoint equally, including infinite ties. NaNs are rejected.
pub fn max(x: &Tensor, ids: &[usize], num_segments: usize) -> Result<Tensor> {
    let (width, shape) = layout(x, ids, num_segments, "segment_max")?;
    let data = x.to_vec();
    if data.iter().any(|v| v.is_nan()) {
        return Err(Error::Unsupported { op: "segment_max", msg: "NaN input".into() });
    }
    let mut out = vec![f32::NEG_INFINITY; num_segments * width];
    for (e, &s) in ids.iter().enumerate() {
        for f in 0..width { out[s * width + f] = out[s * width + f].max(data[e * width + f]); }
    }
    let mut counts = vec![0usize; out.len()];
    for (e, &s) in ids.iter().enumerate() {
        for f in 0..width { if data[e * width + f] == out[s * width + f] { counts[s * width + f] += 1; } }
    }
    let ids = ids.to_vec();
    let input_shape = x.shape().to_vec();
    Ok(Tensor::from_vec(out.clone(), &shape)?.record_fn(vec![x.clone()], move |g| {
        let g = g.to_vec();
        let mut dx = vec![0.; data.len()];
        for (e, &s) in ids.iter().enumerate() {
            for f in 0..width {
                let i = e * width + f;
                let j = s * width + f;
                if data[i] == out[j] { dx[i] = g[j] / counts[j] as f32; }
            }
        }
        vec![Tensor::from_vec(dx, &input_shape).unwrap()]
    }))
}

/// Stable per-segment, per-feature softmax with the same shape as x.
/// Empty segments emit no entries. Inputs must be finite.
pub fn softmax(x: &Tensor, ids: &[usize], num_segments: usize) -> Result<Tensor> {
    let (width, _) = layout(x, ids, num_segments, "segment_softmax")?;
    let data = x.to_vec();
    if data.iter().any(|v| !v.is_finite()) {
        return Err(Error::Unsupported { op: "segment_softmax", msg: "finite logits required".into() });
    }
    let mut maxima = vec![f32::NEG_INFINITY; num_segments * width];
    for (e, &s) in ids.iter().enumerate() {
        for f in 0..width { maxima[s * width + f] = maxima[s * width + f].max(data[e * width + f]); }
    }
    let mut denom = vec![0.; maxima.len()];
    let mut out = vec![0.; data.len()];
    for (e, &s) in ids.iter().enumerate() {
        for f in 0..width {
            out[e * width + f] = (data[e * width + f] - maxima[s * width + f]).exp();
            denom[s * width + f] += out[e * width + f];
        }
    }
    for (e, &s) in ids.iter().enumerate() {
        for f in 0..width { out[e * width + f] /= denom[s * width + f]; }
    }
    let ids = ids.to_vec();
    let shape = x.shape().to_vec();
    Ok(Tensor::from_vec(out.clone(), &shape)?.record_fn(vec![x.clone()], move |g| {
        let g = g.to_vec();
        let mut dot = vec![0.; num_segments * width];
        for (e, &s) in ids.iter().enumerate() {
            for f in 0..width { dot[s * width + f] += g[e * width + f] * out[e * width + f]; }
        }
        let mut dx = vec![0.; out.len()];
        for (e, &s) in ids.iter().enumerate() {
            for f in 0..width { dx[e * width + f] = out[e * width + f] * (g[e * width + f] - dot[s * width + f]); }
        }
        vec![Tensor::from_vec(dx, &shape).unwrap()]
    }))
}
