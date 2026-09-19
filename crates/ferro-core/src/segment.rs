//! Unsorted integer-indexed segment algebra. Sum/softmax support resident backends.
//! Capture/replay is explicitly unsupported. Use PreparedSegments to reuse topology.
use std::sync::Arc;
use crate::dispatch::{backend_for, Backend, SegmentOp, SegmentPlan};
use crate::tensor::Storage;
use crate::tensor::device_leaf;

/// Reusable validated topology. Device preparation uploads integer metadata once;
/// free-function calls prepare per forward, while backward retains this same plan.
/// CUDA uses stable CSR edge order (no floating-point atomics). Numerical parity
/// is tolerance-based, not a cross-device bitwise contract. CUDA softmax preserves
/// finite-logit validation via one explicit four-byte control read per nonempty
/// forward; feature data and adjoints never leave the device. Mean/max remain CPU.
/// The existing infallible autograd API panics on backend backward errors; forward
/// returns them unchanged. Strided f32 CUDA inputs and cotangents materialize
/// on device. Integer metadata is immutable; prepare again when indices change.
#[derive(Clone)]
pub struct PreparedSegments {
    ids: Arc<[usize]>,
    groups: usize,
    device: Device,
    resident: Option<(Arc<dyn Backend>, Arc<dyn SegmentPlan>)>,
}

impl PreparedSegments {
    pub fn new(ids: &[usize], groups: usize, device: Device) -> Result<Self> {
        if ids.iter().any(|&i| i >= groups) {
            return Err(Error::InvalidShape { op: "prepare_segments", msg: "segment id out of bounds".into() });
        }
        let resident = if device == Device::Cpu { None } else {
            let b = backend_for(device)?;
            let plan = b.prepare_segments(ids, groups)?;
            Some((b, plan))
        };
        Ok(Self { ids: Arc::from(ids), groups, device, resident })
    }

    pub fn sum(&self, x: &Tensor) -> Result<Tensor> {
        let (width, shape) = layout_validated(x, self.ids.len(), self.groups, "segment_sum")?;
        if x.device() != self.device {
            return Err(Error::Unsupported { op: "segment_sum", msg: "topology device mismatch".into() });
        }
        if self.resident.is_none() { return sum(x, &self.ids, self.groups); }
        let out = self.run(x, None, width, &shape, SegmentOp::Sum)?;
        let plan = self.clone();
        let shape = x.shape().to_vec();
        Ok(out.record_fn(vec![x.clone()], move |g| {
            vec![plan.run(g, None, width, &shape, SegmentOp::SumBackward)
                .expect("resident segment sum backward failed")]
        }))
    }

    /// Gather rows in original ID order. Its adjoint sums duplicate IDs.
    pub fn gather(&self, x: &Tensor) -> Result<Tensor> {
        let (width, shape) = layout_validated(x, self.groups, self.ids.len(), "prepared_gather")?;
        if x.device() != self.device {
            return Err(Error::DeviceMismatch { op: "prepared_gather", lhs: x.device(), rhs: self.device });
        }
        if self.resident.is_none() { return x.index_select(0, &self.ids); }
        let out = self.run(x, None, width, &shape, SegmentOp::SumBackward)?;
        let plan = self.clone();
        let shape = x.shape().to_vec();
        Ok(out.record_fn(vec![x.clone()], move |g| {
            vec![plan.run(g, None, width, &shape, SegmentOp::Sum)
                .expect("resident gather backward failed")]
        }))
    }

    /// Select an arbitrary axis using this plan's one-dimensional IDs.
    /// The selected axis must have num_segments entries. Result is contiguous.
    pub fn select(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        if dim >= x.ndim() {
            return Err(Error::InvalidShape { op: "prepared_select", msg: "axis out of range".into() });
        }
        if dim == 0 { return self.gather(x); }
        self.gather(&x.transpose(0, dim)?)?.transpose(0, dim)?.resident_contiguous()
    }

    /// Out-of-place base + scatter(src), with repeated IDs accumulated.
    /// src has one entry per ID on dim and matches base on all other axes.
    pub fn scatter_add(&self, base: &Tensor, src: &Tensor, dim: usize) -> Result<Tensor> {
        if dim >= base.ndim() || src.ndim() != base.ndim() || base.shape()[dim] != self.groups
            || src.shape()[dim] != self.ids.len()
            || base.shape().iter().zip(src.shape()).enumerate().any(|(i,(a,b))| i != dim && a != b) {
            return Err(Error::InvalidShape { op: "prepared_scatter_add", msg: "incompatible base/source shapes or axis".into() });
        }
        if base.device() != self.device || src.device() != self.device {
            return Err(Error::Unsupported { op: "prepared_scatter_add", msg: "topology device mismatch".into() });
        }
        if base.dtype() != DType::F32 || src.dtype() != DType::F32 {
            return Err(Error::Unsupported { op: "prepared_scatter_add", msg: "f32 required".into() });
        }
        let summed = if dim == 0 { self.sum(src)? } else {
            self.sum(&src.transpose(0, dim)?)?.transpose(0, dim)?.resident_contiguous()?
        };
        base.resident_contiguous()?.add(&summed)
    }

    pub fn softmax(&self, x: &Tensor) -> Result<Tensor> {
        let (width, _) = layout_validated(x, self.ids.len(), self.groups, "segment_softmax")?;
        if x.device() != self.device {
            return Err(Error::Unsupported { op: "segment_softmax", msg: "topology device mismatch".into() });
        }
        if self.resident.is_none() { return softmax(x, &self.ids, self.groups); }
        let shape = x.shape().to_vec();
        let out = self.run(x, None, width, &shape, SegmentOp::Softmax)?;
        let y = out.detach_copy();
        let plan = self.clone();
        Ok(out.record_fn(vec![x.clone()], move |g| {
            vec![plan.run(g, Some(&y), width, &shape, SegmentOp::SoftmaxBackward)
                .expect("resident segment softmax backward failed")]
        }))
    }

    fn run(&self, x: &Tensor, saved: Option<&Tensor>, width: usize, shape: &[usize], op: SegmentOp) -> Result<Tensor> {
        let (b, plan) = self.resident.as_ref().expect("resident topology");
        if x.device() != self.device || x.dtype() != DType::F32 {
            return Err(Error::Unsupported { op: "segment_dev", msg: "resident f32 input/cotangent required on topology device".into() });
        }
        let whole = x.device_resident_whole() && x.is_contiguous();
        // Match the address-ordered StorageCell sweep used by static replay.
        let mut cells = vec![&x.0.storage];
        if let Some(y) = saved { cells.push(&y.0.storage); }
        cells.sort_unstable_by_key(|s| Arc::as_ptr(s));
        cells.dedup_by_key(|s| Arc::as_ptr(s));
        let guards: Vec<_> = cells.iter().map(|s| s.read()).collect();
        let buf = |t: &Tensor| {
            let i = cells.iter().position(|s| Arc::ptr_eq(s, &t.0.storage)).unwrap();
            let Storage::Device(b) = &*guards[i] else { unreachable!() };
            b.as_ref()
        };
        let materialized = if whole { None } else {
            Some(b.materialize_dev(buf(x), x.shape(), &x.0.stride, x.0.offset)?)
        };
        let input = materialized.as_deref().unwrap_or_else(|| buf(x));
        let out = b.segment_dev(plan.as_ref(), op, input, saved.map(buf), width)?;
        Ok(device_leaf(out, shape, self.device))
    }
}
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
    if ids.iter().any(|&i| i >= n) {
        return Err(Error::InvalidShape { op, msg: "segment id out of bounds".into() });
    }
    layout_validated(x, ids.len(), n, op)
}

fn layout_validated(x: &Tensor, edges: usize, n: usize, op: &'static str) -> Result<(usize, Vec<usize>)> {
    if crate::capture::is_recording() {
        return Err(Error::Unsupported { op, msg: "segment capture/replay is not supported".into() });
    }
    if x.dtype() != DType::F32 {
        return Err(Error::DtypeMismatch { op, expected: DType::F32, got: x.dtype() });
    }
    if x.shape().is_empty() || x.shape()[0] != edges {
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
    if x.device() != Device::Cpu {
        return PreparedSegments::new(ids, num_segments, x.device())?.sum(x);
    }
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
    cpu_f32(x, "segment_mean")?;
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
    cpu_f32(x, "segment_max")?;
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
    if x.device() != Device::Cpu {
        return PreparedSegments::new(ids, num_segments, x.device())?.softmax(x);
    }
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
