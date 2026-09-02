mod dlpack;

use std::time::{SystemTime, UNIX_EPOCH};

use ferro_core::{Device, Rng, Tensor as CoreTensor};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyEllipsis, PyList, PySlice, PyTuple};

fn map_err(e: ferro_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn default_seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

fn parse_device(s: &str) -> PyResult<Device> {
    if s == "cpu" {
        return Ok(Device::Cpu);
    }
    if let Some(rest) = s.strip_prefix("cuda") {
        if rest.is_empty() {
            return Ok(Device::Cuda(0));
        }
        if let Some(idx) = rest.strip_prefix(':') {
            return idx.parse::<u32>().map(Device::Cuda).map_err(|_| {
                PyValueError::new_err(format!("invalid cuda device index in {s:?}"))
            });
        }
    }
    Err(PyValueError::new_err(format!(
        "unknown device {s:?}; expected 'cpu', 'cuda' or 'cuda:N'"
    )))
}

fn norm_dim(dim: isize, ndim: usize) -> PyResult<usize> {
    let d = if dim < 0 { dim + ndim as isize } else { dim };
    if d < 0 || d as usize >= ndim {
        return Err(PyValueError::new_err(format!(
            "dim {dim} out of range for rank-{ndim} tensor"
        )));
    }
    Ok(d as usize)
}

/// Autograd-aware tensor exposed to Python. Thin wrapper over ferro_core::Tensor.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
struct PyTensor {
    inner: CoreTensor,
}

impl PyTensor {
    fn wrap(inner: CoreTensor) -> PyTensor {
        PyTensor { inner }
    }
}

/// Stateful PRNG stream: successive draws advance state instead of forcing a
/// fresh per-call seed.
#[pyclass(name = "Generator")]
struct Generator {
    rng: std::sync::Mutex<Rng>,
}

#[pymethods]
impl Generator {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Generator {
        Generator { rng: std::sync::Mutex::new(Rng::new(seed.unwrap_or_else(default_seed))) }
    }

    /// Re-seed this generator (torch.Generator.manual_seed analogue).
    fn manual_seed(&self, seed: u64) {
        *self.rng.lock().unwrap() = Rng::new(seed);
    }
}

fn tensor_operand(other: &Bound<'_, PyAny>) -> PyResult<Option<CoreTensor>> {
    if let Ok(t) = other.extract::<PyRef<'_, PyTensor>>() {
        return Ok(Some(t.inner.clone()));
    }
    if let Ok(v) = other.extract::<f64>() {
        return Ok(Some(CoreTensor::scalar(v as f32)));
    }
    Ok(None)
}

fn expect_operand(other: &Bound<'_, PyAny>) -> PyResult<CoreTensor> {
    tensor_operand(other)?.ok_or_else(|| {
        let name = other.get_type().name().map(|n| n.to_string()).unwrap_or_default();
        PyTypeError::new_err(format!("unsupported operand type: {name}"))
    })
}

/// Recursively build nested Python lists from row-major data and a shape.
fn to_nested<'py>(py: Python<'py>, data: &[f32], shape: &[usize]) -> Bound<'py, PyAny> {
    if shape.is_empty() {
        return data[0].into_pyobject(py).unwrap().into_any();
    }
    let outer = shape[0];
    let inner_shape = &shape[1..];
    let stride: usize = inner_shape.iter().product();
    let items: Vec<Bound<'py, PyAny>> = (0..outer)
        .map(|i| to_nested(py, &data[i * stride..(i + 1) * stride], inner_shape))
        .collect();
    PyList::new(py, items).unwrap().into_any()
}

fn bin_op(
    a: &CoreTensor,
    b: &CoreTensor,
    f: fn(&CoreTensor, &CoreTensor) -> ferro_core::Result<CoreTensor>,
) -> PyResult<PyTensor> {
    f(a, b).map(PyTensor::wrap).map_err(map_err)
}

/// Recursively format nested data, showing at most 4 entries per dimension
/// with an ellipsis marker where truncation happened (torch-style).
const REPR_ELEMS_PER_DIM: usize = 4;

fn fmt_data(data: &[f32], shape: &[usize]) -> String {
    if shape.is_empty() {
        return data[0].to_string();
    }
    let stride: usize = shape[1..].iter().product();
    let n = shape[0].min(REPR_ELEMS_PER_DIM);
    let mut body: Vec<String> =
        (0..n).map(|i| fmt_data(&data[i * stride..], &shape[1..])).collect();
    if shape[0] > n {
        body.push("...".into());
    }
    format!("[{}]", body.join(", "))
}

// --- basic indexing ---------------------------------------------------------

enum Sel {
    Idx(isize),
    Slice(Option<isize>, Option<isize>, Option<isize>),
}

impl Clone for Sel {
    fn clone(&self) -> Self {
        match self {
            Sel::Idx(i) => Sel::Idx(*i),
            Sel::Slice(a, b, c) => Sel::Slice(*a, *b, *c),
        }
    }
}

fn wrap_index(i: isize, len: usize) -> PyResult<usize> {
    let l = len as isize;
    let j = if i < 0 { i + l } else { i };
    if j < 0 || j >= l {
        return Err(PyValueError::new_err(format!(
            "index {i} out of bounds for dimension of size {len}"
        )));
    }
    Ok(j as usize)
}

/// Python slice semantics resolved to explicit indices; step must be nonzero.
fn slice_indices(
    start: Option<isize>,
    stop: Option<isize>,
    step: isize,
    len: usize,
) -> PyResult<Vec<usize>> {
    if step == 0 {
        return Err(PyValueError::new_err("slice step cannot be zero"));
    }
    let l = len as isize;
    let mut out = Vec::new();
    if step > 0 {
        // Python semantics: negative bounds count from the end before clamping.
        let norm = |v: isize| if v < 0 { v + l } else { v };
        let s = norm(start.unwrap_or(0)).clamp(0, l);
        let e = norm(stop.unwrap_or(l)).clamp(0, l);
        let mut i = s;
        while i < e {
            out.push(i as usize);
            i += step;
        }
    } else {
        let s = start.unwrap_or(l - 1).clamp(-1, l - 1);
        let e = stop.unwrap_or(-(l + 1)).clamp(-1, l);
        let mut i = s;
        while i > e {
            if i >= 0 {
                out.push(i as usize);
            }
            i += step;
        }
    }
    Ok(out)
}

fn resolve_sel(sel: &Sel, len: usize) -> PyResult<(Vec<usize>, bool)> {
    Ok(match sel {
        Sel::Idx(i) => (vec![wrap_index(*i, len)?], false),
        Sel::Slice(s, e, st) => (slice_indices(*s, *e, st.unwrap_or(1), len)?, true),
    })
}

fn parse_key(key: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Vec<Sel>> {
    fn one(obj: &Bound<'_, PyAny>) -> PyResult<Sel> {
        if let Ok(slice) = obj.downcast::<PySlice>() {
            let start = slice.getattr("start")?.extract::<Option<isize>>()?;
            let stop = slice.getattr("stop")?.extract::<Option<isize>>()?;
            let step = slice.getattr("step")?.extract::<Option<isize>>()?;
            return Ok(Sel::Slice(start, stop, step));
        }
        obj.extract::<isize>().map(Sel::Idx).map_err(|_| {
            PyTypeError::new_err("tensor indices must be integers or slices")
        })
    }
    if key.is_instance_of::<PyEllipsis>() {
        return Ok(vec![Sel::Slice(None, None, None); ndim]);
    }
    if let Ok(tuple) = key.downcast::<PyTuple>() {
        // One Ellipsis expands to fill the un-specified dims.
        let mut sels: Vec<Sel> = Vec::new();
        for obj in tuple.iter() {
            if obj.is_instance_of::<PyEllipsis>() {
                let specified = tuple.len().saturating_sub(1);
                let fill = ndim.saturating_sub(specified + sels.len());
                sels.extend(std::iter::repeat(Sel::Slice(None, None, None)).take(fill));
            } else {
                sels.push(one(&obj)?);
            }
        }
        if sels.len() > ndim {
            return Err(PyValueError::new_err(format!(
                "too many indices ({}) for rank-{ndim} tensor",
                sels.len()
            )));
        }
        return Ok(sels);
    }
    Ok(vec![one(key)?])
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<PyTensor> {
        CoreTensor::from_vec(data, &shape).map(PyTensor::wrap).map_err(map_err)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, device="cpu"))]
    fn zeros(shape: Vec<usize>, device: &str) -> PyResult<PyTensor> {
        CoreTensor::full_on(&shape, 0.0, parse_device(device)?)
            .map(PyTensor::wrap)
            .map_err(map_err)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, device="cpu"))]
    fn ones(shape: Vec<usize>, device: &str) -> PyResult<PyTensor> {
        CoreTensor::full_on(&shape, 1.0, parse_device(device)?)
            .map(PyTensor::wrap)
            .map_err(map_err)
    }

    /// Draw standard normals. Pass `seed` for a one-shot reproducible stream,
    /// a `Generator` to advance shared state across calls, or neither for a
    /// time-seeded draw.
    #[staticmethod]
    #[pyo3(signature = (shape, seed=None, generator=None, device="cpu"))]
    fn randn(
        shape: Vec<usize>,
        seed: Option<u64>,
        generator: Option<PyRef<'_, Generator>>,
        device: &str,
    ) -> PyResult<PyTensor> {
        let dev = parse_device(device)?;
        let fallback;
        let gen_guard =
            generator.as_deref().filter(|_| seed.is_none()).map(|g| g.rng.lock().unwrap());
        let rng: &Rng = if let Some(s) = seed {
            fallback = Rng::new(s);
            &fallback
        } else if let Some(g) = &gen_guard {
            g
        } else {
            fallback = Rng::new(default_seed());
            &fallback
        };
        CoreTensor::randn(&shape, rng).to_device(dev).map(PyTensor::wrap).map_err(map_err)
    }

    /// Draw uniform samples in [0, 1) with the same seeding rules as randn.
    #[staticmethod]
    #[pyo3(signature = (shape, seed=None, generator=None, device="cpu"))]
    fn rand(
        shape: Vec<usize>,
        seed: Option<u64>,
        generator: Option<PyRef<'_, Generator>>,
        device: &str,
    ) -> PyResult<PyTensor> {
        let dev = parse_device(device)?;
        let fallback;
        let gen_guard =
            generator.as_deref().filter(|_| seed.is_none()).map(|g| g.rng.lock().unwrap());
        let rng: &Rng = if let Some(s) = seed {
            fallback = Rng::new(s);
            &fallback
        } else if let Some(g) = &gen_guard {
            g
        } else {
            fallback = Rng::new(default_seed());
            &fallback
        };
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.uniform()).collect();
        CoreTensor::from_vec(data, &shape)
            .and_then(|t| t.to_device(dev))
            .map(PyTensor::wrap)
            .map_err(map_err)
    }

    /// I64 index tensor (for gather/rope positions and future index ops).
    #[staticmethod]
    fn from_i64(data: Vec<i64>, shape: Vec<usize>) -> PyResult<PyTensor> {
        CoreTensor::from_vec_i64(data, &shape).map(PyTensor::wrap).map_err(map_err)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &expect_operand(other)?, CoreTensor::add)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &expect_operand(other)?, CoreTensor::add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &expect_operand(other)?, CoreTensor::sub)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&expect_operand(other)?, &self.inner, CoreTensor::sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &expect_operand(other)?, CoreTensor::mul)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &expect_operand(other)?, CoreTensor::mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &expect_operand(other)?, CoreTensor::div)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&expect_operand(other)?, &self.inner, CoreTensor::div)
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner.matmul(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn __matmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = tensor_operand(other)?
            .ok_or_else(|| PyTypeError::new_err("matmul requires a Tensor operand"))?;
        bin_op(&self.inner, &b, CoreTensor::matmul)
    }

    fn __rmatmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = tensor_operand(other)?
            .ok_or_else(|| PyTypeError::new_err("matmul requires a Tensor operand"))?;
        bin_op(&b, &self.inner, CoreTensor::matmul)
    }

    fn __neg__(&self) -> PyTensor {
        PyTensor::wrap(self.inner.neg())
    }

    fn relu(&self) -> PyTensor {
        PyTensor::wrap(self.inner.relu())
    }

    /// Evaluate the pointwise expression that produced this tensor as fused
    /// kernels: capture the op graph rooted here, run the fusion planner, and
    /// execute each detected pointwise chain in ONE launch (via the backend's
    /// `chain_dev`) instead of one launch per op. Returns a NEW detached
    /// tensor with the same values as `self` but computed with fewer launches
    /// and no global-memory round-trips for fused intermediates.
    ///
    /// Detached: the result carries no autograd graph. Use on an inference /
    /// forward-only expression like `(x.relu() * y + z).fuse()`. On a chain
    /// with no fusible run this is a correct no-op copy.
    fn fuse(&self) -> PyResult<PyTensor> {
        let g = ferro_core::graph::Graph::from_root(&self.inner);
        g.eval_fused().map(PyTensor::wrap).map_err(map_err)
    }

    /// (launches_before, launches_after) the fusion planner would issue for the
    /// pointwise graph rooted at this tensor. A structural proof hook: lets a
    /// caller assert fusion actually collapsed launches, not just that numbers
    /// came out equal.
    fn fusion_launches(&self) -> (usize, usize) {
        let g = ferro_core::graph::Graph::from_root(&self.inner);
        let p = g.plan_fusion();
        (p.launches_before, p.launches_after)
    }

    fn sigmoid(&self) -> PyTensor {
        PyTensor::wrap(self.inner.sigmoid())
    }

    fn exp(&self) -> PyTensor {
        PyTensor::wrap(self.inner.exp())
    }

    fn log(&self) -> PyTensor {
        PyTensor::wrap(self.inner.log())
    }

    fn tanh(&self) -> PyTensor {
        PyTensor::wrap(self.inner.tanh())
    }

    fn gelu(&self) -> PyTensor {
        PyTensor::wrap(self.inner.gelu())
    }

    fn sqrt(&self) -> PyTensor {
        PyTensor::wrap(self.inner.sqrt())
    }

    fn abs(&self) -> PyTensor {
        PyTensor::wrap(self.inner.abs())
    }

    fn pow(&self, p: f32) -> PyTensor {
        PyTensor::wrap(self.inner.powf(p))
    }

    fn clamp(&self, min: f32, max: f32) -> PyTensor {
        PyTensor::wrap(self.inner.clamp(min, max))
    }

    fn max(&self) -> PyResult<PyTensor> {
        self.inner.max().map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn sum_dim(&self, dim: isize, keepdim: bool) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.sum_dim(d, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn mean_dim(&self, dim: isize, keepdim: bool) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.mean_dim(d, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    fn softmax(&self, dim: isize) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.softmax(d).map(PyTensor::wrap).map_err(map_err)
    }

    fn log_softmax(&self, dim: isize) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.log_softmax(d).map(PyTensor::wrap).map_err(map_err)
    }

    fn bmm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner.bmm(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn cumsum(&self, dim: isize) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.cumsum(d).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn argmax(&self, dim: isize, keepdim: bool) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.argmax(d, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn argmin(&self, dim: isize, keepdim: bool) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.argmin(d, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    fn topk(&self, k: usize, dim: isize) -> PyResult<(PyTensor, PyTensor)> {
        let d = norm_dim(dim, self.inner.ndim())?;
        let (v, i) = self.inner.topk(k, d).map_err(map_err)?;
        Ok((PyTensor::wrap(v), PyTensor::wrap(i)))
    }

    fn gather(&self, dim: isize, index: &PyTensor) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.gather(d, &index.inner).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (positions, base=10000.0))]
    fn rope(&self, positions: &PyTensor, base: f32) -> PyResult<PyTensor> {
        self.inner.rope(&positions.inner, base).map(PyTensor::wrap).map_err(map_err)
    }

    fn index_select(&self, dim: isize, indices: Vec<isize>) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        let len = self.inner.shape()[d];
        let norm: Vec<usize> = indices.iter().map(|i| wrap_index(*i, len)).collect::<PyResult<_>>()?;
        self.inner.index_select(d, &norm).map(PyTensor::wrap).map_err(map_err)
    }

    fn squeeze(&self, dim: isize) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.squeeze(d).map(PyTensor::wrap).map_err(map_err)
    }

    fn unsqueeze(&self, dim: isize) -> PyResult<PyTensor> {
        let ndim = self.inner.ndim() + 1;
        let d = if dim < 0 { dim + ndim as isize } else { dim };
        if d < 0 || d as usize > ndim {
            return Err(PyValueError::new_err(format!(
                "dim {dim} out of range for insertion into rank-{} tensor",
                ndim - 1
            )));
        }
        self.inner.unsqueeze(d as usize).map(PyTensor::wrap).map_err(map_err)
    }

    fn transpose(&self, d0: isize, d1: isize) -> PyResult<PyTensor> {
        let ndim = self.inner.ndim();
        self.inner.transpose(norm_dim(d0, ndim)?, norm_dim(d1, ndim)?)
            .map(PyTensor::wrap)
            .map_err(map_err)
    }

    #[pyo3(signature = (weight, stride=1, padding=0))]
    fn conv2d(&self, weight: &PyTensor, stride: usize, padding: usize) -> PyResult<PyTensor> {
        self.inner.conv2d(&weight.inner, stride, padding).map(PyTensor::wrap).map_err(map_err)
    }

    fn max_pool2d(&self, kernel: usize, stride: usize) -> PyResult<PyTensor> {
        self.inner.max_pool2d(kernel, stride).map(PyTensor::wrap).map_err(map_err)
    }

    fn sum(&self) -> PyTensor {
        PyTensor::wrap(self.inner.sum())
    }

    fn mean(&self) -> PyTensor {
        PyTensor::wrap(self.inner.mean())
    }

    /// Basic indexing: ints and slices per dimension, negative indices/steps
    /// supported. The result is a detached copy (no indexing autograd yet).
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let shape = self.inner.shape().to_vec();
        let picked = parse_key(key, shape.len())?;
        let mut dims: Vec<Vec<usize>> = Vec::with_capacity(shape.len());
        let mut out_shape: Vec<usize> = Vec::new();
        for (i, len) in shape.iter().enumerate() {
            let sel = picked.get(i).cloned().unwrap_or(Sel::Slice(None, None, None));
            let (idx, keep) = resolve_sel(&sel, *len)?;
            if keep {
                out_shape.push(idx.len());
            }
            dims.push(idx);
        }
        let data = self.inner.to_vec();
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let counts: Vec<usize> = dims.iter().map(|d| d.len()).collect();
        let total: usize = counts.iter().product();
        let mut out = Vec::with_capacity(total);
        for n in 0..total {
            // Last dim varies fastest (row-major), matching the output shape.
            let (mut rem, mut off) = (n, 0usize);
            for d in (0..counts.len()).rev() {
                off += dims[d][rem % counts[d]] * strides[d];
                rem /= counts[d];
            }
            out.push(data[off]);
        }
        CoreTensor::from_vec(out, &out_shape).map(PyTensor::wrap).map_err(map_err)
    }

    /// Move this tensor to `device` ('cpu', 'cuda' or 'cuda:N').
    fn to(&self, device: &str) -> PyResult<PyTensor> {
        self.inner.to_device(parse_device(device)?).map(PyTensor::wrap).map_err(map_err)
    }

    fn cpu(&self) -> PyResult<PyTensor> {
        self.inner.to_device(Device::Cpu).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (index=0))]
    fn cuda(&self, index: u32) -> PyResult<PyTensor> {
        self.inner.to_device(Device::Cuda(index)).map(PyTensor::wrap).map_err(map_err)
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    /// Statement-style flag setter (torch idiom). Returns the same Python
    /// object (a new reference to self, identity preserved), so both
    /// `t.requires_grad_(True)` and `t = t.requires_grad_(True)` are safe;
    /// it is not a copy.
    fn requires_grad_(mut slf: PyRefMut<'_, Self>, req: bool) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner = slf.inner.requires_grad_(req).map_err(map_err)?;
        Ok(slf)
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    fn backward(&self) -> PyResult<()> {
        if self.inner.numel() != 1 {
            return Err(PyValueError::new_err(
                "backward() requires a scalar output; reduce with .sum() or .mean()",
            ));
        }
        self.inner.backward();
        Ok(())
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn detach(&self) -> PyTensor {
        PyTensor::wrap(self.inner.detach_copy())
    }

    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(PyTensor::wrap)
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let dims: Vec<usize> = self.inner.shape().to_vec();
        PyList::new(py, dims).unwrap().into_any()
    }

    fn tolist<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        to_nested(py, &self.inner.to_vec(), self.inner.shape())
    }

    fn item(&self) -> PyResult<f32> {
        if self.inner.numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "item() requires a single-element tensor, got shape {:?}",
                self.inner.shape()
            )));
        }
        Ok(self.inner.item())
    }

    fn __repr__(&self) -> String {
        let dev = self.inner.device().to_string();
        let dev_s = if dev == "cpu" { String::new() } else { format!("device='{dev}', ") };
        format!(
            "Tensor({:?}, {}dtype=f32, data={})",
            self.inner.shape(),
            dev_s,
            fmt_data(&self.inner.to_vec(), self.inner.shape())
        )
    }

    /// DLPack producer. `stream` is accepted for protocol compatibility but
    /// ignored: transfers are synchronous.
    #[pyo3(signature = (stream=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = stream;
        dlpack::export_for(py, &self.inner)
    }

    /// DLPack device: (kDLCPU, 0) or (kDLCUDA, ordinal).
    fn __dlpack_device__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        dlpack::dlpack_device_for(py, &self.inner)
    }
}

/// Consume any object exposing `__dlpack__` into a new ferro Tensor (copy).
#[pyfunction]
fn from_dlpack(obj: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    dlpack::import_from_dlpack(obj).map(PyTensor::wrap)
}

#[pyfunction]
fn cat(tensors: Vec<PyTensor>, dim: isize) -> PyResult<PyTensor> {
    let inners: Vec<CoreTensor> = tensors.iter().map(|t| t.inner.clone()).collect();
    let ndim = inners.first().map(|t| t.ndim()).unwrap_or(0);
    let d = norm_dim(dim, ndim)?;
    CoreTensor::cat(&inners, d).map(PyTensor::wrap).map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "where")]
fn where_(cond: &PyTensor, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    CoreTensor::where_cond(&cond.inner, &a.inner, &b.inner).map(PyTensor::wrap).map_err(map_err)
}

/// Write a `{name: Tensor}` dict to a .safetensors file.
#[pyfunction]
fn save_safetensors(path: &str, tensors: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut pairs: Vec<(String, CoreTensor)> = Vec::new();
    for (k, v) in tensors.iter() {
        pairs.push((k.extract()?, v.extract::<PyTensor>()?.inner));
    }
    let refs: Vec<(&str, &CoreTensor)> = pairs.iter().map(|(n, t)| (n.as_str(), t)).collect();
    ferro_core::save_safetensors(path, &refs).map_err(map_err)
}

/// Read a .safetensors file into a `{name: Tensor}` dict (header order).
#[pyfunction]
fn load_safetensors<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, t) in ferro_core::load_safetensors(path).map_err(map_err)? {
        d.set_item(name, PyTensor::wrap(t))?;
    }
    Ok(d)
}

/// Initialise and register the CUDA backend for device `index` (default 0).
/// Must be called before moving tensors to `cuda`/`cuda:N`. Returns `True` on
/// success; raises with the driver/runtime error string when no usable CUDA
/// device is present (never panics). Idempotent - a second call is a no-op.
#[pyfunction]
#[pyo3(signature = (index=0))]
fn cuda_init(index: u32) -> PyResult<bool> {
    ferro_cuda::install(index).map_err(PyValueError::new_err)?;
    Ok(true)
}

/// Whether a CUDA driver + device are visible to ferro (does not register).
#[pyfunction]
fn cuda_is_available() -> bool {
    ferro_cuda::is_available()
}

/// Block until all queued CUDA work on ferro's stream completes. Pure stream
/// fence (no device->host copy) - use this to bracket GPU benchmark timing so
/// it measures kernel completion, not a PCIe readback.
#[pyfunction]
fn cuda_synchronize() -> PyResult<()> {
    ferro_cuda::device_synchronize().map_err(PyValueError::new_err)
}

#[pymodule]
fn ferro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Route matmul through the optimized CPU backend for the whole process.
    ferro_fastcpu::install();
    m.add_class::<PyTensor>()?;
    m.add_class::<Generator>()?;
    m.add_function(wrap_pyfunction!(from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(where_, m)?)?;
    m.add_function(wrap_pyfunction!(save_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_init, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_synchronize, m)?)?;
    Ok(())
}
