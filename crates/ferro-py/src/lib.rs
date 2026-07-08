mod dlpack;

use ferro_core::{Rng, Tensor as CoreTensor};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

fn map_err(e: ferro_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
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

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<PyTensor> {
        CoreTensor::from_vec(data, &shape).map(PyTensor::wrap).map_err(map_err)
    }

    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> PyTensor {
        PyTensor::wrap(CoreTensor::zeros(&shape))
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>) -> PyTensor {
        PyTensor::wrap(CoreTensor::ones(&shape))
    }

    #[staticmethod]
    fn randn(shape: Vec<usize>, seed: u64) -> PyTensor {
        PyTensor::wrap(CoreTensor::randn(&shape, &Rng::new(seed)))
    }

    /// I64 index tensor (for gather/rope positions and future index ops).
    #[staticmethod]
    fn from_i64(data: Vec<i64>, shape: Vec<usize>) -> PyResult<PyTensor> {
        CoreTensor::from_vec_i64(data, &shape).map(PyTensor::wrap).map_err(map_err)
    }

    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner.add(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner.sub(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner.mul(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn __truediv__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner.div(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner.matmul(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn __neg__(&self) -> PyTensor {
        PyTensor::wrap(self.inner.neg())
    }

    fn relu(&self) -> PyTensor {
        PyTensor::wrap(self.inner.relu())
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
    fn sum_dim(&self, dim: usize, keepdim: bool) -> PyResult<PyTensor> {
        self.inner.sum_dim(dim, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn mean_dim(&self, dim: usize, keepdim: bool) -> PyResult<PyTensor> {
        self.inner.mean_dim(dim, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    fn softmax(&self, dim: usize) -> PyResult<PyTensor> {
        self.inner.softmax(dim).map(PyTensor::wrap).map_err(map_err)
    }

    fn log_softmax(&self, dim: usize) -> PyResult<PyTensor> {
        self.inner.log_softmax(dim).map(PyTensor::wrap).map_err(map_err)
    }

    fn bmm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.inner.bmm(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn cumsum(&self, dim: usize) -> PyResult<PyTensor> {
        self.inner.cumsum(dim).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn argmax(&self, dim: usize, keepdim: bool) -> PyResult<PyTensor> {
        self.inner.argmax(dim, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn argmin(&self, dim: usize, keepdim: bool) -> PyResult<PyTensor> {
        self.inner.argmin(dim, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    fn topk(&self, k: usize, dim: usize) -> PyResult<(PyTensor, PyTensor)> {
        let (v, i) = self.inner.topk(k, dim).map_err(map_err)?;
        Ok((PyTensor::wrap(v), PyTensor::wrap(i)))
    }

    fn gather(&self, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
        self.inner.gather(dim, &index.inner).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (positions, base=10000.0))]
    fn rope(&self, positions: &PyTensor, base: f32) -> PyResult<PyTensor> {
        self.inner.rope(&positions.inner, base).map(PyTensor::wrap).map_err(map_err)
    }

    fn index_select(&self, dim: usize, indices: Vec<usize>) -> PyResult<PyTensor> {
        self.inner.index_select(dim, &indices).map(PyTensor::wrap).map_err(map_err)
    }

    fn squeeze(&self, dim: usize) -> PyResult<PyTensor> {
        self.inner.squeeze(dim).map(PyTensor::wrap).map_err(map_err)
    }

    fn unsqueeze(&self, dim: usize) -> PyResult<PyTensor> {
        self.inner.unsqueeze(dim).map(PyTensor::wrap).map_err(map_err)
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

    fn transpose(&self, d0: usize, d1: usize) -> PyResult<PyTensor> {
        self.inner.transpose(d0, d1).map(PyTensor::wrap).map_err(map_err)
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        self.inner.reshape(&shape).map(PyTensor::wrap).map_err(map_err)
    }

    /// In-place like torch: mutates self and returns it for chaining.
    fn requires_grad_(mut slf: PyRefMut<'_, Self>, req: bool) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.requires_grad_(req);
        slf
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
        format!("Tensor(shape={:?}, data={:?})", self.inner.shape(), self.inner.to_vec())
    }

    /// DLPack producer. `stream` is accepted for protocol compatibility but
    /// ignored: this is a synchronous CPU tensor.
    #[pyo3(signature = (stream=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = stream;
        let (data, shape) = self.inner.to_contiguous();
        dlpack::export_capsule(py, data, shape)
    }

    /// DLPack device: (kDLCPU, 0).
    fn __dlpack_device__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        dlpack::dlpack_device(py)
    }
}

/// Consume any object exposing `__dlpack__` into a new ferro Tensor (copy).
#[pyfunction]
fn from_dlpack(obj: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    dlpack::import_from_dlpack(obj).map(PyTensor::wrap)
}

#[pyfunction]
fn cat(tensors: Vec<PyTensor>, dim: usize) -> PyResult<PyTensor> {
    let inners: Vec<CoreTensor> = tensors.iter().map(|t| t.inner.clone()).collect();
    CoreTensor::cat(&inners, dim).map(PyTensor::wrap).map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "where")]
fn where_(cond: &PyTensor, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    CoreTensor::where_cond(&cond.inner, &a.inner, &b.inner).map(PyTensor::wrap).map_err(map_err)
}

#[pymodule]
fn ferro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Route matmul through the optimized CPU backend for the whole process.
    ferro_fastcpu::install();
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(where_, m)?)?;
    Ok(())
}
