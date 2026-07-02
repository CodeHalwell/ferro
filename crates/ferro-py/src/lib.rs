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

#[pymodule]
fn ferro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(from_dlpack, m)?)?;
    Ok(())
}
