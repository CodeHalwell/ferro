mod dlpack;
mod training_state;
mod architecture;
mod recurrent;
mod basis;
mod convolution;
mod losses;
mod data;
mod linalg;
mod forward_ad;

#[cfg(test)]
#[path = "../tests/support/dlpack_recorded_error.rs"]
mod dlpack_error_tests;

use std::time::{SystemTime, UNIX_EPOCH};

use ferro_core::{Device, Rng, Tensor as CoreTensor};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::types::{PyDict, PyEllipsis, PyList, PySlice, PyTuple};

fn map_err(e: ferro_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn ok(r: ferro_core::Result<CoreTensor>) -> PyResult<PyTensor> {
    r.map(PyTensor::wrap).map_err(map_err)
}

fn validate_dlpack_stream(device: Device, stream: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(stream) = stream.filter(|s| !s.is_none()) else { return Ok(()); };
    if device == Device::Cpu {
        return Err(PyValueError::new_err("CPU DLPack export requires stream=None"));
    }
    if stream.is_instance_of::<pyo3::types::PyBool>() || !stream.is_instance_of::<pyo3::types::PyInt>() {
        return Err(PyTypeError::new_err("CUDA DLPack stream must be None or an integer"));
    }
    let token = stream.extract::<usize>().map_err(|_| PyValueError::new_err(
        "CUDA DLPack stream must be a positive pointer-sized integer; no-sync (-1) is unsupported"))?;
    if token == 0 {
        return Err(PyValueError::new_err("CUDA DLPack stream 0 is ambiguous; use None, 1, or 2"));
    }
    Ok(())
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
#[pyclass(name = "Tensor", subclass)]
#[derive(Clone)]
struct PyTensor {
    inner: CoreTensor,
}

/// A precompiled pointwise graph (see `Tensor.compile_fused`). The historical
/// FusedChain name is retained; graphs may contain branches and shared nodes.
/// Replay executes the compiled schedule without a tape walk or re-planning.
#[pyclass(name = "FusedChain")]
struct PyCompiledChain {
    inner: ferro_core::graph::CompiledChain,
}

#[pymethods]
impl PyCompiledChain {
    fn prepare_static(&self) -> PyResult<PyStaticGraph> {
        self.inner.prepare_static().map(|inner| PyStaticGraph { inner }).map_err(map_err)
    }

    /// Re-run the scheduled graph from current leaf values. Shape-preserving
    /// runs attempt fused backend launches. Returns a new detached tensor.
    fn replay(&self) -> PyResult<PyTensor> {
        self.inner.replay().map(PyTensor::wrap).map_err(map_err)
    }

    /// Number of distinct graph leaves retained by the compiled handle.
    #[getter]
    fn num_operands(&self) -> usize {
        self.inner.num_operands()
    }

    /// Number of scheduled runs; backend fallback may issue more launches.
    #[getter]
    fn num_runs(&self) -> usize {
        self.inner.num_runs()
    }

    /// Number of recorded operations, counting shared nodes once.
    #[getter]
    fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
}

/// Thread-confined CUDA graph. Replay returns no alias to reusable storage.
#[pyclass(name = "StaticGraph", unsendable)]
struct PyStaticGraph { inner: ferro_core::graph::PreparedChain }

#[pymethods]
impl PyStaticGraph {
    fn replay(&mut self) -> PyResult<()> { self.inner.replay().map_err(map_err) }
    fn snapshot(&self) -> PyResult<PyTensor> { self.inner.snapshot().map(PyTensor::wrap).map_err(map_err) }
    #[getter]
    fn replay_count(&self) -> usize { self.inner.replay_count() }
}

impl PyTensor {
    fn wrap(inner: CoreTensor) -> PyTensor {
        PyTensor { inner }
    }

    fn typed_math(&self, integers: bool) -> PyResult<()> {
        use ferro_core::DType;
        if self.inner.dtype() == DType::F32 { return Ok(()); }
        if self.inner.dtype() != DType::F64 && !(integers && self.inner.dtype() == DType::I64) {
            return Err(PyTypeError::new_err("this operation requires a supported numeric dtype"));
        }
        if ferro_core::capture::is_recording() {
            return Err(PyValueError::new_err("typed CPU computation does not support inference capture"));
        }
        if self.inner.device() != Device::Cpu {
            return Err(PyValueError::new_err("typed computation is supported on CPU only"));
        }
        if self.inner.numel() > isize::MAX as usize / 8 {
            return Err(PyValueError::new_err("typed input materialization exceeds allocation capacity"));
        }
        Ok(())
    }

    fn float_math(&self) -> PyResult<()> {
        if self.inner.dtype() != ferro_core::DType::F32 {
            return Err(PyTypeError::new_err("this operation requires float32 storage"));
        }
        Ok(())
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

fn arithmetic_operand(left: &CoreTensor, other: &Bound<'_, PyAny>) -> PyResult<CoreTensor> {
    numeric_operand(left, other, false)
}

fn numpy_number<'py>(value: &Bound<'py, PyAny>, scalar: bool) -> PyResult<Option<Bound<'py, PyAny>>> {
    if scalar && value.is_instance_of::<pyo3::types::PyComplex>() { return Ok(None); }
    let py = value.py();
    let modules = PyModule::import(py, "sys")?.getattr("modules")?.downcast_into::<PyDict>()?;
    let Some(numpy) = modules.get_item("numpy")? else { return Ok(None); };
    if !value.is_instance(&numpy.getattr("generic")?)? { return Ok(None); }
    let kind: String = value.getattr("dtype")?.getattr("kind")?.extract()?;
    match kind.as_str() {
        // NumPy bool and non-Python complex scalars follow Torch's float cast.
        "b" | "f" | "c" => Ok(Some(pyo3::types::PyFloat::new(py, value.call_method0("__float__")?.extract()?).into_any())),
        "i" | "u" => {
            let integer = value.call_method0("item")?;
            Ok(Some(if scalar {
                integer.extract::<i64>().map_err(|_| PyTypeError::new_err("an integer is required for this NumPy scalar"))?.into_pyobject(py)?.into_any()
            } else { pyo3::types::PyFloat::new(py, integer.extract()?).into_any() }))
        }
        _ => Err(PyTypeError::new_err("unsupported NumPy scalar dtype")),
    }
}

#[pyfunction]
#[pyo3(signature = (value, *, scalar=false))]
fn _real_scalar<'py>(value: &Bound<'py, PyAny>, scalar: bool) -> PyResult<Bound<'py, PyAny>> {
    Ok(numpy_number(value, scalar)?.unwrap_or_else(|| value.clone()))
}

fn numeric_operand(left: &CoreTensor, other: &Bound<'_, PyAny>, division: bool) -> PyResult<CoreTensor> {
    use ferro_core::DType;
    if let Ok(tensor) = other.extract::<PyRef<'_, PyTensor>>() { return Ok(tensor.inner.clone()); }
    if other.is_instance_of::<pyo3::types::PyBool>() {
        return CoreTensor::from_vec_bool(vec![other.extract::<bool>()?], &[]).map_err(map_err);
    }
    let floating = matches!(left.dtype(), DType::F32 | DType::F64) || other.is_instance_of::<pyo3::types::PyFloat>();
    if floating && (other.is_instance_of::<pyo3::types::PyInt>() || other.is_instance_of::<pyo3::types::PyFloat>()) {
        if other.is_instance_of::<pyo3::types::PyInt>() {
            let signed = other.extract::<i64>().ok();
            let unsigned = if signed.is_none() { Some(other.extract::<u64>()?) } else { None };
            return if left.dtype() == DType::F64 {
                CoreTensor::from_vec_f64(vec![signed.map(|value| value as f64).unwrap_or_else(|| unsigned.unwrap() as f64)], &[]).map_err(map_err)
            } else {
                Ok(CoreTensor::scalar(signed.map(|value| value as f32).unwrap_or_else(|| unsigned.unwrap() as f32)))
            };
        }
        let value = other.extract::<f64>()?;
        return if left.dtype() == DType::F64 { CoreTensor::from_vec_f64(vec![value], &[]).map_err(map_err) }
            else { Ok(CoreTensor::scalar(value as f32)) };
    }
    if other.is_instance_of::<pyo3::types::PyInt>() {
        let value = match other.extract::<i64>() {
            Ok(value) => value,
            Err(_) => {
                let value = other.extract::<u64>()?;
                if left.dtype() != DType::I64 { return Err(PyTypeError::new_err("unsigned scalar promotion with this dtype is unsupported")); }
                if division { return Ok(CoreTensor::scalar(value as f32)); }
                value as i64
            },
        };
        return CoreTensor::from_vec_i64(vec![value], &[]).map_err(map_err);
    }
    if let Some(value) = numpy_number(other, true)? { return numeric_operand(left, &value, division); }
    Err(PyTypeError::new_err("arithmetic operands must be tensors or real numeric scalars"))
}

struct PyAlpha(ferro_core::ops_ext::add_alpha::Alpha);

struct RoundDecimals(Option<i64>);

impl<'py> FromPyObject<'py> for RoundDecimals {
    fn extract_bound(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(tensor) = value.extract::<PyRef<'_, PyTensor>>() {
            if tensor.inner.ndim() == 0 && tensor.inner.dtype() == ferro_core::DType::I64 {
                return Ok(Self(Some(tensor.inner.to_vec_i64()[0])));
            }
            return Err(PyTypeError::new_err("decimals tensor must be a zero-dimensional int64 tensor"));
        }
        if !value.is_instance_of::<pyo3::types::PyInt>() || value.is_instance_of::<pyo3::types::PyBool>() {
            return Err(PyTypeError::new_err("decimals must be an integer"));
        }
        Ok(Self(Some(value.extract()?)))
    }
}

impl<'py> FromPyObject<'py> for PyAlpha {
    fn extract_bound(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        use ferro_core::ops_ext::add_alpha::Alpha;
        if let Ok(tensor) = value.extract::<PyRef<'_, PyTensor>>() {
            if tensor.inner.ndim() != 0 || tensor.inner.requires_grad() {
                return Err(PyTypeError::new_err("alpha tensor must be zero-dimensional and not require gradients"));
            }
            return Self::extract_bound(&tensor.item(value.py())?);
        }
        let scalar = if value.is_instance_of::<pyo3::types::PyBool>() {
            Alpha::Bool(value.extract()?)
        } else if value.is_instance_of::<pyo3::types::PyInt>() {
            match value.extract::<i64>() {
                Ok(value) => Alpha::I64(value),
                Err(_) => Alpha::U64(value.extract::<u64>()?),
            }
        } else if value.is_instance_of::<pyo3::types::PyFloat>() {
            Alpha::F64(value.extract()?)
        } else if let Some(number) = numpy_number(value, true)? {
            return Self::extract_bound(&number);
        } else { return Err(PyTypeError::new_err("alpha must be a real numeric scalar")); };
        Ok(Self(scalar))
    }
}

fn arithmetic_pair(input: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>, division: bool) -> PyResult<(CoreTensor, CoreTensor)> {
    if let Ok(value) = input.extract::<PyRef<'_, PyTensor>>() {
        return Ok((value.inner.clone(), numeric_operand(&value.inner, other, division)?));
    }
    if let Ok(value) = other.extract::<PyRef<'_, PyTensor>>() {
        return Ok((numeric_operand(&value.inner, input, division)?, value.inner.clone()));
    }
    let normalized_input = numpy_number(input, true)?;
    let normalized_other = numpy_number(other, true)?;
    let input = normalized_input.as_ref().unwrap_or(input);
    let other = normalized_other.as_ref().unwrap_or(other);
    let context = if input.is_instance_of::<pyo3::types::PyFloat>() || other.is_instance_of::<pyo3::types::PyFloat>() {
        CoreTensor::scalar(0.)
    } else {
        let mut unsigned = [None, None];
        for (index, value) in [input, other].into_iter().enumerate() {
            if value.is_instance_of::<pyo3::types::PyInt>() && value.extract::<i64>().is_err() {
                unsigned[index] = Some(value.extract::<u64>()?);
            }
        }
        if let [Some(left), Some(right)] = unsigned {
            if division { return Ok((CoreTensor::scalar(left as f32), CoreTensor::scalar(right as f32))); }
        }
        if unsigned.iter().any(Option::is_some) { return Err(PyTypeError::new_err("unsigned Number/Number promotion is unsupported")); }
        CoreTensor::from_vec_i64(vec![0], &[]).map_err(map_err)?
    };
    Ok((arithmetic_operand(&context, input)?, arithmetic_operand(&context, other)?))
}

#[pyfunction]
#[pyo3(signature = (operation, input, other, *, alpha=PyAlpha(ferro_core::ops_ext::add_alpha::Alpha::I64(1))), text_signature="(operation, input, other, *, alpha=1)")]
fn _arithmetic(operation: &str, input: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>, alpha: PyAlpha) -> PyResult<PyTensor> {
    let (input, other) = arithmetic_pair(input, other, operation == "true_divide")?;
    ok(match operation {
        "add" => input.add_alpha(&other, alpha.0),
        "sub" => input.sub_alpha(&other, alpha.0),
        "mul" => input.mul(&other),
        "true_divide" => input.div(&other),
        _ => return Err(PyValueError::new_err("unknown arithmetic operation")),
    })
}

fn division_mode(value: Option<&str>) -> PyResult<Option<ferro_core::ops_ext::div_rounding::DivisionRounding>> {
    use ferro_core::ops_ext::div_rounding::DivisionRounding;
    match value {
        None => Ok(None), Some("floor") => Ok(Some(DivisionRounding::Floor)), Some("trunc") => Ok(Some(DivisionRounding::Trunc)),
        Some(_) => Err(PyValueError::new_err("rounding_mode must be None, 'floor' or 'trunc'")),
    }
}

fn division_result(input: &CoreTensor, other: &CoreTensor, mode: Option<ferro_core::ops_ext::div_rounding::DivisionRounding>, floor_api: bool) -> PyResult<PyTensor> {
    ok(if floor_api { input.floor_divide(other) } else if let Some(mode) = mode { input.div_rounding(other, mode) } else { input.div(other) })
}

#[pyfunction]
#[pyo3(signature = (operation, input, other, *, rounding_mode=None))]
fn _division(operation: &str, input: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>, rounding_mode: Option<&str>) -> PyResult<PyTensor> {
    let mode = division_mode(rounding_mode)?;
    let floor_api = match operation {
        "div" => false, "floor_divide" if mode.is_none() => true,
        _ => return Err(PyValueError::new_err("unknown division operation or invalid floor_divide rounding mode")),
    };
    let (input, other) = arithmetic_pair(input, other, mode.is_none() && !floor_api)?;
    division_result(&input, &other, mode, floor_api)
}

fn parse_dtype(dtype: &str) -> PyResult<ferro_core::DType> {
    use ferro_core::DType;
    match dtype {
        "float32" => Ok(DType::F32), "float64" => Ok(DType::F64), "int64" => Ok(DType::I64), "bool" => Ok(DType::Bool),
        _ => Err(PyTypeError::new_err("dtype must be float32, float64, int64 or bool")),
    }
}

fn comparison_operand(left: &CoreTensor, other: &Bound<'_, PyAny>) -> PyResult<CoreTensor> {
    arithmetic_operand(left, other)
}

fn checked_inplace_copy(destination: &CoreTensor, source: &CoreTensor) -> PyResult<()> {
    if !ferro_core::autograd::is_grad_enabled() && destination.requires_grad() {
        destination.copy_leaf_from(source).map_err(map_err)
    } else {
        destination.copy_from(source).map_err(map_err)
    }
}

fn inplace_binary(destination: &PyTensor, other: &Bound<'_, PyAny>, alpha: Option<f64>) -> PyResult<()> {
    destination.float_math()?;
    // Whole self-copy runs core's destination gates without a write or version bump.
    checked_inplace_copy(&destination.inner, &destination.inner)?;
    if alpha.is_some_and(|value| value.is_finite() && value.abs() > f32::MAX as f64) {
        return Err(PyValueError::new_err("alpha cannot be converted to float32 without overflow"));
    }
    let rhs = if let Ok(tensor) = other.extract::<PyRef<'_, PyTensor>>() {
        tensor.float_math()?;
        if tensor.inner.requires_grad() && ferro_core::autograd::is_grad_enabled() {
            return Err(PyValueError::new_err("mutation with a differentiable source requires disabled grad recording"));
        }
        tensor.inner.clone()
    } else if other.is_instance_of::<pyo3::types::PyBool>() || other.is_instance_of::<pyo3::types::PyInt>() || other.is_instance_of::<pyo3::types::PyFloat>() {
        CoreTensor::full_on(&[], other.extract::<f64>()? as f32, destination.inner.device()).map_err(map_err)?
    } else {
        return Err(PyTypeError::new_err("in-place arithmetic requires a float32 tensor or real scalar"));
    };
    if rhs.device() != destination.inner.device() {
        return Err(PyValueError::new_err("in-place operands must be on the same device"));
    }
    let shape = ferro_core::shape::broadcast_shapes("in-place arithmetic", destination.inner.shape(), rhs.shape()).map_err(map_err)?;
    if shape != destination.inner.shape() {
        return Err(PyValueError::new_err("in-place broadcasting cannot change the destination shape"));
    }
    let value = if let Some(alpha) = alpha {
        let rhs = if alpha == 1.0 { rhs } else {
            let scale = CoreTensor::full_on(&[], alpha as f32, rhs.device()).map_err(map_err)?;
            rhs.expand(destination.inner.shape()).map_err(map_err)?.mul(&scale).map_err(map_err)?
        };
        destination.inner.add(&rhs).map_err(map_err)?
    } else {
        destination.inner.mul(&rhs).map_err(map_err)?
    };
    checked_inplace_copy(&destination.inner, &value)
}

fn signed_dimensions(value: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    let integer = |item: &Bound<'_, PyAny>| {
        if item.is_instance_of::<pyo3::types::PyBool>() { return Err(PyTypeError::new_err("dimensions must be integers, not bool")); }
        item.extract::<isize>()
    };
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        value.try_iter()?.map(|item| integer(&item?)).collect()
    } else { Ok(vec![integer(value)?]) }
}

fn inferred_shape(dimensions: &[isize], numel: usize) -> PyResult<Vec<usize>> {
    if dimensions.iter().any(|&n| n < -1) || dimensions.iter().filter(|&&n| n == -1).count() > 1 {
        return Err(PyValueError::new_err("shape requires nonnegative sizes and at most one inferred dimension"));
    }
    let mut output = dimensions.iter().map(|&n| n.max(0) as usize).collect::<Vec<_>>();
    if let Some(axis) = dimensions.iter().position(|&n| n == -1) {
        let known = if dimensions.contains(&0) { 0 } else {
            dimensions.iter().filter(|&&n| n != -1).try_fold(1usize, |n, &d| n.checked_mul(d as usize))
                .ok_or_else(|| PyValueError::new_err("shape size overflow"))?
        };
        if known == 0 || numel % known != 0 { return Err(PyValueError::new_err("cannot infer requested shape")); }
        output[axis] = numel / known;
    }
    Ok(output)
}

fn functional_reduction(tensor: &PyTensor, py: Python<'_>, dim: Option<&Bound<'_, PyAny>>, keepdim: bool, dtype: Option<&str>, name: &str) -> PyResult<PyTensor> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("keepdim", keepdim)?;
    if let Some(dim) = dim { kwargs.set_item("dim", dim)?; }
    if let Some(dtype) = dtype { kwargs.set_item("dtype", dtype)?; }
    let input = Py::new(py, PyTensor::wrap(tensor.inner.clone()))?;
    let result = py.import("ferro._ops")?.getattr(name)?.call((input,), Some(&kwargs))?;
    let result = result.extract::<PyRef<'_, PyTensor>>()?;
    Ok(PyTensor::wrap(result.inner.clone()))
}

fn functional_comparison<'py>(tensor: &PyTensor, other: &PyTensor, py: Python<'py>, name: &str,
    args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyAny>> {
    let input = Py::new(py, PyTensor::wrap(tensor.inner.clone()))?;
    let other = Py::new(py, PyTensor::wrap(other.inner.clone()))?;
    let mut values = vec![input.into_any(), other.into_any()];
    values.extend(args.iter().map(Bound::unbind));
    let arguments = PyTuple::new(py, values)?;
    py.import("ferro._comparison")?.getattr(name)?.call(arguments, kwargs)
}

fn tensor_matmul(left: &CoreTensor, right: &CoreTensor) -> PyResult<PyTensor> {
    if left.dtype() != ferro_core::DType::F32 || right.dtype() != ferro_core::DType::F32 || (left.ndim() == 2 && right.ndim() == 2) {
        return ok(left.matmul(right));
    }
    Python::with_gil(|py| {
        let a = Py::new(py, PyTensor::wrap(left.clone()))?;
        let b = Py::new(py, PyTensor::wrap(right.clone()))?;
        let result = py.import("ferro._ops")?.getattr("matmul")?.call1((a, b))?;
        let result = result.extract::<PyRef<'_, PyTensor>>()?;
        Ok(PyTensor::wrap(result.inner.clone()))
    })
}

/// Recursively build nested Python lists from row-major data and a shape.
fn to_nested<'py, T: Copy + IntoPyObject<'py>>(py: Python<'py>, data: &[T], shape: &[usize]) -> Bound<'py, PyAny> {
    if shape.is_empty() {
        return data[0].into_bound_py_any(py).unwrap();
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

fn fmt_data<T: std::fmt::Display>(data: &[T], shape: &[usize]) -> String {
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
    NewAxis,
    Skip,
    TupleSlice(Option<isize>, Option<isize>, Option<isize>),
    Array(CoreTensor),
    Idx(isize),
    Slice(Option<isize>, Option<isize>, Option<isize>),
}

impl Clone for Sel {
    fn clone(&self) -> Self {
        match self {
            Sel::NewAxis => Sel::NewAxis,
            Sel::Skip => Sel::Skip,
            Sel::TupleSlice(a, b, c) => Sel::TupleSlice(*a, *b, *c),
            Sel::Array(index) => Sel::Array(index.clone()),
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
            let Some(next) = i.checked_add(step) else { break; };
            i = next;
        }
    } else {
        let norm = |v: isize| if v < 0 { v + l } else { v };
        let s = start.map(norm).unwrap_or(l - 1).clamp(-1, l - 1);
        let e = stop.map(norm).unwrap_or(-1).clamp(-1, l - 1);
        let mut i = s;
        while i > e {
            if i >= 0 {
                out.push(i as usize);
            }
            let Some(next) = i.checked_add(step) else { break; };
            i = next;
        }
    }
    Ok(out)
}

fn parse_key(key: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Vec<Sel>> {
    fn one(obj: &Bound<'_, PyAny>, tuple: bool) -> PyResult<Sel> {
        if obj.is_none() { return Ok(Sel::NewAxis); }
        if obj.is_instance_of::<pyo3::types::PyBool>() {
            return Err(PyTypeError::new_err("boolean tuple indices are unsupported; pass a single boolean mask"));
        }
        if let Ok(index) = obj.extract::<PyRef<'_, PyTensor>>() {
            if index.inner.dtype() != ferro_core::DType::I64 { return Err(PyTypeError::new_err("array indices must be int64 tensors")); }
            return if index.inner.ndim() == 0 {
                Ok(Sel::Idx(isize::try_from(index.inner.to_vec_i64()[0]).map_err(|_| PyValueError::new_err("index exceeds platform range"))?))
            } else { Ok(Sel::Array(index.inner.clone())) };
        }
        if let Ok(slice) = obj.downcast::<PySlice>() {
            let start = slice.getattr("start")?.extract::<Option<isize>>()?;
            let stop = slice.getattr("stop")?.extract::<Option<isize>>()?;
            let step = slice.getattr("step")?.extract::<Option<isize>>()?;
            return Ok(if tuple { Sel::TupleSlice(start, stop, step) } else { Sel::Slice(start, stop, step) });
        }
        obj.extract::<isize>().map(Sel::Idx).map_err(|_| PyTypeError::new_err("tensor indices must be integers, slices, None, ellipsis or an int64 Tensor"))
    }
    if key.is_instance_of::<PyEllipsis>() { return Ok(vec![Sel::Skip; ndim]); }
    let picked = if let Ok(tuple) = key.downcast::<PyTuple>() {
        if tuple.iter().filter(|obj| obj.is_instance_of::<PyEllipsis>()).count() > 1 {
            return Err(PyValueError::new_err("an index can contain only one ellipsis"));
        }
        let specified = tuple.iter().filter(|obj| !obj.is_none() && !obj.is_instance_of::<PyEllipsis>()).count();
        let mut sels = Vec::new();
        for obj in tuple.iter() {
            if obj.is_instance_of::<PyEllipsis>() {
                sels.extend(std::iter::repeat(Sel::Skip).take(ndim.saturating_sub(specified)));
            } else { sels.push(one(&obj, true)?); }
        }
        sels
    } else { vec![one(key, false)?] };
    if picked.iter().filter(|sel| !matches!(sel, Sel::NewAxis)).count() > ndim {
        return Err(PyValueError::new_err("too many indices for tensor rank"));
    }
    if picked.iter().filter(|sel| matches!(sel, Sel::Array(_))).count() > 1 {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err("multiple integer array indices are not supported"));
    }
    Ok(picked)
}

impl PyTensor {
    fn index_view(&self, picked: &[Sel], _shape: &[usize]) -> PyResult<PyTensor> {
        let mut t = self.inner.clone();
        let mut drop = Vec::new();
        let mut array = None;
        for (d, sel) in picked.iter().enumerate() {
            t = match sel {
                Sel::NewAxis => t.unsqueeze(d).map_err(map_err)?,
                Sel::Skip => t,
                Sel::Array(index) => { array = Some((d, index)); t },
                Sel::Idx(i) => {
                    drop.push(d);
                    t.narrow(d, wrap_index(*i, t.shape()[d])?, 1).map_err(map_err)?
                }
                Sel::Slice(start, stop, step) | Sel::TupleSlice(start, stop, step) => {
                    let st = step.unwrap_or(1);
                    if st > 0 {
                        let len = t.shape()[d] as i128;
                        let norm = |value: i128| (if value < 0 { value + len } else { value }).clamp(0, len) as usize;
                        let start = norm(start.map_or(0, |value| value as i128));
                        let stop = norm(stop.map_or(len, |value| value as i128));
                        if matches!(sel, Sel::TupleSlice(..)) && start == 0 && stop == t.shape()[d] && st == 1 { t }
                        else { t.slice(d, start, stop, st as usize).map_err(map_err)? }
                    } else {
                        let idx = slice_indices(*start, *stop, st, t.shape()[d])?;
                        if idx.is_empty() { t.narrow(d, 0, 0).map_err(map_err)? } else { t.index_select(d, &idx).map_err(map_err)? }
                    }
                }
            };
        }
        for &d in drop.iter().rev() { t = t.squeeze(d).map_err(map_err)?; }
        if let Some((d, index)) = array {
            t = t.index_tensor(d - drop.iter().filter(|&&axis| axis < d).count(), index).map_err(map_err)?;
        }
        if t.id() == self.inner.id() { t = t.alias_view().map_err(map_err)?; }
        Ok(PyTensor::wrap(t))
    }
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
    #[pyo3(signature = (shape, seed=None, generator=None, device="cpu", dtype="float32"))]
    fn randn(
        shape: Vec<usize>,
        seed: Option<u64>,
        generator: Option<PyRef<'_, Generator>>,
        device: &str,
        dtype: &str,
    ) -> PyResult<PyTensor> {
        let dev = parse_device(device)?;
        if !matches!(dtype, "float32" | "float64") { return Err(PyTypeError::new_err("random factories require float32 or float64 dtype")); }
        if dtype == "float64" && dev != Device::Cpu { return Err(PyValueError::new_err("float64 random factories require CPU storage")); }
        ferro_core::forward_ad::check_random_operation().map_err(map_err)?;
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
        if dtype == "float64" { return CoreTensor::randn_f64(&shape, rng).map(PyTensor::wrap).map_err(map_err); }
        CoreTensor::randn(&shape, rng).to_device(dev).map(PyTensor::wrap).map_err(map_err)
    }

    /// Draw uniform samples in [0, 1) with the same seeding rules as randn.
    #[staticmethod]
    #[pyo3(signature = (shape, seed=None, generator=None, device="cpu", dtype="float32"))]
    fn rand(
        shape: Vec<usize>,
        seed: Option<u64>,
        generator: Option<PyRef<'_, Generator>>,
        device: &str,
        dtype: &str,
    ) -> PyResult<PyTensor> {
        let dev = parse_device(device)?;
        if !matches!(dtype, "float32" | "float64") { return Err(PyTypeError::new_err("random factories require float32 or float64 dtype")); }
        if dtype == "float64" && dev != Device::Cpu { return Err(PyValueError::new_err("float64 random factories require CPU storage")); }
        ferro_core::forward_ad::check_random_operation().map_err(map_err)?;
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
        if dtype == "float64" { return CoreTensor::rand_f64(&shape, rng).map(PyTensor::wrap).map_err(map_err); }
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

    /// F64 storage preserves Python float precision and supports CPU first-order autograd.
    #[staticmethod]
    fn from_f64(data: Vec<f64>, shape: Vec<usize>) -> PyResult<PyTensor> {
        CoreTensor::from_vec_f64(data, &shape).map(PyTensor::wrap).map_err(map_err)
    }

    #[staticmethod]
    fn from_bool(data: Vec<bool>, shape: Vec<usize>) -> PyResult<PyTensor> {
        CoreTensor::from_vec_bool(data, &shape).map(PyTensor::wrap).map_err(map_err)
    }

    fn equal(&self, other: &PyTensor, py: Python<'_>) -> PyResult<bool> {
        functional_comparison(self, other, py, "equal", &PyTuple::empty(py), None)?.extract()
    }

    #[pyo3(signature = (other, *args, **kwargs), text_signature="($self, other, rtol=1e-5, atol=1e-8, equal_nan=False)")]
    fn isclose<'py>(&self, other: &PyTensor, py: Python<'py>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<PyTensor> {
        let result = functional_comparison(self, other, py, "isclose", args, kwargs)?;
        let result = result.extract::<PyRef<'_, PyTensor>>()?;
        Ok(PyTensor::wrap(result.inner.clone()))
    }

    #[pyo3(signature = (other, *args, **kwargs), text_signature="($self, other, rtol=1e-5, atol=1e-8, equal_nan=False)")]
    fn allclose<'py>(&self, other: &PyTensor, py: Python<'py>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<bool> {
        functional_comparison(self, other, py, "allclose", args, kwargs)?.extract()
    }

    fn eq(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { ok(self.inner.eq_tensor(&comparison_operand(&self.inner, other)?)) }
    fn ne(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { ok(self.inner.ne_tensor(&comparison_operand(&self.inner, other)?)) }
    fn lt(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { ok(self.inner.lt_tensor(&comparison_operand(&self.inner, other)?)) }
    fn le(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { ok(self.inner.le_tensor(&comparison_operand(&self.inner, other)?)) }
    fn gt(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { ok(self.inner.gt_tensor(&comparison_operand(&self.inner, other)?)) }
    fn ge(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { ok(self.inner.ge_tensor(&comparison_operand(&self.inner, other)?)) }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.eq(other) }
    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.ne(other) }
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.lt(other) }
    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.le(other) }
    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.gt(other) }
    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.ge(other) }
    fn __hash__(slf: PyRef<'_, Self>) -> isize { slf.as_ptr() as isize }
    fn logical_and(&self, other: &PyTensor) -> PyResult<PyTensor> { ok(self.inner.logical_and(&other.inner)) }
    fn logical_or(&self, other: &PyTensor) -> PyResult<PyTensor> { ok(self.inner.logical_or(&other.inner)) }
    fn logical_xor(&self, other: &PyTensor) -> PyResult<PyTensor> { ok(self.inner.logical_xor(&other.inner)) }
    fn logical_not(&self) -> PyResult<PyTensor> { ok(self.inner.logical_not()) }
    fn __and__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        if self.inner.dtype() != ferro_core::DType::Bool || other.inner.dtype() != ferro_core::DType::Bool {
            return Err(PyTypeError::new_err("bitwise tensor operators currently require bool dtype"));
        }
        self.logical_and(other)
    }
    fn __or__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        if self.inner.dtype() != ferro_core::DType::Bool || other.inner.dtype() != ferro_core::DType::Bool {
            return Err(PyTypeError::new_err("bitwise tensor operators currently require bool dtype"));
        }
        self.logical_or(other)
    }
    fn __xor__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        if self.inner.dtype() != ferro_core::DType::Bool || other.inner.dtype() != ferro_core::DType::Bool {
            return Err(PyTypeError::new_err("bitwise tensor operators currently require bool dtype"));
        }
        self.logical_xor(other)
    }
    fn __invert__(&self) -> PyResult<PyTensor> {
        if self.inner.dtype() != ferro_core::DType::Bool { return Err(PyTypeError::new_err("bitwise inversion currently requires bool dtype")); }
        self.logical_not()
    }
    fn isnan(&self) -> PyResult<PyTensor> { ok(self.inner.isnan()) }
    fn isfinite(&self) -> PyResult<PyTensor> { ok(self.inner.isfinite()) }
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn any(&self, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
        if self.inner.ndim() == 0 && matches!(dim, Some(0 | -1)) { return ok(self.inner.bool_reduce(None, keepdim, false)); }
        ok(self.inner.bool_reduce(dim.map(|d| norm_dim(d, self.inner.ndim())).transpose()?, keepdim, false))
    }
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn all(&self, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
        if self.inner.ndim() == 0 && matches!(dim, Some(0 | -1)) { return ok(self.inner.bool_reduce(None, keepdim, true)); }
        ok(self.inner.bool_reduce(dim.map(|d| norm_dim(d, self.inner.ndim())).transpose()?, keepdim, true))
    }
    fn masked_select(&self, mask: &PyTensor) -> PyResult<PyTensor> { ok(self.inner.masked_select(&mask.inner)) }
    #[pyo3(name = "bool")]
    fn to_bool(&self) -> PyResult<PyTensor> {
        ok(self.inner.to_bool())
    }
    #[pyo3(signature = (as_tuple=false))]
    fn nonzero<'py>(&self, py: Python<'py>, as_tuple: bool) -> PyResult<Bound<'py, PyAny>> {
        let indices = self.inner.nonzero().map_err(map_err)?;
        if !as_tuple { return Ok(Py::new(py, PyTensor::wrap(indices))?.into_bound(py).into_any()); }
        let count = indices.shape()[0];
        let dims = self.inner.ndim().max(1);
        let matrix = indices.to_vec_i64();
        let mut columns = Vec::with_capacity(dims);
        for dim in 0..dims {
            let values = if self.inner.ndim() == 0 { vec![0; count] }
                else { matrix.chunks(dims).map(|row| row[dim]).collect() };
            columns.push(Py::new(py, PyTensor::wrap(CoreTensor::from_vec_i64(values, &[count]).map_err(map_err)?))?);
        }
        Ok(PyTuple::new(py, columns)?.into_any())
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &arithmetic_operand(&self.inner, other)?, CoreTensor::add)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &arithmetic_operand(&self.inner, other)?, CoreTensor::add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &arithmetic_operand(&self.inner, other)?, CoreTensor::sub)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&arithmetic_operand(&self.inner, other)?, &self.inner, CoreTensor::sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &arithmetic_operand(&self.inner, other)?, CoreTensor::mul)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &arithmetic_operand(&self.inner, other)?, CoreTensor::mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &numeric_operand(&self.inner, other, true)?, CoreTensor::div)
    }

    #[pyo3(signature = (other, *, alpha=PyAlpha(ferro_core::ops_ext::add_alpha::Alpha::I64(1))), text_signature="($self, other, *, alpha=1)")]
    fn add(&self, other: &Bound<'_, PyAny>, alpha: PyAlpha) -> PyResult<PyTensor> {
        ok(self.inner.add_alpha(&arithmetic_operand(&self.inner, other)?, alpha.0))
    }

    #[pyo3(signature = (other, *, alpha=PyAlpha(ferro_core::ops_ext::add_alpha::Alpha::I64(1))), text_signature="($self, other, *, alpha=1)")]
    fn sub(&self, other: &Bound<'_, PyAny>, alpha: PyAlpha) -> PyResult<PyTensor> {
        ok(self.inner.sub_alpha(&arithmetic_operand(&self.inner, other)?, alpha.0))
    }

    #[pyo3(signature = (other, *, alpha=PyAlpha(ferro_core::ops_ext::add_alpha::Alpha::I64(1))), text_signature="($self, other, *, alpha=1)")]
    fn subtract(&self, other: &Bound<'_, PyAny>, alpha: PyAlpha) -> PyResult<PyTensor> {
        self.sub(other, alpha)
    }

    fn mul(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&self.inner, &arithmetic_operand(&self.inner, other)?, CoreTensor::mul)
    }

    fn multiply(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.mul(other) }

    #[pyo3(signature = (other, *, rounding_mode=None))]
    fn div(&self, other: &Bound<'_, PyAny>, rounding_mode: Option<&str>) -> PyResult<PyTensor> {
        let mode = division_mode(rounding_mode)?;
        division_result(&self.inner, &numeric_operand(&self.inner, other, mode.is_none())?, mode, false)
    }

    #[pyo3(signature = (other, *, rounding_mode=None))]
    fn divide(&self, other: &Bound<'_, PyAny>, rounding_mode: Option<&str>) -> PyResult<PyTensor> { self.div(other, rounding_mode) }

    fn floor_divide(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        division_result(&self.inner, &numeric_operand(&self.inner, other, false)?, None, true)
    }

    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.floor_divide(other) }

    fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        division_result(&numeric_operand(&self.inner, other, false)?, &self.inner, None, true)
    }

    fn true_divide(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.__truediv__(other)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        bin_op(&numeric_operand(&self.inner, other, true)?, &self.inner, CoreTensor::div)
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        tensor_matmul(&self.inner, &other.inner)
    }

    fn __matmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = other.extract::<PyRef<'_, PyTensor>>().map_err(|_| PyTypeError::new_err("matmul requires a Tensor operand"))?;
        tensor_matmul(&self.inner, &b.inner)
    }

    fn __rmatmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = other.extract::<PyRef<'_, PyTensor>>().map_err(|_| PyTypeError::new_err("matmul requires a Tensor operand"))?;
        tensor_matmul(&b.inner, &self.inner)
    }

    fn __neg__(&self) -> PyResult<PyTensor> {
        self.typed_math(true)?;
        Ok(PyTensor::wrap(self.inner.neg()))
    }

    fn neg(&self) -> PyResult<PyTensor> { self.__neg__() }

    fn negative(&self) -> PyResult<PyTensor> { self.__neg__() }

    fn relu(&self) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        Ok(PyTensor::wrap(self.inner.relu()))
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

    /// Compile the supported inference DAG that produced this tensor into a
    /// reusable fused handle, resolved once here. `FusedChain.replay()` runs
    /// the compiled schedule without a tape walk or re-planning. Benchmark
    /// compilation separately from replay; replay outputs are detached.
    ///
    /// Replay includes every recorded operation over current graph-leaf values.
    /// Build with `ferro.capture(lambda: expression)`; gradients are not needed.
    /// Outside capture, detached/no-grad computations are opaque input values.
    /// Tagged pointwise branches, shared nodes and broadcasts are supported.
    /// Matmul, bmm, reshape, transpose, softmax, sum_dim and LayerNorm have
    /// forward replay metadata; other untagged operations raise ValueError.
    /// FusedChain.num_runs reports scheduled runs, not kernel launches.
    fn compile_fused(&self) -> PyResult<PyCompiledChain> {
        ferro_core::graph::CompiledChain::compile(&self.inner)
            .map(|c| PyCompiledChain { inner: c })
            .map_err(map_err)
    }

    fn sigmoid(&self) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        Ok(PyTensor::wrap(self.inner.sigmoid()))
    }

    fn exp(&self) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        Ok(PyTensor::wrap(self.inner.exp()))
    }

    fn log(&self) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        Ok(PyTensor::wrap(self.inner.log()))
    }

    fn tanh(&self) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        Ok(PyTensor::wrap(self.inner.tanh()))
    }

    fn gelu(&self) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        Ok(PyTensor::wrap(self.inner.gelu()))
    }

    fn sqrt(&self) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        Ok(PyTensor::wrap(self.inner.sqrt()))
    }

    fn abs(&self) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        Ok(PyTensor::wrap(self.inner.abs()))
    }

    fn pow(&self, p: f64) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        ok(self.inner.pow_scalar(p))
    }

    fn clamp(&self, min: f64, max: f64) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        if self.inner.dtype() == ferro_core::DType::F64 { return ok(self.inner.clamp_f64(min, max)); }
        Ok(PyTensor::wrap(self.inner.clamp(min as f32, max as f32)))
    }

    fn __pow__(&self, exponent: f64, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
        if modulo.is_some() { return Err(PyTypeError::new_err("tensor power does not accept modulo")); }
        self.pow(exponent)
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
        self.typed_math(false)?;
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.mean_dim(d, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    fn softmax(&self, dim: isize) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        let d = norm_dim(dim, self.inner.ndim().max(1))?;
        if self.inner.ndim() == 0 {
            return ok(self.inner.reshape(&[1]).and_then(|t| t.softmax(d)).and_then(|t| t.reshape(&[])));
        }
        self.inner.softmax(d).map(PyTensor::wrap).map_err(map_err)
    }

    fn log_softmax(&self, dim: isize) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        let d = norm_dim(dim, self.inner.ndim().max(1))?;
        if self.inner.ndim() == 0 {
            return ok(self.inner.reshape(&[1]).and_then(|t| t.log_softmax(d)).and_then(|t| t.reshape(&[])));
        }
        self.inner.log_softmax(d).map(PyTensor::wrap).map_err(map_err)
    }

    fn prelu(&self, weight: &PyTensor) -> PyResult<PyTensor> { ok(self.inner.prelu(&weight.inner)) }

    fn bmm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.typed_math(false)?;
        other.typed_math(false)?;
        self.inner.bmm(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn cummax(&self, py: Python<'_>, dim: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let dim = _real_scalar(dim, true)?;
        if !dim.is_instance_of::<pyo3::types::PyInt>() || dim.is_instance_of::<pyo3::types::PyBool>() {
            return Err(PyTypeError::new_err("cummax dim must be an integer, not bool"));
        }
        let d = norm_dim(dim.extract()?, self.inner.ndim().max(1))?;
        let (values, indices) = self.inner.cummax(d).map_err(map_err)?;
        Ok(PyModule::import(py, "ferro._cumulative")?.getattr("CummaxResult")?.call1((PyTensor::wrap(values), PyTensor::wrap(indices)))?.unbind())
    }

    fn cummin(&self, py: Python<'_>, dim: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let dim = _real_scalar(dim, true)?;
        if !dim.is_instance_of::<pyo3::types::PyInt>() || dim.is_instance_of::<pyo3::types::PyBool>() {
            return Err(PyTypeError::new_err("cummin dim must be an integer, not bool"));
        }
        let d = norm_dim(dim.extract()?, self.inner.ndim().max(1))?;
        let (values, indices) = self.inner.cummin(d).map_err(map_err)?;
        Ok(PyModule::import(py, "ferro._cumulative")?.getattr("CumminResult")?.call1((PyTensor::wrap(values), PyTensor::wrap(indices)))?.unbind())
    }

    #[pyo3(signature = (dim, *, dtype=None))]
    fn cumsum(&self, dim: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyTensor> {
        let dim = _real_scalar(dim, true)?;
        if !dim.is_instance_of::<pyo3::types::PyInt>() || dim.is_instance_of::<pyo3::types::PyBool>() {
            return Err(PyTypeError::new_err("cumsum dim must be an integer, not bool"));
        }
        let d = norm_dim(dim.extract()?, self.inner.ndim().max(1))?;
        self.inner.cumsum_dtype(d, dtype.map(parse_dtype).transpose()?).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn prod<'py>(&self, py: Python<'py>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<PyTensor> {
        let input = Py::new(py, PyTensor::wrap(self.inner.clone()))?;
        let mut values = vec![input.into_any()]; values.extend(args.iter().map(Bound::unbind));
        let arguments = PyTuple::new(py, values)?;
        let result = py.import("ferro._reductions")?.getattr("prod")?.call(arguments, kwargs)?;
        let result = result.extract::<PyRef<'_, PyTensor>>()?;
        Ok(PyTensor::wrap(result.inner.clone()))
    }

    #[pyo3(signature = (dim=None, keepdim=false, dtype=None))]
    fn _prod(&self, dim: Option<isize>, keepdim: bool, dtype: Option<&str>) -> PyResult<PyTensor> {
        let dim = dim.map(|dim| norm_dim(dim, self.inner.ndim().max(1))).transpose()?;
        ok(self.inner.product(dim, keepdim, dtype.map(parse_dtype).transpose()?))
    }

    #[pyo3(signature = (dim, *, dtype=None))]
    fn cumprod(&self, dim: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyTensor> {
        let dim = _real_scalar(dim, true)?;
        if !dim.is_instance_of::<pyo3::types::PyInt>() || dim.is_instance_of::<pyo3::types::PyBool>() {
            return Err(PyTypeError::new_err("cumprod dim must be an integer, not bool"));
        }
        let d = norm_dim(dim.extract()?, self.inner.ndim().max(1))?;
        self.inner.cumprod_dtype(d, dtype.map(parse_dtype).transpose()?).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn argmax(&self, dim: isize, keepdim: bool) -> PyResult<PyTensor> {
        self.float_math()?;
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.argmax(d, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn argmin(&self, dim: isize, keepdim: bool) -> PyResult<PyTensor> {
        self.float_math()?;
        let d = norm_dim(dim, self.inner.ndim())?;
        self.inner.argmin(d, keepdim).map(PyTensor::wrap).map_err(map_err)
    }

    fn topk(&self, k: usize, dim: isize) -> PyResult<(PyTensor, PyTensor)> {
        self.float_math()?;
        let d = norm_dim(dim, self.inner.ndim())?;
        let (v, i) = self.inner.topk(k, d).map_err(map_err)?;
        Ok((PyTensor::wrap(v), PyTensor::wrap(i)))
    }

    #[pyo3(signature = (dim=-1, descending=false, *, stable=false))]
    fn sort<'py>(&self, py: Python<'py>, dim: isize, descending: bool, stable: bool) -> PyResult<Bound<'py, PyAny>> {
        let dim = norm_dim(dim, self.inner.ndim().max(1))?;
        let (values, indices) = self.inner.sort(dim, descending, stable).map_err(map_err)?;
        py.import("ferro._sorting")?.getattr("_SortResult")?.call1((PyTensor::wrap(values), PyTensor::wrap(indices)))
    }

    #[pyo3(signature = (dim=-1, descending=false, *, stable=false))]
    fn argsort(&self, dim: isize, descending: bool, stable: bool) -> PyResult<PyTensor> {
        let dim = norm_dim(dim, self.inner.ndim().max(1))?;
        ok(self.inner.argsort(dim, descending, stable))
    }

    fn take(&self, index: &PyTensor) -> PyResult<PyTensor> { ok(self.inner.take(&index.inner)) }

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

    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, dim: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
        let ndim = self.inner.ndim();
        let axes = match dim {
            None => (0..ndim).collect::<Vec<_>>(),
            Some(dim) => signed_dimensions(dim)?.iter().map(|&d| norm_dim(d, ndim.max(1))).collect::<PyResult<Vec<_>>>()?,
        };
        let mut axes = axes;
        axes.sort_unstable();
        if axes.windows(2).any(|a| a[0] == a[1]) { return Err(PyValueError::new_err("squeeze dimensions must be unique")); }
        let mut output = self.inner.clone();
        if ndim != 0 {
            for axis in axes.into_iter().rev() {
                if self.inner.shape()[axis] == 1 { output = output.squeeze(axis).map_err(map_err)?; }
            }
        }
        Ok(PyTensor::wrap(output))
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

    #[pyo3(signature = (*sizes, shape=None))]
    fn reshape(&self, sizes: &Bound<'_, PyTuple>, shape: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
        if shape.is_some() && !sizes.is_empty() { return Err(PyTypeError::new_err("reshape received both positional and keyword shapes")); }
        let dimensions = if let Some(shape) = shape { signed_dimensions(shape)? }
            else if sizes.is_empty() { return Err(PyTypeError::new_err("reshape requires a shape")); }
            else if sizes.len() == 1 { signed_dimensions(&sizes.get_item(0)?)? }
            else { signed_dimensions(sizes.as_any())? };
        let output = inferred_shape(&dimensions, self.inner.numel())?;
        self.inner.reshape(&output).map(PyTensor::wrap).map_err(map_err)
    }

    /// Shape-only view; incompatible strides reject. Eager/CPU capture aliases;
    /// device-leaf inference recording keeps the existing isolated snapshot policy.
    #[pyo3(signature = (*sizes, size=None))]
    fn view(&self, sizes: &Bound<'_, PyTuple>, size: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
        if size.is_some() && !sizes.is_empty() { return Err(PyTypeError::new_err("view received both positional and keyword shapes")); }
        let dimensions = if let Some(size) = size { signed_dimensions(size)? }
            else if sizes.is_empty() { return Err(PyTypeError::new_err("view requires a shape")); }
            else if sizes.len() == 1 { signed_dimensions(&sizes.get_item(0)?)? }
            else { signed_dimensions(sizes.as_any())? };
        let shape = inferred_shape(&dimensions, self.inner.numel())?;
        self.inner.view(&shape).map(PyTensor::wrap).map_err(map_err)
    }

    fn view_as(&self, other: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
        self.inner.view_as(&other.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn unflatten(&self, dim: isize, sizes: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if !sizes.is_instance_of::<PyTuple>() && !sizes.is_instance_of::<PyList>() {
            return Err(PyTypeError::new_err("unflatten sizes must be a tuple or list"));
        }
        self.inner.unflatten(dim, &signed_dimensions(sizes)?).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (weight=None, bias=None, eps=1e-5))]
    fn layer_norm(&self, weight: Option<&PyTensor>, bias: Option<&PyTensor>, eps: f32) -> PyResult<PyTensor> {
        self.inner.layer_norm(weight.map(|t| &t.inner), bias.map(|t| &t.inner), eps).map(PyTensor::wrap).map_err(map_err)
    }

    fn transpose(&self, d0: isize, d1: isize) -> PyResult<PyTensor> {
        let ndim = self.inner.ndim();
        self.inner.transpose(norm_dim(d0, ndim)?, norm_dim(d1, ndim)?)
            .map(PyTensor::wrap)
            .map_err(map_err)
    }

    #[getter(T)]
    fn reversed_transpose(&self) -> PyResult<PyTensor> {
        let ndim = self.inner.ndim();
        let mut output = match ndim {
            0 => self.inner.reshape(&[]).map_err(map_err)?,
            1 => self.inner.transpose(0, 0).map_err(map_err)?,
            _ => self.inner.clone(),
        };
        for axis in 0..ndim / 2 {
            output = output.transpose(axis, ndim - 1 - axis).map_err(map_err)?;
        }
        Ok(PyTensor::wrap(output))
    }

    fn t(&self) -> PyResult<PyTensor> {
        if self.inner.ndim() > 2 { return Err(PyValueError::new_err("t() requires a tensor with at most two dimensions")); }
        self.reversed_transpose()
    }

    #[getter(mT)]
    fn matrix_transpose(slf: PyRef<'_, Self>) -> PyResult<Py<PyTensor>> {
        let ndim = slf.inner.ndim();
        if ndim == 0 { return Ok(slf.into()); }
        if ndim == 1 { return Err(PyValueError::new_err("mT requires a matrix or batch of matrices")); }
        Py::new(slf.py(), PyTensor::wrap(slf.inner.transpose(ndim - 2, ndim - 1).map_err(map_err)?))
    }

    #[getter(H)]
    fn hermitian_transpose(slf: PyRef<'_, Self>) -> PyResult<Py<PyTensor>> {
        let ndim = slf.inner.ndim();
        if ndim == 0 { return Ok(slf.into()); }
        if ndim != 2 { return Err(PyValueError::new_err("H requires a two-dimensional matrix")); }
        Py::new(slf.py(), PyTensor::wrap(slf.inner.transpose(0, 1).map_err(map_err)?))
    }

    #[getter(mH)]
    fn matrix_hermitian_transpose(slf: PyRef<'_, Self>) -> PyResult<Py<PyTensor>> {
        Self::matrix_transpose(slf)
    }

    #[pyo3(signature = (*dims))]
    fn permute(&self, dims: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        let dims = if dims.len() == 1 { signed_dimensions(&dims.get_item(0)?)? } else { signed_dimensions(dims.as_any())? };
        let ndim = self.inner.ndim();
        let d: Vec<usize> = dims.iter().map(|&x| norm_dim(x, ndim)).collect::<PyResult<_>>()?;
        ok(self.inner.permute(&d))
    }

    fn narrow(&self, dim: isize, start: isize, length: usize) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        let len = self.inner.shape()[d] as isize;
        let s = if start < 0 { start + len } else { start };
        if s < 0 {
            return Err(PyValueError::new_err(format!("start {start} out of range for dimension of size {len}")));
        }
        ok(self.inner.narrow(d, s as usize, length))
    }

    #[pyo3(signature = (split_size_or_sections, dim=0))]
    fn split(&self, split_size_or_sections: &Bound<'_, PyAny>, dim: isize) -> PyResult<Vec<PyTensor>> {
        let d = norm_dim(dim, self.inner.ndim())?;
        let parts = match split_size_or_sections.extract::<usize>() {
            Ok(size) => self.inner.split(size, d),
            Err(_) => self.inner.split_sizes(&split_size_or_sections.extract::<Vec<usize>>()?, d),
        };
        Ok(parts.map_err(map_err)?.into_iter().map(PyTensor::wrap).collect())
    }

    #[pyo3(signature = (chunks, dim=0))]
    fn chunk(&self, chunks: usize, dim: isize) -> PyResult<Vec<PyTensor>> {
        let d = norm_dim(dim, self.inner.ndim())?;
        Ok(self.inner.chunk(chunks, d).map_err(map_err)?.into_iter().map(PyTensor::wrap).collect())
    }

    /// torch semantics: -1 keeps an existing size; new leading dims may be added.
    #[pyo3(signature = (*sizes))]
    fn expand(&self, sizes: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        let sizes = if sizes.len() == 1 { signed_dimensions(&sizes.get_item(0)?)? } else { signed_dimensions(sizes.as_any())? };
        let shape = self.inner.shape();
        if sizes.len() < shape.len() {
            return Err(PyValueError::new_err(format!("expand: {} sizes given for rank-{} tensor", sizes.len(), shape.len())));
        }
        let lead = sizes.len() - shape.len();
        let mut out = Vec::with_capacity(sizes.len());
        for (i, &s) in sizes.iter().enumerate() {
            out.push(match s {
                -1 if i >= lead => shape[i - lead],
                s if s >= 0 => s as usize,
                _ => return Err(PyValueError::new_err(format!("expand: invalid size {s} at dim {i}"))),
            });
        }
        let mut src = self.inner.clone();
        for _ in 0..lead { src = src.unsqueeze(0).map_err(map_err)?; }
        ok(src.expand(&out))
    }

    #[pyo3(signature = (*reps))]
    fn repeat(&self, reps: Vec<usize>) -> PyResult<PyTensor> { ok(self.inner.repeat(&reps)) }

    #[pyo3(signature = (*dims))]
    fn flip(&self, dims: Vec<isize>) -> PyResult<PyTensor> {
        let ndim = self.inner.ndim();
        let d: Vec<usize> = dims.iter().map(|&x| norm_dim(x, ndim)).collect::<PyResult<_>>()?;
        ok(self.inner.flip(&d))
    }

    #[pyo3(signature = (shifts, dims=None))]
    fn roll(&self, shifts: &Bound<'_, PyAny>, dims: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
        let shifts: Vec<isize> = shifts.extract::<isize>().map(|s| vec![s]).or_else(|_| shifts.extract())?;
        let ndim = self.inner.ndim();
        let dims: Vec<isize> = match dims {
            None => Vec::new(),
            Some(d) => d.extract::<isize>().map(|x| vec![x]).or_else(|_| d.extract())?,
        };
        let d: Vec<usize> = dims.iter().map(|&x| norm_dim(x, ndim)).collect::<PyResult<_>>()?;
        ok(self.inner.roll(&shifts, &d))
    }

    fn masked_fill(&self, mask: &PyTensor, value: f32) -> PyResult<PyTensor> { ok(self.inner.masked_fill(&mask.inner, value)) }

    fn index_add(&self, dim: isize, index: &PyTensor, source: &PyTensor) -> PyResult<PyTensor> {
        let d = norm_dim(dim, self.inner.ndim())?;
        ok(self.inner.index_add(d, &index.inner, &source.inner))
    }

    #[pyo3(signature = (weight, stride=1, padding=0))]
    fn conv2d(&self, weight: &PyTensor, stride: usize, padding: usize) -> PyResult<PyTensor> {
        self.inner.conv2d(&weight.inner, stride, padding).map(PyTensor::wrap).map_err(map_err)
    }

    fn max_pool2d(&self, kernel: usize, stride: usize) -> PyResult<PyTensor> {
        self.inner.max_pool2d(kernel, stride).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (dim=None, keepdim=false, *, dtype=None))]
    fn sum(&self, py: Python<'_>, dim: Option<&Bound<'_, PyAny>>, keepdim: bool, dtype: Option<&str>) -> PyResult<PyTensor> {
        if dim.is_some() || keepdim || dtype.is_some() { return functional_reduction(self, py, dim, keepdim, dtype, "sum"); }
        if self.inner.dtype() == ferro_core::DType::Bool { return ok(self.inner.bool_sum(None, false)); }
        self.typed_math(true)?;
        Ok(PyTensor::wrap(self.inner.sum()))
    }

    #[pyo3(signature = (dim=None, keepdim=false, *, dtype=None))]
    fn mean(&self, py: Python<'_>, dim: Option<&Bound<'_, PyAny>>, keepdim: bool, dtype: Option<&str>) -> PyResult<PyTensor> {
        if dim.is_some() || keepdim || dtype.is_some() { return functional_reduction(self, py, dim, keepdim, dtype, "mean"); }
        self.typed_math(false)?;
        Ok(PyTensor::wrap(self.inner.mean()))
    }

    fn acos(&self) -> PyResult<PyTensor> { ok(self.inner.acos()) }
    fn acosh(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.acosh()) }
    fn asin(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.asin()) }
    fn asinh(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.asinh()) }
    fn atan(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.atan()) }
    fn atanh(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.atanh()) }
    fn cos(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.cos()) }
    fn cosh(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.cosh()) }
    fn sin(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.sin()) }
    fn sinh(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.sinh()) }
    fn tan(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.tan()) }
    fn sinc(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.sinc()) }
    fn deg2rad(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.deg2rad()) }
    fn rad2deg(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.rad2deg()) }
    fn erf(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.erf()) }
    fn erfc(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.erfc()) }
    fn exp2(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.exp2()) }
    fn expm1(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.expm1()) }
    fn log10(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.log10()) }
    fn log1p(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.log1p()) }
    fn log2(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.log2()) }
    fn logit(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.logit()) }
    fn reciprocal(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.reciprocal()) }
    fn rsqrt(&self) -> PyResult<PyTensor> { ok(self.inner.rsqrt()) }
    fn square(&self) -> PyResult<PyTensor> { self.typed_math(true)?; ok(self.inner.square()) }
    fn sign(&self) -> PyResult<PyTensor> { self.typed_math(true)?; ok(self.inner.sign()) }
    fn ceil(&self) -> PyResult<PyTensor> { ok(self.inner.ceil()) }
    fn floor(&self) -> PyResult<PyTensor> { ok(self.inner.floor()) }
    #[pyo3(signature = (*, decimals=RoundDecimals(None)), text_signature="($self, *, decimals=0)")]
    fn round(&self, decimals: RoundDecimals) -> PyResult<PyTensor> {
        ok(if let Some(decimals) = decimals.0 { self.inner.round_decimals(decimals) } else { self.inner.round() })
    }
    fn trunc(&self) -> PyResult<PyTensor> { ok(self.inner.trunc()) }
    fn frac(&self) -> PyResult<PyTensor> { ok(self.inner.frac()) }
    #[pyo3(signature = (offset=0, dim1=0, dim2=1))]
    fn diagonal(&self, offset: isize, dim1: isize, dim2: isize) -> PyResult<PyTensor> {
        ok(ferro_core::linalg::diagonal(&self.inner, offset, dim1, dim2))
    }
    fn trace(&self) -> PyResult<PyTensor> {
        if self.inner.dtype() == ferro_core::DType::F64 { return ok(ferro_core::linalg::trace(&self.inner)); }
        self.float_math()?; ok(self.inner.trace())
    }
    fn silu(&self) -> PyResult<PyTensor> { self.typed_math(false)?; Ok(PyTensor::wrap(self.inner.silu())) }
    fn gelu_erf(&self) -> PyResult<PyTensor> { self.float_math()?; Ok(PyTensor::wrap(self.inner.gelu_erf())) }
    fn mish(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.mish()) }
    fn selu(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.selu()) }
    fn relu6(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.relu6()) }
    fn hardsigmoid(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.hardsigmoid()) }
    fn hardswish(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.hardswish()) }
    fn log_sigmoid(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.log_sigmoid()) }
    #[pyo3(signature = (beta=PyAlpha(ferro_core::ops_ext::add_alpha::Alpha::F64(1.0)), threshold=PyAlpha(ferro_core::ops_ext::add_alpha::Alpha::F64(20.0))), text_signature="($self, beta=1.0, threshold=20.0)")]
    fn softplus(&self, beta: PyAlpha, threshold: PyAlpha) -> PyResult<PyTensor> {
        use ferro_core::ops_ext::add_alpha::Alpha;
        self.typed_math(false)?;
        let scalar = |value| match value {
            Alpha::Bool(value) => u8::from(value) as f64,
            Alpha::I64(value) if self.inner.dtype() == ferro_core::DType::F32 => value as f32 as f64,
            Alpha::U64(value) if self.inner.dtype() == ferro_core::DType::F32 => value as f32 as f64,
            Alpha::I64(value) => value as f64, Alpha::U64(value) => value as f64,
            Alpha::F64(value) => value,
        };
        ok(self.inner.softplus_options(scalar(beta.0), scalar(threshold.0)))
    }
    fn softsign(&self) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.softsign()) }
    fn tanhshrink(&self) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.tanhshrink()) }
    #[pyo3(signature = (alpha=PyAlpha(ferro_core::ops_ext::add_alpha::Alpha::F64(1.0))), text_signature="($self, alpha=1.0)")]
    fn elu(&self, alpha: PyAlpha) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.elu_alpha(alpha.0)) }
    #[pyo3(signature = (alpha=PyAlpha(ferro_core::ops_ext::add_alpha::Alpha::F64(1.0))), text_signature="($self, alpha=1.0)")]
    fn celu(&self, alpha: PyAlpha) -> PyResult<PyTensor> { self.typed_math(false)?; ok(self.inner.celu_alpha(alpha.0)) }
    #[pyo3(signature = (negative_slope=0.01))]
    fn leaky_relu(&self, negative_slope: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.leaky_relu(negative_slope)) }
    #[pyo3(signature = (lambd=0.5))]
    fn hardshrink(&self, lambd: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.hardshrink(lambd)) }
    #[pyo3(signature = (lambd=0.5))]
    fn softshrink(&self, lambd: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.softshrink(lambd)) }
    #[pyo3(signature = (min_val=-1.0, max_val=1.0))]
    fn hardtanh(&self, min_val: f32, max_val: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.hardtanh(min_val, max_val)) }
    fn threshold(&self, threshold: f32, value: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.threshold(threshold, value)) }

    fn atan2(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { ok(self.inner.atan2(&expect_operand(other)?)) }
    fn copysign(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.copysign(&expect_operand(other)?)) }
    fn fmod(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.fmod(&expect_operand(other)?)) }
    fn remainder(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.remainder(&expect_operand(other)?)) }
    fn hypot(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.hypot(&expect_operand(other)?)) }
    fn heaviside(&self, values: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.heaviside(&expect_operand(values)?)) }
    fn logaddexp(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.logaddexp(&expect_operand(other)?)) }
    fn maximum(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.maximum(&expect_operand(other)?)) }
    fn minimum(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.minimum(&expect_operand(other)?)) }
    fn xlogy(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.xlogy(&expect_operand(other)?)) }
    fn outer(&self, other: &PyTensor) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.outer(&other.inner)) }
    fn lerp(&self, end: &PyTensor, weight: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.lerp(&end.inner, weight)) }
    #[pyo3(signature = (tensor1, tensor2, value=1.0))]
    fn addcmul(&self, tensor1: &PyTensor, tensor2: &PyTensor, value: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.addcmul(&tensor1.inner, &tensor2.inner, value)) }
    #[pyo3(signature = (tensor1, tensor2, value=1.0))]
    fn addcdiv(&self, tensor1: &PyTensor, tensor2: &PyTensor, value: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.addcdiv(&tensor1.inner, &tensor2.inner, value)) }
    #[pyo3(signature = (other, p=2.0))]
    fn dist(&self, other: &PyTensor, p: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.dist(&other.inner, p)) }

    fn logsumexp(&self, dim: isize) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.logsumexp(norm_dim(dim, self.inner.ndim())?)) }
    fn softmin(&self, dim: isize) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.softmin(norm_dim(dim, self.inner.ndim())?)) }
    #[pyo3(signature = (dim=-1))]
    fn diff(&self, dim: isize) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.diff(norm_dim(dim, self.inner.ndim())?)) }
    #[pyo3(signature = (dim, keepdim=false))]
    fn prod_dim(&self, dim: isize, keepdim: bool) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.prod_dim(norm_dim(dim, self.inner.ndim())?, keepdim)) }
    #[pyo3(signature = (dim, correction=1, keepdim=false))]
    fn var_dim(&self, dim: isize, correction: usize, keepdim: bool) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.var_dim(norm_dim(dim, self.inner.ndim())?, correction, keepdim)) }
    #[pyo3(signature = (dim, correction=1, keepdim=false))]
    fn std_dim(&self, dim: isize, correction: usize, keepdim: bool) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.std_dim(norm_dim(dim, self.inner.ndim())?, correction, keepdim)) }
    #[pyo3(signature = (dim=1, eps=1e-12))]
    fn normalize(&self, dim: isize, eps: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.normalize(norm_dim(dim, self.inner.ndim())?, eps)) }
    #[pyo3(signature = (other, dim=1, eps=1e-8))]
    fn cosine_similarity(&self, other: &PyTensor, dim: isize, eps: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.cosine_similarity(&other.inner, norm_dim(dim, self.inner.ndim())?, eps)) }
    #[pyo3(signature = (other, p=2.0, eps=1e-6))]
    fn pairwise_distance(&self, other: &PyTensor, p: f32, eps: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.pairwise_distance(&other.inner, p, eps)) }
    #[pyo3(signature = (start_dim=0, end_dim=-1))]
    fn flatten(&self, start_dim: isize, end_dim: isize) -> PyResult<PyTensor> {
        let n = self.inner.ndim().max(1);
        ok(self.inner.flatten(norm_dim(start_dim, n)?, norm_dim(end_dim, n)?))
    }
    #[pyo3(signature = (diagonal=0))]
    fn triu(&self, diagonal: i64) -> PyResult<PyTensor> { ok(self.inner.triu(diagonal)) }
    #[pyo3(signature = (diagonal=0))]
    fn tril(&self, diagonal: i64) -> PyResult<PyTensor> { ok(self.inner.tril(diagonal)) }
    /// `pads` holds one (before, after) pair per dimension in dim order
    /// (not torch's last-dim-first order; `ferro.nn.functional.pad` converts).
    #[pyo3(signature = (pads, value=0.0))]
    fn pad_constant(&self, pads: Vec<usize>, value: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.pad_constant(&pads, value)) }
    fn scatter(&self, dim: isize, index: &PyTensor, src: &PyTensor) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.scatter(norm_dim(dim, self.inner.ndim())?, &index.inner, &src.inner)) }
    fn scatter_add(&self, dim: isize, index: &PyTensor, src: &PyTensor) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.scatter_add(norm_dim(dim, self.inner.ndim())?, &index.inner, &src.inner)) }
    #[pyo3(signature = (kernel, stride=None))]
    fn avg_pool2d(&self, kernel: usize, stride: Option<usize>) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.avg_pool2d(kernel, stride.unwrap_or(kernel))) }
    #[pyo3(signature = (weight=None, eps=1e-6))]
    fn rms_norm(&self, weight: Option<&PyTensor>, eps: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.rms_norm(weight.map(|t| &t.inner), eps)) }
    #[pyo3(signature = (num_groups, weight, bias, eps=1e-5))]
    fn group_norm(&self, num_groups: usize, weight: &PyTensor, bias: &PyTensor, eps: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.group_norm(num_groups, &weight.inner, &bias.inner, eps)) }
    /// Returns (output, new_running_mean, new_running_var); inputs are not mutated.
    #[pyo3(signature = (weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5))]
    fn batch_norm(&self, weight: &PyTensor, bias: &PyTensor, running_mean: &PyTensor, running_var: &PyTensor, training: bool, momentum: f32, eps: f32) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
        let r = self.inner.batch_norm(&weight.inner, &bias.inner, &running_mean.inner, &running_var.inner, eps, training, momentum).map_err(map_err)?;
        Ok((PyTensor::wrap(r.output), PyTensor::wrap(r.running_mean), PyTensor::wrap(r.running_var)))
    }
    /// Counter-based (Philox) inverted dropout: the mask is a pure function of
    /// (seed, offset, element index); callers advance `offset` between steps.
    #[pyo3(signature = (p, training, seed, offset=0))]
    fn dropout(&self, p: f64, training: bool, seed: u64, offset: u64) -> PyResult<PyTensor> {
        if self.inner.dtype() == ferro_core::DType::F64 { return ok(self.inner.dropout_f64(p, training, seed, offset)); }
        self.float_math()?;
        ok(self.inner.dropout(p as f32, training, seed, offset))
    }
    /// RoPE at positions 0..seq over [batch, seq, head_dim] from the shared table cache.
    #[pyo3(signature = (base=10000.0))]
    fn rope_cached(&self, base: f32) -> PyResult<PyTensor> { self.float_math()?; ok(self.inner.rope_cached(base)) }

    /// Basic views and one broadcast integer array or a single boolean mask.
    /// Typed floating selections retain native reverse and forward AD.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if key.is_instance_of::<pyo3::types::PyBool>() {
            let mask = CoreTensor::from_vec_bool(vec![key.extract::<bool>()?], &[]).map_err(map_err)?;
            return ok(self.inner.mask_index(&mask));
        }
        if let Ok(index) = key.extract::<PyRef<'_, PyTensor>>() {
            if index.inner.dtype() == ferro_core::DType::Bool { return ok(self.inner.mask_index(&index.inner)); }
            if index.inner.dtype() != ferro_core::DType::I64 || self.inner.ndim() == 0 {
                return Err(PyTypeError::new_err("tensor indexing requires a boolean mask or int64 index on a nonscalar input"));
            }
            if index.inner.device() != self.inner.device() { return Err(PyValueError::new_err("index and input devices must match")); }
            if index.inner.ndim() == 0 {
                let i = isize::try_from(index.inner.to_vec_i64()[0]).map_err(|_| PyValueError::new_err("index exceeds platform range"))?;
                return self.index_view(&[Sel::Idx(i)], self.inner.shape());
            }
            if self.inner.device() != Device::Cpu {
                if index.inner.ndim() != 1 { return Err(pyo3::exceptions::PyNotImplementedError::new_err("multidimensional integer indices currently require CPU storage")); }
                let indices = index.inner.to_vec_i64().into_iter().map(|i| {
                    let i = isize::try_from(i).map_err(|_| PyValueError::new_err("index exceeds platform range"))?;
                    wrap_index(i, self.inner.shape()[0])
                }).collect::<PyResult<Vec<_>>>()?;
                return ok(self.inner.index_select(0, &indices));
            }
            return ok(self.inner.index_tensor(0, &index.inner));
        }
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            if tuple.len() == 1 {
                let item = tuple.get_item(0)?;
                let direct = item.is_instance_of::<pyo3::types::PyBool>() || item.extract::<PyRef<'_, PyTensor>>().is_ok();
                if direct { return self.__getitem__(&item); }
            }
        }
        let shape = self.inner.shape().to_vec();
        let picked = parse_key(key, shape.len())?;
        self.index_view(&picked, &shape)
    }

    /// Convert dtype/device; floating CPU casts retain the autograd graph.
    #[pyo3(signature = (device=None, dtype=None, *, copy=false))]
    fn to(slf: PyRef<'_, Self>, device: Option<&Bound<'_, PyAny>>, dtype: Option<&str>, copy: bool) -> PyResult<Py<PyTensor>> {
        let mut target_device = slf.inner.device();
        let mut target_dtype = dtype.map(parse_dtype).transpose()?.unwrap_or_else(|| slf.inner.dtype());
        if let Some(value) = device.filter(|v| !v.is_none()) {
            if let Ok(other) = value.extract::<PyRef<'_, PyTensor>>() {
                if dtype.is_some() { return Err(PyTypeError::new_err("to(other) does not accept a separate dtype")); }
                target_device = other.inner.device();
                target_dtype = other.inner.dtype();
            } else {
                let value = value.extract::<String>()?;
                if matches!(value.as_str(), "float32" | "float64" | "int64" | "bool") {
                    if dtype.is_some() { return Err(PyTypeError::new_err("dtype specified twice")); }
                    target_dtype = parse_dtype(&value)?;
                } else { target_device = parse_device(&value)?; }
            }
        }
        if target_device == slf.inner.device() && target_dtype == slf.inner.dtype() && !copy { return Ok(slf.into()); }
        if target_device != slf.inner.device() && slf.inner.requires_grad() {
            return Err(PyValueError::new_err("differentiable device transfers are not supported; detach explicitly before transfer"));
        }
        let result = slf.inner.to_dtype_autograd(target_dtype, copy).and_then(|t| t.to_device(target_device)).map_err(map_err)?;
        Py::new(slf.py(), PyTensor::wrap(result))
    }

    fn float(slf: PyRef<'_, Self>) -> PyResult<Py<PyTensor>> { Self::to(slf, None, Some("float32"), false) }
    fn double(slf: PyRef<'_, Self>) -> PyResult<Py<PyTensor>> { Self::to(slf, None, Some("float64"), false) }
    fn long(slf: PyRef<'_, Self>) -> PyResult<Py<PyTensor>> { Self::to(slf, None, Some("int64"), false) }

    fn cpu(&self) -> PyResult<PyTensor> {
        if self.inner.requires_grad() && self.inner.device() != Device::Cpu {
            return Err(PyValueError::new_err("differentiable device transfers are not supported; detach explicitly before transfer"));
        }
        self.inner.to_device(Device::Cpu).map(PyTensor::wrap).map_err(map_err)
    }

    #[pyo3(signature = (index=0))]
    fn cuda(&self, index: u32) -> PyResult<PyTensor> {
        if self.inner.requires_grad() && self.inner.device() != Device::Cuda(index) {
            return Err(PyValueError::new_err("differentiable device transfers are not supported; detach explicitly before transfer"));
        }
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
    #[pyo3(signature = (requires_grad=None, *, req=None))]
    fn requires_grad_(slf: PyRefMut<'_, Self>, requires_grad: Option<bool>, req: Option<bool>) -> PyResult<PyRefMut<'_, Self>> {
        if requires_grad.is_some() && req.is_some() { return Err(PyTypeError::new_err("requires_grad_ received both requires_grad and req")); }
        slf.inner.set_requires_grad(requires_grad.or(req).unwrap_or(true)).map_err(map_err)?;
        Ok(slf)
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    fn is_leaf(&self) -> bool { self.inner.is_leaf() }

    fn _depends_on(&self, other: &PyTensor) -> bool {
        self.inner.depends_on(&other.inner)
    }

    fn _id(&self) -> usize { self.inner.id() }
    fn _is_structural_zero(&self) -> bool { self.inner.is_structural_zero() }
    fn _shares_storage(&self, other: &PyTensor) -> bool { self.inner._storage_ptr() == other.inner._storage_ptr() }

    fn _has_grad_leaf_except(&self, excluded: Vec<PyRef<'_, PyTensor>>) -> bool {
        self.inner.has_grad_leaf_except(&excluded.iter().map(|t| &t.inner).collect::<Vec<_>>())
    }

    fn _has_current_grad_leaf_except(&self, excluded: Vec<PyRef<'_, PyTensor>>) -> bool {
        self.inner.has_current_grad_leaf_except(&excluded.iter().map(|t| &t.inner).collect::<Vec<_>>())
    }

    #[staticmethod]
    #[pyo3(signature = (outputs, inputs, seeds, create_graph=false, preserve_passthrough=false, cutoff=None, excluded=None, functional=false, caller_grad_mode=false))]
    fn _vjp_many_wrt_transform(outputs: Vec<PyRef<'_, PyTensor>>, inputs: Vec<PyRef<'_, PyTensor>>, seeds: Vec<PyRef<'_, PyTensor>>, create_graph: bool, preserve_passthrough: bool, cutoff: Option<usize>, excluded: Option<Vec<PyRef<'_, PyTensor>>>, functional: bool, caller_grad_mode: bool) -> PyResult<Vec<Option<PyTensor>>> {
        let outputs: Vec<_> = outputs.iter().map(|value| &value.inner).collect();
        let inputs: Vec<_> = inputs.iter().map(|value| &value.inner).collect();
        let seeds: Vec<_> = seeds.iter().map(|value| &value.inner).collect();
        let excluded = excluded.unwrap_or_default();
        let excluded: Vec<_> = excluded.iter().map(|value| &value.inner).collect();
        if cutoff.is_none() && !excluded.is_empty() {
            return Err(PyValueError::new_err("excluded primals require a frozen cutoff"));
        }
        let frozen = cutoff.map(|value| (value, excluded.as_slice()));
        let result = CoreTensor::vjp_many_wrt_transform_policy(&outputs, &inputs, &seeds, create_graph, preserve_passthrough, frozen, functional, caller_grad_mode);
        result.map(|values| values.into_iter().map(|value| value.map(PyTensor::wrap)).collect()).map_err(map_err)
    }

    #[staticmethod]
    #[pyo3(signature = (outputs, inputs, seeds, create_graph=false))]
    fn _vjp_many_wrt_optional(outputs: Vec<PyRef<'_, PyTensor>>, inputs: Vec<PyRef<'_, PyTensor>>, seeds: Vec<PyRef<'_, PyTensor>>, create_graph: bool) -> PyResult<Vec<Option<PyTensor>>> {
        let outputs: Vec<_> = outputs.iter().map(|t| &t.inner).collect();
        let inputs: Vec<_> = inputs.iter().map(|t| &t.inner).collect();
        let seeds: Vec<_> = seeds.iter().map(|t| &t.inner).collect();
        CoreTensor::vjp_many_wrt_optional(&outputs, &inputs, &seeds, create_graph)
            .map(|v| v.into_iter().map(|t| t.map(PyTensor::wrap)).collect()).map_err(map_err)
    }

    #[pyo3(signature = (inputs, create_graph=false))]
    fn grad_wrt(&self, inputs: Vec<PyRef<'_, PyTensor>>, create_graph: bool) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| &t.inner).collect();
        self.inner.grad_wrt(&inputs, create_graph).map(|v| v.into_iter().map(PyTensor::wrap).collect()).map_err(map_err)
    }

    #[pyo3(signature = (inputs, seed, create_graph=false))]
    fn vjp_wrt(&self, inputs: Vec<PyRef<'_, PyTensor>>, seed: &PyTensor, create_graph: bool) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| &t.inner).collect();
        self.inner.vjp_wrt(&inputs, &seed.inner, create_graph).map(|v| v.into_iter().map(PyTensor::wrap).collect()).map_err(map_err)
    }

    fn _vjp_wrt_preserve(&self, inputs: Vec<PyRef<'_, PyTensor>>, seed: &PyTensor) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| &t.inner).collect();
        self.inner.vjp_wrt_preserve(&inputs, &seed.inner).map(|v| v.into_iter().map(PyTensor::wrap).collect()).map_err(map_err)
    }

    #[pyo3(signature = (inputs, seed, create_graph=false))]
    fn _vjp_wrt_optional(&self, inputs: Vec<PyRef<'_, PyTensor>>, seed: &PyTensor, create_graph: bool) -> PyResult<Vec<Option<PyTensor>>> {
        let inputs: Vec<_> = inputs.iter().map(|t| &t.inner).collect();
        self.inner.vjp_wrt_optional(&inputs, &seed.inner, create_graph)
            .map(|v| v.into_iter().map(|t| t.map(PyTensor::wrap)).collect()).map_err(map_err)
    }

    #[pyo3(signature = (inputs, seed, cutoff, excluded=None))]
    fn _vjp_wrt_frozen(&self, inputs: Vec<PyRef<'_, PyTensor>>, seed: &PyTensor, cutoff: usize, excluded: Option<Vec<PyRef<'_, PyTensor>>>) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| &t.inner).collect();
        let excluded = excluded.unwrap_or_default();
        let excluded: Vec<_> = excluded.iter().map(|t| &t.inner).collect();
        self.inner.vjp_wrt_frozen_excluding(&inputs, &seed.inner, cutoff, &excluded)
            .map(|v| v.into_iter().map(PyTensor::wrap).collect()).map_err(map_err)
    }

    fn backward_with(&self, seed: &PyTensor) -> PyResult<()> {
        if seed.inner.shape() != self.inner.shape() || seed.inner.device() != self.inner.device() || seed.inner.dtype() != self.inner.dtype() {
            return Err(PyValueError::new_err("backward seed must match output shape, dtype, and device"));
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.inner.backward_with(&seed.inner)))
            .map_err(|_| PyValueError::new_err("native backward failed (invalid or stale graph)"))
    }

    fn backward(&self) -> PyResult<()> {
        if self.inner.numel() != 1 {
            return Err(PyValueError::new_err(
                "backward() requires a scalar output; reduce with .sum() or .mean()",
            ));
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.inner.backward()))
            .map_err(|payload| {
                let message = payload.downcast_ref::<String>().map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("native backward failed (invalid or stale graph)");
                PyValueError::new_err(message.to_owned())
            })
    }

    /// Copy values into this tensor's existing storage and return self.
    /// Shape and dtype must match. Uses core's checked in-place API: whole
    /// contiguous destinations, no op history or shared device snapshots. Typed
    /// copies require CPU; device copies require f32. Grad-requiring
    /// leaves additionally require an active no_grad context.
    /// Captured leaf handles observe the new values on their next replay.
    fn copy_<'py>(slf: PyRef<'py, Self>, src: &PyTensor) -> PyResult<PyRef<'py, Self>> {
        checked_inplace_copy(&slf.inner, &src.inner)?;
        Ok(slf)
    }

    #[pyo3(signature = (other, *, alpha=1.0))]
    fn add_<'py>(slf: PyRef<'py, Self>, other: &Bound<'_, PyAny>, alpha: f64) -> PyResult<PyRef<'py, Self>> {
        inplace_binary(&slf, other, Some(alpha))?;
        Ok(slf)
    }

    fn mul_<'py>(slf: PyRef<'py, Self>, other: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
        inplace_binary(&slf, other, None)?;
        Ok(slf)
    }

    fn zero_(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.inner.zero_().map_err(map_err)?;
        Ok(slf)
    }

    fn fill_<'py>(slf: PyRef<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<PyRef<'py, Self>> {
        use ferro_core::{DType, inplace::FillValue};
        checked_inplace_copy(&slf.inner, &slf.inner)?;
        let scalar;
        let (value, from_tensor) = if let Ok(tensor) = value.extract::<PyRef<'_, PyTensor>>() {
            if tensor.inner.ndim() != 0 { return Err(PyValueError::new_err("fill_ requires a zero-dimensional value tensor")); }
            if tensor.inner.requires_grad() && ferro_core::autograd::is_grad_enabled() {
                return Err(PyValueError::new_err("fill_ with a differentiable value tensor requires disabled grad recording"));
            }
            scalar = tensor.item(value.py())?;
            (&scalar, true)
        } else { (value, false) };
        let integer = value.is_instance_of::<pyo3::types::PyInt>();
        if !integer && !value.is_instance_of::<pyo3::types::PyFloat>() {
            return Err(PyTypeError::new_err("fill_ requires a real numeric scalar or zero-dimensional tensor"));
        }
        let signed = if integer { value.extract::<i64>().ok() } else { None };
        let unsigned = if integer && signed.is_none() { Some(value.extract::<u64>()?) } else { None };
        let scalar = match slf.inner.dtype() {
            DType::F32 => {
                let number = if let Some(value) = signed { value as f32 }
                    else if let Some(value) = unsigned { value as f32 }
                    else {
                        let value = value.extract::<f64>()?;
                        if !from_tensor && value.is_finite() && value.abs() > f32::MAX as f64 {
                            return Err(PyValueError::new_err("value cannot be converted to float32 without overflow"));
                        }
                        value as f32
                    };
                FillValue::F32(number)
            },
            DType::F64 => FillValue::F64(value.extract::<f64>()?),
            DType::I64 => {
                let number = if let Some(value) = signed { value } else if integer {
                    return Err(PyValueError::new_err("value cannot be converted to int64 without overflow"));
                } else {
                    let value = value.extract::<f64>()?;
                    let limit = 9_223_372_036_854_775_808.0;
                    let invalid = !value.is_finite() || value < -limit || value > limit;
                    if invalid && !from_tensor {
                        return Err(PyValueError::new_err("value cannot be converted to int64 without overflow"));
                    }
                    if invalid || value == limit { i64::MIN } else { value as i64 }
                };
                FillValue::I64(number)
            },
            DType::Bool => FillValue::Bool(value.is_truthy()?),
            _ => return Err(PyTypeError::new_err("fill_ supports float32, float64, int64 and bool storage")),
        };
        slf.inner.fill_typed_(scalar).map_err(map_err)?;
        Ok(slf)
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn detach(&self) -> PyTensor {
        PyTensor::wrap(self.inner.detach())
    }

    fn _zeros_like_layout(&self) -> PyResult<PyTensor> {
        ok(self.inner.zeros_like_layout())
    }

    #[pyo3(name = "clone", signature = (*, memory_format=None))]
    fn clone_tensor(&self, memory_format: Option<&str>) -> PyResult<PyTensor> {
        let preserve = match memory_format {
            None | Some("preserve_format") => true,
            Some("contiguous_format") => false,
            _ => return Err(PyValueError::new_err("clone supports preserve_format and contiguous_format")),
        };
        ok(self.inner.clone_tensor(preserve))
    }

    #[pyo3(signature = (*, memory_format=None))]
    fn contiguous<'py>(slf: PyRef<'py, Self>, memory_format: Option<&str>) -> PyResult<Py<PyTensor>> {
        if !matches!(memory_format, None | Some("contiguous_format") | Some("preserve_format")) {
            return Err(PyValueError::new_err("contiguous supports contiguous_format"));
        }
        if slf.inner.is_contiguous() { return Ok(slf.into()); }
        if memory_format == Some("preserve_format") {
            return Err(PyValueError::new_err("preserve_format is unsupported for a noncontiguous tensor"));
        }
        let output = slf.inner.contiguous().map_err(map_err)?;
        Py::new(slf.py(), PyTensor::wrap(output))
    }

    #[pyo3(signature = (*, memory_format=None))]
    fn is_contiguous(&self, memory_format: Option<&str>) -> PyResult<bool> {
        if !matches!(memory_format, None | Some("contiguous_format") | Some("preserve_format")) {
            return Err(PyValueError::new_err("is_contiguous supports contiguous_format and preserve_format"));
        }
        Ok(self.inner.is_contiguous())
    }

    #[pyo3(signature = (dim=None))]
    fn stride<'py>(&self, py: Python<'py>, dim: Option<isize>) -> PyResult<Bound<'py, PyAny>> {
        match dim {
            None => Ok(PyTuple::new(py, self.inner.strides())?.into_any()),
            Some(dim) => self.inner.strides()[norm_dim(dim, self.inner.ndim())?].into_bound_py_any(py),
        }
    }

    fn storage_offset(&self) -> usize {
        self.inner.storage_offset()
    }

    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(PyTensor::wrap)
    }

    #[setter(grad)]
    fn set_grad(&self, gradient: Option<&PyTensor>) -> PyResult<()> {
        self.inner.set_leaf_grad(gradient.map(|value| value.inner.clone())).map_err(map_err)
    }

    #[pyo3(signature = (gradient=None))]
    fn _set_leaf_grad(&self, gradient: Option<&PyTensor>) -> PyResult<()> {
        self.inner.set_leaf_grad(gradient.map(|value| value.inner.clone())).map_err(map_err)
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let dims: Vec<usize> = self.inner.shape().to_vec();
        PyList::new(py, dims).unwrap().into_any()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        use ferro_core::DType;
        match self.inner.dtype() {
            DType::F32 => "float32", DType::F64 => "float64", DType::I64 => "int64",
            DType::Bool => "bool",
            DType::F16 => "float16", DType::BF16 => "bfloat16",
        }
    }

    #[getter]
    fn ndim(&self) -> usize { self.inner.ndim() }

    fn dim(&self) -> usize { self.inner.ndim() }

    fn numel(&self) -> usize { self.inner.numel() }

    #[pyo3(signature = (dim=None))]
    fn size<'py>(&self, py: Python<'py>, dim: Option<isize>) -> PyResult<Bound<'py, PyAny>> {
        match dim {
            Some(d) => self.inner.shape()[norm_dim(d, self.inner.ndim())?].into_bound_py_any(py),
            None => Ok(PyTuple::new(py, self.inner.shape())?.into_any()),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner.shape().first().copied().ok_or_else(|| PyTypeError::new_err("len() of a scalar tensor"))
    }

    fn __bool__(&self) -> PyResult<bool> {
        if self.inner.numel() != 1 {
            return Err(PyValueError::new_err("truth value requires a single-element tensor"));
        }
        use ferro_core::DType;
        Ok(match self.inner.dtype() {
            DType::I64 => self.inner.to_vec_i64()[0] != 0,
            DType::Bool => self.inner.to_vec_bool()[0],
            DType::F64 => self.inner.to_vec_f64()[0] != 0.0,
            DType::F32 | DType::F16 | DType::BF16 => self.inner.item() != 0.0,
        })
    }

    fn tolist<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        use ferro_core::DType;
        match self.inner.dtype() {
            DType::I64 => to_nested(py, &self.inner.to_vec_i64(), self.inner.shape()),
            DType::Bool => to_nested(py, &self.inner.to_vec_bool(), self.inner.shape()),
            DType::F64 => to_nested(py, &self.inner.to_vec_f64(), self.inner.shape()),
            DType::F32 | DType::F16 | DType::BF16 => to_nested(py, &self.inner.to_vec(), self.inner.shape()),
        }
    }

    fn item<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.inner.numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "item() requires a single-element tensor, got shape {:?}",
                self.inner.shape()
            )));
        }
        use ferro_core::DType;
        match self.inner.dtype() {
            DType::I64 => self.inner.to_vec_i64()[0].into_bound_py_any(py),
            DType::Bool => self.inner.to_vec_bool()[0].into_bound_py_any(py),
            DType::F64 => self.inner.to_vec_f64()[0].into_bound_py_any(py),
            DType::F32 | DType::F16 | DType::BF16 => self.inner.item().into_bound_py_any(py),
        }
    }

    fn __repr__(&self) -> String {
        let dev = self.inner.device().to_string();
        let dev_s = if dev == "cpu" { String::new() } else { format!("device='{dev}', ") };
        let data = match self.inner.dtype() {
            ferro_core::DType::I64 => fmt_data(&self.inner.to_vec_i64(), self.inner.shape()),
            ferro_core::DType::Bool => fmt_data(&self.inner.to_vec_bool(), self.inner.shape()),
            ferro_core::DType::F64 => fmt_data(&self.inner.to_vec_f64(), self.inner.shape()),
            _ => fmt_data(&self.inner.to_vec(), self.inner.shape()),
        };
        format!(
            "Tensor({:?}, {}dtype={}, data={})",
            self.inner.shape(),
            dev_s,
            self.dtype(),
            data
        )
    }

    /// DLPack producer. CUDA None/1 (legacy), 2 (per-thread), and positive
    /// stream handles use a conservative allocation-stream fence. The optional
    /// -1 no-sync extension is not supported; no consumer handle is dereferenced.
    #[pyo3(signature = (stream=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        validate_dlpack_stream(self.inner.device(), stream.as_ref())?;
        if self.inner.device() != Device::Cpu {
            let view = self.inner.dlpack_device_view().map_err(map_err)?;
            let buffer = view.device_buffer().as_any().downcast_ref::<ferro_cuda::CudaBuf>()
                .ok_or_else(|| PyValueError::new_err("device buffer was not allocated by CUDA"))?;
            // Use allocation ownership, not the replaceable registry/LAST_BACKEND.
            // DevicePtr joins tracked foreign writes onto this stream before the fence.
            let producer = buffer.stream();
            let (_ptr, usage) = cudarc::driver::DevicePtr::device_ptr(&**buffer, producer);
            #[cfg(test)]
            dlpack_error_tests::inject(1, producer.context());
            let ctx = producer.context();
            // Infallible dependency/Drop APIs record errors on the allocation context.
            let ready = ctx.check_err().and_then(|()| producer.synchronize());
            drop(usage);
            #[cfg(test)]
            dlpack_error_tests::inject(2, ctx);
            let released = ctx.check_err();
            ready.and(released).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        #[cfg(test)]
        dlpack_error_tests::EXPORT_CALLS.with(|n| n.set(n.get() + 1));
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
#[pyo3(signature = (tensors, dim=0))]
fn stack(tensors: Vec<PyTensor>, dim: isize) -> PyResult<PyTensor> {
    let inners: Vec<CoreTensor> = tensors.iter().map(|t| t.inner.clone()).collect();
    let ndim = inners.first().map(|t| t.ndim() + 1).unwrap_or(1);
    let d = if dim < 0 { dim + ndim as isize } else { dim };
    if d < 0 || d as usize >= ndim {
        return Err(PyValueError::new_err(format!("dim {dim} out of range for stacking rank-{} tensors", ndim - 1)));
    }
    ok(CoreTensor::stack(&inners, d as usize))
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
/// device is present (never panics). Each call replaces the registered backend.
/// Older allocations remain readable via tolist()/cpu() on their owning stream;
/// compute, mutation and device-to-device copies still require the owning backend.
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

/// Execute a zero-argument callable eagerly while recording op inputs and tags.
/// Returns its Tensor result without enabling requires_grad. Use
/// `ferro.capture(lambda: (x.relu() * w + b)).compile_fused()` then `replay()`.
/// Recording is thread-local, nestable, and restored even if the callable raises.
/// This is synchronous (not an async context); spawned threads do not inherit it.
/// Inputs are retained by the recorded graph; existing autograd behavior is
/// preserved. Capture-only nodes do not retain backward closures. Compilation
/// rejects unsupported graphs; replay returns detached outputs from current
/// leaf storage values. Python variable rebinding does not replace those leaves.
#[pyfunction]
fn capture(build: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    ferro_core::capture(|| build.call0()?.extract::<PyTensor>())
}

#[pyfunction]
fn _is_capturing() -> bool { ferro_core::capture::is_recording() }

#[pyfunction]
fn _tensor_id_watermark() -> usize { ferro_core::tensor::tensor_id_watermark() }

#[pyclass(name = "Parameter", module = "ferro.nn", extends=PyTensor, unsendable)]
struct PyParameter { inner: ferro_core::params::Param }

impl PyParameter {
    fn parts(inner: ferro_core::params::Param) -> (Self, PyTensor) {
        let tensor = PyTensor::wrap(inner.tensor());
        (Self { inner }, tensor)
    }

    fn wrap(py: Python<'_>, inner: ferro_core::params::Param) -> PyResult<Py<Self>> {
        Py::new(py, Self::parts(inner))
    }
}

#[pymethods]
impl PyParameter {
    #[staticmethod]
    #[pyo3(signature = (tensor, requires_grad=true))]
    fn _from_shared(py: Python<'_>, tensor: &PyTensor, requires_grad: bool) -> PyResult<Py<Self>> {
        Self::wrap(py, ferro_core::params::Param::from_shared(&tensor.inner, requires_grad).map_err(map_err)?)
    }

    #[new]
    #[pyo3(signature = (data=None, requires_grad=true, *, tensor=None))]
    fn new(data: Option<&PyTensor>, requires_grad: bool, tensor: Option<&PyTensor>) -> PyResult<(Self, PyTensor)> {
        if ferro_core::capture::is_recording() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err("Parameter construction during capture is unsupported"));
        }
        if data.is_some() && tensor.is_some() {
            return Err(PyTypeError::new_err("Parameter received both data and tensor"));
        }
        let tensor = match data.or(tensor) {
            Some(tensor) => {
                if !matches!(tensor.inner.dtype(), ferro_core::DType::F32 | ferro_core::DType::F64) {
                    return Err(PyTypeError::new_err("Parameter requires float32 or float64 storage"));
                }
                tensor.inner.detach()
            },
            None => CoreTensor::from_vec(Vec::new(), &[0]).map_err(map_err)?,
        };
        let parameter = ferro_core::params::Param::try_new(tensor).map_err(map_err)?;
        parameter.tensor().set_requires_grad(requires_grad).map_err(map_err)?;
        Ok(Self::parts(parameter))
    }
    fn tensor(&self) -> PyTensor { PyTensor::wrap(self.inner.tensor()) }
    #[getter]
    fn requires_grad(&self) -> bool { self.inner.tensor().requires_grad() }
    #[pyo3(signature = (requires_grad=true))]
    fn requires_grad_<'py>(slf: PyRef<'py, Self>, requires_grad: bool) -> PyResult<PyRef<'py, Self>> {
        if ferro_core::capture::is_recording() {
            return Err(PyValueError::new_err("requires_grad_ is not supported during capture"));
        }
        slf.inner.tensor().set_requires_grad(requires_grad).map_err(map_err)?;
        Ok(slf)
    }
    #[getter]
    fn trainable(&self) -> bool { self.inner.is_trainable() }
    #[setter]
    fn set_trainable(&self, value: bool) { self.inner.set_trainable(value); }
    fn zero_grad(&self) { self.inner.zero_grad(); }
    #[getter]
    fn grad(&self) -> Option<PyTensor> { self.inner.grad().map(PyTensor::wrap) }
    #[setter(grad)]
    fn set_grad(&self, gradient: Option<&PyTensor>) -> PyResult<()> {
        self.inner.tensor().set_leaf_grad(gradient.map(|value| value.inner.clone())).map_err(map_err)
    }
}

#[pyclass(name = "_SGD", unsendable)]
struct PySgd { inner: ferro_core::optim::Sgd }

#[pymethods]
impl PySgd {
    #[new]
    #[pyo3(signature = (params, lr, momentum=0.0, nesterov=false))]
    fn new(params: Vec<PyRef<'_, PyParameter>>, lr: f32, momentum: f32, nesterov: bool) -> PyResult<Self> {
        if !lr.is_finite() || lr < 0.0 || !momentum.is_finite() || momentum < 0.0 || (nesterov && momentum == 0.0) {
            return Err(PyValueError::new_err("invalid SGD learning rate or momentum"));
        }
        let slots = params.iter().map(|p| p.inner.clone()).collect();
        Ok(Self { inner: ferro_core::optim::Sgd::new(slots, lr).with_momentum(momentum).with_nesterov(nesterov) })
    }
    fn zero_grad(&self) { self.inner.zero_grad(); }
    fn step(&mut self) -> PyResult<()> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.inner.step()))
            .map_err(|_| PyValueError::new_err("native SGD step failed"))
    }
}

#[pyfunction]
fn is_grad_enabled() -> bool { ferro_core::autograd::is_grad_enabled() }

#[pyfunction]
fn set_grad_enabled(enabled: bool) -> bool { ferro_core::autograd::set_grad_enabled(enabled) }

#[pyfunction]
fn no_grad(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(py.import("ferro._grad_mode")?.getattr("no_grad")?.call0()?.unbind())
}

#[pyfunction]
fn enable_grad(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(py.import("ferro._grad_mode")?.getattr("enable_grad")?.call0()?.unbind())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Route matmul and elementwise/reduction ops through the optimized CPU
    // backend for the whole process.
    ferro_fastcpu::install();
    ferro_fastcpu::install_backend();
    training_state::register(m)?;
    architecture::register(m)?;
    recurrent::register(m)?;
    basis::register(m)?;
    convolution::register(m)?;
    losses::register(m)?;
    data::register(m)?;
    linalg::register(m)?;
    forward_ad::register(m)?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyParameter>()?;
    m.add_class::<PySgd>()?;
    m.add_function(wrap_pyfunction!(_arithmetic, m)?)?;
    m.add_function(wrap_pyfunction!(_division, m)?)?;
    m.add_function(wrap_pyfunction!(_real_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(is_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(no_grad, m)?)?;
    m.add_function(wrap_pyfunction!(enable_grad, m)?)?;
    m.add_function(wrap_pyfunction!(capture, m)?)?;
    m.add_function(wrap_pyfunction!(_is_capturing, m)?)?;
    m.add_function(wrap_pyfunction!(_tensor_id_watermark, m)?)?;
    m.add_class::<PyCompiledChain>()?;
    m.add_class::<PyStaticGraph>()?;
    m.add_class::<Generator>()?;
    m.add_function(wrap_pyfunction!(from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(where_, m)?)?;
    m.add_function(wrap_pyfunction!(save_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_init, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_synchronize, m)?)?;
    Ok(())
}
