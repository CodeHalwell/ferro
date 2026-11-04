use super::*;

#[pyclass(name = "_ForwardAD", unsendable)]
struct PyForwardAD { level: Option<ferro_core::forward_ad::DualLevel> }

#[pyclass(name = "_ForwardColumns", unsendable)]
struct PyForwardColumns { level: Option<ferro_core::forward_ad::DualLevel> }

#[pyclass(name = "_ForwardBatched", unsendable)]
struct PyForwardBatched { level: Option<ferro_core::forward_ad::DualLevel> }

#[pymethods]
impl PyForwardBatched {
    #[new]
    #[pyo3(signature = (primals, tangents, batch, randomness_same=false))]
    fn new(primals: Vec<PyRef<'_, PyTensor>>, tangents: Vec<PyRef<'_, PyTensor>>, batch: usize, randomness_same: bool) -> PyResult<Self> {
        let primals = primals.iter().map(|value| &value.inner).collect::<Vec<_>>();
        let tangents = tangents.iter().map(|value| &value.inner).collect::<Vec<_>>();
        let level = ferro_core::forward_ad::DualLevel::functional_batched(&primals, &tangents, batch).map_err(map_err)?;
        level.set_randomness_same(randomness_same).map_err(map_err)?;
        Ok(Self { level: Some(level) })
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __exit__(&mut self, _kind: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) -> bool { self.level.take(); false }
    fn tangents(&self, outputs: Vec<PyRef<'_, PyTensor>>) -> PyResult<Vec<Option<PyTensor>>> {
        let level = self.level.as_ref().ok_or_else(|| PyValueError::new_err("forward AD level is closed"))?;
        outputs.iter().map(|value| level.batched_tangent(&value.inner).map(|value| value.map(PyTensor::wrap)).map_err(map_err)).collect()
    }
    fn primals(&self, outputs: Vec<PyRef<'_, PyTensor>>, auxiliary: bool) -> PyResult<Vec<PyTensor>> {
        let level = self.level.as_ref().ok_or_else(|| PyValueError::new_err("forward AD level is closed"))?;
        outputs.iter().map(|value| level.primal(&value.inner, auxiliary).map(PyTensor::wrap).map_err(map_err)).collect()
    }
}

#[pyclass(name = "_ForwardReverse", unsendable)]
struct PyForwardReverse { frame: Option<ferro_core::forward_ad::ReverseFrameGuard> }

#[pymethods]
impl PyForwardReverse {
    #[new]
    fn new() -> PyResult<Self> { Ok(Self { frame: Some(ferro_core::forward_ad::ReverseFrameGuard::new().map_err(map_err)?) }) }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __exit__(&mut self, _kind: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) -> bool {
        self.frame.take(); false
    }
}

#[pymethods]
impl PyForwardAD {
    #[new]
    fn new(primals: Vec<PyRef<'_, PyTensor>>, tangents: Vec<PyRef<'_, PyTensor>>) -> PyResult<Self> {
        let primals = primals.iter().map(|value| &value.inner).collect::<Vec<_>>();
        let tangents = tangents.iter().map(|value| &value.inner).collect::<Vec<_>>();
        Ok(Self { level: Some(ferro_core::forward_ad::DualLevel::functional(&primals, &tangents).map_err(map_err)?) })
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __exit__(&mut self, _kind: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) -> bool {
        self.level.take();
        false
    }
    fn tangents(&self, outputs: Vec<PyRef<'_, PyTensor>>) -> PyResult<Vec<Option<PyTensor>>> {
        let level = self.level.as_ref().ok_or_else(|| PyValueError::new_err("forward AD level is closed"))?;
        outputs.iter().map(|value| level.tangent(&value.inner).map(|tangent| tangent.map(PyTensor::wrap)).map_err(map_err)).collect()
    }
    fn primals(&self, outputs: Vec<PyRef<'_, PyTensor>>, auxiliary: bool) -> PyResult<Vec<PyTensor>> {
        let level = self.level.as_ref().ok_or_else(|| PyValueError::new_err("forward AD level is closed"))?;
        outputs.iter().map(|value| level.primal(&value.inner, auxiliary).map(PyTensor::wrap).map_err(map_err)).collect()
    }
}

#[pymethods]
impl PyForwardColumns {
    #[new]
    #[pyo3(signature = (primals, columns, randomness_same=false))]
    fn new(primals: Vec<PyRef<'_, PyTensor>>, columns: Vec<Vec<PyRef<'_, PyTensor>>>, randomness_same: bool) -> PyResult<Self> {
        let primals = primals.iter().map(|value| &value.inner).collect::<Vec<_>>();
        let columns = columns.iter().map(|column| column.iter().map(|value| &value.inner).collect::<Vec<_>>()).collect::<Vec<_>>();
        let refs = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let level = ferro_core::forward_ad::DualLevel::functional_columns(&primals, &refs).map_err(map_err)?;
        level.set_randomness_same(randomness_same).map_err(map_err)?;
        Ok(Self { level: Some(level) })
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __exit__(&mut self, _kind: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) -> bool {
        self.level.take(); false
    }
    fn tangents(&self, outputs: Vec<PyRef<'_, PyTensor>>) -> PyResult<Vec<Option<Vec<PyTensor>>>> {
        let level = self.level.as_ref().ok_or_else(|| PyValueError::new_err("forward AD level is closed"))?;
        outputs.iter().map(|value| level.tangents(&value.inner).map(|columns| columns.map(|columns| columns.into_iter().map(PyTensor::wrap).collect())).map_err(map_err)).collect()
    }
    fn primals(&self, outputs: Vec<PyRef<'_, PyTensor>>, auxiliary: bool) -> PyResult<Vec<PyTensor>> {
        let level = self.level.as_ref().ok_or_else(|| PyValueError::new_err("forward AD level is closed"))?;
        outputs.iter().map(|value| level.primal(&value.inner, auxiliary).map(PyTensor::wrap).map_err(map_err)).collect()
    }
}

#[pyfunction]
fn _forward_ad_check_random() -> PyResult<()> { ferro_core::forward_ad::check_random_operation().map_err(map_err) }

#[pyfunction]
fn _forward_ad_tangents(source: PyRef<'_, PyTensor>) -> PyResult<Option<Vec<PyTensor>>> {
    ferro_core::forward_ad::current_tangents(&source.inner).map(|values| values.map(|values| values.into_iter().map(PyTensor::wrap).collect())).map_err(map_err)
}

#[pyfunction]
fn _forward_ad_post_columns(source: PyRef<'_, PyTensor>, target: PyRef<'_, PyTensor>, tangents: Option<Vec<PyRef<'_, PyTensor>>>) -> PyResult<()> {
    let tangents = tangents.as_ref().map(|values| values.iter().map(|value| &value.inner).collect::<Vec<_>>());
    ferro_core::forward_ad::post_alias_columns(&source.inner, &target.inner, tangents.as_deref()).map_err(map_err)
}

#[pyfunction]
fn _forward_ad_is_active() -> bool { ferro_core::forward_ad::is_active() }

#[pyfunction]
fn _forward_ad_prepare(source: PyRef<'_, PyTensor>, target: PyRef<'_, PyTensor>) -> PyResult<()> {
    ferro_core::forward_ad::prepare_alias(&source.inner, &target.inner).map_err(map_err)
}

#[pyfunction]
fn _forward_ad_tangent(source: PyRef<'_, PyTensor>) -> PyResult<Option<PyTensor>> {
    ferro_core::forward_ad::current_tangent(&source.inner).map(|value| value.map(PyTensor::wrap)).map_err(map_err)
}

#[pyfunction]
fn _forward_ad_unwrap(values: Vec<PyRef<'_, PyTensor>>) -> PyResult<Vec<PyTensor>> {
    let values = values.iter().map(|value| &value.inner).collect::<Vec<_>>();
    ferro_core::forward_ad::unwrap_values(&values).map(|values| values.into_iter().map(PyTensor::wrap).collect()).map_err(map_err)
}

#[pyfunction]
fn _forward_ad_freeze(source: PyRef<'_, PyTensor>, cutoff: usize, excluded: Vec<PyRef<'_, PyTensor>>) -> PyResult<PyTensor> {
    let excluded = excluded.iter().map(|value| &value.inner).collect::<Vec<_>>();
    ferro_core::forward_ad::freeze_post(&source.inner, cutoff, &excluded).map(PyTensor::wrap).map_err(map_err)
}

#[pyfunction]
fn _forward_ad_post(source: PyRef<'_, PyTensor>, target: PyRef<'_, PyTensor>, tangent: Option<PyRef<'_, PyTensor>>) -> PyResult<()> {
    ferro_core::forward_ad::post_alias(&source.inner, &target.inner, tangent.as_ref().map(|value| &value.inner)).map_err(map_err)
}

#[pyfunction]
fn _forward_ad_vjp(output: PyRef<'_, PyTensor>, inputs: Vec<PyRef<'_, PyTensor>>, seed: PyRef<'_, PyTensor>) -> PyResult<Vec<Option<PyTensor>>> {
    let inputs = inputs.iter().map(|value| &value.inner).collect::<Vec<_>>();
    output.inner.vjp_wrt_forward(&inputs, &seed.inner).map(|values| values.into_iter().map(|value| value.map(PyTensor::wrap)).collect()).map_err(map_err)
}

#[pyfunction]
fn _forward_ad_vjp_many(outputs: Vec<PyRef<'_, PyTensor>>, inputs: Vec<PyRef<'_, PyTensor>>, seeds: Vec<PyRef<'_, PyTensor>>) -> PyResult<Vec<Option<PyTensor>>> {
    let outputs = outputs.iter().map(|value| &value.inner).collect::<Vec<_>>();
    let inputs = inputs.iter().map(|value| &value.inner).collect::<Vec<_>>();
    let seeds = seeds.iter().map(|value| &value.inner).collect::<Vec<_>>();
    ferro_core::Tensor::vjp_many_wrt_forward(&outputs, &inputs, &seeds)
        .map(|values| values.into_iter().map(|value| value.map(PyTensor::wrap)).collect()).map_err(map_err)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForwardAD>()?;
    m.add_class::<PyForwardColumns>()?;
    m.add_class::<PyForwardBatched>()?;
    m.add_function(wrap_pyfunction!(_forward_ad_check_random, m)?)?;
    m.add_function(wrap_pyfunction!(_forward_ad_tangents, m)?)?;
    m.add_function(wrap_pyfunction!(_forward_ad_post_columns, m)?)?;
    m.add_class::<PyForwardReverse>()?;
    m.add_function(wrap_pyfunction!(_forward_ad_is_active, m)?)?;
    m.add_function(wrap_pyfunction!(_forward_ad_prepare, m)?)?;
    m.add_function(wrap_pyfunction!(_forward_ad_tangent, m)?)?;
    m.add_function(wrap_pyfunction!(_forward_ad_unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(_forward_ad_freeze, m)?)?;
    m.add_function(wrap_pyfunction!(_forward_ad_post, m)?)?;
    m.add_function(wrap_pyfunction!(_forward_ad_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(_forward_ad_vjp_many, m)?)
}
