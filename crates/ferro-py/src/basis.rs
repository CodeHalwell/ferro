use super::*;
use ferro_core::basis::{self, BSplineBasis, OutsideDomain};

#[pyclass(name = "BSplineBasis", module = "ferro.basis", frozen)]
struct PyBasis { inner: BSplineBasis }
#[pymethods]
impl PyBasis {
    #[new]
    #[pyo3(signature = (knots, degree, outside="zero"))]
    fn new(knots: Vec<f32>, degree: usize, outside: &str) -> PyResult<Self> {
        let outside=match outside { "zero"=>OutsideDomain::Zero, "error"=>OutsideDomain::Error, _=>return Err(PyValueError::new_err("outside must be zero or error")) };
        BSplineBasis::new(knots,degree,outside).map(|inner| Self { inner }).map_err(map_err)
    }
    #[getter]
    fn knots(&self) -> Vec<f32> { self.inner.knots().to_vec() }
    #[getter]
    fn degree(&self) -> usize { self.inner.degree() }
    #[getter]
    fn outside(&self) -> &'static str {
        match self.inner.outside() { OutsideDomain::Zero => "zero", OutsideDomain::Error => "error" }
    }
    #[getter]
    fn num_basis(&self) -> usize { self.inner.num_basis() }
    #[getter]
    fn domain(&self) -> (f32,f32) { self.inner.domain() }
    fn evaluate(&self, x: &PyTensor) -> PyResult<PyTensor> { self.inner.evaluate(&x.inner).map(PyTensor::wrap).map_err(map_err) }
}
#[pyfunction]
fn kan(x: &PyTensor, coefficients: &PyTensor, basis: &PyBasis) -> PyResult<PyTensor> {
    basis::kan(&x.inner,&coefficients.inner,&basis.inner).map(PyTensor::wrap).map_err(map_err)
}
#[pyfunction]
fn _kan_parameter(py: Python<'_>, basis: &PyBasis, coefficients: &PyTensor) -> PyResult<Py<PyParameter>> {
    basis::KanLayer::from_coefficients(basis.inner.clone(),coefficients.inner.clone())
        .map_err(map_err).and_then(|layer| PyParameter::wrap(py, layer.coefficients()))
}
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBasis>()?;
    m.add_function(wrap_pyfunction!(kan,m)?)?;
    m.add_function(wrap_pyfunction!(_kan_parameter,m)?)?;
    Ok(())
}
