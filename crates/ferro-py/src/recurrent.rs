use super::*;
use ferro_core::recurrent::{self, RecurrentState};

#[pyclass(name = "RecurrentState", module = "ferro.recurrent")]
struct PyState { inner: RecurrentState }
#[pymethods]
impl PyState {
    #[new]
    #[pyo3(signature = (h, c=None))]
    fn new(h: &PyTensor, c: Option<&PyTensor>) -> Self { Self { inner: RecurrentState { h:h.inner.clone(), c:c.map(|t| t.inner.clone()) } } }
    #[getter]
    fn h(&self) -> PyTensor { PyTensor::wrap(self.inner.h.clone()) }
    #[getter]
    fn c(&self) -> Option<PyTensor> { self.inner.c.clone().map(PyTensor::wrap) }
    fn detach(&self) -> Self { Self { inner:self.inner.detach() } }
}
#[pyclass(name = "UnrollOutput", module = "ferro.recurrent")]
struct PyUnroll { inner: recurrent::UnrollOutput }
#[pymethods]
impl PyUnroll {
    #[getter]
    fn outputs(&self) -> PyTensor { PyTensor::wrap(self.inner.outputs.clone()) }
    #[getter]
    fn state(&self) -> PyState { PyState { inner:self.inner.state.clone() } }
}
#[pyfunction]
#[pyo3(signature = (x, h, weight_ih, weight_hh, bias_ih=None, bias_hh=None))]
fn rnn_cell(x: &PyTensor, h: &PyTensor, weight_ih: &PyTensor, weight_hh: &PyTensor, bias_ih: Option<&PyTensor>, bias_hh: Option<&PyTensor>) -> PyResult<PyTensor> {
    recurrent::rnn_cell(&x.inner,&h.inner,&weight_ih.inner,&weight_hh.inner,bias_ih.map(|t| &t.inner),bias_hh.map(|t| &t.inner)).map(PyTensor::wrap).map_err(map_err)
}
#[pyfunction]
#[pyo3(signature = (x, h, weight_ih, weight_hh, bias_ih=None, bias_hh=None))]
fn gru_cell(x: &PyTensor, h: &PyTensor, weight_ih: &PyTensor, weight_hh: &PyTensor, bias_ih: Option<&PyTensor>, bias_hh: Option<&PyTensor>) -> PyResult<PyTensor> {
    recurrent::gru_cell(&x.inner,&h.inner,&weight_ih.inner,&weight_hh.inner,bias_ih.map(|t| &t.inner),bias_hh.map(|t| &t.inner)).map(PyTensor::wrap).map_err(map_err)
}
#[pyfunction]
#[pyo3(signature = (x, h, c, weight_ih, weight_hh, bias_ih=None, bias_hh=None))]
fn lstm_cell(x: &PyTensor, h: &PyTensor, c: &PyTensor, weight_ih: &PyTensor, weight_hh: &PyTensor, bias_ih: Option<&PyTensor>, bias_hh: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
    recurrent::lstm_cell(&x.inner,&h.inner, &c.inner,&weight_ih.inner,&weight_hh.inner,bias_ih.map(|t| &t.inner),bias_hh.map(|t| &t.inner)).map(|(h,c)| (PyTensor::wrap(h),PyTensor::wrap(c))).map_err(map_err)
}
#[pyfunction]
#[pyo3(signature = (kind, input, initial, weight_ih, weight_hh, bias_ih=None, bias_hh=None, *, lengths, reset=None, truncate=None))]
fn unroll(kind: &str, input: &PyTensor, initial: &PyState, weight_ih: &PyTensor, weight_hh: &PyTensor, bias_ih: Option<&PyTensor>, bias_hh: Option<&PyTensor>, lengths: Vec<usize>, reset: Option<Vec<bool>>, truncate: Option<usize>) -> PyResult<PyUnroll> {
    if !matches!(kind,"rnn"|"gru"|"lstm") || (kind=="lstm") != initial.inner.c.is_some() {
        return Err(PyValueError::new_err("kind must be rnn/gru with hidden-only state, or lstm with hidden and cell state"));
    }
    let wi=&weight_ih.inner; let wh=&weight_hh.inner;
    let bi=bias_ih.map(|t| &t.inner); let bh=bias_hh.map(|t| &t.inner);
    recurrent::unroll(&input.inner,&initial.inner,&lengths,reset.as_deref(),truncate,|x,s| {
        if kind=="lstm" {
            let (h,c)=recurrent::lstm_cell(x,&s.h,s.c.as_ref().unwrap(),wi,wh,bi,bh)?;
            Ok(RecurrentState { h,c:Some(c) })
        } else {
            let h=if kind=="rnn" { recurrent::rnn_cell(x,&s.h,wi,wh,bi,bh)? } else { recurrent::gru_cell(x,&s.h,wi,wh,bi,bh)? };
            Ok(RecurrentState { h,c:None })
        }
    }).map(|inner| PyUnroll { inner }).map_err(map_err)
}
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyState>()?;
    m.add_class::<PyUnroll>()?;
    m.add_function(wrap_pyfunction!(rnn_cell,m)?)?;
    m.add_function(wrap_pyfunction!(gru_cell,m)?)?;
    m.add_function(wrap_pyfunction!(lstm_cell,m)?)?;
    m.add_function(wrap_pyfunction!(unroll,m)?)?;
    Ok(())
}
