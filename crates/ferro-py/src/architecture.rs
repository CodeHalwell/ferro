use super::*;

/// Immutable integer topology for reusable first-order sum/softmax.
/// Device backends retain their original topology allocations and stream owner.
#[pyclass(name = "PreparedSegments", module = "ferro.graph")]
struct PyPreparedSegments { inner: ferro_core::segment::PreparedSegments }

#[pymethods]
impl PyPreparedSegments {
    #[new]
    #[pyo3(signature = (ids, num_segments, device="cpu"))]
    fn new(ids: Vec<i64>, num_segments: usize, device: &str) -> PyResult<Self> {
        let ids = ids.into_iter().map(|id| usize::try_from(id)
            .map_err(|_| PyValueError::new_err("segment ids must be nonnegative i64 integers")))
            .collect::<PyResult<Vec<_>>>()?;
        ferro_core::segment::PreparedSegments::new(&ids, num_segments, parse_device(device)?)
            .map(|inner| Self { inner }).map_err(map_err)
    }

    fn sum(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner.sum(&x.inner).map(PyTensor::wrap).map_err(map_err)
    }

    fn gather(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner.gather(&x.inner).map(PyTensor::wrap).map_err(map_err)
    }
    #[pyo3(signature = (x, dim=0))]
    fn select(&self, x: &PyTensor, dim: usize) -> PyResult<PyTensor> {
        self.inner.select(&x.inner, dim).map(PyTensor::wrap).map_err(map_err)
    }
    #[pyo3(signature = (base, src, dim=0))]
    fn scatter_add(&self, base: &PyTensor, src: &PyTensor, dim: usize) -> PyResult<PyTensor> {
        self.inner.scatter_add(&base.inner, &src.inner, dim).map(PyTensor::wrap).map_err(map_err)
    }
    fn softmax(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.inner.softmax(&x.inner).map(PyTensor::wrap).map_err(map_err)
    }
}

#[pyfunction]
fn segment_sum(x: &PyTensor, ids: Vec<usize>, num_segments: usize) -> PyResult<PyTensor> {
    ferro_core::segment::sum(&x.inner, &ids, num_segments).map(PyTensor::wrap).map_err(map_err)
}

#[pyfunction]
fn segment_mean(x: &PyTensor, ids: Vec<usize>, num_segments: usize) -> PyResult<PyTensor> {
    ferro_core::segment::mean(&x.inner, &ids, num_segments).map(PyTensor::wrap).map_err(map_err)
}

#[pyfunction]
fn segment_max(x: &PyTensor, ids: Vec<usize>, num_segments: usize) -> PyResult<PyTensor> {
    ferro_core::segment::max(&x.inner, &ids, num_segments).map(PyTensor::wrap).map_err(map_err)
}

#[pyfunction]
fn segment_softmax(x: &PyTensor, ids: Vec<usize>, num_segments: usize) -> PyResult<PyTensor> {
    ferro_core::segment::softmax(&x.inner, &ids, num_segments).map(PyTensor::wrap).map_err(map_err)
}

#[pyclass(name = "COO", module = "ferro.graph")]
struct PyCOO { inner: ferro_core::sparse::Coo }
#[pymethods]
impl PyCOO {
    #[new]
    fn new(nrows: usize, ncols: usize, rows: Vec<usize>, cols: Vec<usize>) -> PyResult<Self> {
        ferro_core::sparse::Coo::new(nrows,ncols,rows,cols).map(|inner| Self { inner }).map_err(map_err)
    }
    #[getter]
    fn shape(&self) -> Vec<usize> { self.inner.shape().to_vec() }
    #[getter]
    fn nnz(&self) -> usize { self.inner.nnz() }
    #[getter]
    fn rows(&self) -> Vec<usize> { self.inner.rows().to_vec() }
    #[getter]
    fn cols(&self) -> Vec<usize> { self.inner.cols().to_vec() }
    fn spmm(&self, values: &PyTensor, dense: &PyTensor) -> PyResult<PyTensor> { self.inner.spmm(&values.inner,&dense.inner).map(PyTensor::wrap).map_err(map_err) }
    fn sddmm(&self, left: &PyTensor, right: &PyTensor) -> PyResult<PyTensor> { self.inner.sddmm(&left.inner,&right.inner).map(PyTensor::wrap).map_err(map_err) }
    #[pyo3(signature = (device="cpu"))]
    fn prepare(&self, device: &str) -> PyResult<PyPreparedCOO> {
        self.inner.prepare(parse_device(device)?).map(|inner| PyPreparedCOO { inner }).map_err(map_err)
    }
    #[staticmethod]
    fn batch(graphs: Vec<PyRef<'_, PyCOO>>) -> PyResult<(Self, Vec<[usize; 2]>)> {
        let graphs: Vec<_> = graphs.iter().map(|g| g.inner.clone()).collect();
        ferro_core::sparse::Coo::disjoint_union(&graphs).map(|(inner, offsets)| (Self { inner }, offsets)).map_err(map_err)
    }
    fn to_csr(&self) -> PyResult<(PyCSR, Vec<usize>)> { self.inner.to_csr().map(|(inner,order)| (PyCSR { inner },order)).map_err(map_err) }
    fn coalesce(&self, values: &PyTensor) -> PyResult<(Self, PyTensor)> { self.inner.coalesce(&values.inner).map(|(inner,v)| (Self { inner },PyTensor::wrap(v))).map_err(map_err) }
}

#[pyclass(name = "CSR", module = "ferro.graph")]
struct PyCSR { inner: ferro_core::sparse::Csr }
#[pymethods]
impl PyCSR {
    #[new]
    fn new(nrows: usize, ncols: usize, row_offsets: Vec<usize>, col_indices: Vec<usize>) -> PyResult<Self> {
        ferro_core::sparse::Csr::new(nrows,ncols,row_offsets,col_indices).map(|inner| Self { inner }).map_err(map_err)
    }
    #[getter]
    fn shape(&self) -> Vec<usize> { self.inner.shape().to_vec() }
    #[getter]
    fn nnz(&self) -> usize { self.inner.nnz() }
    #[getter]
    fn row_offsets(&self) -> Vec<usize> { self.inner.row_offsets().to_vec() }
    #[getter]
    fn col_indices(&self) -> Vec<usize> { self.inner.col_indices().to_vec() }
    fn spmm(&self, values: &PyTensor, dense: &PyTensor) -> PyResult<PyTensor> { self.inner.spmm(&values.inner,&dense.inner).map(PyTensor::wrap).map_err(map_err) }
    fn sddmm(&self, left: &PyTensor, right: &PyTensor) -> PyResult<PyTensor> { self.inner.to_coo().sddmm(&left.inner,&right.inner).map(PyTensor::wrap).map_err(map_err) }
    fn to_coo(&self) -> PyCOO { PyCOO { inner:self.inner.to_coo() } }
}

#[pyclass(name = "PreparedCOO", module = "ferro.graph")]
struct PyPreparedCOO { inner: ferro_core::sparse::PreparedCoo }
#[pymethods]
impl PyPreparedCOO {
    fn spmm(&self, values: &PyTensor, dense: &PyTensor) -> PyResult<PyTensor> {
        self.inner.spmm(&values.inner, &dense.inner).map(PyTensor::wrap).map_err(map_err)
    }
    fn sddmm(&self, left: &PyTensor, right: &PyTensor) -> PyResult<PyTensor> {
        self.inner.sddmm(&left.inner, &right.inner).map(PyTensor::wrap).map_err(map_err)
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(segment_sum, m)?)?;
    m.add_function(wrap_pyfunction!(segment_mean, m)?)?;
    m.add_function(wrap_pyfunction!(segment_max, m)?)?;
    m.add_function(wrap_pyfunction!(segment_softmax, m)?)?;
    m.add_class::<PyPreparedSegments>()?;
    m.add_class::<PyPreparedCOO>()?;
    m.add_class::<PyCOO>()?;
    m.add_class::<PyCSR>()?;
    Ok(())
}
