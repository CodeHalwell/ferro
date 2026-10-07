use super::*;
use ferro_core::data::{DataLoader, Dataset, TensorDataset};
use std::sync::Arc;

#[pyclass(name = "TensorDataset", module = "ferro.data")]
struct PyTensorDataset { inner: Arc<TensorDataset> }

#[pymethods]
impl PyTensorDataset {
    #[new]
    fn new(inputs: &PyTensor, targets: &PyTensor) -> PyResult<Self> {
        TensorDataset::new(inputs.inner.clone(), targets.inner.clone()).map(|d| Self { inner: Arc::new(d) }).map_err(map_err)
    }
    fn __len__(&self) -> usize { self.inner.len() }
    fn __getitem__(&self, index: isize) -> PyResult<(PyTensor, PyTensor)> {
        let (x, y) = self.inner.get(wrap_index(index, self.inner.len())?).map_err(map_err)?;
        Ok((PyTensor::wrap(x), PyTensor::wrap(y)))
    }
}

/// One epoch of stacked (inputs, targets) batches from the native loader.
/// The epoch is collated with the GIL released (worker threads when
/// num_workers > 0) and then yielded in sampler order.
#[pyfunction]
fn _load_epoch(py: Python<'_>, dataset: &PyTensorDataset, batch_size: usize, shuffle_seed: Option<u64>, drop_last: bool, num_workers: usize) -> PyResult<Vec<(PyTensor, PyTensor)>> {
    if batch_size == 0 {
        return Err(PyValueError::new_err("batch_size must be positive"));
    }
    let mut loader = DataLoader::new(dataset.inner.clone(), batch_size).drop_last(drop_last).workers(num_workers);
    if let Some(seed) = shuffle_seed {
        loader = loader.shuffle(seed);
    }
    let batches = py.allow_threads(|| loader.iter().collect::<ferro_core::Result<Vec<_>>>()).map_err(map_err)?;
    Ok(batches.into_iter().map(|(x, y)| (PyTensor::wrap(x), PyTensor::wrap(y))).collect())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensorDataset>()?;
    m.add_function(wrap_pyfunction!(_load_epoch, m)?)?;
    Ok(())
}
