//! Bounded public CPU classification loss over core's existing implementation.
use super::*;

#[pyfunction]
fn cross_entropy(logits: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
    use ferro_core::DType;
    let x = &logits.inner;
    let y = &targets.inner;
    if x.device() != Device::Cpu || y.device() != Device::Cpu {
        return Err(PyValueError::new_err("cross_entropy supports CPU tensors only"));
    }
    if x.dtype() != DType::F32 || y.dtype() != DType::I64 {
        return Err(PyValueError::new_err("cross_entropy requires f32 logits and i64 targets"));
    }
    if x.shape().len() != 2 || x.shape()[0] == 0 || x.shape()[1] == 0 || y.shape() != [x.shape()[0]] {
        return Err(PyValueError::new_err("cross_entropy requires nonempty [batch, classes] logits and [batch] targets"));
    }
    ferro_core::nn::cross_entropy_indices(x, y).map(PyTensor::wrap).map_err(map_err)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    Ok(())
}
