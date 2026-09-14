use super::*;

fn cpu(tensors: &[&PyTensor]) -> PyResult<()> {
    if tensors.iter().any(|t| t.inner.device()!=Device::Cpu) {
        return Err(PyValueError::new_err("convolution foundations are CPU only"));
    }
    Ok(())
}
#[pyfunction]
#[pyo3(signature = (x, weight, bias, stride, padding, dilation, groups))]
fn conv2d_options(x: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>, stride: [usize;2], padding: [usize;2], dilation: [usize;2], groups: usize) -> PyResult<PyTensor> {
    cpu(&[x,weight])?;
    if let Some(b)=bias {
        cpu(&[b])?;
        if weight.inner.ndim()!=4 || b.inner.shape()!=[weight.inner.shape()[0]] {
            return Err(PyValueError::new_err("bias must have shape [out_channels]"));
        }
    }
    let y=x.inner.conv2d_with_options(&weight.inner,stride,padding,dilation,groups).map_err(map_err)?;
    let y=if let Some(b)=bias { y.add(&b.inner.reshape(&[1,b.inner.shape()[0],1,1]).map_err(map_err)?).map_err(map_err)? } else { y };
    Ok(PyTensor::wrap(y))
}
#[pyfunction]
fn unfold2d(x: &PyTensor, kernel: [usize;2], stride: [usize;2], padding: [usize;2], dilation: [usize;2]) -> PyResult<PyTensor> {
    cpu(&[x])?;
    x.inner.unfold2d(kernel,stride,padding,dilation).map(PyTensor::wrap).map_err(map_err)
}
#[pyfunction]
fn fold2d(x: &PyTensor, output_size: [usize;2], kernel: [usize;2], stride: [usize;2], padding: [usize;2], dilation: [usize;2]) -> PyResult<PyTensor> {
    cpu(&[x])?;
    x.inner.fold2d(output_size,kernel,stride,padding,dilation).map(PyTensor::wrap).map_err(map_err)
}
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(conv2d_options,m)?)?;
    m.add_function(wrap_pyfunction!(unfold2d,m)?)?;
    m.add_function(wrap_pyfunction!(fold2d,m)?)?;
    Ok(())
}
