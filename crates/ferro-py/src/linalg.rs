use super::*;

#[pyfunction]
#[pyo3(signature = (a, b, left=true))]
fn _linalg_solve(a: &PyTensor, b: &PyTensor, left: bool) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::solve(&a.inner, &b.inner, left))
}

#[pyfunction]
fn _linalg_inv(a: &PyTensor) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::inv(&a.inner))
}

#[pyfunction]
#[pyo3(signature = (a, b, left=true, check_errors=false))]
fn _linalg_solve_ex(a: &PyTensor, b: &PyTensor, left: bool, check_errors: bool) -> PyResult<(PyTensor, PyTensor)> {
    let (result, info) = ferro_core::linalg::solve_ex(&a.inner, &b.inner, left, check_errors).map_err(map_err)?;
    Ok((PyTensor::wrap(result), PyTensor::wrap(info)))
}

#[pyfunction]
#[pyo3(signature = (a, check_errors=false))]
fn _linalg_inv_ex(a: &PyTensor, check_errors: bool) -> PyResult<(PyTensor, PyTensor)> {
    let (result, info) = ferro_core::linalg::inv_ex(&a.inner, check_errors).map_err(map_err)?;
    Ok((PyTensor::wrap(result), PyTensor::wrap(info)))
}

#[pyfunction]
#[pyo3(signature = (a, b, upper, left=true, unitriangular=false))]
fn _linalg_solve_triangular(a: &PyTensor, b: &PyTensor, upper: bool, left: bool, unitriangular: bool) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::solve_triangular(&a.inner, &b.inner, upper, left, unitriangular))
}

#[pyfunction]
#[pyo3(signature = (a, upper=false, check_errors=false))]
fn _linalg_cholesky_ex(a: &PyTensor, upper: bool, check_errors: bool) -> PyResult<(PyTensor, PyTensor)> {
    let (factor, info) = ferro_core::linalg::cholesky_ex(&a.inner, upper, check_errors).map_err(map_err)?;
    Ok((PyTensor::wrap(factor), PyTensor::wrap(info)))
}

#[pyfunction]
#[pyo3(signature = (b, a, upper=false))]
fn _linalg_cholesky_solve(b: &PyTensor, a: &PyTensor, upper: bool) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::cholesky_solve(&b.inner, &a.inner, upper))
}

#[pyfunction]
#[pyo3(signature = (a, upper=false))]
fn _linalg_cholesky_inverse(a: &PyTensor, upper: bool) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::cholesky_inverse(&a.inner, upper))
}

#[pyfunction]
fn _linalg_det(a: &PyTensor) -> PyResult<PyTensor> { ok(ferro_core::linalg::det(&a.inner)) }

#[pyfunction]
fn _linalg_slogdet(a: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
    let (sign, logarithm) = ferro_core::linalg::slogdet(&a.inner).map_err(map_err)?;
    Ok((PyTensor::wrap(sign), PyTensor::wrap(logarithm)))
}

#[pyfunction]
#[pyo3(signature = (a, order=2.0, dims=None, keepdim=false))]
fn _linalg_vector_norm(a: &PyTensor, order: f64, dims: Option<Vec<isize>>, keepdim: bool) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::vector_norm(&a.inner, order, dims.as_deref(), keepdim))
}

#[pyfunction]
#[pyo3(signature = (a, offset=0, dim1=-2, dim2=-1))]
fn _linalg_diagonal(a: &PyTensor, offset: isize, dim1: isize, dim2: isize) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::diagonal(&a.inner, offset, dim1, dim2))
}

#[pyfunction]
fn _linalg_qr(a: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
    let (q, r) = ferro_core::linalg::qr(&a.inner).map_err(map_err)?;
    Ok((PyTensor::wrap(q), PyTensor::wrap(r)))
}

#[pyfunction]
#[pyo3(signature = (a, upper=false))]
fn _linalg_eigh(a: &PyTensor, upper: bool) -> PyResult<(PyTensor, PyTensor)> {
    let (values, vectors) = ferro_core::linalg::eigh(&a.inner, upper).map_err(map_err)?;
    Ok((PyTensor::wrap(values), PyTensor::wrap(vectors)))
}

#[pyfunction]
#[pyo3(signature = (a, full_matrices=true))]
fn _linalg_svd(a: &PyTensor, full_matrices: bool) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
    let (u, s, vh) = ferro_core::linalg::svd(&a.inner, full_matrices).map_err(map_err)?;
    Ok((PyTensor::wrap(u), PyTensor::wrap(s), PyTensor::wrap(vh)))
}

#[pyfunction]
#[pyo3(signature = (a, atol=None, rtol=None, hermitian=false))]
fn _linalg_pinv(a: &PyTensor, atol: Option<&PyTensor>, rtol: Option<&PyTensor>, hermitian: bool) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::pinv(&a.inner, atol.map(|x| &x.inner), rtol.map(|x| &x.inner), hermitian))
}

#[pyfunction]
#[pyo3(signature = (a, atol=None, rtol=None, hermitian=false))]
fn _linalg_matrix_rank(a: &PyTensor, atol: Option<&PyTensor>, rtol: Option<&PyTensor>, hermitian: bool) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::matrix_rank(&a.inner, atol.map(|x| &x.inner), rtol.map(|x| &x.inner), hermitian))
}

#[pyfunction]
#[pyo3(signature = (a, b, rcond=None))]
fn _linalg_lstsq(a: &PyTensor, b: &PyTensor, rcond: Option<f64>) -> PyResult<(PyTensor, PyTensor, PyTensor, PyTensor)> {
    let (solution, residuals, rank, singular_values) = ferro_core::linalg::lstsq(&a.inner, &b.inner, rcond).map_err(map_err)?;
    Ok((PyTensor::wrap(solution), PyTensor::wrap(residuals), PyTensor::wrap(rank), PyTensor::wrap(singular_values)))
}

#[pyfunction]
fn _linalg_lu_factor(a: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
    let (packed, pivots) = ferro_core::linalg::lu_factor(&a.inner).map_err(map_err)?;
    Ok((PyTensor::wrap(packed), PyTensor::wrap(pivots)))
}

#[pyfunction]
fn _linalg_lu_factor_ex(a: &PyTensor, check_errors: bool) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
    let (packed, pivots, info) = ferro_core::linalg::lu_factor_ex(&a.inner, check_errors).map_err(map_err)?;
    Ok((PyTensor::wrap(packed), PyTensor::wrap(pivots), PyTensor::wrap(info)))
}

#[pyfunction]
fn _linalg_lu(a: &PyTensor) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
    let (p, l, u) = ferro_core::linalg::lu(&a.inner).map_err(map_err)?;
    Ok((PyTensor::wrap(p), PyTensor::wrap(l), PyTensor::wrap(u)))
}

#[pyfunction]
fn _linalg_lu_solve(packed: &PyTensor, pivots: &PyTensor, b: &PyTensor, left: bool, adjoint: bool) -> PyResult<PyTensor> {
    ok(ferro_core::linalg::lu_solve(&packed.inner, &pivots.inner, &b.inner, left, adjoint))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_linalg_solve, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_inv, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_solve_ex, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_inv_ex, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_solve_triangular, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_cholesky_ex, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_cholesky_solve, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_cholesky_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_det, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_slogdet, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_vector_norm, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_qr, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_eigh, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_svd, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_pinv, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_matrix_rank, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_lstsq, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_lu_factor, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_lu_factor_ex, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_lu, m)?)?;
    m.add_function(wrap_pyfunction!(_linalg_lu_solve, m)?)?;
    Ok(())
}
