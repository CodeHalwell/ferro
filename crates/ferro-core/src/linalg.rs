//! Eager CPU real linear algebra with first-order analytic adjoints.
//!
//! Inputs and gradients retain f32/f64 dtype. Both use f64 factorization work;
//! f32 results are rounded at the tensor boundary. No external BLAS/LAPACK or
//! tensor runtime is used. Inference capture and higher-order AD are unsupported.
use crate::{DType, Error, Result, Tensor};
use crate::shape::{broadcast_shapes, checked_numel};
use crate::typed_math::{check, check_shape, filled};

fn invalid(op: &'static str, message: impl Into<String>) -> Error {
    Error::InvalidShape { op, msg: message.into() }
}

fn capacity(op: &'static str, shape: &[usize]) -> Result<usize> {
    let count = check_shape(op, shape)?;
    let layout = shape.iter().try_fold(1usize, |size, &dim| size.checked_mul(dim.max(1)))
        .ok_or_else(|| invalid(op, "layout capacity overflow"))?;
    if layout > isize::MAX as usize { return Err(invalid(op, "layout exceeds addressable capacity")); }
    Ok(count)
}

fn real(op: &'static str, inputs: &[&Tensor]) -> Result<DType> {
    let dtype = inputs[0].dtype();
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::DtypeMismatch { op, expected: DType::F32, got: dtype });
    }
    check(op, inputs, dtype)?;
    for input in inputs { capacity(op, input.shape())?; }
    Ok(dtype)
}

fn values(input: &Tensor) -> Vec<f64> {
    if input.dtype() == DType::F64 { input.to_vec_f64() }
    else { input.to_vec().into_iter().map(f64::from).collect() }
}

fn tensor(data: Vec<f64>, shape: &[usize], dtype: DType) -> Result<Tensor> {
    if dtype == DType::F64 { Tensor::from_vec_f64(data, shape) }
    else { Tensor::from_vec(data.into_iter().map(|x| x as f32).collect(), shape) }
}

fn square(op: &'static str, input: &Tensor) -> Result<(usize, Vec<usize>)> {
    let shape = input.shape();
    if shape.len() < 2 || shape[shape.len() - 2] != shape[shape.len() - 1] {
        return Err(invalid(op, "A must have shape [..., n, n]"));
    }
    Ok((shape[shape.len() - 1], shape[..shape.len() - 2].to_vec()))
}

fn batch_index(mut flat: usize, output: &[usize], source: &[usize]) -> usize {
    let mut index = 0;
    let mut stride = 1;
    for axis in (0..output.len()).rev() {
        let coordinate = flat % output[axis];
        flat /= output[axis];
        if axis + source.len() >= output.len() {
            let size = source[axis + source.len() - output.len()];
            if size != 1 { index += coordinate * stride; }
            stride *= size;
        }
    }
    index
}

fn transpose(data: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    let mut output = vec![0.; data.len()];
    for row in 0..rows { for column in 0..columns {
        output[column * rows + row] = data[row * columns + column];
    } }
    output
}

#[derive(Clone)]
struct Lu {
    data: Vec<f64>,
    swaps: Vec<usize>,
    n: usize,
}

fn lu_factor_data(mut data: Vec<f64>, rows: usize, columns: usize, op: &'static str) -> Result<(Vec<f64>, Vec<usize>, usize)> {
    if data.iter().any(|x| !x.is_finite()) {
        return Err(invalid(op, "coefficient matrices must contain finite values"));
    }
    let mut swaps = Vec::with_capacity(rows.min(columns));
    let mut info = 0;
    for column in 0..rows.min(columns) {
        let mut pivot = column;
        for row in column + 1..rows {
            if data[row * columns + column].abs() > data[pivot * columns + column].abs() { pivot = row; }
        }
        if data[pivot * columns + column] == 0. {
            if info == 0 { info = column + 1; }
            swaps.push(pivot);
            continue;
        }
        swaps.push(pivot);
        if pivot != column {
            for j in 0..columns { data.swap(column * columns + j, pivot * columns + j); }
        }
        for row in column + 1..rows {
            data[row * columns + column] /= data[column * columns + column];
            for j in column + 1..columns {
                data[row * columns + j] -= data[row * columns + column] * data[column * columns + j];
            }
        }
    }
    Ok((data, swaps, info))
}

impl Lu {
    fn factor_info(data: Vec<f64>, n: usize, op: &'static str) -> Result<(Self, usize)> {
        let (data, swaps, info) = lu_factor_data(data, n, n, op)?;
        Ok((Self { data, swaps, n }, info))
    }

    fn factor(data: Vec<f64>, n: usize, op: &'static str) -> Result<Self> {
        let (factor, info) = Self::factor_info(data, n, op)?;
        if info != 0 { return Err(invalid(op, format!("singular coefficient matrix at pivot {info}"))); }
        Ok(factor)
    }

    fn determinant(&self) -> f64 {
        let mut value = 1.;
        for row in 0..self.n {
            value *= self.data[row * self.n + row];
            if self.swaps[row] != row { value = -value; }
        }
        value
    }

    fn signed_logdet(&self) -> (f64, f64) {
        let (mut sign, mut logarithm) = (1., 0.);
        for row in 0..self.n {
            let diagonal = self.data[row * self.n + row];
            sign *= diagonal.signum();
            if self.swaps[row] != row { sign = -sign; }
            logarithm += diagonal.abs().ln();
        }
        (sign, logarithm)
    }

    fn solve(&self, rhs: &[f64], columns: usize, transposed: bool) -> Vec<f64> {
        let n = self.n;
        let mut result = rhs.to_vec();
        if !transposed {
            for row in 0..n {
                if self.swaps[row] != row {
                    for column in 0..columns { result.swap(row * columns + column, self.swaps[row] * columns + column); }
                }
            }
            for row in 0..n { for column in 0..columns {
                for k in 0..row { result[row * columns + column] -= self.data[row * n + k] * result[k * columns + column]; }
            } }
            for row in (0..n).rev() { for column in 0..columns {
                for k in row + 1..n { result[row * columns + column] -= self.data[row * n + k] * result[k * columns + column]; }
                result[row * columns + column] /= self.data[row * n + row];
            } }
        } else {
            for row in 0..n { for column in 0..columns {
                for k in 0..row { result[row * columns + column] -= self.data[k * n + row] * result[k * columns + column]; }
                result[row * columns + column] /= self.data[row * n + row];
            } }
            for row in (0..n).rev() { for column in 0..columns {
                for k in row + 1..n { result[row * columns + column] -= self.data[k * n + row] * result[k * columns + column]; }
            } }
            for row in (0..n).rev() {
                if self.swaps[row] != row {
                    for column in 0..columns { result.swap(row * columns + column, self.swaps[row] * columns + column); }
                }
            }
        }
        result
    }
}

struct SolveShape {
    n: usize,
    columns: usize,
    a_batch: Vec<usize>,
    b_batch: Vec<usize>,
    batch: Vec<usize>,
    output: Vec<usize>,
}

fn solve_shape(a: &Tensor, b: &Tensor, left: bool, vectors: bool, op: &'static str) -> Result<SolveShape> {
    let (n, a_batch) = square(op, a)?;
    let vector = vectors && (b.ndim() == 1 || (b.ndim() + 1 == a.ndim() && b.shape() == &a.shape()[..a.ndim() - 1]));
    if vector && !left { return Err(invalid(op, "right solves require matrix right-hand sides")); }
    let (columns, b_batch) = if vector {
        if b.shape().last() != Some(&n) { return Err(invalid(op, "right-hand side length must match A")); }
        (1, b.shape()[..b.ndim() - 1].to_vec())
    } else {
        if b.ndim() < 2 { return Err(invalid(op, "B must have matrix dimensions")); }
        let (rows, columns) = (b.shape()[b.ndim() - 2], b.shape()[b.ndim() - 1]);
        if if left { rows != n } else { columns != n } {
            return Err(invalid(op, "A and B matrix dimensions are incompatible"));
        }
        (if left { columns } else { rows }, b.shape()[..b.ndim() - 2].to_vec())
    };
    let batch = broadcast_shapes(op, &a_batch, &b_batch)?;
    let mut output = batch.clone();
    if vector { output.push(n); }
    else if left { output.extend([n, columns]); }
    else { output.extend([columns, n]); }
    capacity(op, &output)?;
    capacity(op, &[n, n])?;
    capacity(op, &[n, columns])?;
    Ok(SolveShape { n, columns, a_batch, b_batch, batch, output })
}

#[derive(Clone, Copy)]
enum SolveOutput { Plain, WithInfo { check_errors: bool, inverse: bool } }

fn solve_impl(a: &Tensor, b: &Tensor, left: bool, triangular: Option<(bool, bool)>, vectors: bool, mode: SolveOutput) -> Result<(Tensor, Option<Tensor>)> {
    let op = match mode {
        SolveOutput::WithInfo { inverse: true, .. } => "linalg.inv_ex",
        SolveOutput::WithInfo { .. } => "linalg.solve_ex",
        SolveOutput::Plain if triangular.is_some() => "linalg.solve_triangular",
        SolveOutput::Plain => "linalg.solve",
    };
    let dtype = real(op, &[a, b])?;
    let shape = solve_shape(a, b, left, vectors, op)?;
    let n = shape.n;
    let columns = shape.columns;
    let a_count = checked_numel(op, &shape.a_batch)?;
    let batch_count = checked_numel(op, &shape.batch)?;
    let with_info = matches!(mode, SolveOutput::WithInfo { .. });
    let mut info = if with_info { capacity(op, &shape.a_batch)?; Some(filled(op, &shape.a_batch, 0i64)?) } else { None };
    // Status codes belong to A's factorization, including right solves.
    let transpose_factor = !left && !with_info;
    let transpose_solve = !left && !transpose_factor;
    let mut output = filled(op, &shape.output, 0.0f64)?;
    let mut factors = Vec::new();
    if n > 0 {
        factors.try_reserve_exact(a_count).map_err(|_| invalid(op, "cannot allocate matrix factors"))?;
        let original = values(a);
        for batch in 0..a_count {
            let mut matrix = original[batch * n * n..(batch + 1) * n * n].to_vec();
            if let Some((upper, unit)) = triangular {
                for row in 0..n { for column in 0..n {
                    if if upper { row > column } else { row < column } { matrix[row * n + column] = 0.; }
                    if unit && row == column { matrix[row * n + column] = 1.; }
                } }
            }
            if transpose_factor { matrix = transpose(&matrix, n, n); }
            let (factor, failure) = Lu::factor_info(matrix, n, op)?;
            if failure != 0 && !matches!(mode, SolveOutput::WithInfo { check_errors: false, .. }) {
                return Err(invalid(op, format!("singular coefficient matrix at pivot {failure}")));
            }
            if let Some(info) = &mut info { info[batch] = failure as i64; }
            factors.push(factor);
        }
        let rhs = values(b);
        for batch in 0..if columns > 0 { batch_count } else { 0 } {
            let ai = batch_index(batch, &shape.batch, &shape.a_batch);
            let bi = batch_index(batch, &shape.batch, &shape.b_batch);
            let right = &rhs[bi * n * columns..(bi + 1) * n * columns];
            let work = if left { right.to_vec() } else { transpose(right, columns, n) };
            let solution = factors[ai].solve(&work, columns, transpose_solve);
            let result = if left { solution } else { transpose(&solution, n, columns) };
            output[batch * n * columns..(batch + 1) * n * columns].copy_from_slice(&result);
        }
    }
    let output = tensor(output, &shape.output, dtype)?;
    let saved = output.detach_copy();
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    let info = info.map(|values| Tensor::from_vec_i64(values, &shape.a_batch)).transpose()?;
    let backward = move |gradient: &Tensor| {
        let mut da = filled(op, &a_shape, 0.0f64).unwrap();
        let mut db = filled(op, &b_shape, 0.0f64).unwrap();
        if n > 0 && columns > 0 {
            let x = values(&saved);
            let cotangent = values(gradient);
            for batch in 0..batch_count {
                let ai = batch_index(batch, &shape.batch, &shape.a_batch);
                let bi = batch_index(batch, &shape.batch, &shape.b_batch);
                let range = batch * n * columns..(batch + 1) * n * columns;
                let g = if left { cotangent[range.clone()].to_vec() } else { transpose(&cotangent[range.clone()], columns, n) };
                let solution = if left { x[range].to_vec() } else { transpose(&x[range], columns, n) };
                let rhs_gradient = factors[ai].solve(&g, columns, !transpose_solve);
                for row in 0..n { for column in 0..n {
                    let mut contribution = 0.;
                    for k in 0..columns { contribution -= rhs_gradient[row * columns + k] * solution[column * columns + k]; }
                    let (r, c) = if left { (row, column) } else { (column, row) };
                    if let Some((upper, unit)) = triangular {
                        if (unit && r == c) || if upper { r > c } else { r < c } { continue; }
                    }
                    da[ai * n * n + r * n + c] += contribution;
                } }
                let original = if left { rhs_gradient } else { transpose(&rhs_gradient, n, columns) };
                for (index, value) in original.into_iter().enumerate() { db[bi * n * columns + index] += value; }
            }
        }
        vec![tensor(da, &a_shape, dtype).unwrap(), tensor(db, &b_shape, dtype).unwrap()]
    };
    let inputs = vec![a.clone(), b.clone()];
    let output = match mode {
        SolveOutput::Plain => output.record_fn(inputs, backward).save_output_version(),
        SolveOutput::WithInfo { inverse, .. } => output.record_fn_saved(inputs, if inverse { &[] } else { &[0] }, true, backward),
    };
    Ok((output, info))
}

/// Solve AX=B (left) or XA=B (right), with broadcast batches and Torch vector rules.
pub fn solve(a: &Tensor, b: &Tensor, left: bool) -> Result<Tensor> { Ok(solve_impl(a, b, left, None, true, SolveOutput::Plain)?.0) }

/// Solve with an untracked i64 status per original A batch. Positive status
/// identifies the first zero LU pivot; failed solutions/adjoints are undefined.
pub fn solve_ex(a: &Tensor, b: &Tensor, left: bool, check_errors: bool) -> Result<(Tensor, Tensor)> {
    let (result, info) = solve_impl(a, b, left, None, true, SolveOutput::WithInfo { check_errors, inverse: false })?;
    Ok((result, info.unwrap()))
}

/// Solve with only the selected triangle; unit diagonal values are never read.
pub fn solve_triangular(a: &Tensor, b: &Tensor, upper: bool, left: bool, unitriangular: bool) -> Result<Tensor> {
    Ok(solve_impl(a, b, left, Some((upper, unitriangular)), false, SolveOutput::Plain)?.0)
}

/// Batched matrix inverse using partial-pivot LU solves of the identity.
pub fn inv(a: &Tensor) -> Result<Tensor> {
    let dtype = real("linalg.inv", &[a])?;
    let (n, _) = square("linalg.inv", a)?;
    let mut identity = filled("linalg.inv", &[n, n], 0.0f64)?;
    for index in 0..n { identity[index * n + index] = 1.; }
    Ok(solve_impl(a, &tensor(identity, &[n, n], dtype)?, true, None, false, SolveOutput::Plain)?.0)
}

/// Inverse with original-batch i64 LU status and output-only version dependency.
pub fn inv_ex(a: &Tensor, check_errors: bool) -> Result<(Tensor, Tensor)> {
    let dtype = real("linalg.inv_ex", &[a])?;
    let (n, _) = square("linalg.inv_ex", a)?;
    let mut identity = filled("linalg.inv_ex", &[n, n], 0.0f64)?;
    for index in 0..n { identity[index * n + index] = 1.; }
    let (result, info) = solve_impl(a, &tensor(identity, &[n, n], dtype)?, true, None, false, SolveOutput::WithInfo { check_errors, inverse: true })?;
    Ok((result, info.unwrap()))
}

fn cholesky_factor(matrix: &[f64], n: usize, upper: bool) -> (Vec<f64>, i64) {
    let mut lower = vec![0.; n * n];
    for row in 0..n { for column in 0..=row {
        lower[row * n + column] = if upper { matrix[column * n + row] } else { matrix[row * n + column] };
    } }
    let mut info = 0;
    for column in 0..n {
        let mut diagonal = lower[column * n + column];
        for k in 0..column { diagonal -= lower[column * n + k].powi(2); }
        lower[column * n + column] = diagonal;
        if diagonal <= 0. || !diagonal.is_finite() {
            info = (column + 1) as i64;
            break;
        }
        lower[column * n + column] = diagonal.sqrt();
        for row in column + 1..n {
            let mut value = lower[row * n + column];
            for k in 0..column { value -= lower[row * n + k] * lower[column * n + k]; }
            lower[row * n + column] = value / lower[column * n + column];
        }
    }
    (if upper { transpose(&lower, n, n) } else { lower }, info)
}

fn cholesky_adjoint(factor: &[f64], gradient: &[f64], n: usize, upper: bool) -> Vec<f64> {
    let lower = if upper { transpose(factor, n, n) } else { factor.to_vec() };
    let g = if upper { transpose(gradient, n, n) } else { gradient.to_vec() };
    let mut phi = vec![0.; n * n];
    for row in 0..n { for column in 0..=row {
        for k in row..n { phi[row * n + column] += lower[k * n + row] * g[k * n + column]; }
        if row == column { phi[row * n + column] *= 0.5; }
    } }
    let mut inverse = vec![0.; n * n];
    for column in 0..n { for row in 0..n {
        let mut value = if row == column { 1. } else { 0. };
        for k in 0..row { value -= lower[row * n + k] * inverse[k * n + column]; }
        inverse[row * n + column] = value / lower[row * n + row];
    } }
    let mut product = vec![0.; n * n];
    for row in 0..n { for column in 0..n {
        for k in 0..n { product[row * n + column] += phi[row * n + k] * inverse[k * n + column]; }
    } }
    let mut output = vec![0.; n * n];
    for row in 0..n { for column in 0..n {
        for k in 0..n { output[row * n + column] += inverse[k * n + row] * product[k * n + column]; }
    } }
    for row in 0..n { for column in 0..row {
        let value = (output[row * n + column] + output[column * n + row]) * 0.5;
        output[row * n + column] = value;
        output[column * n + row] = value;
    } }
    output
}

/// Cholesky factor and per-matrix first non-positive minor (zero on success).
/// Only the selected input triangle is read. Gradients are defined for SPD inputs.
pub fn cholesky_ex(a: &Tensor, upper: bool, check_errors: bool) -> Result<(Tensor, Tensor)> {
    let op = "linalg.cholesky_ex";
    let dtype = real(op, &[a])?;
    let (n, batch) = square(op, a)?;
    capacity(op, &[n, n])?;
    capacity(op, &batch)?;
    let matrices = checked_numel(op, &batch)?;
    let mut output = filled(op, a.shape(), 0.0f64)?;
    let mut info = filled(op, &batch, 0i64)?;
    if n > 0 {
        let data = values(a);
        for matrix in 0..matrices {
            let range = matrix * n * n..(matrix + 1) * n * n;
            let (factor, failure) = cholesky_factor(&data[range.clone()], n, upper);
            if check_errors && failure != 0 { return Err(invalid(op, format!("matrix {matrix} is not positive definite at leading minor {failure}"))); }
            output[range].copy_from_slice(&factor);
            info[matrix] = failure;
        }
    }
    let out = tensor(output, a.shape(), dtype)?;
    let saved = out.detach_copy();
    let shape = a.shape().to_vec();
    let out = out.record_fn(vec![a.clone()], move |gradient| {
        let factor = values(&saved);
        let g = values(gradient);
        let mut result = filled(op, &shape, 0.0f64).unwrap();
        if n > 0 { for matrix in 0..matrices {
            let range = matrix * n * n..(matrix + 1) * n * n;
            let derivative = cholesky_adjoint(&factor[range.clone()], &g[range.clone()], n, upper);
            result[range].copy_from_slice(&derivative);
        } }
        vec![tensor(result, &shape, dtype).unwrap()]
    }).save_output_version();
    Ok((out, Tensor::from_vec_i64(info, &batch)?))
}

pub fn cholesky(a: &Tensor, upper: bool) -> Result<Tensor> { Ok(cholesky_ex(a, upper, true)?.0) }
fn cholesky_solve_impl(a: &Tensor, b: &Tensor, upper: bool, inverse: bool) -> Result<Tensor> {
    let op = if inverse { "cholesky_inverse" } else { "cholesky_solve" };
    let dtype = real(op, &[a, b])?;
    let shape = solve_shape(a, b, true, false, op)?;
    let n = shape.n;
    let columns = shape.columns;
    let a_count = checked_numel(op, &shape.a_batch)?;
    let batch_count = checked_numel(op, &shape.batch)?;
    let original = values(a);
    let mut factors = Vec::new();
    let mut output = filled(op, &shape.output, 0.0f64)?;
    if n > 0 {
        factors.try_reserve_exact(a_count).map_err(|_| invalid(op, "cannot allocate Cholesky factors"))?;
        for batch in 0..a_count {
            let mut triangle = original[batch * n * n..(batch + 1) * n * n].to_vec();
            for row in 0..n { for column in 0..n {
                if if upper { row > column } else { row < column } { triangle[row * n + column] = 0.; }
            } }
            let (factor, failure) = Lu::factor_info(triangle, n, op)?;
            if inverse && failure != 0 { return Err(invalid(op, format!("singular factor at pivot {failure}"))); }
            factors.push(factor);
        }
        let rhs = values(b);
        for batch in 0..if columns > 0 { batch_count } else { 0 } {
            let ai = batch_index(batch, &shape.batch, &shape.a_batch);
            let bi = batch_index(batch, &shape.batch, &shape.b_batch);
            let right = &rhs[bi * n * columns..(bi + 1) * n * columns];
            let first = factors[ai].solve(right, columns, upper);
            let mut result = factors[ai].solve(&first, columns, !upper);
            if inverse { for row in 0..n { for column in 0..row {
                let value = if upper { result[column * n + row] } else { result[row * n + column] };
                result[row * n + column] = value;
                result[column * n + row] = value;
            } } }
            output[batch * n * columns..(batch + 1) * n * columns].copy_from_slice(&result);
        }
    }
    let output = tensor(output, &shape.output, dtype)?;
    let saved = output.detach_copy();
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    Ok(output.record_fn(vec![a.clone(), b.clone()], move |gradient| {
        let mut da = filled(op, &a_shape, 0.0f64).unwrap();
        let mut db = filled(op, &b_shape, 0.0f64).unwrap();
        if n > 0 && columns > 0 {
            let solution = values(&saved);
            let cotangent = values(gradient);
            for batch in 0..batch_count {
                let ai = batch_index(batch, &shape.batch, &shape.a_batch);
                let bi = batch_index(batch, &shape.batch, &shape.b_batch);
                let range = batch * n * columns..(batch + 1) * n * columns;
                let first = factors[ai].solve(&cotangent[range.clone()], columns, upper);
                let rhs_gradient = factors[ai].solve(&first, columns, !upper);
                let x = &solution[range];
                let mut symmetric = vec![0.; n * n];
                for row in 0..n { for column in 0..n { for k in 0..columns {
                    symmetric[row * n + column] += rhs_gradient[row * columns + k] * x[column * columns + k]
                        + x[row * columns + k] * rhs_gradient[column * columns + k];
                } } }
                let factor = &original[ai * n * n..(ai + 1) * n * n];
                // Torch's factor-manifold adjoint includes the ignored triangle.
                for row in 0..n { for column in 0..n { for k in 0..n {
                    da[ai * n * n + row * n + column] -= if upper {
                        factor[row * n + k] * symmetric[k * n + column]
                    } else {
                        symmetric[row * n + k] * factor[k * n + column]
                    };
                } } }
                for (index, value) in rhs_gradient.into_iter().enumerate() { db[bi * n * columns + index] += value; }
            }
        } else if n > 0 && columns == 0 && batch_count > 0 {
            // A zero symmetric product still propagates unused nonfinite entries.
            // Visit original factors only, never the potentially huge RHS batch.
            for batch in 0..a_count {
                let factor = &original[batch * n * n..(batch + 1) * n * n];
                let nonfinite: Vec<_> = (0..n).map(|index| (0..n).any(|k| {
                    !factor[if upper { index * n + k } else { k * n + index }].is_finite()
                })).collect();
                for row in 0..n { for column in 0..n {
                    if nonfinite[if upper { row } else { column }] { da[batch * n * n + row * n + column] = f64::NAN; }
                } }
            }
        }
        vec![tensor(da, &a_shape, dtype).unwrap(), tensor(db, &b_shape, dtype).unwrap()]
    }).save_output_version())
}

/// Solve from a triangular Cholesky factor; selected-triangle forward, full
/// unmasked factor adjoint matching Torch's factor-manifold convention.
pub fn cholesky_solve(b: &Tensor, a: &Tensor, upper: bool) -> Result<Tensor> {
    cholesky_solve_impl(a, b, upper, false)
}

/// Invert from a nonsingular triangular Cholesky factor. The factor adjoint
/// uses all original entries, not finite differences of the ignored triangle.
pub fn cholesky_inverse(a: &Tensor, upper: bool) -> Result<Tensor> {
    let dtype = real("cholesky_inverse", &[a])?;
    let (n, _) = square("cholesky_inverse", a)?;
    let mut identity = filled("cholesky_inverse", &[n, n], 0.0f64)?;
    for index in 0..n { identity[index * n + index] = 1.; }
    cholesky_solve_impl(a, &tensor(identity, &[n, n], dtype)?, upper, true)
}


fn identity(n: usize) -> Vec<f64> {
    let mut result = vec![0.; n * n];
    for index in 0..n { result[index * n + index] = 1.; }
    result
}

fn cofactors(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.; n * n];
    if n == 0 { return result; }
    let mut minor = Vec::with_capacity((n - 1) * (n - 1));
    for row in 0..n { for column in 0..n {
        minor.clear();
        for i in 0..n { if i != row { for j in 0..n { if j != column { minor.push(matrix[i * n + j]); } } } }
        let (factor, info) = Lu::factor_info(minor.clone(), n - 1, "linalg.det").unwrap();
        let value = if info == 0 { factor.determinant() } else { 0. };
        result[row * n + column] = if (row + column) % 2 == 0 { value } else { -value };
    } }
    result
}

/// Determinant, including the polynomial cofactor adjoint at singular inputs.
/// Singular or numerically degenerate adjoints use an O(n^5) minor fallback.
pub fn det(a: &Tensor) -> Result<Tensor> {
    let op = "linalg.det";
    let dtype = real(op, &[a])?;
    let (n, batch) = square(op, a)?;
    capacity(op, &[n, n])?;
    let matrices = capacity(op, &batch)?;
    let data = values(a);
    let mut output = filled(op, &batch, 1.0f64)?;
    let mut derivatives = filled(op, a.shape(), 0.0f64)?;
    if n > 0 { for matrix in 0..matrices {
        let range = matrix * n * n..(matrix + 1) * n * n;
        let original = &data[range.clone()];
        let (factor, info) = Lu::factor_info(original.to_vec(), n, op)?;
        let value = if info == 0 { factor.determinant() } else { 0. };
        output[matrix] = value;
        if a.requires_grad() {
            let mut derivative = if info == 0 && value != 0. && value.is_finite() {
                factor.solve(&identity(n), n, true).into_iter().map(|x| x * value).collect::<Vec<_>>()
            } else { cofactors(original, n) };
            if derivative.iter().any(|x| !x.is_finite()) { derivative = cofactors(original, n); }
            derivatives[range].copy_from_slice(&derivative);
        }
    } }
    let shape = a.shape().to_vec();
    Ok(tensor(output, &batch, dtype)?.record_fn(vec![a.clone()], move |gradient| {
        let mut result = derivatives.clone();
        let g = values(gradient);
        if n > 0 { for matrix in 0..matrices {
            for value in &mut result[matrix * n * n..(matrix + 1) * n * n] { *value *= g[matrix]; }
        } }
        vec![tensor(result, &shape, dtype).unwrap()]
    }).save_output_version())
}

/// Sign and log absolute determinant. Log gradients require nonsingular inputs.
/// A singular log determinant is -inf with an explicitly undefined NaN adjoint.
pub fn slogdet(a: &Tensor) -> Result<(Tensor, Tensor)> {
    let op = "linalg.slogdet";
    let dtype = real(op, &[a])?;
    let (n, batch) = square(op, a)?;
    capacity(op, &[n, n])?;
    let matrices = capacity(op, &batch)?;
    let data = values(a);
    let mut signs = filled(op, &batch, 1.0f64)?;
    let mut logs = filled(op, &batch, 0.0f64)?;
    let mut inverse_transposes = filled(op, a.shape(), 0.0f64)?;
    if n > 0 { for matrix in 0..matrices {
        let range = matrix * n * n..(matrix + 1) * n * n;
        let (factor, info) = Lu::factor_info(data[range.clone()].to_vec(), n, op)?;
        if info == 0 {
            (signs[matrix], logs[matrix]) = factor.signed_logdet();
            if a.requires_grad() { inverse_transposes[range].copy_from_slice(&factor.solve(&identity(n), n, true)); }
        } else {
            signs[matrix] = 0.;
            logs[matrix] = f64::NEG_INFINITY;
            inverse_transposes[range].fill(f64::NAN);
        }
    } }
    let shape = a.shape().to_vec();
    let sign = tensor(signs, &batch, dtype)?.record_fn_optional(vec![a.clone()], |_| vec![None]).save_output_version();
    let logarithm = tensor(logs, &batch, dtype)?.record_fn(vec![a.clone()], move |gradient| {
        let mut result = inverse_transposes.clone();
        let g = values(gradient);
        if n > 0 { for matrix in 0..matrices {
            for value in &mut result[matrix * n * n..(matrix + 1) * n * n] { *value *= g[matrix]; }
        } }
        vec![tensor(result, &shape, dtype).unwrap()]
    }).save_tensor_versions(&[&sign]);
    Ok((sign, logarithm))
}

fn reduction_index(mut flat: usize, shape: &[usize], reduced: &[bool]) -> usize {
    let (mut result, mut stride) = (0, 1);
    for axis in (0..shape.len()).rev() {
        let coordinate = flat % shape[axis];
        flat /= shape[axis];
        if !reduced[axis] { result += coordinate * stride; stride *= shape[axis]; }
    }
    result
}

/// Stable vector p-norms. Infinite norms split gradients equally at ties.
/// The zero-norm count has an unused input gradient, matching ordinary Torch AD.
pub fn vector_norm(a: &Tensor, order: f64, dims: Option<&[isize]>, keepdim: bool) -> Result<Tensor> {
    let op = "linalg.vector_norm";
    let dtype = real(op, &[a])?;
    if order.is_nan() { return Err(invalid(op, "order must not be NaN")); }
    let rank = a.ndim();
    let mut reduced = vec![false; rank];
    if let Some(dims) = dims.filter(|axes| !axes.is_empty()) {
        let mut scalar_seen = false;
        for &dim in dims {
            let logical_rank = rank.max(1) as isize;
            if dim < -logical_rank || dim >= logical_rank { return Err(invalid(op, "dimension is out of range")); }
            let axis = if dim < 0 { (dim + logical_rank) as usize } else { dim as usize };
            if rank == 0 {
                if scalar_seen { return Err(invalid(op, "dimensions must be unique")); }
                scalar_seen = true;
            } else {
                if reduced[axis] { return Err(invalid(op, "dimensions must be unique")); }
                reduced[axis] = true;
            }
        }
    } else { reduced.fill(true); }
    if (order < 0. || order.is_infinite()) && a.shape().iter().zip(&reduced).any(|(&n, &r)| r && n == 0) {
        return Err(invalid(op, "this norm has no identity for an empty reduction"));
    }
    let output_shape = a.shape().iter().zip(&reduced).filter_map(|(&n, &r)| {
        if !r { Some(n) } else if keepdim { Some(1) } else { None }
    }).collect::<Vec<_>>();
    capacity(op, &output_shape)?;
    let data = values(a);
    if order == 0. {
        let mut output = filled(op, &output_shape, 0.0f64)?;
        for (index, &value) in data.iter().enumerate() {
            if value != 0. { output[reduction_index(index, a.shape(), &reduced)] += 1.; }
        }
        return Ok(tensor(output, &output_shape, dtype)?.record_fn_optional(vec![a.clone()], |_| vec![None]).save_output_version());
    }
    let mut scale = filled(op, &output_shape, if order < 0. { f64::INFINITY } else { 0. })?;
    for (index, &value) in data.iter().enumerate() {
        let group = reduction_index(index, a.shape(), &reduced);
        let absolute = value.abs();
        if absolute.is_nan() || scale[group].is_nan() { scale[group] = f64::NAN; }
        else if order < 0. { scale[group] = scale[group].min(absolute); }
        else { scale[group] = scale[group].max(absolute); }
    }
    let mut output = scale.clone();
    if order == 1. {
        output.fill(0.);
        for (index, &value) in data.iter().enumerate() {
            output[reduction_index(index, a.shape(), &reduced)] += value.abs();
        }
    } else if !order.is_infinite() {
        let mut sums = filled(op, &output_shape, 0.0f64)?;
        for (index, &value) in data.iter().enumerate() {
            let group = reduction_index(index, a.shape(), &reduced);
            if scale[group] > 0. && scale[group].is_finite() { sums[group] += (value.abs() / scale[group]).powf(order); }
        }
        for group in 0..output.len() {
            if scale[group] > 0. && scale[group].is_finite() { output[group] = scale[group] * sums[group].powf(order.recip()); }
        }
    }
    let mut counts = filled(op, &output_shape, 0usize)?;
    if order.is_infinite() { for (index, &value) in data.iter().enumerate() {
        let group = reduction_index(index, a.shape(), &reduced);
        if value.abs() == output[group] { counts[group] += 1; }
    } }
    let shape = a.shape().to_vec();
    let saved = output.clone();
    Ok(tensor(output, &output_shape, dtype)?.record_fn(vec![a.clone()], move |gradient| {
        let g = values(gradient);
        let mut result = filled(op, &shape, 0.0f64).unwrap();
        for (index, &value) in data.iter().enumerate() {
            let group = reduction_index(index, &shape, &reduced);
            let norm = saved[group];
            let sign = if value > 0. { 1. } else if value < 0. { -1. } else { 0. };
            let derivative = if order.is_infinite() {
                if value.abs() == norm { sign / counts[group] as f64 } else { 0. }
            } else if order == 1. { sign }
            else if norm == 0. { 0. }
            else if value == 0. && norm.is_finite() { 0. }
            else { sign * (value.abs() / norm).powf(order - 1.) };
            result[index] = derivative * g[group];
        }
        vec![tensor(result, &shape, dtype).unwrap()]
    }).save_output_version())
}

/// Shared-storage diagonal view; backward scatters into the input's logical shape.
pub fn diagonal(a: &Tensor, offset: isize, dim1: isize, dim2: isize) -> Result<Tensor> {
    let op = "linalg.diagonal";
    let dtype = real(op, &[a])?;
    let rank = a.ndim();
    if rank < 2 { return Err(invalid(op, "diagonal requires at least two dimensions")); }
    let normalize = |dim: isize| -> Result<usize> {
        if dim < -(rank as isize) || dim >= rank as isize { return Err(invalid(op, "dimension is out of range")); }
        Ok(if dim < 0 { (dim + rank as isize) as usize } else { dim as usize })
    };
    let (dim1, dim2) = (normalize(dim1)?, normalize(dim2)?);
    if dim1 == dim2 { return Err(invalid(op, "diagonal dimensions must be distinct")); }
    let (row, column) = if offset >= 0 { (0, offset as usize) } else { (offset.unsigned_abs(), 0) };
    let length = if row >= a.shape()[dim1] || column >= a.shape()[dim2] { 0 }
        else { (a.shape()[dim1] - row).min(a.shape()[dim2] - column) };
    let axes = (0..rank).filter(|&axis| axis != dim1 && axis != dim2).collect::<Vec<_>>();
    let mut shape = axes.iter().map(|&axis| a.shape()[axis]).collect::<Vec<_>>();
    shape.push(length);
    capacity(op, &shape)?;
    let mut strides = axes.iter().map(|&axis| a.0.stride[axis]).collect::<Vec<_>>();
    strides.push(a.0.stride[dim1].checked_add(a.0.stride[dim2]).ok_or_else(|| invalid(op, "diagonal stride overflow"))?);
    let storage_offset = if length == 0 { a.0.offset } else {
        row.checked_mul(a.0.stride[dim1]).and_then(|x| column.checked_mul(a.0.stride[dim2]).and_then(|y| x.checked_add(y)))
            .and_then(|x| a.0.offset.checked_add(x)).ok_or_else(|| invalid(op, "diagonal offset overflow"))?
    };
    let output = Tensor::from_parts(a.0.storage.clone(), shape, strides, storage_offset, a.device(), false, None).inherit_no_grad_view(a);
    let input_shape = a.shape().to_vec();
    let logical_strides = crate::shape::default_strides(&input_shape);
    Ok(output.record_fn(vec![a.clone()], move |gradient| {
        let mut result = filled(op, &input_shape, 0.0f64).unwrap();
        for (flat, value) in values(gradient).into_iter().enumerate() {
            let diagonal = flat % length;
            let mut outer = flat / length;
            let mut index = (row + diagonal) * logical_strides[dim1] + (column + diagonal) * logical_strides[dim2];
            for &axis in axes.iter().rev() {
                let coordinate = outer % input_shape[axis];
                outer /= input_shape[axis];
                index += coordinate * logical_strides[axis];
            }
            result[index] += value;
        }
        vec![tensor(result, &input_shape, dtype).unwrap()]
    }))
}

/// Sum the main diagonal of a rank-two real matrix.
pub fn trace(a: &Tensor) -> Result<Tensor> {
    if a.ndim() != 2 { return Err(invalid("linalg.trace", "trace requires a rank-two input")); }
    Ok(diagonal(a, 0, 0, 1)?.sum())
}

struct QrFactor {
    q: Vec<f64>,
    r: Vec<f64>,
    tail: Vec<f64>,
    triangular: Lu,
}

fn householder_qr(mut matrix: Vec<f64>, rows: usize, columns: usize) -> Result<QrFactor> {
    let op = "linalg.qr";
    if matrix.iter().any(|x| !x.is_finite()) { return Err(invalid(op, "QR inputs must contain finite values")); }
    let mut tau = vec![0.; columns];
    for column in 0..columns {
        let mut tail = 0.0f64;
        for row in column + 1..rows { tail = tail.hypot(matrix[row * columns + column]); }
        let alpha = matrix[column * columns + column];
        if tail != 0. {
            let beta = -alpha.hypot(tail).copysign(alpha);
            tau[column] = 1. - alpha / beta;
            for row in column + 1..rows {
                matrix[row * columns + column] = (matrix[row * columns + column] / beta) / (alpha / beta - 1.);
            }
            matrix[column * columns + column] = beta;
        }
        for k in column + 1..columns {
            let mut product = matrix[column * columns + k];
            for row in column + 1..rows { product += matrix[row * columns + column] * matrix[row * columns + k]; }
            product *= tau[column];
            matrix[column * columns + k] -= product;
            for row in column + 1..rows { matrix[row * columns + k] -= matrix[row * columns + column] * product; }
        }
    }
    let mut q = vec![0.; rows * columns];
    for column in 0..columns { q[column * columns + column] = 1.; }
    for column in (0..columns).rev() {
        for k in 0..columns {
            let mut product = q[column * columns + k];
            for row in column + 1..rows { product += matrix[row * columns + column] * q[row * columns + k]; }
            product *= tau[column];
            q[column * columns + k] -= product;
            for row in column + 1..rows { q[row * columns + k] -= matrix[row * columns + column] * product; }
        }
    }
    let mut r = vec![0.; columns * columns];
    for row in 0..columns { for column in row..columns { r[row * columns + column] = matrix[row * columns + column]; } }
    let triangular = Lu::factor(r.clone(), columns, op)?;
    Ok(QrFactor { q, r, tail: Vec::new(), triangular })
}

fn qr_adjoint(factor: &QrFactor, rows: usize, columns: usize, gq: Option<&[f64]>, gr: Option<&[f64]>) -> Vec<f64> {
    let mut middle = vec![0.; columns * columns];
    for row in 0..columns { for column in 0..=row {
        let mut value = 0.;
        if let Some(g) = gr { for k in 0..columns { value += factor.r[row * columns + k] * g[column * columns + k]; } }
        if let Some(g) = gq { for k in 0..rows { value -= g[k * columns + row] * factor.q[k * columns + column]; } }
        middle[row * columns + column] = value;
        middle[column * columns + row] = value;
    } }
    let mut direction = gq.map(|g| g.to_vec()).unwrap_or_else(|| vec![0.; rows * columns]);
    for row in 0..rows { for column in 0..columns {
        for k in 0..columns { direction[row * columns + column] += factor.q[row * columns + k] * middle[k * columns + column]; }
    } }
    let solved = factor.triangular.solve(&transpose(&direction, rows, columns), rows, false);
    transpose(&solved, columns, rows)
}

/// Reduced Householder QR with linearly independent leading min(m,n) columns.
/// Exact zero R pivots reject; rank-deficient
/// gradients are unsupported. Both factors retain first-order input adjoints.
pub fn qr(a: &Tensor) -> Result<(Tensor, Tensor)> {
    let op = "linalg.qr";
    let dtype = real(op, &[a])?;
    if a.ndim() < 2 { return Err(invalid(op, "QR requires at least two dimensions")); }
    let rows = a.shape()[a.ndim() - 2];
    let columns = a.shape()[a.ndim() - 1];
    let width = rows.min(columns);
    let tail_width = columns - width;
    let batch = &a.shape()[..a.ndim() - 2];
    let matrices = checked_numel(op, batch)?;
    let mut q_shape = batch.to_vec();
    q_shape.extend([rows, width]);
    let mut r_shape = batch.to_vec();
    r_shape.extend([width, columns]);
    capacity(op, &[rows, columns])?;
    capacity(op, &[width, width])?;
    capacity(op, &q_shape)?;
    capacity(op, &r_shape)?;
    let mut q_values = filled(op, &q_shape, 0.0f64)?;
    let mut r_values = filled(op, &r_shape, 0.0f64)?;
    let mut factors = Vec::new();
    if width > 0 {
        factors.try_reserve_exact(matrices).map_err(|_| invalid(op, "cannot allocate QR factors"))?;
        let data = values(a);
        for matrix in 0..matrices {
            let input_start = matrix * rows * columns;
            let q_range = matrix * rows * width..(matrix + 1) * rows * width;
            let r_start = matrix * width * columns;
            let mut leading = vec![0.; rows * width];
            let mut tail = vec![0.; rows * tail_width];
            for row in 0..rows {
                let start = input_start + row * columns;
                leading[row * width..(row + 1) * width].copy_from_slice(&data[start..start + width]);
                tail[row * tail_width..(row + 1) * tail_width].copy_from_slice(&data[start + width..start + columns]);
            }
            if tail.iter().any(|x| !x.is_finite()) { return Err(invalid(op, "QR inputs must contain finite values")); }
            let mut factor = householder_qr(leading, rows, width)?;
            q_values[q_range].copy_from_slice(&factor.q);
            for row in 0..width {
                let start = r_start + row * columns;
                r_values[start..start + width].copy_from_slice(&factor.r[row * width..(row + 1) * width]);
                for column in 0..tail_width {
                    let mut value = 0.;
                    for k in 0..rows { value += factor.q[k * width + row] * tail[k * tail_width + column]; }
                    if !value.is_finite() { return Err(invalid(op, "QR factors must be representable")); }
                    r_values[start + width + column] = value;
                }
            }
            factor.tail = tail;
            factors.push(factor);
        }
    }
    let q = tensor(q_values, &q_shape, dtype)?;
    let r = tensor(r_values, &r_shape, dtype)?;
    let cache = std::sync::Arc::new(factors);
    let q_cache = cache.clone();
    let input_shape = a.shape().to_vec();
    let q_input_shape = input_shape.clone();
    let q = q.record_fn(vec![a.clone()], move |gradient| {
        let g = values(gradient);
        let mut result = filled(op, &q_input_shape, 0.0f64).unwrap();
        if width > 0 { for matrix in 0..matrices {
            let range = matrix * rows * width..(matrix + 1) * rows * width;
            let adjoint = qr_adjoint(&q_cache[matrix], rows, width, Some(&g[range]), None);
            for row in 0..rows {
                let start = (matrix * rows + row) * columns;
                result[start..start + width].copy_from_slice(&adjoint[row * width..(row + 1) * width]);
            }
        } }
        vec![tensor(result, &q_input_shape, dtype).unwrap()]
    }).save_output_version();
    let r = r.record_fn(vec![a.clone()], move |gradient| {
        let g = values(gradient);
        let mut result = filled(op, &input_shape, 0.0f64).unwrap();
        if width > 0 { for matrix in 0..matrices {
            let factor = &cache[matrix];
            let g = &g[matrix * width * columns..(matrix + 1) * width * columns];
            let mut leading = vec![0.; width * width];
            let mut effective_q = vec![0.; rows * width];
            for row in 0..width {
                leading[row * width..(row + 1) * width].copy_from_slice(&g[row * columns..row * columns + width]);
            }
            for row in 0..rows { for column in 0..width {
                for k in 0..tail_width { effective_q[row * width + column] += factor.tail[row * tail_width + k] * g[column * columns + width + k]; }
            } }
            let adjoint = qr_adjoint(factor, rows, width, Some(&effective_q), Some(&leading));
            for row in 0..rows {
                let start = (matrix * rows + row) * columns;
                result[start..start + width].copy_from_slice(&adjoint[row * width..(row + 1) * width]);
                for column in 0..tail_width {
                    for k in 0..width { result[start + width + column] += factor.q[row * width + k] * g[k * columns + width + column]; }
                }
            }
        } }
        vec![tensor(result, &input_shape, dtype).unwrap()]
    }).save_output_version().save_tensor_versions(&[&q]);
    let q = q.save_tensor_versions(&[&r]);
    Ok((q, r))
}

struct EigenFactor {
    values: Vec<f64>,
    vectors: Vec<f64>,
}

fn symmetric_eigen(mut matrix: Vec<f64>, n: usize) -> Result<EigenFactor> {
    let op = "linalg.eigh";
    if matrix.iter().any(|x| !x.is_finite()) { return Err(invalid(op, "selected triangle must contain finite values")); }
    let mut vectors = vec![0.; n * n];
    for i in 0..n { vectors[i * n + i] = 1.; }
    let mut converged = n <= 1;
    for _ in 0..100 {
        for p in 0..n { for q in p + 1..n {
            let off = matrix[p * n + q];
            if off == 0. { continue; }
            let delta = matrix[q * n + q] * 0.5 - matrix[p * n + p] * 0.5;
            let scale = delta.abs().max(off.abs());
            let d = delta / scale;
            let b = off / scale;
            let t = b / (d + d.hypot(b).copysign(delta));
            let c = 1. / 1.0f64.hypot(t);
            let s = t * c;
            matrix[p * n + p] -= t * off;
            matrix[q * n + q] += t * off;
            matrix[p * n + q] = 0.;
            matrix[q * n + p] = 0.;
            for k in 0..n {
                if k != p && k != q {
                    let left = matrix[k * n + p];
                    let right = matrix[k * n + q];
                    matrix[k * n + p] = c * left - s * right;
                    matrix[p * n + k] = matrix[k * n + p];
                    matrix[k * n + q] = s * left + c * right;
                    matrix[q * n + k] = matrix[k * n + q];
                }
                let left = vectors[k * n + p];
                let right = vectors[k * n + q];
                vectors[k * n + p] = c * left - s * right;
                vectors[k * n + q] = s * left + c * right;
            }
        } }
        if matrix.iter().any(|x| !x.is_finite()) { return Err(invalid(op, "eigenvalues exceed f64 working capacity")); }
        converged = true;
        for p in 0..n { for q in p + 1..n {
            let tolerance = f64::EPSILON * matrix[p * n + p].abs() + f64::EPSILON * matrix[q * n + q].abs();
            if matrix[p * n + q].abs() > tolerance { converged = false; }
        } }
        if converged { break; }
    }
    if !converged { return Err(invalid(op, "Jacobi eigensolver did not converge in 100 sweeps")); }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| matrix[i * n + i].total_cmp(&matrix[j * n + j]));
    let values: Vec<f64> = order.iter().map(|&i| matrix[i * n + i]).collect();
    if values.iter().any(|x| !x.is_finite()) { return Err(invalid(op, "eigenvalues exceed f64 working capacity")); }
    let mut sorted = vec![0.; n * n];
    for (column, &source) in order.iter().enumerate() {
        for row in 0..n { sorted[row * n + column] = vectors[row * n + source]; }
    }
    Ok(EigenFactor { values, vectors: sorted })
}

fn eigen_adjoint(factor: &EigenFactor, n: usize, gw: Option<&[f64]>, gv: Option<&[f64]>) -> Vec<f64> {
    let mut middle = vec![0.; n * n];
    if let Some(g) = gw { for i in 0..n { middle[i * n + i] = g[i]; } }
    if let Some(g) = gv {
        for i in 0..n { for j in i + 1..n {
            let mut ij = 0.;
            let mut ji = 0.;
            for k in 0..n {
                ij += factor.vectors[k * n + i] * g[k * n + j];
                ji += factor.vectors[k * n + j] * g[k * n + i];
            }
            let value = (ij - ji) * 0.5 / (factor.values[j] - factor.values[i]);
            middle[i * n + j] = value;
            middle[j * n + i] = value;
        } }
    }
    let mut temporary = vec![0.; n * n];
    let mut result = vec![0.; n * n];
    for row in 0..n { for column in 0..n { for k in 0..n {
        temporary[row * n + column] += factor.vectors[row * n + k] * middle[k * n + column];
    } } }
    for row in 0..n { for column in 0..n { for k in 0..n {
        result[row * n + column] += temporary[row * n + k] * factor.vectors[column * n + k];
    } } }
    result
}

/// Real symmetric eigendecomposition, reading only one selected triangle.
/// Eigenvectors have arbitrary column signs. The adjoint is symmetric, as in
/// Torch; eigenvector derivatives are undefined at repeated eigenvalues.
pub fn eigh(a: &Tensor, upper: bool) -> Result<(Tensor, Tensor)> {
    let op = "linalg.eigh";
    let dtype = real(op, &[a])?;
    let (n, batch) = square(op, a)?;
    let matrices = checked_numel(op, &batch)?;
    let mut value_shape = batch;
    value_shape.push(n);
    capacity(op, &[n, n])?;
    capacity(op, &value_shape)?;
    let mut eigenvalues = filled(op, &value_shape, 0.0f64)?;
    let mut eigenvectors = filled(op, a.shape(), 0.0f64)?;
    let mut factors = Vec::new();
    if n > 0 {
        factors.try_reserve_exact(matrices).map_err(|_| invalid(op, "cannot allocate eigen factors"))?;
        let data = values(a);
        for matrix in 0..matrices {
            let start = matrix * n * n;
            let mut symmetric = vec![0.; n * n];
            for row in 0..n { for column in row..n {
                let value = data[start + if upper { row * n + column } else { column * n + row }];
                symmetric[row * n + column] = value;
                symmetric[column * n + row] = value;
            } }
            let factor = symmetric_eigen(symmetric, n)?;
            eigenvalues[matrix * n..(matrix + 1) * n].copy_from_slice(&factor.values);
            eigenvectors[start..start + n * n].copy_from_slice(&factor.vectors);
            factors.push(factor);
        }
    }
    let w = tensor(eigenvalues, &value_shape, dtype)?;
    let v = tensor(eigenvectors, a.shape(), dtype)?;
    let cache = std::sync::Arc::new(factors);
    let w_cache = cache.clone();
    let input_shape = a.shape().to_vec();
    let w_input_shape = input_shape.clone();
    let w = w.record_fn(vec![a.clone()], move |gradient| {
        let g = values(gradient);
        let mut result = filled(op, &w_input_shape, 0.0f64).unwrap();
        if n > 0 { for matrix in 0..matrices {
            let adjoint = eigen_adjoint(&w_cache[matrix], n, Some(&g[matrix * n..(matrix + 1) * n]), None);
            result[matrix * n * n..(matrix + 1) * n * n].copy_from_slice(&adjoint);
        } }
        vec![tensor(result, &w_input_shape, dtype).unwrap()]
    }).save_output_version();
    let v = v.record_fn(vec![a.clone()], move |gradient| {
        let g = values(gradient);
        let mut result = filled(op, &input_shape, 0.0f64).unwrap();
        if n > 0 { for matrix in 0..matrices {
            let range = matrix * n * n..(matrix + 1) * n * n;
            let adjoint = eigen_adjoint(&cache[matrix], n, None, Some(&g[range.clone()]));
            result[range].copy_from_slice(&adjoint);
        } }
        vec![tensor(result, &input_shape, dtype).unwrap()]
    }).save_output_version().save_tensor_versions(&[&w]);
    let w = w.save_tensor_versions(&[&v]);
    Ok((w, v))
}

struct SvdFactor {
    u: Vec<f64>,
    s: Vec<f64>,
    vh: Vec<f64>,
}

fn complete_columns(thin: &[f64], rows: usize, columns: usize, width: usize) -> Result<Vec<f64>> {
    if columns == width { return Ok(thin.to_vec()); }
    let mut result = filled("linalg.svd", &[rows, width], 0.0f64)?;
    for row in 0..rows { result[row * width..row * width + columns].copy_from_slice(&thin[row * columns..(row + 1) * columns]); }
    for column in columns..width {
        let mut best = vec![0.; rows];
        let mut trial = vec![0.; rows];
        let mut best_norm = -1.;
        for axis in 0..rows {
            trial.fill(0.);
            trial[axis] = 1.;
            for _ in 0..2 { for prior in 0..column {
                let mut projection = 0.;
                for row in 0..rows { projection += result[row * width + prior] * trial[row]; }
                for row in 0..rows { trial[row] -= projection * result[row * width + prior]; }
            } }
            let length = trial.iter().fold(0.0f64, |norm, &value| norm.hypot(value));
            if length > best_norm { best.copy_from_slice(&trial); best_norm = length; }
        }
        if !best_norm.is_finite() || best_norm <= 0. { return Err(invalid("linalg.svd", "cannot complete orthogonal singular-vector basis")); }
        for row in 0..rows { result[row * width + column] = best[row] / best_norm; }
    }
    Ok(result)
}

fn tall_svd(mut matrix: Vec<f64>, rows: usize, columns: usize) -> Result<SvdFactor> {
    let op = "linalg.svd";
    if matrix.iter().any(|x| !x.is_finite()) { return Err(invalid(op, "SVD inputs must contain finite values")); }
    let reference = matrix.iter().map(|value| value.abs()).fold(0.0f64, f64::max);
    let mut v = vec![0.; columns * columns];
    for i in 0..columns { v[i * columns + i] = 1.; }
    let mut converged = false;
    for _ in 0..100 {
        let mut changed = false;
        for p in 0..columns { for q in p + 1..columns {
            let mut left = 0.0f64;
            let mut right = 0.0f64;
            for row in 0..rows {
                left = left.hypot(matrix[row * columns + p]);
                right = right.hypot(matrix[row * columns + q]);
            }
            if !left.is_finite() || !right.is_finite() { return Err(invalid(op, "singular values exceed f64 working capacity")); }
            if left == 0. || right == 0. { continue; }
            let mut correlation = 0.;
            for row in 0..rows { correlation += (matrix[row * columns + p] / left) * (matrix[row * columns + q] / right); }
            if correlation.abs() <= 4. * f64::EPSILON { continue; }
            changed = true;
            let scale = left.max(right);
            let (l, r) = (left / scale, right / scale);
            let off = correlation * l * r;
            let deflatable = left.min(right).is_subnormal() && left.min(right) < reference * f64::EPSILON;
            if off == 0. {
                if deflatable {
                    let column = if left < right { p } else { q };
                    for row in 0..rows { matrix[row * columns + column] = 0.; }
                }
                continue;
            }
            let delta = (r * r - l * l) * 0.5;
            let t = off / (delta + delta.hypot(off).copysign(delta));
            let c = 1. / 1.0f64.hypot(t);
            let s = t * c;
            for row in 0..rows {
                let (left, right) = (matrix[row * columns + p], matrix[row * columns + q]);
                matrix[row * columns + p] = c * left - s * right;
                matrix[row * columns + q] = s * left + c * right;
            }
            // A null column can stall at the smallest subnormal. Deflate only
            // when the rotation cannot decrease its norm; orthogonal tiny
            // singular values never enter this correlated-column path.
            if deflatable {
                let (mut new_left, mut new_right) = (0.0f64, 0.0f64);
                for row in 0..rows {
                    new_left = new_left.hypot(matrix[row * columns + p]);
                    new_right = new_right.hypot(matrix[row * columns + q]);
                }
                if new_left.min(new_right) >= left.min(right) {
                    let column = if new_left < new_right { p } else { q };
                    for row in 0..rows { matrix[row * columns + column] = 0.; }
                }
            }
            for row in 0..columns {
                let (left, right) = (v[row * columns + p], v[row * columns + q]);
                v[row * columns + p] = c * left - s * right;
                v[row * columns + q] = s * left + c * right;
            }
        } }
        if !changed { converged = true; break; }
    }
    if !converged { return Err(invalid(op, "one-sided Jacobi SVD did not converge in 100 sweeps")); }
    let mut norms = vec![0.0f64; columns];
    for column in 0..columns { for row in 0..rows { norms[column] = norms[column].hypot(matrix[row * columns + column]); } }
    if norms.iter().any(|x| !x.is_finite()) { return Err(invalid(op, "singular values exceed f64 working capacity")); }
    let mut order: Vec<usize> = (0..columns).collect();
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]));
    let rank = order.iter().filter(|&&i| norms[i] != 0.).count();
    let mut thin = vec![0.; rows * rank];
    let mut vh = vec![0.; columns * columns];
    for (column, &source) in order.iter().enumerate() {
        if column < rank { for row in 0..rows { thin[row * rank + column] = matrix[row * columns + source] / norms[source]; } }
        for row in 0..columns { vh[column * columns + row] = v[row * columns + source]; }
    }
    let u = complete_columns(&thin, rows, rank, columns)?;
    let s = order.iter().map(|&i| norms[i]).collect();
    Ok(SvdFactor { u, s, vh })
}

fn svd_adjoint(factor: &SvdFactor, rows: usize, columns: usize, gu: Option<&[f64]>, gs: Option<&[f64]>, gvh: Option<&[f64]>) -> Vec<f64> {
    let width = factor.s.len();
    let mut eu = vec![0.; width * width];
    let mut ev = vec![0.; width * width];
    if let Some(g) = gu { for i in 0..width { for j in 0..width { for row in 0..rows {
        eu[i * width + j] += factor.u[row * width + i] * g[row * width + j];
    } } } }
    if let Some(g) = gvh { for i in 0..width { for j in 0..width { for column in 0..columns {
        ev[i * width + j] += factor.vh[i * columns + column] * g[j * columns + column];
    } } } }
    let mut middle = vec![0.; width * width];
    if let Some(g) = gs { for i in 0..width { middle[i * width + i] = g[i]; } }
    if gu.is_some() || gvh.is_some() { for i in 0..width { for j in 0..width {
        if i == j { continue; }
        let gap = (factor.s[j] - factor.s[i]) * (factor.s[j] + factor.s[i]);
        if gu.is_some() { middle[i * width + j] += (eu[i * width + j] - eu[j * width + i]) * factor.s[j] / gap; }
        if gvh.is_some() { middle[i * width + j] += factor.s[i] * (ev[i * width + j] - ev[j * width + i]) / gap; }
    } } }
    let mut left = vec![0.; rows * width];
    let mut result = vec![0.; rows * columns];
    for row in 0..rows { for column in 0..width { for k in 0..width {
        left[row * width + column] += factor.u[row * width + k] * middle[k * width + column];
    } } }
    if rows > width { if let Some(g) = gu { for row in 0..rows { for column in 0..width {
        let mut projected = 0.;
        for k in 0..width { projected += factor.u[row * width + k] * eu[k * width + column]; }
        left[row * width + column] += (g[row * width + column] - projected) / factor.s[column];
    } } } }
    for row in 0..rows { for column in 0..columns { for k in 0..width {
        result[row * columns + column] += left[row * width + k] * factor.vh[k * columns + column];
    } } }
    if columns > width { if let Some(g) = gvh {
        let mut residual = vec![0.; width * columns];
        for row in 0..width { for column in 0..columns {
            let mut projected = 0.;
            for k in 0..width { projected += ev[k * width + row] * factor.vh[k * columns + column]; }
            residual[row * columns + column] = (g[row * columns + column] - projected) / factor.s[row];
        } }
        for row in 0..rows { for column in 0..columns { for k in 0..width {
            result[row * columns + column] += factor.u[row * width + k] * residual[k * columns + column];
        } } }
    } }
    result
}

/// One-sided Jacobi SVD with first-order real adjoints. Complementary columns
/// of full factors have ignored adjoints, matching Torch. Vector derivatives
/// at repeated/zero singular values are undefined; values retain their own VJP.
/// Unresolved correlated subnormal values below the original matrix precision
/// may be deflated; orthogonal tiny singular values remain representable.
pub fn svd(a: &Tensor, full_matrices: bool) -> Result<(Tensor, Tensor, Tensor)> {
    let op = "linalg.svd";
    let dtype = real(op, &[a])?;
    if a.ndim() < 2 { return Err(invalid(op, "SVD requires at least two dimensions")); }
    let (rows, columns) = (a.shape()[a.ndim() - 2], a.shape()[a.ndim() - 1]);
    let width = rows.min(columns);
    let (u_columns, v_rows) = if full_matrices { (rows, columns) } else { (width, width) };
    let batch = &a.shape()[..a.ndim() - 2];
    let matrices = checked_numel(op, batch)?;
    let mut u_shape = batch.to_vec(); u_shape.extend([rows, u_columns]);
    let mut s_shape = batch.to_vec(); s_shape.push(width);
    let mut v_shape = batch.to_vec(); v_shape.extend([v_rows, columns]);
    for shape in [&u_shape[..], &s_shape, &v_shape, &[rows, u_columns], &[v_rows, columns], &[width, width]] { capacity(op, shape)?; }
    let mut u_values = filled(op, &u_shape, 0.0f64)?;
    let mut s_values = filled(op, &s_shape, 0.0f64)?;
    let mut v_values = filled(op, &v_shape, 0.0f64)?;
    let mut factors = Vec::new();
    if width > 0 {
        factors.try_reserve_exact(matrices).map_err(|_| invalid(op, "cannot allocate SVD factors"))?;
        let data = values(a);
        for matrix in 0..matrices {
            let input = &data[matrix * rows * columns..(matrix + 1) * rows * columns];
            let factor = if rows >= columns { tall_svd(input.to_vec(), rows, columns)? } else {
                let transposed = tall_svd(transpose(input, rows, columns), columns, rows)?;
                SvdFactor { u: transpose(&transposed.vh, rows, rows), s: transposed.s, vh: transpose(&transposed.u, columns, rows) }
            };
            let u = complete_columns(&factor.u, rows, width, u_columns)?;
            let v = complete_columns(&transpose(&factor.vh, width, columns), columns, width, v_rows)?;
            u_values[matrix * rows * u_columns..(matrix + 1) * rows * u_columns].copy_from_slice(&u);
            s_values[matrix * width..(matrix + 1) * width].copy_from_slice(&factor.s);
            v_values[matrix * v_rows * columns..(matrix + 1) * v_rows * columns].copy_from_slice(&transpose(&v, columns, v_rows));
            factors.push(factor);
        }
    } else if full_matrices && (rows > 0 || columns > 0) {
        for matrix in 0..matrices {
            for i in 0..rows { u_values[matrix * rows * rows + i * rows + i] = 1.; }
            for i in 0..columns { v_values[matrix * columns * columns + i * columns + i] = 1.; }
        }
    }
    let u = tensor(u_values, &u_shape, dtype)?;
    let s = tensor(s_values, &s_shape, dtype)?;
    let vh = tensor(v_values, &v_shape, dtype)?;
    let cache = std::sync::Arc::new(factors);
    let u_cache = cache.clone();
    let s_cache = cache.clone();
    let input_shape = a.shape().to_vec();
    let u_input_shape = input_shape.clone();
    let s_input_shape = input_shape.clone();
    let u = u.record_fn(vec![a.clone()], move |gradient| {
        let g = values(gradient);
        let mut result = filled(op, &u_input_shape, 0.0f64).unwrap();
        if width > 0 { for matrix in 0..matrices {
            let mut thin = vec![0.; rows * width];
            for row in 0..rows {
                let start = (matrix * rows + row) * u_columns;
                thin[row * width..(row + 1) * width].copy_from_slice(&g[start..start + width]);
            }
            let adjoint = svd_adjoint(&u_cache[matrix], rows, columns, Some(&thin), None, None);
            result[matrix * rows * columns..(matrix + 1) * rows * columns].copy_from_slice(&adjoint);
        } }
        vec![tensor(result, &u_input_shape, dtype).unwrap()]
    }).save_output_version();
    let s = s.record_fn(vec![a.clone()], move |gradient| {
        let g = values(gradient);
        let mut result = filled(op, &s_input_shape, 0.0f64).unwrap();
        if width > 0 { for matrix in 0..matrices {
            let adjoint = svd_adjoint(&s_cache[matrix], rows, columns, None, Some(&g[matrix * width..(matrix + 1) * width]), None);
            result[matrix * rows * columns..(matrix + 1) * rows * columns].copy_from_slice(&adjoint);
        } }
        vec![tensor(result, &s_input_shape, dtype).unwrap()]
    }).save_output_version().save_tensor_versions(&[&u]);
    let vh = vh.record_fn(vec![a.clone()], move |gradient| {
        let g = values(gradient);
        let mut result = filled(op, &input_shape, 0.0f64).unwrap();
        if width > 0 { for matrix in 0..matrices {
            let start = matrix * v_rows * columns;
            let adjoint = svd_adjoint(&cache[matrix], rows, columns, None, None, Some(&g[start..start + width * columns]));
            result[matrix * rows * columns..(matrix + 1) * rows * columns].copy_from_slice(&adjoint);
        } }
        vec![tensor(result, &input_shape, dtype).unwrap()]
    }).save_output_version().save_tensor_versions(&[&u, &s]);
    let u = u.save_tensor_versions(&[&s, &vh]);
    let s = s.save_tensor_versions(&[&vh]);
    Ok((u, s, vh))
}

enum ToleranceValues { Real(Vec<f64>), Integer(Vec<i64>) }

struct Tolerance {
    shape: Vec<usize>,
    values: ToleranceValues,
    dtype: DType,
}

impl Tolerance {
    fn read(op: &'static str, value: &Tensor, differentiable: bool) -> Result<Self> {
        check(op, &[value], value.dtype())?;
        capacity(op, value.shape())?;
        if differentiable && value.requires_grad() && crate::autograd::is_grad_enabled() {
            return Err(invalid(op, "atol and rtol are not differentiable"));
        }
        let values = match value.dtype() {
            DType::F32 | DType::F64 => ToleranceValues::Real(values(value)),
            DType::I64 => ToleranceValues::Integer(value.to_vec_i64()),
            DType::Bool => ToleranceValues::Real(value.to_vec_bool().into_iter().map(|x| if x { 1. } else { 0. }).collect()),
            _ => return Err(invalid(op, "tolerances require a real CPU tensor")),
        };
        Ok(Self { shape: value.shape().to_vec(), values, dtype: value.dtype() })
    }
    fn at(&self, index: usize, batch: &[usize]) -> f64 {
        let index = batch_index(index, batch, &self.shape);
        match &self.values { ToleranceValues::Real(values) => values[index], ToleranceValues::Integer(values) => values[index] as f64 }
    }
    fn narrow(&self, index: usize, batch: &[usize]) -> f32 {
        let index = batch_index(index, batch, &self.shape);
        match &self.values { ToleranceValues::Real(values) => values[index] as f32, ToleranceValues::Integer(values) => values[index] as f32 }
    }
}

struct PseudoShape {
    rows: usize,
    columns: usize,
    input_batch: Vec<usize>,
    batch: Vec<usize>,
    absolute: Option<Tolerance>,
    relative: Option<Tolerance>,
    dtype: DType,
}

impl PseudoShape {
    fn new(op: &'static str, a: &Tensor, absolute: Option<&Tensor>, relative: Option<&Tensor>, hermitian: bool, inverse: bool) -> Result<Self> {
        let dtype = real(op, &[a])?;
        if a.ndim() < 2 { return Err(invalid(op, "input requires at least two dimensions")); }
        let rows = a.shape()[a.ndim() - 2];
        let columns = a.shape()[a.ndim() - 1];
        if hermitian && rows != columns { return Err(invalid(op, "hermitian input must be square")); }
        let input_batch = a.shape()[..a.ndim() - 2].to_vec();
        let mut batch = input_batch.clone();
        let absolute = absolute.map(|value| Tolerance::read(op, value, inverse)).transpose()?;
        let relative = relative.map(|value| Tolerance::read(op, value, inverse)).transpose()?;
        for tolerance in [&absolute, &relative].into_iter().flatten() {
            batch = broadcast_shapes(op, &batch, &tolerance.shape)?;
        }
        let width = rows.min(columns);
        for shape in [&[rows, columns][..], &[width, width], &[rows, width], &[width, columns]] { capacity(op, shape)?; }
        Ok(Self { rows, columns, input_batch, batch, absolute, relative, dtype })
    }
    fn cutoff(&self, index: usize, largest: f64) -> f64 {
        let mut absolute = self.absolute.as_ref().map_or(0., |value| value.at(index, &self.batch));
        let default = self.rows.max(self.columns) as f64 * if self.dtype == DType::F32 { f32::EPSILON as f64 } else { f64::EPSILON };
        let relative = self.relative.as_ref().map_or(if absolute > 0. { 0. } else { default }, |value| value.at(index, &self.batch));
        let narrow = self.dtype == DType::F32 && self.relative.as_ref().is_some_and(|value| value.dtype != DType::F64);
        let scaled = if narrow {
            (self.relative.as_ref().unwrap().narrow(index, &self.batch) * (largest as f32)) as f64
        } else { relative * largest };
        if narrow && self.absolute.as_ref().is_none_or(|value| value.dtype != DType::F64) {
            absolute = self.absolute.as_ref().map_or(0., |value| value.narrow(index, &self.batch) as f64);
        }
        if absolute.is_nan() || scaled.is_nan() { f64::NAN } else { absolute.max(scaled) }
    }
    fn output(&self) -> Vec<usize> {
        let mut shape = self.batch.clone(); shape.extend([self.columns, self.rows]); shape
    }
}

fn pseudo_factors(op: &'static str, data: &[f64], shape: &PseudoShape, hermitian: bool) -> Result<Vec<SvdFactor>> {
    let (rows, columns) = (shape.rows, shape.columns);
    let matrices = checked_numel(op, &shape.input_batch)?;
    let mut factors = Vec::new();
    if rows.min(columns) == 0 { return Ok(factors); }
    factors.try_reserve_exact(matrices).map_err(|_| invalid(op, "cannot allocate inverse factors"))?;
    for matrix in 0..matrices {
        let input = &data[matrix * rows * columns..(matrix + 1) * rows * columns];
        let mut factor = if hermitian {
            let mut symmetric = vec![0.; rows * rows];
            for row in 0..rows { for column in row..rows {
                let value = input[column * rows + row];
                symmetric[row * rows + column] = value; symmetric[column * rows + row] = value;
            } }
            let eigen = symmetric_eigen(symmetric, rows)?;
            SvdFactor { vh: transpose(&eigen.vectors, rows, rows), u: eigen.vectors, s: eigen.values }
        } else if rows >= columns { tall_svd(input.to_vec(), rows, columns)? } else {
            let transposed = tall_svd(transpose(input, rows, columns), columns, rows)?;
            SvdFactor { u: transpose(&transposed.vh, rows, rows), s: transposed.s, vh: transpose(&transposed.u, columns, rows) }
        };
        if shape.dtype == DType::F32 {
            // Threshold comparisons use the singular/eigenvalue output dtype.
            for value in &mut factor.s { *value = *value as f32 as f64; }
        }
        factors.push(factor);
    }
    Ok(factors)
}

fn dense_product(a: &[f64], b: &[f64], rows: usize, shared: usize, columns: usize) -> Vec<f64> {
    let mut output = vec![0.; rows * columns];
    for row in 0..rows { for column in 0..columns { for k in 0..shared {
        output[row * columns + column] += a[row * shared + k] * b[k * columns + column];
    } } }
    output
}

fn pseudo_adjoint(a: &[f64], p: &[f64], g: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    if rows > columns {
        return transpose(&pseudo_adjoint(&transpose(a, rows, columns), &transpose(p, columns, rows), &transpose(g, columns, rows), columns, rows), columns, rows);
    }
    let pt = transpose(p, columns, rows);
    let gt = transpose(g, columns, rows);
    let ptg = dense_product(&pt, g, rows, columns, rows);
    let mut result = dense_product(&ptg, &pt, rows, rows, columns);
    for value in &mut result { *value = -*value; }
    let ap = dense_product(a, p, rows, columns, rows);
    let gtp = dense_product(&gt, p, rows, columns, rows);
    let apgtp = dense_product(&ap, &gtp, rows, rows, rows);
    let residual: Vec<_> = gtp.iter().zip(&apgtp).map(|(x, y)| x - y).collect();
    let second = dense_product(&residual, &pt, rows, rows, columns);
    let ptp = dense_product(&pt, p, rows, columns, rows);
    let third = dense_product(&ptp, &gt, rows, rows, columns);
    let third_p = dense_product(&third, p, rows, columns, rows);
    let projected = dense_product(&third_p, a, rows, rows, columns);
    for i in 0..result.len() { result[i] += second[i] + third[i] - projected[i]; }
    result
}

/// Moore-Penrose pseudoinverse with the real constant-rank analytic adjoint.
/// Tolerance tensors broadcast with the input batch and remain nondifferentiable.
/// When nonzero singular values are discarded, this follows Torch's backward
/// convention, which can differ from the truncated forward map's derivative.
pub fn pinv(a: &Tensor, absolute: Option<&Tensor>, relative: Option<&Tensor>, hermitian: bool) -> Result<Tensor> {
    let op = "linalg.pinv";
    let shape = PseudoShape::new(op, a, absolute, relative, hermitian, true)?;
    let output_shape = shape.output();
    capacity(op, &output_shape)?;
    let matrices = checked_numel(op, &shape.batch)?;
    let data = values(a);
    let factors = pseudo_factors(op, &data, &shape, hermitian)?;
    let mut output = filled(op, &output_shape, 0.0f64)?;
    let (rows, columns) = (shape.rows, shape.columns);
    let width = rows.min(columns);
    if width > 0 { for matrix in 0..matrices {
        let source = batch_index(matrix, &shape.batch, &shape.input_batch);
        let factor = &factors[source];
        let largest = factor.s.iter().map(|x| x.abs()).fold(0., f64::max);
        let cutoff = shape.cutoff(matrix, largest);
        for k in 0..width {
            let inverse = if factor.s[k].abs() > cutoff { 1. / factor.s[k] } else { 0. };
            for row in 0..columns { for column in 0..rows {
                output[matrix * columns * rows + row * rows + column] += factor.vh[k * columns + row] * inverse * factor.u[column * width + k];
            } }
        }
    } }
    let input_shape = a.shape().to_vec();
    let dtype = shape.dtype;
    let result = tensor(output.clone(), &output_shape, dtype)?;
    Ok(result.record_fn(vec![a.clone()], move |gradient| {
        let g = values(gradient);
        let mut adjoint = filled(op, &input_shape, 0.0f64).unwrap();
        if width > 0 { for matrix in 0..matrices {
            let source = batch_index(matrix, &shape.batch, &shape.input_batch);
            let input_range = source * rows * columns..(source + 1) * rows * columns;
            let output_range = matrix * rows * columns..(matrix + 1) * rows * columns;
            let current = pseudo_adjoint(&data[input_range.clone()], &output[output_range.clone()], &g[output_range], rows, columns);
            for (dst, value) in adjoint[input_range].iter_mut().zip(current) { *dst += value; }
        } }
        vec![tensor(adjoint, &input_shape, dtype).unwrap()]
    }).save_output_version())
}

/// Numerical matrix rank; no differentiable output or tolerance adjoint.
pub fn matrix_rank(a: &Tensor, absolute: Option<&Tensor>, relative: Option<&Tensor>, hermitian: bool) -> Result<Tensor> {
    let op = "linalg.matrix_rank";
    let shape = PseudoShape::new(op, a, absolute, relative, hermitian, false)?;
    let matrices = capacity(op, &shape.batch)?;
    let data = values(a);
    let factors = pseudo_factors(op, &data, &shape, hermitian)?;
    let mut output = filled(op, &shape.batch, 0i64)?;
    if shape.rows.min(shape.columns) > 0 { for matrix in 0..matrices {
        let source = batch_index(matrix, &shape.batch, &shape.input_batch);
        let singular = &factors[source].s;
        let largest = singular.iter().map(|x| x.abs()).fold(0., f64::max);
        let cutoff = shape.cutoff(matrix, largest);
        output[matrix] = singular.iter().filter(|value| value.abs() > cutoff).count() as i64;
    } }
    Tensor::from_vec_i64(output, &shape.batch)
}

/// Native Jacobi-SVD minimum-norm least squares with CPU-default result fields.
/// This is not a LAPACK gelsy implementation. Positive cutoffs select constant
/// rank; singular-value and residual metadata follow the default empty fields.
/// Backward follows Torch's default-cutoff pseudoinverse convention even when
/// rcond truncates the forward solution, not that truncated map's derivative.
pub fn lstsq(a: &Tensor, b: &Tensor, rcond: Option<f64>) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let op = "linalg.lstsq";
    let dtype = real(op, &[a, b])?;
    if rcond.is_some_and(|value| !value.is_finite() || value <= 0. || value >= 1.) {
        return Err(invalid(op, "rcond must be None or strictly between zero and one; other LAPACK driver conventions are unsupported"));
    }
    let relative = rcond.map(|value| Tensor::from_vec_f64(vec![value], &[])).transpose()?;
    let shape = PseudoShape::new(op, a, None, relative.as_ref(), false, false)?;
    let (rows, columns) = (shape.rows, shape.columns);
    let vector = b.ndim() + 1 == a.ndim() && b.shape() == &a.shape()[..a.ndim() - 1];
    if !vector && (b.ndim() != a.ndim() || b.shape()[b.ndim() - 2] != rows) {
        return Err(invalid(op, "matrix RHS must match input rank and row count; vector RHS must match the input batch and row shape"));
    }
    let rhs = if vector { 1 } else { b.shape()[b.ndim() - 1] };
    let b_batch = b.shape()[..b.ndim() - if vector { 1 } else { 2 }].to_vec();
    let batch = broadcast_shapes(op, &shape.input_batch, &b_batch)?;
    let matrices = checked_numel(op, &batch)?;
    let input_matrices = capacity(op, &shape.input_batch)?;
    let mut output_shape = batch.clone(); output_shape.push(columns);
    if !vector { output_shape.push(rhs); }
    let output_count = capacity(op, &output_shape)?;
    for working in [&[columns, rhs][..], &[rows, rhs]] { capacity(op, working)?; }
    let av = values(a);
    let bv = values(b);
    let factors = pseudo_factors(op, &av, &shape, false)?;
    let mut inverse = filled(op, a.shape(), 0.0f64)?;
    let mut backward_inverse = filled(op, a.shape(), 0.0f64)?;
    let mut ranks = filled(op, &shape.input_batch, 0i64)?;
    let width = rows.min(columns);
    if width > 0 { for matrix in 0..input_matrices {
        let factor = &factors[matrix];
        let largest = factor.s.iter().copied().fold(0., f64::max);
        let cutoff = shape.cutoff(matrix, largest);
        let default_cutoff = rows.max(columns) as f64 * if dtype == DType::F32 { f32::EPSILON as f64 } else { f64::EPSILON } * largest;
        for k in 0..width {
            let selected = factor.s[k] > cutoff;
            let backward_selected = factor.s[k] > default_cutoff;
            if selected { ranks[matrix] += 1; }
            if !selected && !backward_selected { continue; }
            for row in 0..columns { for column in 0..rows {
                let index = matrix * columns * rows + row * rows + column;
                let value = factor.vh[k * columns + row] / factor.s[k] * factor.u[column * width + k];
                if selected { inverse[index] += value; }
                if backward_selected { backward_inverse[index] += value; }
            } }
        }
    } }
    let mut output = filled(op, &output_shape, 0.0f64)?;
    if output_count > 0 { for matrix in 0..matrices {
        let ai = batch_index(matrix, &batch, &shape.input_batch);
        let bi = batch_index(matrix, &batch, &b_batch);
        let current = dense_product(&inverse[ai * columns * rows..(ai + 1) * columns * rows], &bv[bi * rows * rhs..(bi + 1) * rows * rhs], columns, rows, rhs);
        output[matrix * columns * rhs..(matrix + 1) * columns * rhs].copy_from_slice(&current);
    } }
    let (a_shape, b_shape) = (a.shape().to_vec(), b.shape().to_vec());
    let rank = Tensor::from_vec_i64(ranks, &shape.input_batch)?;
    let solution = tensor(output, &output_shape, dtype)?.record_fn(vec![a.clone(), b.clone()], move |gradient| {
        let g = values(gradient);
        let mut da = filled(op, &a_shape, 0.0f64).unwrap();
        let mut db = filled(op, &b_shape, 0.0f64).unwrap();
        if output_count > 0 { for matrix in 0..matrices {
            let ai = batch_index(matrix, &batch, &shape.input_batch);
            let bi = batch_index(matrix, &batch, &b_batch);
            let a_range = ai * rows * columns..(ai + 1) * rows * columns;
            let b_range = bi * rows * rhs..(bi + 1) * rows * rhs;
            let p = &backward_inverse[a_range.clone()];
            let current_g = &g[matrix * columns * rhs..(matrix + 1) * columns * rhs];
            let b_adjoint = dense_product(&transpose(p, columns, rows), current_g, rows, columns, rhs);
            let p_adjoint = dense_product(current_g, &transpose(&bv[b_range.clone()], rows, rhs), columns, rhs, rows);
            let a_adjoint = pseudo_adjoint(&av[a_range.clone()], p, &p_adjoint, rows, columns);
            for (dst, value) in da[a_range].iter_mut().zip(a_adjoint) { *dst += value; }
            for (dst, value) in db[b_range].iter_mut().zip(b_adjoint) { *dst += value; }
        } }
        vec![tensor(da, &a_shape, dtype).unwrap(), tensor(db, &b_shape, dtype).unwrap()]
    }).save_output_version();
    let residuals = tensor(Vec::new(), &[0], dtype)?.record_fn_optional(vec![a.clone(), b.clone()], |_| vec![None, None]).save_tensor_versions(&[&solution]);
    Ok((solution, residuals, rank, tensor(Vec::new(), &[0], dtype)?))
}

fn lu_factor_adjoint(packed: &[f64], swaps: &[i64], gradient: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    let width = rows.min(columns);
    let mut lower = vec![0.; width * width];
    let mut upper = lower.clone();
    let mut gl = lower.clone();
    let mut gu = lower.clone();
    for row in 0..width { for column in 0..width {
        let index = row * width + column;
        lower[index] = if row == column { 1. } else if row > column { packed[row * columns + column] } else { 0. };
        if row > column { gl[index] = gradient[row * columns + column]; }
        if row <= column { upper[index] = packed[row * columns + column]; gu[index] = gradient[row * columns + column]; }
    } }
    let l = Lu { data: lower.clone(), swaps: (0..width).collect(), n: width };
    let u = Lu { data: upper.clone(), swaps: (0..width).collect(), n: width };
    let mut result = vec![0.; rows * columns];
    if rows > width {
        let tail = rows - width;
        let lower_gradient = &gradient[width * columns..];
        let solved = transpose(&u.solve(&transpose(lower_gradient, tail, width), tail, false), width, tail);
        result[width * columns..].copy_from_slice(&solved);
        for row in 0..width { for column in 0..width { for i in 0..tail {
            gu[row * width + column] -= packed[(width + i) * columns + row] * solved[i * width + column];
        } } }
    }
    if columns > width {
        let tail = columns - width;
        let mut right_gradient = vec![0.; width * tail];
        for row in 0..width { right_gradient[row * tail..(row + 1) * tail].copy_from_slice(&gradient[row * columns + width..(row + 1) * columns]); }
        let solved = l.solve(&right_gradient, tail, true);
        for row in 0..width {
            result[row * columns + width..(row + 1) * columns].copy_from_slice(&solved[row * tail..(row + 1) * tail]);
            for column in 0..width { for i in 0..tail {
                gl[row * width + column] -= solved[row * tail + i] * packed[column * columns + width + i];
            } }
        }
    }
    let ltgl = dense_product(&transpose(&lower, width, width), &gl, width, width, width);
    let guut = dense_product(&gu, &transpose(&upper, width, width), width, width, width);
    let middle: Vec<_> = (0..width * width).map(|i| if i / width > i % width { ltgl[i] } else { guut[i] }).collect();
    let left = l.solve(&middle, width, true);
    let leading = transpose(&u.solve(&transpose(&left, width, width), width, false), width, width);
    for row in 0..width { result[row * columns..row * columns + width].copy_from_slice(&leading[row * width..(row + 1) * width]); }
    for row in (0..width).rev() {
        let pivot = swaps[row] as usize - 1;
        for column in 0..columns { result.swap(row * columns + column, pivot * columns + column); }
    }
    result
}

fn lu_factor_impl(a: &Tensor, with_info: bool, check_errors: bool) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    let op = "linalg.lu_factor";
    let dtype = real(op, &[a])?;
    if a.ndim() < 2 { return Err(invalid(op, "LU factorization requires matrix dimensions")); }
    let rows = a.shape()[a.ndim() - 2];
    let columns = a.shape()[a.ndim() - 1];
    let width = rows.min(columns);
    let batch = &a.shape()[..a.ndim() - 2];
    let matrices = checked_numel(op, batch)?;
    let mut pivot_shape = batch.to_vec(); pivot_shape.push(width);
    for shape in [&pivot_shape[..], &[width, width], &[rows, width], &[width, columns]] { capacity(op, shape)?; }
    let mut pivots = filled(op, &pivot_shape, 0i64)?;
    let mut info = if with_info { capacity(op, batch)?; Some(filled(op, batch, 0i64)?) } else { None };
    let mut packed = values(a);
    if width > 0 { for matrix in 0..matrices {
        let range = matrix * rows * columns..(matrix + 1) * rows * columns;
        let (data, swaps, failure) = lu_factor_data(packed[range.clone()].to_vec(), rows, columns, op)?;
        if check_errors && failure != 0 { return Err(invalid(op, format!("singular coefficient matrix at pivot {failure}"))); }
        packed[range].copy_from_slice(&data);
        for (column, pivot) in swaps.into_iter().enumerate() { pivots[matrix * width + column] = pivot as i64 + 1; }
        if let Some(info) = &mut info { info[matrix] = failure as i64; }
    } }
    let out = tensor(packed, a.shape(), dtype)?;
    let saved = out.detach_copy();
    let saved_pivots = pivots.clone();
    let pivots = Tensor::from_vec_i64(pivots, &pivot_shape)?;
    let info = info.map(|values| Tensor::from_vec_i64(values, batch)).transpose()?;
    let shape = a.shape().to_vec();
    let out = out.record_fn_saved(vec![a.clone()], &[], true, move |gradient| {
        let mut result = filled(op, &shape, 0.0f64).unwrap();
        if width > 0 {
            let packed = values(&saved);
            let g = values(gradient);
            for matrix in 0..matrices {
                let range = matrix * rows * columns..(matrix + 1) * rows * columns;
                let adjoint = lu_factor_adjoint(&packed[range.clone()], &saved_pivots[matrix * width..(matrix + 1) * width], &g[range.clone()], rows, columns);
                result[range].copy_from_slice(&adjoint);
            }
        }
        vec![tensor(result, &shape, dtype).unwrap()]
    }).save_tensor_versions(&[&pivots]);
    Ok((out, pivots, info))
}

/// Rectangular partial-pivot LU with one-based i64 pivots. Pivot choices are
/// locally constant; derivatives require a full-rank, nonsingular factor.
pub fn lu_factor(a: &Tensor) -> Result<(Tensor, Tensor)> {
    let (factor, pivots, _) = lu_factor_impl(a, false, true)?;
    Ok((factor, pivots))
}

/// Packed LU, one-based i64 pivots and first-zero-pivot i64 status.
pub fn lu_factor_ex(a: &Tensor, check_errors: bool) -> Result<(Tensor, Tensor, Tensor)> {
    let (factor, pivots, info) = lu_factor_impl(a, true, check_errors)?;
    Ok((factor, pivots, info.unwrap()))
}

/// P,L,U satisfying A=P*L*U. Permutation values are nondifferentiable;
/// factor gradients retain dependencies on all three returned storages.
pub fn lu(a: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let op = "linalg.lu";
    let dtype = real(op, &[a])?;
    if a.ndim() < 2 { return Err(invalid(op, "LU requires matrix dimensions")); }
    let rank = a.ndim();
    let (rows, columns) = (a.shape()[rank - 2], a.shape()[rank - 1]);
    let width = rows.min(columns);
    let batch = &a.shape()[..rank - 2];
    let mut p_shape = batch.to_vec(); p_shape.extend([rows, rows]);
    capacity(op, &p_shape)?;
    capacity(op, &[rows, width])?;
    let (packed, pivots, _) = lu_factor_impl(a, false, false)?;
    let mut identity = filled(op, &[rows, width], 0.0f64)?;
    for i in 0..width { identity[i * width + i] = 1.; }
    let lower = packed.narrow(rank - 1, 0, width)?.tril(-1)?.add(&tensor(identity, &[rows, width], dtype)?)?;
    let upper = packed.narrow(rank - 2, 0, width)?.triu(0)?;
    let mut permutation = filled(op, &p_shape, 0.0f64)?;
    if rows > 0 {
        let indices = pivots.to_vec_i64();
        for matrix in 0..checked_numel(op, batch)? {
            let mut order: Vec<_> = (0..rows).collect();
            for column in 0..width { order.swap(column, indices[matrix * width + column] as usize - 1); }
            for (column, row) in order.into_iter().enumerate() { permutation[matrix * rows * rows + row * rows + column] = 1.; }
        }
    }
    let p = tensor(permutation, &p_shape, dtype)?;
    packed.save_tensor_versions(&[&p, &lower, &upper]);
    Ok((p, lower, upper))
}

/// Solve from independent packed LU factors and one-based i64 pivots, allowing
/// broadcast matrix RHS batches. Singular factors have undefined results/VJPs.
pub fn lu_solve(packed: &Tensor, pivots: &Tensor, b: &Tensor, left: bool, adjoint: bool) -> Result<Tensor> {
    let op = "linalg.lu_solve";
    let dtype = real(op, &[packed, b])?;
    check(op, &[pivots], DType::I64)?;
    capacity(op, pivots.shape())?;
    let shape = solve_shape(packed, b, left, false, op)?;
    let (n, columns) = (shape.n, shape.columns);
    if pivots.shape() != &packed.shape()[..packed.ndim() - 1] { return Err(invalid(op, "pivots must match LU batch and row dimensions")); }
    let indices = pivots.to_vec_i64();
    if indices.iter().any(|&value| value < 1 || value as u64 > n as u64) { return Err(invalid(op, "one-based pivots must be between 1 and the matrix size")); }
    let data = values(packed);
    if data.iter().any(|value| !value.is_finite()) { return Err(invalid(op, "packed factors must be finite")); }
    let matrices = checked_numel(op, &shape.a_batch)?;
    let count = checked_numel(op, &shape.batch)?;
    let transposed = adjoint != !left;
    let mut factors = Vec::new();
    if n > 0 {
        factors.try_reserve_exact(matrices).map_err(|_| invalid(op, "cannot allocate LU factors"))?;
        for matrix in 0..matrices {
            factors.push(Lu { data: data[matrix * n * n..(matrix + 1) * n * n].to_vec(),
                swaps: indices[matrix * n..(matrix + 1) * n].iter().map(|&value| value as usize - 1).collect(), n });
        }
    }
    let mut output = filled(op, &shape.output, 0.0f64)?;
    if n > 0 && columns > 0 {
        let rhs = values(b);
        for matrix in 0..count {
            let ai = batch_index(matrix, &shape.batch, &shape.a_batch);
            let bi = batch_index(matrix, &shape.batch, &shape.b_batch);
            let source = &rhs[bi * n * columns..(bi + 1) * n * columns];
            let rhs = if left { source.to_vec() } else { transpose(source, columns, n) };
            let solved = factors[ai].solve(&rhs, columns, transposed);
            let result = if left { solved } else { transpose(&solved, n, columns) };
            output[matrix * n * columns..(matrix + 1) * n * columns].copy_from_slice(&result);
        }
    }
    let out = tensor(output, &shape.output, dtype)?;
    let saved = out.detach_copy();
    let packed_shape = packed.shape().to_vec();
    let b_shape = b.shape().to_vec();
    Ok(out.record_fn_saved(vec![packed.clone(), b.clone()], &[0], true, move |gradient| {
        let mut da = filled(op, &packed_shape, 0.0f64).unwrap();
        let mut db = filled(op, &b_shape, 0.0f64).unwrap();
        if n > 0 && columns > 0 {
            let solution = values(&saved);
            let g = values(gradient);
            for matrix in 0..count {
                let ai = batch_index(matrix, &shape.batch, &shape.a_batch);
                let bi = batch_index(matrix, &shape.batch, &shape.b_batch);
                let range = matrix * n * columns..(matrix + 1) * n * columns;
                let gw = if left { g[range.clone()].to_vec() } else { transpose(&g[range.clone()], columns, n) };
                let x = if left { solution[range].to_vec() } else { transpose(&solution[range], columns, n) };
                let rhs = factors[ai].solve(&gw, columns, !transposed);
                let mut product = dense_product(&rhs, &transpose(&x, n, columns), n, columns, n);
                for value in &mut product { *value = -*value; }
                let mut coefficient = if transposed { transpose(&product, n, n) } else { product };
                for row in 0..n { for column in 0..n { coefficient.swap(row * n + column, factors[ai].swaps[row] * n + column); } }
                let factor = &factors[ai].data;
                for row in 0..n { for column in 0..n {
                    let mut value = 0.;
                    if row > column { for k in column..n { value += coefficient[row * n + k] * factor[column * n + k]; } }
                    else { for k in row..n { value += (if k == row { 1. } else { factor[k * n + row] }) * coefficient[k * n + column]; } }
                    da[ai * n * n + row * n + column] += value;
                } }
                let rhs = if left { rhs } else { transpose(&rhs, n, columns) };
                for (i, value) in rhs.into_iter().enumerate() { db[bi * n * columns + i] += value; }
            }
        }
        vec![tensor(da, &packed_shape, dtype).unwrap(), tensor(db, &b_shape, dtype).unwrap()]
    }).save_tensor_versions(&[pivots]))
}
