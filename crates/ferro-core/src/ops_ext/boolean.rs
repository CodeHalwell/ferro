//! CPU boolean storage operations. Predicates and masks never require gradients.
//! Selection keeps floating gather adjoints; capture is rejected before
//! reading a mask so its values cannot become frozen replay constants.

use crate::{DType, Device, Error, Result, Tensor};
use crate::shape::{broadcast_shapes, checked_numel};

fn cpu(op: &'static str, tensors: &[&Tensor]) -> Result<()> {
    if crate::capture::is_recording() {
        return Err(Error::Unsupported { op, msg: "boolean operations do not support inference capture".into() });
    }
    if tensors.iter().any(|t| t.device() != Device::Cpu) {
        return Err(Error::Unsupported { op, msg: "boolean operations currently require CPU tensors".into() });
    }
    Ok(())
}

fn filled<T: Clone>(op: &'static str, len: usize, value: T) -> Result<Vec<T>> {
    if len > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err(Error::InvalidShape { op, msg: "output exceeds typed allocation capacity".into() });
    }
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|e| Error::InvalidShape { op, msg: format!("output allocation failed: {e}") })?;
    output.resize(len, value);
    Ok(output)
}

impl Tensor {
    pub fn to_bool(&self) -> Result<Tensor> {
        cpu("to_bool", &[self])?;
        Ok(self.to_dtype(DType::Bool))
    }

    fn compare(&self, other: &Tensor, kind: u8) -> Result<Tensor> {
        cpu("compare", &[self, other])?;
        let dtype = if self.dtype() == other.dtype() { self.dtype() } else { crate::typed_math::result_type(self, other)? };
        let shape = broadcast_shapes("compare", self.shape(), other.shape())?;
        let len = checked_numel("compare", &shape)?;
        let width = [dtype, self.dtype(), other.dtype()].into_iter().map(|dtype| match dtype {
            DType::Bool => 1, DType::F64 | DType::I64 => 8, _ => 4,
        }).max().expect("comparison has input dtypes");
        if len > isize::MAX as usize / width {
            return Err(Error::InvalidShape { op: "compare", msg: "broadcast materialization exceeds allocation capacity".into() });
        }
        let stride = crate::typed_math::binary_pointwise_layout(self, other, &shape, dtype);
        let a = crate::typed_math::pointwise_source(self, &shape, &stride)?;
        let b = crate::typed_math::pointwise_source(other, &shape, &stride)?;
        let a = if a.dtype() == dtype { a } else { a.to_dtype(dtype) };
        let b = if b.dtype() == dtype { b } else { b.to_dtype(dtype) };
        macro_rules! compare {
            ($a:expr, $b:expr) => {
                $a.into_iter().zip($b).map(|(x, y)| match kind {
                    0 => x == y, 1 => x != y, 2 => x < y, 3 => x <= y, 4 => x > y, _ => x >= y,
                }).collect()
            };
        }
        let values = match dtype {
            DType::I64 => compare!(a.to_vec_i64(), b.to_vec_i64()),
            DType::Bool => compare!(a.to_vec_bool(), b.to_vec_bool()),
            DType::F64 => compare!(a.to_vec_f64(), b.to_vec_f64()),
            _ => compare!(a.to_vec(), b.to_vec()),
        };
        let raw = Tensor::from_vec_bool(values, &shape)?;
        Ok(Tensor::from_parts(raw.0.storage.clone(), shape, stride, 0, Device::Cpu, false, None))
    }

    pub fn eq_tensor(&self, other: &Tensor) -> Result<Tensor> { self.compare(other, 0) }
    pub fn ne_tensor(&self, other: &Tensor) -> Result<Tensor> { self.compare(other, 1) }
    pub fn lt_tensor(&self, other: &Tensor) -> Result<Tensor> { self.compare(other, 2) }
    pub fn le_tensor(&self, other: &Tensor) -> Result<Tensor> { self.compare(other, 3) }
    pub fn gt_tensor(&self, other: &Tensor) -> Result<Tensor> { self.compare(other, 4) }
    pub fn ge_tensor(&self, other: &Tensor) -> Result<Tensor> { self.compare(other, 5) }

    fn logical(&self, other: &Tensor, kind: u8) -> Result<Tensor> {
        cpu("logical", &[self, other])?;
        let shape = broadcast_shapes("logical", self.shape(), other.shape())?;
        // Numeric truth conversion materializes an I64/F64 intermediate.
        let width = if self.dtype() == DType::Bool && other.dtype() == DType::Bool { 1 } else { 8 };
        if checked_numel("logical", &shape)? > isize::MAX as usize / width {
            return Err(Error::InvalidShape { op: "logical", msg: "broadcast materialization exceeds allocation capacity".into() });
        }
        let stride = crate::typed_math::iterator_pointwise_layout(&[self, other], &[self.strides().to_vec(), other.strides().to_vec()], &shape);
        let a = crate::typed_math::pointwise_source(self, &shape, &stride)?.to_vec_bool();
        let b = crate::typed_math::pointwise_source(other, &shape, &stride)?.to_vec_bool();
        let raw = Tensor::from_vec_bool(a.into_iter().zip(b).map(|(x, y)| match kind {
            0 => x && y, 1 => x || y, _ => x != y,
        }).collect(), &shape)?;
        Ok(Tensor::from_parts(raw.0.storage.clone(), shape, stride, 0, Device::Cpu, false, None))
    }

    pub fn logical_and(&self, other: &Tensor) -> Result<Tensor> { self.logical(other, 0) }
    pub fn logical_or(&self, other: &Tensor) -> Result<Tensor> { self.logical(other, 1) }
    pub fn logical_xor(&self, other: &Tensor) -> Result<Tensor> { self.logical(other, 2) }
    pub fn logical_not(&self) -> Result<Tensor> {
        cpu("logical_not", &[self])?;
        Tensor::from_vec_bool(self.to_vec_bool().into_iter().map(|x| !x).collect(), self.shape())
    }

    pub fn isnan(&self) -> Result<Tensor> {
        cpu("isnan", &[self])?;
        Tensor::from_vec_bool(self.to_vec_f64().into_iter().map(f64::is_nan).collect(), self.shape())
    }

    pub fn isfinite(&self) -> Result<Tensor> {
        cpu("isfinite", &[self])?;
        Tensor::from_vec_bool(self.to_vec_f64().into_iter().map(f64::is_finite).collect(), self.shape())
    }

    pub fn bool_reduce(&self, dim: Option<usize>, keepdim: bool, all: bool) -> Result<Tensor> {
        cpu("bool_reduce", &[self])?;
        let dim = if self.ndim() == 0 && dim == Some(0) { None } else { dim };
        if let Some(dim) = dim {
            if dim >= self.ndim() {
                return Err(Error::InvalidShape { op: "bool_reduce", msg: "dimension is out of range".into() });
            }
        }
        let values = self.to_vec_bool();
        let Some(dim) = dim else {
            let value = if all { values.into_iter().all(|x| x) } else { values.into_iter().any(|x| x) };
            return Tensor::from_vec_bool(vec![value], &if keepdim { vec![1; self.ndim()] } else { vec![] });
        };
        let mut shape = self.shape().to_vec();
        let inner = shape[dim + 1..].iter().product::<usize>();
        let outer = shape[..dim].iter().product::<usize>();
        let extent = shape[dim];
        if keepdim { shape[dim] = 1; } else { shape.remove(dim); }
        let mut result = filled("bool_reduce", checked_numel("bool_reduce", &shape)?, all)?;
        if result.is_empty() { return Tensor::from_vec_bool(result, &shape); }
        for o in 0..outer {
            for k in 0..extent {
                for i in 0..inner {
                    let value = values[(o * extent + k) * inner + i];
                    let slot = &mut result[o * inner + i];
                    *slot = if all { *slot && value } else { *slot || value };
                }
            }
        }
        Tensor::from_vec_bool(result, &shape)
    }

    pub fn bool_sum(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
        cpu("bool_sum", &[self])?;
        let dim = if self.ndim() == 0 && dim == Some(0) { None } else { dim };
        if self.dtype() != DType::Bool {
            return Err(Error::DtypeMismatch { op: "bool_sum", expected: DType::Bool, got: self.dtype() });
        }
        let values = self.to_vec_bool();
        let Some(dim) = dim else {
            let count = i64::try_from(values.into_iter().filter(|&x| x).count())
                .map_err(|_| Error::InvalidShape { op: "bool_sum", msg: "count exceeds int64".into() })?;
            return Tensor::from_vec_i64(vec![count], &if keepdim { vec![1; self.ndim()] } else { vec![] });
        };
        if dim >= self.ndim() {
            return Err(Error::InvalidShape { op: "bool_sum", msg: "dimension is out of range".into() });
        }
        let mut shape = self.shape().to_vec();
        let inner = shape[dim + 1..].iter().product::<usize>();
        let outer = shape[..dim].iter().product::<usize>();
        let extent = shape[dim];
        if keepdim { shape[dim] = 1; } else { shape.remove(dim); }
        let mut result = filled("bool_sum", checked_numel("bool_sum", &shape)?, 0i64)?;
        if result.is_empty() { return Tensor::from_vec_i64(result, &shape); }
        for o in 0..outer {
            for k in 0..extent {
                for i in 0..inner { result[o * inner + i] += i64::from(values[(o * extent + k) * inner + i]); }
            }
        }
        Tensor::from_vec_i64(result, &shape)
    }

    pub fn nonzero(&self) -> Result<Tensor> {
        cpu("nonzero", &[self])?;
        let values = self.to_vec_bool();
        let rows = values.iter().filter(|&&x| x).count();
        let shape = [rows, self.ndim()];
        let mut indices = filled("nonzero", checked_numel("nonzero", &shape)?, 0i64)?;
        let mut row = 0;
        for (offset, value) in values.into_iter().enumerate() {
            if !value { continue; }
            let mut remaining = offset;
            for dim in (0..self.ndim()).rev() {
                indices[row * self.ndim() + dim] = i64::try_from(remaining % self.shape()[dim])
                    .map_err(|_| Error::InvalidShape { op: "nonzero", msg: "index exceeds int64".into() })?;
                remaining /= self.shape()[dim];
            }
            row += 1;
        }
        Tensor::from_vec_i64(indices, &shape)
    }

    pub fn masked_select(&self, mask: &Tensor) -> Result<Tensor> {
        cpu("masked_select", &[self, mask])?;
        if mask.dtype() != DType::Bool {
            return Err(Error::DtypeMismatch { op: "masked_select", expected: DType::Bool, got: mask.dtype() });
        }
        let shape = broadcast_shapes("masked_select", self.shape(), mask.shape())?;
        let indices: Vec<_> = mask.broadcast_to(&shape)?.to_vec_bool().into_iter().enumerate()
            .filter_map(|(i, keep)| keep.then_some(i)).collect();
        self.expand(&shape)?.reshape(&[checked_numel("masked_select", &shape)?])?.index_select(0, &indices)
    }

    /// Boolean indexing consumes matching leading dimensions; remaining
    /// feature dimensions are retained. Scalar masks prepend a zero/one axis.
    pub fn mask_index(&self, mask: &Tensor) -> Result<Tensor> {
        cpu("mask_index", &[self, mask])?;
        if mask.dtype() != DType::Bool {
            return Err(Error::DtypeMismatch { op: "mask_index", expected: DType::Bool, got: mask.dtype() });
        }
        if mask.ndim() > self.ndim() || mask.shape() != &self.shape()[..mask.ndim()] {
            return Err(Error::ShapeMismatch { op: "mask_index", lhs: self.shape().to_vec(), rhs: mask.shape().to_vec() });
        }
        let indices: Vec<_> = mask.to_vec_bool().into_iter().enumerate().filter_map(|(i, keep)| keep.then_some(i)).collect();
        let mut shape = vec![mask.numel()];
        shape.extend_from_slice(&self.shape()[mask.ndim()..]);
        self.reshape(&shape)?.index_select(0, &indices)
    }
}
