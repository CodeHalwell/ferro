//! CPU ordering with exact typed values, stable ties and saved index versions.
use std::cmp::Ordering;
use std::sync::Arc;
use crate::{DType, Device, Error, Result, Tensor};

fn order<T: Copy>(data: &[T], shape: &[usize], dim: usize, descending: bool, stable: bool,
    compare: impl Fn(T, T) -> Ordering) -> Result<(Vec<T>, Vec<i64>, Vec<usize>)> {
    let mut values = data.to_vec();
    let mut indices = crate::typed_math::filled("sort", shape, 0i64)?;
    let mut source = crate::typed_math::filled("sort", shape, 0usize)?;
    if data.is_empty() { return Ok((values, indices, source)); }
    if shape.is_empty() { return Ok((values, indices, source)); }
    let size = shape[dim];
    let inner: usize = shape[dim+1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    let mut permutation = crate::typed_math::filled("sort", &[size], 0usize)?;
    for batch in 0..outer { for lane in 0..inner {
        let base = batch * size * inner + lane;
        for (position, item) in permutation.iter_mut().enumerate() { *item = position; }
        let cmp = |&a: &usize, &b: &usize| {
            let value = compare(data[base+a*inner], data[base+b*inner]);
            if descending { value.reverse() } else { value }
        };
        if stable { permutation.sort_by(cmp); } else { permutation.sort_unstable_by(cmp); }
        for (position, &item) in permutation.iter().enumerate() {
            let destination = base + position * inner;
            let input = base + item * inner;
            values[destination] = data[input];
            indices[destination] = item as i64;
            source[destination] = input;
        }
    } }
    Ok((values, indices, source))
}

impl Tensor {
    /// Eager CPU F32/F64/I64/Bool sorting. Float values have first-order
    /// adjoints; higher-order queries, inference capture and forward AD reject.
    /// Stable ties retain input order, including signed zeros and NaNs. With
    /// stable=false, tied indices/adjoints have unspecified order.
    pub fn sort(&self, dim: usize, descending: bool, stable: bool) -> Result<(Tensor, Tensor)> {
        const OP: &str = "sort";
        if crate::capture::is_recording() || crate::forward_ad::is_active() {
            return Err(Error::Unsupported { op: OP, msg: "sort currently requires eager execution outside forward AD".into() });
        }
        if self.device() != Device::Cpu {
            return Err(Error::Unsupported { op: OP, msg: "sort currently supports CPU storage".into() });
        }
        if !matches!(self.dtype(), DType::F32 | DType::F64 | DType::I64 | DType::Bool) {
            return Err(Error::Unsupported { op: OP, msg: "sort supports float32, float64, int64 and bool".into() });
        }
        if dim >= self.ndim().max(1) {
            return Err(Error::InvalidShape { op: OP, msg: "sort dimension is outside the input rank".into() });
        }
        let shape = self.shape().to_vec();
        crate::typed_math::check_shape(OP, &shape)?;
        macro_rules! float_order {
            ($a:expr, $b:expr) => {
                match ($a.is_nan(), $b.is_nan()) {
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    _ => $a.partial_cmp(&$b).unwrap_or(Ordering::Equal),
                }
            };
        }
        let (values, indices, source) = match self.dtype() {
            DType::F32 => {
                let (values, indices, source) = order(&self.to_vec(), &shape, dim, descending, stable, |a, b| float_order!(a,b))?;
                (Tensor::from_vec(values, &shape)?, indices, source)
            },
            DType::F64 => {
                let (values, indices, source) = order(&self.to_vec_f64(), &shape, dim, descending, stable, |a, b| float_order!(a,b))?;
                (Tensor::from_vec_f64(values, &shape)?, indices, source)
            },
            DType::I64 => {
                let (values, indices, source) = order(&self.to_vec_i64(), &shape, dim, descending, stable, |a, b| a.cmp(&b))?;
                (Tensor::from_vec_i64(values, &shape)?, indices, source)
            },
            DType::Bool => {
                let (values, indices, source) = order(&self.to_vec_bool(), &shape, dim, descending, stable, |a, b| a.cmp(&b))?;
                (Tensor::from_vec_bool(values, &shape)?, indices, source)
            },
            _ => unreachable!(),
        };
        let indices = Tensor::from_vec_i64(indices, &shape)?;
        if !matches!(self.dtype(), DType::F32 | DType::F64) { return Ok((values, indices)); }
        let dtype = self.dtype();
        let mut values = values.record_fn(vec![self.clone()], move |g| {
            macro_rules! scatter {
                ($ty:ty, $read:ident, $make:ident) => {{
                    let mut gradient = crate::typed_math::filled(OP, &shape, 0.0 as $ty).unwrap();
                    for (index, &value) in g.$read().iter().enumerate() { gradient[source[index]] = value; }
                    Tensor::$make(gradient, &shape).unwrap()
                }};
            }
            vec![match dtype {
                DType::F32 => scatter!(f32, to_vec, from_vec),
                DType::F64 => scatter!(f64, to_vec_f64, from_vec_f64),
                _ => unreachable!(),
            }]
        });
        if let Some(inner) = Arc::get_mut(&mut values.0) {
            if let Some(op) = &mut inner.op { op.omit_saved_input_values(); }
        }
        Ok((values.save_tensor_versions(&[&indices]), indices))
    }

    pub fn argsort(&self, dim: usize, descending: bool, stable: bool) -> Result<Tensor> {
        self.sort(dim, descending, stable).map(|(_, indices)| indices)
    }
}
