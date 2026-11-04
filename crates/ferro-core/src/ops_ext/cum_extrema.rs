//! CPU cumulative extrema with exact typed values and latest-index ties.
use crate::{DType, Error, Result, Tensor};
use crate::autograd::ForwardOp;
use super::indexed::IndexMap;

pub(crate) fn scan(input: &Tensor, dim: usize, maximum: bool) -> Result<(Tensor, Tensor)> {
    let op = if maximum { "cummax" } else { "cummin" };
    if dim >= input.ndim().max(1) {
        return Err(Error::InvalidShape { op, msg: format!("dim {dim} out of range for rank {}", input.ndim()) });
    }
    if !matches!(input.dtype(), DType::Bool | DType::I64 | DType::F32 | DType::F64) {
        return Err(Error::Unsupported { op, msg: "cumulative extrema require bool, int64, float32 or float64 input".into() });
    }
    crate::typed_math::check(op, &[input], input.dtype())?;
    if input.shape().iter().map(|size| (*size).max(1)).product::<usize>() > isize::MAX as usize {
        return Err(Error::InvalidShape { op, msg: "cumulative extrema layout exceeds signed addressable capacity".into() });
    }
    let (outer, width, inner) = if input.ndim() == 0 { (1, 1, 1) }
        else { (input.shape()[..dim].iter().product(), input.shape()[dim], input.shape()[dim + 1..].iter().product()) };
    let mut indices = crate::typed_math::filled(op, input.shape(), 0i64)?;
    macro_rules! select {
        ($values:expr, $ctor:ident, $zero:expr, $nan:expr) => {{
            let values = $values;
            let mut output = crate::typed_math::filled(op, input.shape(), $zero)?;
            if input.numel() != 0 { for group in 0..outer { for lane in 0..inner {
                let base = group * width * inner + lane;
                let mut selected = values[base];
                let mut index = 0i64;
                for position in 0..width {
                    let at = base + position * inner;
                    let value = values[at];
                    if ($nan)(value) || !($nan)(selected) && if maximum { value >= selected } else { value <= selected } {
                        selected = value; index = position as i64;
                    }
                    output[at] = selected; indices[at] = index;
                }
            } } }
            Tensor::$ctor(output, input.shape())?
        }};
    }
    let values = match input.dtype() {
        DType::F32 => select!(input.to_vec(), from_vec, 0.0f32, |value: f32| value.is_nan()),
        DType::F64 => select!(input.to_vec_f64(), from_vec_f64, 0.0f64, |value: f64| value.is_nan()),
        DType::I64 => select!(input.to_vec_i64(), from_vec_i64, 0i64, |_: i64| false),
        DType::Bool => select!(input.to_vec_bool(), from_vec_bool, false, |_: bool| false),
        _ => unreachable!("validated cumulative extrema dtype"),
    };
    Ok((values, Tensor::from_vec_i64(indices, input.shape())?))
}

impl Tensor {
    pub fn cummax(&self, dim: usize) -> Result<(Tensor, Tensor)> { self.cumulative_extrema(dim, true) }
    pub fn cummin(&self, dim: usize) -> Result<(Tensor, Tensor)> { self.cumulative_extrema(dim, false) }

    pub(crate) fn cumulative_extrema(&self, dim: usize, maximum: bool) -> Result<(Tensor, Tensor)> {
        let (values, indices) = scan(self, dim, maximum)?;
        if !matches!(self.dtype(), DType::F32 | DType::F64) { return Ok((values, indices)); }
        let saved = indices.detach_copy();
        let shape = self.shape().to_vec();
        let values = values.record_fn_forward_optional(vec![self.clone(), indices.clone()], ForwardOp::CumExtrema { dim, maximum }, move |g| {
            vec![Some(Tensor::indexed_scatter(g, &saved, &shape, IndexMap::Axis(dim)).expect("validated cumulative extrema adjoint")), None]
        });
        Ok((values, indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    include!("cum_extrema_goldens.rs");
    #[test]
    fn exact_integer_and_boolean_indices() {
        let x = Tensor::from_vec_i64(vec![i64::MAX - 1, i64::MAX, i64::MAX, i64::MIN], &[4]).unwrap();
        let (v, i) = scan(&x, 0, true).unwrap();
        assert_eq!(v.to_vec_i64(), vec![i64::MAX - 1, i64::MAX, i64::MAX, i64::MAX]);
        assert_eq!(i.to_vec_i64(), vec![0, 1, 2, 2]);
        let x = Tensor::from_vec_bool(vec![false, true, true, false], &[4]).unwrap();
        let (v, i) = scan(&x, 0, true).unwrap();
        assert_eq!(v.to_vec_bool(), vec![false, true, true, true]);
        assert_eq!(i.to_vec_i64(), vec![0, 1, 2, 2]);
        assert_eq!(scan(&x, 0, false).unwrap().1.to_vec_i64(), vec![0, 0, 0, 3]);
    }
    #[test]
    fn latest_signed_zero_and_nan_keep_value_bits() {
        for maximum in [false, true] {
            let x = Tensor::from_vec_f64(vec![0., -0., 0., -0.], &[4]).unwrap();
            let (v, i) = scan(&x, 0, maximum).unwrap();
            assert_eq!(v.to_vec_f64().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), x.to_vec_f64().iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            assert_eq!(i.to_vec_i64(), vec![0, 1, 2, 3]);
            let first = f64::from_bits(0x7ff8000000000011); let last = f64::from_bits(0x7ff8000000000033);
            let x = Tensor::from_vec_f64(vec![1., first, 2., last, f64::NEG_INFINITY], &[5]).unwrap();
            let (v, i) = scan(&x, 0, maximum).unwrap();
            assert_eq!(i.to_vec_i64(), vec![0, 1, 1, 3, 3]);
            assert_eq!(v.to_vec_f64().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), [1., first, first, last, last].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
    }
    #[test]
    fn strided_axis_coordinates_are_logical() {
        let x = Tensor::from_vec(vec![3., 1., 3., 0., 4., 4.], &[2, 3]).unwrap().transpose(0, 1).unwrap();
        let (v, i) = scan(&x, 0, true).unwrap();
        assert_eq!(v.to_vec(), vec![3., 0., 3., 4., 3., 4.]);
        assert_eq!(i.to_vec_i64(), vec![0, 0, 0, 1, 2, 2]);
        let (v, i) = scan(&x, 0, false).unwrap();
        assert_eq!(v.to_vec(), vec![3., 0., 1., 0., 1., 0.]);
        assert_eq!(i.to_vec_i64(), vec![0, 0, 1, 0, 1, 0]);
    }
    #[test]
    fn scalar_and_huge_empty_axes_skip_loops() {
        for maximum in [false, true] {
            let (v, i) = scan(&Tensor::scalar(3.), 0, maximum).unwrap();
            assert_eq!(v.shape(), &[]); assert_eq!(v.item(), 3.); assert_eq!(i.to_vec_i64(), vec![0]);
            for shape in [[0, 1usize << 60], [1usize << 60, 0]] { for dim in [0, 1] {
                let x = Tensor::from_vec(vec![], &shape).unwrap();
                let (v, i) = scan(&x, dim, maximum).unwrap();
                assert_eq!(v.shape(), &shape); assert_eq!(i.shape(), &shape); assert_eq!(i.numel(), 0);
            } }
        }
    }
    fn leaf(values: &[f32], shape: &[usize]) -> Tensor { Tensor::from_vec(values.to_vec(), shape).unwrap().requires_grad_(true).unwrap() }

    #[test]
    fn selected_gradient_and_live_seed_have_real_indexed_transposes() {
        for maximum in [false, true] {
            let x = leaf(&[2., 1., 2., 0.], &[4]);
            let seed = leaf(&[1., 3., 5., 7.], &[4]);
            let (values, _) = x.cumulative_extrema(0, maximum).unwrap();
            let first = values.vjp_wrt_optional(&[&x], &seed, true).unwrap().remove(0).unwrap();
            assert_eq!(first.to_vec(), if maximum { vec![4., 0., 12., 0.] } else { vec![1., 8., 0., 7.] });
            assert!(first.requires_grad() && !first.depends_on(&x));
            let query = leaf(&[11., 13., 17., 19.], &[4]);
            let second = first.vjp_wrt_optional(&[&x, &seed], &query, true).unwrap();
            assert!(second[0].is_none());
            assert_eq!(second[1].as_ref().unwrap().to_vec(), if maximum { vec![11., 11., 17., 17.] } else { vec![11., 13., 13., 19.] });
            assert!(x.grad().is_none() && seed.grad().is_none());
        }
        let x = Tensor::from_vec(vec![0.2, 0.7, 0.5], &[3]).unwrap();
        crate::testkit::grad_check(&[x], |xs| xs[0].cummax(0).unwrap().0.square().unwrap().sum());
    }

    #[test]
    fn ordinary_guards_input_indices_not_values_even_empty_scalar() {
        for shape in [vec![], vec![0], vec![3]] { for graph in [false, true] { for changed in 0..3 {
            let x = Tensor::ones(&shape).requires_grad_(true).unwrap();
            let (values, indices) = x.cummax(0).unwrap();
            [&x, &indices, &values][changed]._bump_version_for_test();
            assert_eq!(values.vjp_wrt_optional(&[&x], &Tensor::ones(&shape), graph).is_err(), changed != 2);
        } } }
    }

    #[test]
    fn empty_adjoint_is_original_input_not_seed_or_fake_zero() {
        for graph in [false, true] {
            let x = leaf(&[], &[2, 0]); let seed = leaf(&[], &[2, 0]);
            let y = x.cummin(1).unwrap().0;
            let result = if graph { y.vjp_wrt(&[&x], &seed, true) } else { y.vjp_wrt_preserve(&[&x], &seed) }.unwrap().remove(0);
            assert_eq!(result.id(), x.id()); assert!(result.requires_grad());
            let second = result.vjp_wrt_optional(&[&x, &seed], &Tensor::ones(&[2, 0]), false).unwrap();
            assert!(second[0].is_some()); assert!(second[1].is_none());
        }
    }

    #[test]
    fn jvp_preserves_f64_direction_and_saves_direction_and_indices() {
        for changed in 0..4 {
            let x = leaf(&[2., 1., 2., 0.], &[4]);
            let seed = Tensor::from_vec_f64(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap().requires_grad_(true).unwrap();
            let level = crate::forward_ad::DualLevel::new(&[&x], &[&seed]).unwrap();
            let (values, indices) = x.cummax(0).unwrap();
            let tangent = level.tangent(&values).unwrap().unwrap(); drop(level);
            assert_eq!(tangent.dtype(), DType::F64); assert_eq!(tangent.to_vec_f64(), vec![0.1, 0.1, 0.3, 0.3]);
            [&x, &values, &indices, &seed][changed]._bump_version_for_test();
            let result = tangent.vjp_wrt_optional(&[&x, &seed], &Tensor::from_vec_f64(vec![1.; 4], &[4]).unwrap(), false);
            assert_eq!(result.is_err(), changed == 2 || changed == 3);
            if let Ok(result) = result { assert!(result[0].is_none()); assert_eq!(result[1].as_ref().unwrap().to_vec_f64(), vec![2., 0., 2., 0.]); }
        }
    }

    #[test]
    fn empty_ordinary_backward_publishes_detached_gradient_without_self_cycle() {
        for dtype in [DType::F32, DType::F64] { for maximum in [false, true] {
            let x = Tensor::from_vec_f64(vec![], &[2, 0]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
            let weak = std::sync::Arc::downgrade(&x.0);
            let (values, indices) = x.cumulative_extrema(1, maximum).unwrap();
            values.sum().backward();
            let gradient = x.grad().unwrap();
            assert_ne!(gradient.id(), x.id()); assert!(!gradient.requires_grad());
            assert_eq!(gradient.dtype(), dtype); assert_eq!(gradient.shape(), &[2, 0]);
            drop(gradient); drop(values); drop(indices); drop(x);
            assert!(weak.upgrade().is_none(), "empty gradient publication retained its own input");
        } }
    }

}
