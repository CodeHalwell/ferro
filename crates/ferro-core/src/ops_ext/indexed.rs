//! Typed CPU indexed gathers and their linear scatter adjoints.
use crate::{DType, Device, Error, Result, Tensor};
use crate::autograd::ForwardOp;

#[derive(Clone, Copy, Debug)]
pub(crate) enum IndexMap { Flat, Axis(usize), Select { axis: usize, index_dims: usize } }

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum IndexKind { Take, Gather, Index }

fn offsets(index: &Tensor, shape: &[usize], map: IndexMap) -> Result<Vec<usize>> {
    let size = crate::shape::checked_numel("indexed selection", shape)?;
    let axis = match map {
        IndexMap::Flat => None,
        IndexMap::Axis(axis) => {
            if shape.is_empty() {
                if axis != 0 || index.ndim() != 0 { return Err(Error::InvalidShape { op: "indexed selection", msg: "scalar axis indexing requires a scalar index and axis zero".into() }); }
            } else if axis >= shape.len() || index.ndim() != shape.len() || shape.iter().zip(index.shape()).enumerate().any(|(d, (&a, &b))| d != axis && b > a) {
                return Err(Error::InvalidShape { op: "indexed selection", msg: "axis index rank/shape is incompatible with the source".into() });
            }
            Some(axis)
        }
        IndexMap::Select { axis, index_dims } => {
            if axis >= shape.len() || index_dims == 0 || index.ndim() != shape.len() - 1 + index_dims {
                return Err(Error::InvalidShape { op: "index", msg: "integer array index shape/rank is incompatible with the input".into() });
            }
            for (d, &width) in shape.iter().enumerate() {
                if d != axis && width != index.shape()[if d < axis { d } else { d + index_dims - 1 }] {
                    return Err(Error::InvalidShape { op: "index", msg: "expanded integer array does not match the unselected axes".into() });
                }
            }
            Some(axis)
        }
    };
    let selected_dims = match map { IndexMap::Select { index_dims, .. } => Some(index_dims), _ => None };
    let mut coordinate = vec![0; index.ndim()];
    let mut result = Vec::new();
    result.try_reserve_exact(index.numel()).map_err(|e| Error::InvalidShape { op: "indexed selection", msg: format!("index allocation failed: {e}") })?;
    for raw in index.to_vec_i64() {
        let extent = axis.map(|d| if shape.is_empty() { 1 } else { shape[d] }).unwrap_or(size);
        let normalized = if raw < 0 && (axis.is_none() || selected_dims.is_some()) { raw as i128 + extent as i128 } else { raw as i128 };
        if normalized < 0 || normalized >= extent as i128 { return Err(Error::InvalidShape { op: "indexed selection", msg: format!("index {raw} is outside extent {extent}") }); }
        let selected = normalized as usize;
        let offset = if let Some(axis) = axis {
            shape.iter().enumerate().fold(0, |offset, (d, &width)| offset * width + if d == axis { selected } else { coordinate[if d > axis { d + selected_dims.unwrap_or(1) - 1 } else { d }] })
        } else { selected };
        result.push(offset);
        for d in (0..coordinate.len()).rev() {
            coordinate[d] += 1;
            if coordinate[d] < index.shape()[d] { break; }
            coordinate[d] = 0;
        }
    }
    Ok(result)
}


fn validate(input: &Tensor, index: &Tensor) -> Result<()> {
    if crate::capture::is_recording() { return Err(Error::Unsupported { op: "indexed selection", msg: "inference capture is not supported".into() }); }
    if input.device() != Device::Cpu || index.device() != Device::Cpu { return Err(Error::Unsupported { op: "indexed selection", msg: "typed indexed selection currently requires CPU storage".into() }); }
    if index.dtype() != DType::I64 { return Err(Error::DtypeMismatch { op: "indexed selection", expected: DType::I64, got: index.dtype() }); }
    Ok(())
}

fn selection_layout(input: &Tensor, index: &Tensor, map: IndexMap) -> Vec<usize> {
    let IndexMap::Select { axis, index_dims } = map else { return crate::shape::default_strides(index.shape()); };
    let shape = index.shape();
    if index.numel() == 0 && input.shape().iter().enumerate().all(|(d, &width)| d == axis || width == 1) {
        return crate::shape::default_strides(shape);
    }
    let mut source = input.strides()[..axis].to_vec();
    source.extend(std::iter::repeat(0).take(index_dims));
    source.extend_from_slice(&input.strides()[axis + 1..]);
    let strides = [source.as_slice(), index.strides()];
    let mut axes: Vec<_> = (0..shape.len()).rev().collect();
    for n in 1..axes.len() {
        let mut current = n;
        for previous in (0..n).rev() {
            let (left, right) = (axes[previous], axes[current]);
            let mut comparison = 0;
            for stride in strides {
                let (ls, rs) = (stride[left], stride[right]);
                if ls == 0 || rs == 0 { continue; }
                if ls < rs { comparison = -1; break; }
                if ls > rs || shape[left] > shape[right] { comparison = 1; break; }
            }
            if comparison < 0 { break; }
            if comparison > 0 { axes.swap(previous, current); current = previous; }
        }
    }
    if axes.iter().copied().eq((0..shape.len()).rev()) { return crate::shape::default_strides(shape); }
    let mut output = vec![0; shape.len()];
    let mut size = 1;
    for axis in axes { output[axis] = size; size *= shape[axis]; }
    output
}

fn physical(mut position: usize, shape: &[usize], strides: &[usize], mut offset: usize) -> usize {
    for (&width, &stride) in shape.iter().zip(strides).rev() {
        offset += position % width * stride;
        position /= width;
    }
    offset
}

fn raw_gather(input: &Tensor, index: &Tensor, positions: &[usize], map: IndexMap) -> Result<Tensor> {
    let strides = selection_layout(input, index, map);
    let storage = input.0.storage.read();
    macro_rules! gather { ($values:expr, $zero:expr, $ctor:ident) => {{
        let mut values = crate::typed_math::filled("indexed selection", index.shape(), $zero)?;
        for (logical, &position) in positions.iter().enumerate() {
            let from = physical(position, input.shape(), input.strides(), input.0.offset);
            let to = physical(logical, index.shape(), &strides, 0);
            values[to] = $values[from];
        }
        let output = Tensor::$ctor(values, index.shape())?;
        Ok(if output.strides() == strides { output }
            else { Tensor::from_parts(output.0.storage.clone(), index.shape().to_vec(), strides, 0, Device::Cpu, false, None) })
    }}; }
    match &*storage {
        crate::tensor::Storage::F32(values) => gather!(values, 0.0f32, from_vec),
        crate::tensor::Storage::F64(values) => gather!(values, 0.0f64, from_vec_f64),
        crate::tensor::Storage::I64(values) => gather!(values, 0i64, from_vec_i64),
        crate::tensor::Storage::Bool(values) => gather!(values, false, from_vec_bool),
        crate::tensor::Storage::F16(values) => gather!(values, 0u16, from_vec_f16_bits),
        crate::tensor::Storage::BF16(values) => gather!(values, 0u16, from_vec_bf16_bits),
        _ => Err(Error::Unsupported { op: "indexed selection", msg: "CPU storage is required".into() }),
    }
}

fn raw_scatter(input: &Tensor, shape: &[usize], positions: &[usize]) -> Result<Tensor> {
    macro_rules! scatter { ($data:expr, $zero:expr, $ctor:ident) => {{
        let mut values = crate::typed_math::filled("indexed scatter", shape, $zero)?;
        for (value, &position) in $data.into_iter().zip(positions) { values[position] += value; }
        Tensor::$ctor(values, shape)
    }}; }
    match input.dtype() {
        DType::F32 => scatter!(input.to_vec(), 0.0f32, from_vec),
        DType::F64 => scatter!(input.to_vec_f64(), 0.0f64, from_vec_f64),
        _ => Err(Error::Unsupported { op: "indexed scatter", msg: "scatter adjoints require floating storage".into() }),
    }
}

pub(crate) fn gather_vjp(inputs: &[Tensor], g: &Tensor, map: IndexMap, kind: IndexKind) -> Result<Vec<Option<Tensor>>> {
    let save_input = kind == IndexKind::Take;
    let strides = (save_input && inputs[0].has_dense_layout()).then(|| inputs[0].strides());
    let gradient = Tensor::indexed_scatter_saved(g, &inputs[1], inputs[0].shape(), map, kind, save_input, strides)?;
    Ok(vec![Some(gradient), None])
}

impl Tensor {
    pub fn index_tensor(&self, axis: usize, index: &Tensor) -> Result<Tensor> {
        validate(self, index)?;
        if axis >= self.ndim() || index.ndim() == 0 {
            return Err(Error::InvalidShape { op: "index", msg: "integer array selection needs an existing axis and a nonscalar index".into() });
        }
        let mut shape = self.shape()[..axis].to_vec();
        shape.extend_from_slice(index.shape());
        shape.extend_from_slice(&self.shape()[axis + 1..]);
        crate::shape::checked_numel("index", &shape)?;
        let mut expanded_shape = vec![1; axis];
        expanded_shape.extend_from_slice(index.shape());
        expanded_shape.extend(std::iter::repeat(1).take(self.ndim() - axis - 1));
        let expanded = index.view(&expanded_shape)?;
        self.indexed_gather_kind(&expanded.expand(&shape)?, IndexMap::Select { axis, index_dims: index.ndim() }, IndexKind::Index, false)
    }

    pub fn take(&self, index: &Tensor) -> Result<Tensor> { self.indexed_gather_kind(index, IndexMap::Flat, IndexKind::Take, true) }

    pub(crate) fn indexed_gather(&self, index: &Tensor, map: IndexMap, save_input: bool) -> Result<Tensor> {
        self.indexed_gather_kind(index, map, IndexKind::Gather, save_input)
    }

    pub(crate) fn indexed_gather_kind(&self, index: &Tensor, map: IndexMap, kind: IndexKind, save_input: bool) -> Result<Tensor> {
        validate(self, index)?;
        let positions = offsets(index, self.shape(), map)?;
        let output = raw_gather(self, index, &positions, map)?;
        if !matches!(self.dtype(), DType::F32 | DType::F64) { return Ok(output); }
        let shape = self.shape().to_vec();
        let strides = (kind == IndexKind::Take && self.has_dense_layout()).then(|| self.strides().to_vec());
        Ok(output.record_fn_forward_optional(vec![self.clone(), index.detach()], ForwardOp::IndexedGather { map, kind, save_input }, move |g| {
            let gradient = raw_scatter(g, &shape, &positions).expect("validated indexed scatter");
            let gradient = match &strides { Some(strides) => gradient.copy_grad_layout(strides.clone()), None => gradient };
            vec![Some(gradient), None]
        }))
    }

    pub(crate) fn indexed_scatter(input: &Tensor, index: &Tensor, shape: &[usize], map: IndexMap) -> Result<Tensor> {
        Tensor::indexed_scatter_saved(input, index, shape, map, IndexKind::Gather, false, None)
    }

    pub(crate) fn indexed_scatter_saved(input: &Tensor, index: &Tensor, shape: &[usize], map: IndexMap, kind: IndexKind, save_input: bool, strides: Option<&[usize]>) -> Result<Tensor> {
        validate(input, index)?;
        if input.shape() != index.shape() { return Err(Error::InvalidShape { op: "indexed scatter", msg: "source and index shapes must match".into() }); }
        let positions = offsets(index, shape, map)?;
        let output = raw_scatter(input, shape, &positions)?;
        let output = match strides { Some(strides) => output.copy_grad_layout(strides.to_vec()), None => output };
        let index_shape = index.shape().to_vec();
        Ok(output.record_fn_forward_optional(vec![input.clone(), index.detach()], ForwardOp::IndexedScatter { map, kind, shape: shape.to_vec(), save_input, strides: strides.map(<[usize]>::to_vec) }, move |g| {
            let values = if g.dtype() == DType::F64 {
                let values = if positions.is_empty() { Vec::new() } else { g.to_vec_f64() };
                Tensor::from_vec_f64(positions.iter().map(|&p| values[p]).collect(), &index_shape).unwrap()
            } else {
                let values = g.to_vec();
                Tensor::from_vec(positions.iter().map(|&p| values[p]).collect(), &index_shape).unwrap()
            };
            vec![Some(values), None]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::with_grad_enabled;
    use crate::forward_ad::DualLevel;

    fn leaf(values: &[f32], shape: &[usize]) -> Tensor { Tensor::from_vec(values.to_vec(), shape).unwrap().requires_grad_(true).unwrap() }
    fn index(values: &[i64], shape: &[usize]) -> Tensor { Tensor::from_vec_i64(values.to_vec(), shape).unwrap() }

    include!("../../tests/fixtures/indexed_layout_goldens.rs");

    #[test]
    fn actual_torch_array_index_layouts_cover_views_empty_and_singleton_axes() {
        fn length(shape: &[usize], strides: &[usize]) -> usize {
            if shape.contains(&0) { 0 } else { 1 + shape.iter().zip(strides).map(|(&d, &s)| (d - 1) * s).sum::<usize>() }
        }
        assert_eq!(INDEX_LAYOUT_CASES.len(), 2864);
        for &(shape, strides, axis, ishape, istrides, expected) in INDEX_LAYOUT_CASES {
            let n = length(shape, strides);
            let storage = Tensor::zeros(&[n]);
            let input = Tensor::from_parts(storage.0.storage.clone(), shape.to_vec(), strides.to_vec(), 0, Device::Cpu, false, None);
            let n = length(ishape, istrides);
            let storage = Tensor::from_vec_i64(vec![0; n], &[n]).unwrap();
            let indices = Tensor::from_parts(storage.0.storage.clone(), ishape.to_vec(), istrides.to_vec(), 0, Device::Cpu, false, None);
            let output = input.index_tensor(axis, &indices).unwrap();
            assert_eq!(output.strides(), expected, "source={shape:?}/{strides:?} axis={axis} index={ishape:?}/{istrides:?}");
        }
    }

    #[test]
    fn take_preserves_typed_values_and_logical_strided_order() {
        let ids = index(&[-1, 1, 1, 0], &[2, 2]);
        let x = Tensor::from_vec_f64(vec![1. + f64::EPSILON, 2., 3., -0.], &[2, 2]).unwrap().transpose(0, 1).unwrap();
        let y = x.take(&ids).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.to_vec_f64().iter().map(|x| x.to_bits()).collect::<Vec<_>>(), [-0.0f64, 3., 3., 1. + f64::EPSILON].iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert_eq!(Tensor::from_vec_i64(vec![i64::MAX, i64::MIN], &[2]).unwrap().take(&index(&[1, 0, 1], &[3])).unwrap().to_vec_i64(), [i64::MIN, i64::MAX, i64::MIN]);
        assert_eq!(Tensor::from_vec_bool(vec![true, false], &[2]).unwrap().take(&index(&[-1, 0], &[2])).unwrap().to_vec_bool(), [false, true]);
    }

    #[test]
    fn take_duplicate_adjoints_and_live_seed_have_real_transposed_graphs() {
        let x = leaf(&[0.2, 0.5, -0.7], &[3]);
        let ids = index(&[1, 0, 1], &[3]);
        let seed = leaf(&[2., 3., 4.], &[3]);
        let y = x.take(&ids).unwrap();
        let gradient = y.vjp_wrt_optional(&[&x], &seed, true).unwrap().remove(0).unwrap();
        assert_eq!(gradient.to_vec(), [3., 6., 0.]);
        assert!(gradient.requires_grad() && !gradient.depends_on(&x));
        let second = gradient.vjp_wrt_optional(&[&x, &seed], &leaf(&[7., 11., 13.], &[3]), true).unwrap();
        assert!(second[0].is_none());
        assert_eq!(second[1].as_ref().unwrap().to_vec(), [11., 7., 11.]);
        y.sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), [1., 2., 0.]);
    }

    #[test]
    fn take_matches_saved_input_and_index_versions_but_not_output() {
        for graph in [false, true] { for changed in 0..3 {
            let x = leaf(&[1., 2., 3.], &[3]);
            let ids = index(&[1, 0, 1], &[3]);
            let y = x.take(&ids).unwrap();
            [&x, &ids, &y][changed]._bump_version_for_test();
            let result = y.vjp_wrt_optional(&[&x], &Tensor::ones(&[3]), graph);
            assert_eq!(result.is_err(), changed != 2);
        } }
        let x = leaf(&[1., 2., 3.], &[3]);
        let ids = index(&[1, 0, 1], &[3]);
        let seed = leaf(&[2., 3., 4.], &[3]);
        let gradient = x.take(&ids).unwrap().vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
        x._bump_version_for_test();
        assert_eq!(gradient.sum().grad_wrt(&[&seed], false).unwrap()[0].to_vec(), [1., 1., 1.]);
        seed._bump_version_for_test();
        assert!(gradient.sum().grad_wrt(&[&seed], false).is_err());
        ids._bump_version_for_test();
        assert!(gradient.sum().grad_wrt(&[&seed], false).is_err());
    }

    #[test]
    fn axis_map_scatter_handles_nonleading_axes_and_duplicates() {
        let x = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let ids = index(&[2, 0, 2, 1], &[2, 2]);
        let selected = x.indexed_gather(&ids, IndexMap::Axis(1), false).unwrap();
        assert_eq!(selected.to_vec(), [3., 1., 6., 5.]);
        let g = leaf(&[2., 3., 4., 5.], &[2, 2]);
        let vjp = selected.vjp_wrt(&[&x], &g, true).unwrap().remove(0);
        assert_eq!(vjp.to_vec(), [3., 0., 2., 0., 5., 4.]);
        x._bump_version_for_test(); selected._bump_version_for_test(); g._bump_version_for_test();
        assert_eq!(vjp.sum().grad_wrt(&[&g], false).unwrap()[0].to_vec(), [1., 1., 1., 1.]);
        ids._bump_version_for_test();
        assert!(vjp.sum().grad_wrt(&[&g], false).is_err());
    }

    #[test]
    fn jvp_preserves_f64_directions_and_take_direction_dependency() {
        let x = leaf(&[1., 2., 3.], &[3]);
        let direction = Tensor::from_vec_f64(vec![0.2, 0.4, 0.7], &[3]).unwrap().requires_grad_(true).unwrap();
        let ids = index(&[1, 0, 1], &[3]);
        let level = DualLevel::new(&[&x], &[&direction]).unwrap();
        let y = x.take(&ids).unwrap();
        let tangent = level.tangent(&y).unwrap().unwrap();
        drop(level);
        assert_eq!(tangent.dtype(), DType::F64);
        assert_eq!(tangent.to_vec_f64(), [0.4, 0.2, 0.4]);
        x._bump_version_for_test(); y._bump_version_for_test();
        assert_eq!(tangent.sum().grad_wrt(&[&direction], false).unwrap()[0].to_vec_f64(), [1., 2., 0.]);
        direction._bump_version_for_test();
        assert!(tangent.sum().grad_wrt(&[&direction], false).is_err());
        ids._bump_version_for_test();
        assert!(tangent.sum().grad_wrt(&[&direction], false).is_err());
    }

    #[test]
    fn no_grad_modes_and_nonfinite_scatter_do_not_add_input_dependencies() {
        let x = leaf(&[1., 2., 3.], &[3]);
        let ids = index(&[1, 1], &[2]);
        assert!(!with_grad_enabled(false, || x.take(&ids).unwrap()).requires_grad());
        let y = x.take(&ids).unwrap();
        for value in [f32::NAN, f32::INFINITY] {
            let gradient = y.vjp_wrt(&[&x], &leaf(&[value, value], &[2]), true).unwrap().remove(0);
            let values = gradient.to_vec();
            assert_eq!(values[0], 0.); assert_eq!(values[2], 0.);
            assert!(values[1] == value || values[1].is_nan() && value.is_nan());
        }
    }

    #[test]
    fn empty_selection_avoids_reading_enormous_expanded_sources() {
        let x = leaf(&[0.2], &[1]).expand(&[1usize << 45]).unwrap();
        let ids = index(&[], &[1usize << 45, 0]);
        let output = x.take(&ids).unwrap();
        assert_eq!(output.shape(), ids.shape()); assert_eq!(output.numel(), 0);
        let tiny = x.take(&index(&[0, -1], &[2])).unwrap();
        assert_eq!(tiny.to_vec(), [0.2, 0.2]);
        let empty = leaf(&[], &[2, 0]);
        let output = empty.take(&index(&[], &[0])).unwrap();
        let gradient = output.vjp_wrt(&[&empty], &Tensor::zeros(&[0]), true).unwrap().remove(0);
        assert_eq!(gradient.shape(), &[2, 0]);
    }

    #[test]
    fn malformed_indices_reject_and_take_passes_finite_difference() {
        let x = leaf(&[0.2, -0.3, 0.7], &[3]);
        assert!(x.take(&index(&[3], &[1])).is_err());
        assert!(x.take(&index(&[-4], &[1])).is_err());
        assert!(x.take(&index(&[i64::MIN], &[1])).is_err());
        assert!(x.take(&Tensor::ones(&[1])).is_err());
        assert!(x.indexed_gather(&index(&[-1], &[1]), IndexMap::Axis(0), false).is_err());
        crate::testkit::grad_check(&[x], |inputs| inputs[0].take(&index(&[1, 0, 1], &[3])).unwrap().square().unwrap().sum());
    }

    #[test]
    fn take_adjoint_preserves_only_dense_source_layouts() {
        for double in [false, true] { for gapped in [false, true] {
            let x = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
            let x = if double { x.to_dtype_autograd(DType::F64, true).unwrap() } else { x };
            let x = if gapped { x.slice(1, 0, 3, 2).unwrap() } else { x.transpose(0, 1).unwrap() };
            let ids = index(&[1, 0, 1], &[3]);
            let seed = Tensor::ones(&[3]).to_dtype(x.dtype());
            let out = x.take(&ids).unwrap();
            let gradient = Tensor::vjp_many_wrt_optional(&[&out], &[&x], &[&seed], false).unwrap().remove(0).unwrap();
            let expected = if gapped { vec![2, 1] } else { vec![1, 3] };
            assert_eq!(gradient.strides(), expected);
            if !double {
                let gradient = out.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
                assert_eq!(gradient.strides(), expected);
            }
        } }
    }

    #[test]
    fn axis_gather_and_scatter_keep_distinct_seed_version_policies() {
        let x = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]).transpose(0, 1).unwrap();
        let ids = index(&[1, 0, 0, 1, 1, 0], &[3, 2]);
        let seed = leaf(&[2., 3., 4., 5., 6., 7.], &[3, 2]);
        let y = x.indexed_gather(&ids, IndexMap::Axis(1), true).unwrap();
        let grad = y.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
        assert_eq!(grad.strides(), &[2, 1]);
        seed._bump_version_for_test();
        assert_eq!(grad.sum().grad_wrt(&[&seed], false).unwrap()[0].to_vec(), [1.; 6]);
        let gg = leaf(&[2.; 6], &[3, 2]);
        let next = grad.vjp_wrt(&[&seed], &gg, true).unwrap().remove(0);
        gg._bump_version_for_test();
        assert!(next.sum().grad_wrt(&[&gg], false).is_err());
        x._bump_version_for_test();
        assert!(y.vjp_wrt(&[&x], &Tensor::ones(&[3, 2]), false).is_err());
    }

    #[test]
    fn integer_arrays_insert_all_index_dimensions_and_preserve_duplicates() {
        let x = leaf(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &[2, 3, 2]);
        let ids = index(&[2, 0, 1, -1], &[2, 2]);
        let y = x.index_tensor(1, &ids).unwrap();
        assert_eq!(y.shape(), &[2, 2, 2, 2]);
        assert_eq!(y.to_vec(), [4., 5., 0., 1., 2., 3., 4., 5., 10., 11., 6., 7., 8., 9., 10., 11.]);
        assert_eq!(x.index_tensor(1, &index(&[], &[2, 0])).unwrap().shape(), &[2, 2, 0, 2]);
        assert!(x.index_tensor(3, &ids).is_err());
        assert!(x.index_tensor(0, &index(&[0], &[])).is_err());
        assert!(x.index_tensor(1, &index(&[3], &[1])).is_err());
        let transposed = x.transpose(0, 1).unwrap();
        assert_eq!(transposed.index_tensor(0, &ids).unwrap().to_vec(), [4., 5., 10., 11., 0., 1., 6., 7., 2., 3., 8., 9., 4., 5., 10., 11.]);
    }

    #[test]
    fn array_index_reverse_and_higher_rules_save_only_index_values() {
        let x = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let ids = index(&[2, 0, 2, 1], &[2, 2]);
        let y = x.index_tensor(1, &ids).unwrap();
        let seed = leaf(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2]);
        let gradient = y.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
        assert_eq!(gradient.to_vec(), [2., 4., 4., 6., 8., 12.]);
        x._bump_version_for_test(); y._bump_version_for_test(); seed._bump_version_for_test();
        let gg = leaf(&[10., 11., 12., 13., 14., 15.], &[2, 3]);
        let next = gradient.vjp_wrt(&[&seed], &gg, true).unwrap().remove(0);
        assert_eq!(next.to_vec(), [12., 10., 12., 11., 15., 13., 15., 14.]);
        gg._bump_version_for_test();
        assert_eq!(next.sum().grad_wrt(&[&gg], false).unwrap()[0].to_vec(), [1., 1., 2., 1., 1., 2.]);
        ids._bump_version_for_test();
        assert!(next.sum().grad_wrt(&[&gg], false).is_err());
        assert!(y.vjp_wrt(&[&x], &Tensor::ones(y.shape()), false).is_err());
    }

    #[test]
    fn array_jvp_preserves_typed_tangents_and_alias_index_guards() {
        let x = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let v = leaf(&[2., 3., 4., 5., 6., 7.], &[2, 3]);
        let ids = index(&[2, 0, 2, 1], &[2, 2]);
        let level = DualLevel::new(&[&x], &[&v]).unwrap();
        let y = x.index_tensor(1, &ids).unwrap();
        let tangent = level.tangent(&y).unwrap().unwrap();
        drop(level);
        assert_eq!(tangent.shape(), &[2, 2, 2]);
        assert_eq!(tangent.to_vec(), [4., 2., 4., 3., 7., 5., 7., 6.]);
        x._bump_version_for_test(); v._bump_version_for_test(); y._bump_version_for_test();
        assert_eq!(tangent.sum().grad_wrt(&[&v], false).unwrap()[0].to_vec(), [1., 1., 2., 1., 1., 2.]);
        ids._bump_version_for_test();
        assert!(tangent.sum().grad_wrt(&[&v], false).is_err());
    }

    #[test]
    fn true_alias_has_a_fresh_forward_identity_and_exact_live_seed_pullback() {
        for graph in [false, true] {
            let x = leaf(&[1., 2., 3., 4.], &[2, 2]);
            let y = x.alias_view().unwrap();
            assert_ne!(x.id(), y.id());
            assert_eq!(x._storage_ptr(), y._storage_ptr());
            assert!(!y.is_leaf());
            let seed = leaf(&[2., 3., 4., 5.], &[2, 2]);
            x._bump_version_for_test();
            let gradient = Tensor::vjp_many_wrt_optional(&[&y], &[&x], &[&seed], graph).unwrap().remove(0).unwrap();
            assert_eq!(gradient.id(), seed.id());
            assert!(gradient.requires_grad());
            y.sum().backward();
            assert!(!x.grad().unwrap().requires_grad());
            assert_ne!(x.id(), x.grad().unwrap().id());
        }
    }

    #[test]
    fn alias_jvp_preserves_real_direction_identity_without_value_guards() {
        let x = leaf(&[1., 2.], &[2]);
        let v = leaf(&[2., 3.], &[2]);
        let level = DualLevel::new(&[&x], &[&v]).unwrap();
        let y = x.alias_view().unwrap();
        let tangent = level.tangent(&y).unwrap().unwrap();
        assert_eq!(tangent.id(), v.id());
        drop(level);
        x._bump_version_for_test(); y._bump_version_for_test();
        assert_eq!(tangent.sum().grad_wrt(&[&v], false).unwrap()[0].to_vec(), [1., 1.]);
    }
}
