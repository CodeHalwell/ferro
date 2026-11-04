use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::DualLevel;
use ferro_core::testkit::grad_check;

fn tensor(values: &[f64], shape: &[usize], dtype: DType, grad: bool) -> Tensor {
    Tensor::from_vec_f64(values.to_vec(), shape).unwrap().to_dtype(dtype).requires_grad_(grad).unwrap()
}
fn values(value: &Tensor) -> Vec<f64> { value.to_dtype(DType::F64).to_vec_f64() }

#[test]
fn typed_cumulative_values_wrapping_and_widened_accumulation() {
    let mask = Tensor::from_vec_bool(vec![true, false, true], &[3]).unwrap();
    assert_eq!(mask.cumsum(0).unwrap().to_vec_i64(), vec![1, 1, 2]);
    assert_eq!(mask.cumprod(0).unwrap().to_vec_i64(), vec![1, 0, 0]);
    let integers = Tensor::from_vec_i64(vec![i64::MAX, 2, 3], &[3]).unwrap();
    assert_eq!(integers.cumprod(0).unwrap().to_vec_i64(), vec![i64::MAX, -2, -6]);
    assert_eq!(integers.cumsum(0).unwrap().to_vec_i64(), vec![i64::MAX, i64::MIN + 1, i64::MIN + 4]);
    let x = tensor(&[1e30, 1e30, 1e-30], &[3], DType::F32, false);
    let product = x.cumprod(0).unwrap().to_vec();
    assert!(product[1].is_infinite()); assert!(product[2].is_finite());
    let x = tensor(&[16777216., 1., -16777216.], &[3], DType::F32, false);
    assert_eq!(x.cumsum(0).unwrap().to_vec(), vec![16777216., 16777216., 1.]);
    let x = tensor(&[2., 3., 4., 5., 6., 7.], &[2, 3], DType::F64, false).transpose(0, 1).unwrap();
    assert_eq!(x.cumprod(0).unwrap().to_vec_f64(), vec![2., 5., 6., 30., 24., 210.]);
    assert_eq!(x.cumsum(1).unwrap().to_vec_f64(), vec![2., 7., 3., 9., 4., 11.]);
}

#[test]
fn ordinary_cumulative_adjoint_has_real_zero_and_nonfinite_seed_semantics() {
    for dtype in [DType::F32, DType::F64] {
        for (input, seed, expected) in [
            (vec![2., 0., 4.], vec![1., 2., 3.], vec![1., 28., 0.]),
            (vec![0., 3., 0.], vec![1., 2., 3.], vec![7., 0., 0.]),
            (vec![0., 2., 0., 3.], vec![1., 1., f64::INFINITY, 1.], vec![3., 0., 0., 0.]),
            (vec![f64::INFINITY, 0., 0.], vec![1., 2., 3.], vec![f64::NAN, f64::INFINITY, 0.]),
            (vec![0., f64::INFINITY, 0.], vec![1., 2., 3.], vec![f64::NAN, 0., 0.]),
        ] {
            let x = tensor(&input, &[input.len()], dtype, true);
            let g = tensor(&seed, &[seed.len()], dtype, false);
            let got = x.cumprod(0).unwrap().vjp_wrt(&[&x], &g, false).unwrap();
            for (actual, expected) in values(&got[0]).iter().zip(expected) {
                assert!((*actual == expected) || (actual.is_nan() && expected.is_nan()), "{actual} != {expected}");
            }
            assert!(x.grad().is_none());
        }
    }
    let x = tensor(&[0.5, -1.2, 0.8, 1.4, -0.3, 0.9], &[2, 3], DType::F32, false);
    for dim in [0, 1] { grad_check(&[x.clone()], |xs| xs[0].cumprod(dim).unwrap().mul(&xs[0]).unwrap().sum()); }
}

#[test]
fn huge_empty_cumulative_axes_skip_all_outer_and_inner_loops() {
    for dtype in [DType::F32, DType::F64] { for shape in [[0, 1usize << 60], [1usize << 60, 0]] { for dim in [0, 1] {
        let x = tensor(&[], &shape, dtype, true);
        for output in [x.cumsum(dim).unwrap(), x.cumprod(dim).unwrap()] {
            let seed = tensor(&[], &shape, dtype, true);
            let gradient = output.vjp_wrt_preserve(&[&x], &seed).unwrap();
            assert_eq!(gradient[0].shape(), &shape); assert_eq!(gradient[0].id(), seed.id());
        }
    } } }
}

#[test]
fn cumulative_dtype_options_scalar_identity_and_integer_ad_rejection() {
    let x = tensor(&[2., 3., 4.], &[3], DType::F32, true);
    let y = x.cumprod_dtype(0, Some(DType::F64)).unwrap();
    assert_eq!(y.dtype(), DType::F64);
    assert_eq!(y.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec(), vec![16., 10., 6.]);
    assert!(x.cumsum_dtype(0, Some(DType::I64)).is_err());
    assert!(x.cumprod_dtype(0, Some(DType::I64)).is_err());
    assert_eq!(with_grad_enabled(false, || x.cumprod_dtype(0, Some(DType::I64))).unwrap().to_vec_i64(), vec![2, 6, 24]);
    for shape in [vec![], vec![1], vec![3, 1]] {
        let x = Tensor::ones(&shape).requires_grad_(true).unwrap();
        let g = Tensor::ones(&shape).requires_grad_(true).unwrap();
        let dim = shape.len().saturating_sub(1);
        for y in [x.cumsum(dim).unwrap(), x.cumprod(dim).unwrap()] { for graph in [false, true] {
            let result = if graph { y.vjp_wrt(&[&x], &g, true) } else { y.vjp_wrt_preserve(&[&x], &g) }.unwrap();
            assert_eq!(result[0].id(), g.id()); assert!(result[0].requires_grad());
        } }
    }
}

#[test]
fn explicit_dtype_adjoint_uses_original_values_and_casts_seed_before_scans() {
    let x = tensor(&[1e-50, 2.], &[2], DType::F64, true);
    let y = x.cumprod_dtype(0, Some(DType::F32)).unwrap();
    assert_eq!(values(&y.sum().grad_wrt(&[&x], false).unwrap()[0]), vec![0., 0.]);
    let x = tensor(&[1.00000003, 2.00000003, 0.], &[3], DType::F64, true);
    let y = x.cumprod_dtype(0, Some(DType::F32)).unwrap();
    let gradient = values(&y.sum().grad_wrt(&[&x], false).unwrap()[0]);
    for (actual, expected) in gradient.iter().zip([2.9999999100000028, 0.9999999850000003, 2.]) {
        assert!((actual - expected).abs() < 1e-14);
    }
    let x = tensor(&[1., 2., 3.], &[3], DType::F64, true);
    let y = x.cumsum_dtype(0, Some(DType::F32)).unwrap();
    let seed = tensor(&[1e8, 1., -1e8], &[3], DType::F32, false);
    assert_eq!(values(&y.vjp_wrt(&[&x], &seed, false).unwrap()[0]), vec![1., -99999999., -1e8]);
    let x = tensor(&[0., 1., 1.], &[3], DType::F32, true);
    let seed = tensor(&[1e8, 1., -1e8], &[3], DType::F64, false);
    let y = x.cumprod_dtype(0, Some(DType::F64)).unwrap();
    assert_eq!(values(&y.vjp_wrt(&[&x], &seed, false).unwrap()[0]), vec![0., 0., 0.]);
    for shape in [vec![], vec![1], vec![2, 1]] {
        let count = shape.iter().product();
        let x = tensor(&vec![f64::INFINITY; count], &shape, DType::F32, false);
        let seed = Tensor::ones(&shape);
        let level = DualLevel::new(&[&x], &[&seed]).unwrap();
        let y = x.cumprod(shape.len().saturating_sub(1)).unwrap();
        assert!(values(&level.tangent(&y).unwrap().unwrap()).iter().all(|value| value.is_nan()));
    }
}

#[test]
fn zero_cumprod_second_and_third_derivatives_keep_input_and_seed_graphs() {
    let x = tensor(&[2., 0., 4.], &[3], DType::F32, true);
    let seed = tensor(&[1., 2., 3.], &[3], DType::F32, true);
    let first = x.cumprod(0).unwrap().vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
    assert_eq!(first.to_vec(), vec![1., 28., 0.]);
    let second = first.sum().grad_wrt(&[&x, &seed], true).unwrap();
    assert_eq!(second[0].to_vec(), vec![14., 20., 6.]);
    assert_eq!(second[1].to_vec(), vec![1., 2., 8.]);
    assert_eq!(second[0].sum().grad_wrt(&[&x], false).unwrap()[0].to_vec(), vec![6., 6., 6.]);
    let large = Tensor::zeros(&[2048]).requires_grad_(true).unwrap();
    assert!(large.cumprod(0).unwrap().sum().grad_wrt(&[&large], true).is_err());
    assert!(large.grad().is_none());
}

#[test]
fn cumulative_saved_values_differ_from_linear_metadata() {
    for dtype in [DType::F32, DType::F64] { for product in [false, true] { for change_output in [false, true] {
        let x = tensor(&[2., 3.], &[2], dtype, true);
        let y = if product { x.cumprod(0) } else { x.cumsum(0) }.unwrap();
        with_grad_enabled(false, || {
            if change_output { y.detach().copy_from(&tensor(&[5., 7.], &[2], dtype, false)) }
            else { x.copy_leaf_from(&tensor(&[5., 7.], &[2], dtype, false)) }
        }).unwrap();
        let gradients = y.sum().grad_wrt(&[&x], false);
        if product { assert!(gradients.is_err()); }
        else { assert_eq!(values(&gradients.unwrap()[0]), vec![2., 1.]); }
        assert!(x.grad().is_none());
    } } }
}

#[test]
fn cumsum_dual_dtype_and_cumprod_promoted_tangent_arithmetic() {
    let x = tensor(&[1e-30, 1e30], &[2], DType::F32, false);
    let v = tensor(&[1e-50, 0.], &[2], DType::F64, true);
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let default = x.cumsum(0).unwrap();
    let explicit = x.cumsum_dtype(0, Some(DType::F32)).unwrap();
    let product = x.cumprod(0).unwrap();
    let ds = level.tangent(&default).unwrap().unwrap();
    let cast_ds = level.tangent(&explicit).unwrap().unwrap();
    let dp = level.tangent(&product).unwrap().unwrap();
    drop(level);
    assert_eq!(ds.dtype(), DType::F64); assert_eq!(ds.to_vec_f64(), vec![1e-50, 1e-50]);
    assert_eq!(cast_ds.dtype(), DType::F32); assert_eq!(cast_ds.to_vec(), vec![0., 0.]);
    assert_eq!(dp.dtype(), DType::F32); assert_eq!(dp.to_vec()[0], 0.); assert!(dp.to_vec()[1] > 9.9e-21);
    assert!(dp.sum().grad_wrt(&[&v], false).is_ok());
}

#[test]
fn zero_cumprod_duals_preserve_masked_tangent_and_reverse_nan_convention() {
    for dtype in [DType::F32] {
        let x = tensor(&[2., 0., 4.], &[3], dtype, true);
        let v = tensor(&[0.1, 0.2, 0.3], &[3], dtype, true);
        let level = DualLevel::new(&[&x], &[&v]).unwrap();
        let y = x.cumprod(0).unwrap();
        let dy = level.tangent(&y).unwrap().unwrap();
        drop(level);
        for (got, expected) in values(&dy).iter().zip([0.1, 0.4, 1.6]) { assert!((*got - expected).abs() < 1e-6); }
        let gradients = dy.sum().grad_wrt(&[&x, &v], false).unwrap();
        let dx = values(&gradients[0]); let dv = values(&gradients[1]);
        assert!((dx[0] - 1.).abs() < 1e-6); assert!(dx[1].is_nan()); assert!((dx[2] - 0.4).abs() < 1e-6);
        assert_eq!(dv[0], 1.); assert!(dv[1].is_nan()); assert_eq!(dv[2], 0.);
    }
    let x = tensor(&[1., 0., 1e30], &[3], DType::F32, false);
    let v = tensor(&[0., 1e-50, 0.], &[3], DType::F64, true);
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let y = x.cumprod(0).unwrap(); let dy = level.tangent(&y).unwrap().unwrap(); drop(level);
    assert_eq!(dy.dtype(), DType::F32); assert_eq!(&dy.to_vec()[..2], &[0., 0.]); assert!(dy.to_vec()[2] > 9.9e-21);
}
