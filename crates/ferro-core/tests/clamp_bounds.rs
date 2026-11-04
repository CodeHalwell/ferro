use ferro_core::{DType, Tensor};
use ferro_core::forward_ad::DualLevel;
include!("fixtures/clamp_bounds_goldens.rs");

fn exact32(actual: &[f32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!(a.to_bits() == b || a.is_nan() && f32::from_bits(b).is_nan(), "{a:?} {:x} != {:x}", a.to_bits(), b); }
}
fn exact64(actual: &[f64], expected: &[u64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!(a.to_bits() == b || a.is_nan() && f64::from_bits(b).is_nan(), "{a:?} {:x} != {:x}", a.to_bits(), b); }
}

#[test]
fn scalar_bound_clamp_matches_actual_torch_nonfinite_and_signed_zero_goldens() {
    let x = Tensor::from_vec(INPUT32.iter().map(|&v| f32::from_bits(v)).collect(), &[9]).unwrap();
    for &(min, max, expected) in CLAMP32 { exact32(&x.clamp(f32::from_bits(min), f32::from_bits(max)).to_vec(), expected); }
    let x = Tensor::from_vec_f64(INPUT64.iter().map(|&v| f64::from_bits(v)).collect(), &[9]).unwrap();
    for &(min, max, expected) in CLAMP64 { exact64(&x.clamp_f64(f64::from_bits(min), f64::from_bits(max)).unwrap().to_vec_f64(), expected); }
}

#[test]
fn clamp_live_seed_vjp_masks_nonfinite_values_and_keeps_only_seed_provenance() {
    for &(min, max, _) in CLAMP32 { for value in [0.7, f32::INFINITY, f32::NAN, -0.] { for graph in [false, true] {
        let min = f32::from_bits(min); let max = f32::from_bits(max);
        let x = Tensor::from_vec(INPUT32.iter().map(|&v| f32::from_bits(v)).collect(), &[3, 3]).unwrap().requires_grad_(true).unwrap();
        let g = Tensor::full(x.shape(), value).requires_grad_(true).unwrap();
        let y = x.clamp(min, max);
        let derivative = Tensor::vjp_many_wrt_optional(&[&y], &[&x], &[&g], graph).unwrap().remove(0).unwrap();
        let expected = x.to_vec().iter().map(|&x| if x >= min && x <= max { value.to_bits() } else { 0.0f32.to_bits() }).collect::<Vec<_>>();
        exact32(&derivative.to_vec(), &expected); assert_eq!(derivative.requires_grad(), graph);
        if graph {
            let mixed = Tensor::vjp_many_wrt_optional(&[&derivative.sum()], &[&x, &g], &[&Tensor::scalar(1.)], false).unwrap();
            assert!(mixed[0].is_none());
            let mask = x.to_vec().iter().map(|&x| if x >= min && x <= max { 1. } else { 0. }).collect::<Vec<_>>();
            assert_eq!(mixed[1].as_ref().unwrap().to_vec(), mask);
        }
        assert!(x.grad().is_none() && g.grad().is_none());
    } } }
    for &(min, max, _) in CLAMP64 { for value in [0.7, f64::INFINITY, f64::NAN, -0.] {
        let min = f64::from_bits(min); let max = f64::from_bits(max);
        let x = Tensor::from_vec_f64(INPUT64.iter().map(|&v| f64::from_bits(v)).collect(), &[9]).unwrap().requires_grad_(true).unwrap();
        let g = Tensor::from_vec_f64(vec![value; 9], &[9]).unwrap().requires_grad_(true).unwrap();
        let y = x.clamp_f64(min, max).unwrap();
        let derivative = Tensor::vjp_many_wrt_optional(&[&y], &[&x], &[&g], false).unwrap().remove(0).unwrap();
        let expected = x.to_vec_f64().iter().map(|&x| if x >= min && x <= max { value.to_bits() } else { 0.0f64.to_bits() }).collect::<Vec<_>>();
        exact64(&derivative.to_vec_f64(), &expected); assert!(!derivative.requires_grad());
        y._bump_version_for_test(); assert!(Tensor::vjp_many_wrt_optional(&[&y], &[&x], &[&g], false).is_ok());
        x._bump_version_for_test(); assert!(Tensor::vjp_many_wrt_optional(&[&y], &[&x], &[&g], false).is_err());
    } }
}

#[test]
fn finite_clamp_directions_match_central_differences_away_from_boundaries() {
    let values = [-1.2, -0.6, -0.2, 0.3, 0.6, 1.2]; let direction = [0.3, -0.7, 0.5, 0.9, 0.2, -0.4];
    let x = Tensor::from_vec(values.to_vec(), &[2, 3]).unwrap().requires_grad_(true).unwrap();
    let v = Tensor::from_vec(direction.to_vec(), &[2, 3]).unwrap();
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let y = x.clamp(-0.8, 0.8); let dy = level.tangent(&y).unwrap().unwrap(); drop(level);
    let eps = 0.001;
    let evaluate = |sign| Tensor::from_vec(values.iter().zip(direction).map(|(&x, v)| x + sign * eps * v).collect(), &[2, 3]).unwrap().clamp(-0.8, 0.8).to_vec();
    for ((got, plus), minus) in dy.to_vec().into_iter().zip(evaluate(1.)).zip(evaluate(-1.)) { assert!((got - (plus - minus) / (2. * eps)).abs() < 3e-5); }
}

#[test]
fn clamp_jvp_bounds_empty_scalar_and_saved_value_guards() {
    for &(min, max, _) in CLAMP32 {
        let min = f32::from_bits(min); let max = f32::from_bits(max);
        let x = Tensor::from_vec(INPUT32.iter().map(|&v| f32::from_bits(v)).collect(), &[3, 3]).unwrap().transpose(0, 1).unwrap().detach().requires_grad_(true).unwrap();
        let direction = Tensor::full(&[3, 3], f32::INFINITY).transpose(0, 1).unwrap().requires_grad_(true).unwrap();
        let level = DualLevel::new(&[&x], &[&direction]).unwrap();
        let output = x.clamp(min, max); let tangent = level.tangent(&output).unwrap().unwrap(); drop(level);
        assert_eq!(tangent.strides(), &[1, 3]);
        let expected = x.to_vec().iter().map(|&x| if x >= min && x <= max { f32::INFINITY.to_bits() } else { 0.0f32.to_bits() }).collect::<Vec<_>>();
        exact32(&tangent.to_vec(), &expected);
        output._bump_version_for_test(); assert!(output.vjp_wrt(&[&x], &Tensor::ones(output.shape()), false).is_ok());
        x._bump_version_for_test(); assert!(output.vjp_wrt(&[&x], &Tensor::ones(output.shape()), false).is_err());
        for shape in [vec![], vec![0], vec![2, 0, 3]] {
            let input = Tensor::full(&shape, 0.).requires_grad_(true).unwrap();
            let seed = Tensor::ones(&shape);
            let level = DualLevel::new(&[&input], &[&seed]).unwrap();
            let y = input.clamp(min, max); let tangent = level.tangent(&y).unwrap().unwrap(); drop(level);
            assert_eq!(y.shape(), shape); assert_eq!(tangent.shape(), shape);
            assert_eq!(y.dtype(), DType::F32);
        }
    }
}
