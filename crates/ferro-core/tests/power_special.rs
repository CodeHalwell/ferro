use ferro_core::{DType, Tensor};
use ferro_core::forward_ad::DualLevel;

include!("fixtures/power_special_reference.rs");

fn scalar(wide: bool, bits: u64) -> Tensor {
    if wide { Tensor::from_vec_f64(vec![f64::from_bits(bits)], &[]).unwrap() }
    else { Tensor::scalar(f32::from_bits(bits as u32)) }
}
fn exact(value: &Tensor, wide: bool, bits: u64) {
    assert_eq!(value.dtype(), if wide { DType::F64 } else { DType::F32 });
    if wide {
        let value = value.to_vec_f64()[0]; let expected = f64::from_bits(bits);
        assert!(value.is_nan() && expected.is_nan() || value.to_bits() == bits, "{value:?} != {expected:?}");
    } else {
        let value = value.to_vec()[0]; let expected = f32::from_bits(bits as u32);
        assert!(value.is_nan() && expected.is_nan() || value.to_bits() == bits as u32, "{value:?} != {expected:?}");
    }
}

#[test]
fn special_power_forwards_match_actual_torch_scalar_and_vector_bits() {
    for &(wide, power, bits, expected) in FORWARD {
        let x = scalar(wide, bits);
        exact(&x.pow_scalar(power).unwrap(), wide, expected);
        if !wide { exact(&x.powf(power as f32), false, expected); }
    }
}

#[test]
fn special_derivative_exponents_match_actual_torch_in_all_supported_ad_paths() {
    for &(wide, power, bits, gradient, expected) in BACKWARD {
        let x = scalar(wide, bits).requires_grad_(true).unwrap();
        let g = scalar(wide, gradient).requires_grad_(true).unwrap();
        let y = x.pow_scalar(power).unwrap();
        for create_graph in [false, true] {
            if wide && create_graph { continue; }
            let result = y.vjp_wrt(&[&x], &g, create_graph).unwrap().remove(0);
            exact(&result, wide, expected);
            assert_eq!(result.requires_grad(), create_graph);
        }
        y.backward_with(&g);
        exact(&x.grad().unwrap(), wide, expected);
        assert!(!x.grad().unwrap().requires_grad());
        if !wide {
            let level = DualLevel::new(&[&x], &[&g]).unwrap();
            let result = x.pow_scalar(power).unwrap();
            let tangent = level.tangent(&result).unwrap().unwrap();
            drop(level);
            exact(&tangent, false, expected);
        }
    }
}

#[test]
fn original_neighbor_exponents_do_not_enter_exact_integer_specializations() {
    // Adjacent f64 exponents retain their noninteger domain even when f32
    // conversion would round to a specialized integer kernel.
    for power in [-2.0f64, -1.0, 2.0, 3.0] {
        for bits in [power.to_bits() - 1, power.to_bits() + 1] {
            let p = f64::from_bits(bits);
            let x = Tensor::scalar(-4.);
            assert!(x.pow_scalar(p).unwrap().item().is_nan(), "original exponent {p:?}");
            assert!(!x.pow_scalar(power).unwrap().item().is_nan());
        }
    }
    let x = Tensor::scalar(1e20).requires_grad_(true).unwrap();
    let y = x.pow_scalar(-1.).unwrap();
    let g = Tensor::scalar(2e38);
    for graph in [false, true] {
        let gradient = y.vjp_wrt(&[&x], &g, graph).unwrap().remove(0);
        assert_eq!(gradient.item().to_bits(), (-0.0f32).to_bits());
    }
}
