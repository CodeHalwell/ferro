use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn sqrt_values() {
    let a = Tensor::from_vec(vec![4.0, 9.0, 16.0], &[3]).unwrap();
    let got = a.sqrt().to_vec();
    assert!((got[0] - 2.0).abs() < 1e-5);
    assert!((got[1] - 3.0).abs() < 1e-5);
    assert!((got[2] - 4.0).abs() < 1e-5);
}

#[test]
fn sqrt_grad() {
    let a = Tensor::from_vec(vec![0.5, 1.0, 2.0, 4.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].sqrt().sum());
}

#[test]
fn sqrt_subnormal_cotangents_divide_before_underflow_in_both_precisions() {
    let x = Tensor::from_vec(vec![f32::from_bits(1)], &[1]).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::from_vec(vec![f32::from_bits(1)], &[1]).unwrap().requires_grad_(true).unwrap();
    for graph in [false, true] {
        let gradient = x.sqrt().vjp_wrt(&[&x], &seed, graph).unwrap().remove(0);
        assert_eq!(gradient.to_vec()[0].to_bits(), 1.871696033254608e-23f32.to_bits());
    }
    let level = ferro_core::forward_ad::DualLevel::new(&[&x], &[&seed]).unwrap();
    assert_eq!(level.tangent(&x.sqrt()).unwrap().unwrap().to_vec()[0].to_bits(), 1.871696033254608e-23f32.to_bits());
    drop(level);
    let x = Tensor::from_vec_f64(vec![f64::from_bits(1)], &[1]).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::from_vec_f64(vec![f64::from_bits(1)], &[1]).unwrap();
    let gradient = x.sqrt().vjp_wrt(&[&x], &seed, false).unwrap().remove(0);
    assert_eq!(gradient.to_vec_f64()[0].to_bits(), 1.1113793747425387e-162f64.to_bits());
    assert!(!gradient.requires_grad());
    assert!(x.grad().is_none());
}

#[test]
fn sqrt_reads_original_output_after_input_edits_in_ordinary_and_higher_derivatives() {
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::from_vec_f64(vec![0.5, 1.5], &[2]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let output = x.sqrt();
        let initial = output.sum().grad_wrt(&[&x], false).unwrap().remove(0);
        ferro_core::autograd::with_grad_enabled(false, || x.copy_leaf_from(&Tensor::full(&[2], 8.).to_dtype(dtype))).unwrap();
        let after = output.sum().grad_wrt(&[&x], false).unwrap().remove(0);
        assert_eq!(initial.to_vec_f64(), after.to_vec_f64());
        if dtype == ferro_core::DType::F32 {
            let first = output.sum().grad_wrt(&[&x], true).unwrap().remove(0);
            assert_eq!(first.to_vec(), initial.to_vec());
            let second = first.sum().grad_wrt(&[&x], false).unwrap().remove(0);
            for (actual, expected) in second.to_vec().into_iter().zip([-0.70710677f32, -0.13608275]) {
                assert!((actual - expected).abs() < 1e-7);
            }
        }
        output.sum().backward();
        assert_eq!(x.grad().unwrap().to_vec_f64(), initial.to_vec_f64());
        let before = x.grad().unwrap().to_vec_f64();
        output.detach().copy_from(&Tensor::ones(output.shape()).to_dtype(dtype)).unwrap();
        assert!(output.sum().grad_wrt(&[&x], false).is_err());
        if dtype == ferro_core::DType::F32 { assert!(output.sum().grad_wrt(&[&x], true).is_err()); }
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| output.sum().backward())).is_err());
        assert_eq!(x.grad().unwrap().to_vec_f64(), before);
    }
}

#[test]
fn sqrt_extreme_adjoint_stages_signed_zero_and_empty_strides_are_preserved() {
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::from_vec_f64(vec![0., -0., f64::INFINITY, f64::NAN], &[2, 2]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let gradient = x.sqrt().sum().grad_wrt(&[&x], false).unwrap().remove(0).to_vec_f64();
        assert_eq!(gradient[0], f64::INFINITY);
        assert_eq!(gradient[1], f64::NEG_INFINITY);
        assert_eq!(gradient[2].to_bits(), 0.0f64.to_bits());
        assert!(gradient[3].is_nan());
        let empty = Tensor::zeros(&[0, 8]).to_dtype(dtype).requires_grad_(true).unwrap();
        let output = empty.narrow(1, 4, 4).unwrap().sqrt();
        output.sum().backward();
        assert_eq!(empty.grad().unwrap().shape(), &[0, 8]);
        assert!(output.to_vec_f64().is_empty());
    }
    let x = Tensor::from_vec(vec![f32::MAX], &[1]).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::from_vec(vec![f32::MAX], &[1]).unwrap();
    assert_eq!(x.sqrt().vjp_wrt(&[&x], &seed, false).unwrap()[0].to_vec()[0], 9.223372036854776e18f32);
}
