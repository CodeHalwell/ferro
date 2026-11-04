use ferro_core::{DType, Tensor};

fn leaf(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec_f64(v.to_vec(), shape).unwrap().requires_grad_(true).unwrap()
}

#[test]
fn f64_broadcast_strided_values_and_finite_difference_gradients() {
    let av = [0.2, -0.4, 0.8, 1.2, -1.1, 0.7];
    let a = leaf(&av, &[3, 2]).transpose(0, 1).unwrap();
    let b = leaf(&[0.3, -0.7, 1.4], &[1, 3]);
    let out = a.atan2(&b).unwrap();
    assert_eq!(out.dtype(), DType::F64);
    let gradient = out.sum().grad_wrt(&[&a, &b], false).unwrap();
    let ax = a.to_vec_f64(); let bx = b.to_vec_f64();
    for i in 0..6 {
        let step = 1e-6;
        let reference = ((ax[i] + step).atan2(bx[i % 3]) - (ax[i] - step).atan2(bx[i % 3])) / (2.0 * step);
        assert!((gradient[0].to_vec_f64()[i] - reference).abs() < 2e-10);
    }
    for j in 0..3 {
        let step = 1e-6;
        let reference: f64 = [j, j + 3].iter().map(|&i| (ax[i].atan2(bx[j] + step) - ax[i].atan2(bx[j] - step)) / (2.0 * step)).sum();
        assert!((gradient[1].to_vec_f64()[j] - reference).abs() < 2e-10);
    }
}

#[test]
fn torch_zero_radius_signed_zero_and_overflow_staging() {
    for dtype in [DType::F32, DType::F64] {
        let small = if dtype == DType::F32 { 1e-30 } else { 1e-200 };
        let large = if dtype == DType::F32 { 1e30 } else { 1e200 };
        let a = leaf(&[0., -0., small, large], &[4]).to_dtype_autograd(dtype, false).unwrap();
        let b = leaf(&[-0., 0., small, large], &[4]).to_dtype_autograd(dtype, false).unwrap();
        let seed = Tensor::from_vec_f64(vec![1., 1., 1., large], &[4]).unwrap().to_dtype(dtype);
        let output = a.atan2(&b).unwrap();
        let grad = output.vjp_wrt(&[&a, &b], &seed, false).unwrap();
        let ga = grad[0].to_vec_f64(); let gb = grad[1].to_vec_f64();
        assert!(ga[0] == 0. && ga[0].is_sign_negative());
        assert!(ga[1] == 0. && !ga[1].is_sign_negative());
        assert!(gb[0] == 0. && gb[0].is_sign_negative());
        assert!(gb[1] == 0. && !gb[1].is_sign_negative());
        assert_eq!(ga[2], 0.); assert!(gb[2] == 0. && gb[2].is_sign_negative());
        assert!(ga[3].is_nan() && gb[3].is_nan());
    }
}

#[test]
fn promotion_keeps_dimensional_precedence_and_parameter_gradient_dtypes() {
    let scalar = leaf(&[0.7], &[]);
    let dimensional = Tensor::from_vec(vec![-0.3, 0.4], &[2]).unwrap().requires_grad_(true).unwrap();
    let output = scalar.atan2(&dimensional).unwrap();
    assert_eq!(output.dtype(), DType::F32);
    let grads = output.sum().grad_wrt(&[&scalar, &dimensional], false).unwrap();
    assert_eq!(grads[0].dtype(), DType::F64); assert_eq!(grads[1].dtype(), DType::F32);
    let tiny = leaf(&[1e-200], &[1]);
    let one = Tensor::scalar(1.);
    let out = tiny.atan2(&one).unwrap();
    assert_eq!(out.dtype(), DType::F64); assert_eq!(out.to_vec_f64(), [1e-200]);
    assert!(Tensor::from_vec_i64(vec![1], &[1]).unwrap().atan2(&one).is_err());
}

#[test]
fn both_primal_versions_are_needed_but_output_version_is_not() {
    for changed in 0..3 {
        let a = leaf(&[0.3], &[1]); let b = leaf(&[0.8], &[1]);
        let out = a.atan2(&b).unwrap();
        [&a, &b, &out][changed]._bump_version_for_test();
        let result = out.sum().grad_wrt(&[&a, &b], false);
        assert_eq!(result.is_err(), changed < 2);
    }
}

#[test]
fn huge_empty_broadcast_skips_primal_copies_and_keeps_real_zero_adjoints() {
    let size = 1_000_000_000_000usize;
    let a = leaf(&[0.5], &[1, 1]).expand(&[size, 1]).unwrap();
    let b = Tensor::from_vec_f64(vec![], &[1, 0]).unwrap().requires_grad_(true).unwrap().expand(&[size, 0]).unwrap();
    let out = a.atan2(&b).unwrap();
    assert_eq!(out.shape(), [size, 0]);
    let seed = Tensor::from_vec_f64(vec![], &[size, 0]).unwrap();
    let grads = out.vjp_wrt_preserve(&[&a, &b], &seed).unwrap();
    assert_eq!(grads[0].shape(), [size, 1]); assert_eq!(grads[1].shape(), [size, 0]);
    assert!(!grads[0].is_structural_zero()); assert_eq!(grads[0].strides()[0], 0);
    let huge = leaf(&[1.], &[1, 1]).expand(&[isize::MAX as usize / 8 + 1, 1]).unwrap();
    assert!(huge.atan2(&Tensor::from_vec_f64(vec![1., 2.], &[1, 2]).unwrap()).is_err());
}
