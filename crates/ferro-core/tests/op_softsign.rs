use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn softsign_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -2.0], &[3]).unwrap();
    let got = a.softsign().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 0.5).abs() < 1e-5);
    assert!((got[2] - (-2.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn softsign_grad() {
    let a = Tensor::from_vec(vec![-1.7, -0.6, 0.9, 2.3], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].softsign().unwrap().sum());
}

#[test]
fn typed_quotient_staging_preserves_nonfinite_and_extreme_adjoints() {
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::from_vec_f64(vec![f64::NEG_INFINITY, -2., -0., 0., 2., f64::INFINITY], &[6]).unwrap()
            .to_dtype(dtype).requires_grad_(true).unwrap();
        let output = x.softsign().unwrap();
        let values = output.to_vec_f64();
        assert!(values[0].is_nan() && values[5].is_nan());
        assert!(values[2].is_sign_negative() && !values[3].is_sign_negative());
        output.sum().backward();
        let gradients = x.grad().unwrap().to_vec_f64();
        assert!(gradients[0].is_nan() && gradients[5].is_nan());
        let expected = if dtype == ferro_core::DType::F32 { 0.111111119389534 } else { 0.1111111111111111 };
        assert_eq!(gradients[1], expected);
        assert_eq!(gradients[4], expected);
        assert_eq!(&gradients[2..4], &[1., 1.]);
    }
}

#[test]
fn typed_softsign_empty_and_capacity_preflight() {
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::zeros(&[0, 8]).to_dtype(dtype).requires_grad_(true).unwrap();
        let output = x.narrow(1, 4, 4).unwrap().softsign().unwrap();
        assert_eq!(output.shape(), &[0, 4]);
        output.sum().backward();
        assert_eq!(x.grad().unwrap().shape(), &[0, 8]);
        let huge = Tensor::scalar(1.).to_dtype(dtype).expand(&[isize::MAX as usize / 8 + 1]).unwrap();
        assert!(huge.softsign().is_err());
    }
    assert!(ferro_core::capture(|| Tensor::scalar(1.).softsign()).is_err());
    assert!(Tensor::from_vec_i64(vec![1], &[1]).unwrap().softsign().is_err());
}
