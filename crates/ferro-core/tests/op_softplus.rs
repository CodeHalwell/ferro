use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn softplus_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let got = a.softplus().unwrap().to_vec();
    assert!((got[0] - 2.0f32.ln()).abs() < 1e-5);
    assert!((got[1] - (1.0 + 1.0f32.exp()).ln()).abs() < 1e-5);
    assert!((got[2] - (1.0 + (-1.0f32).exp()).ln()).abs() < 1e-5);
}

#[test]
fn softplus_large_input_no_overflow() {
    let a = Tensor::from_vec(vec![100.0, 1000.0, 50.0], &[3]).unwrap();
    let got = a.softplus().unwrap().to_vec();
    assert!(got[0].is_finite());
    assert!(got[1].is_finite());
    assert!(got[2].is_finite());
    assert!((got[0] - 100.0).abs() < 1e-5);
    assert!((got[1] - 1000.0).abs() < 1e-5);
    assert!((got[2] - 50.0).abs() < 1e-5);
}

#[test]
fn softplus_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, -1.5], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].softplus().unwrap().sum());
}

#[test]
fn softplus_tiny_values_and_beta_zero_keep_torch_adjoints() {
    let x = Tensor::from_vec(vec![-100., 2.], &[2]).unwrap().requires_grad_(true).unwrap();
    let y = x.softplus().unwrap();
    assert_eq!(y.to_vec()[0], 3.783505853677006e-44f32);
    y.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec()[0], 3.783505853677006e-44f32);
    let x = Tensor::from_vec(vec![-2., 0., 2.], &[3]).unwrap().requires_grad_(true).unwrap();
    let y = x.softplus_options(0., 20.).unwrap();
    assert!(y.to_vec().iter().all(|value| *value == f32::INFINITY));
    y.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![0.5; 3]);
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::scalar(2.).to_dtype(dtype).requires_grad_(true).unwrap();
        let seed = if dtype == ferro_core::DType::F32 { Tensor::scalar(1e38) }
            else { Tensor::from_vec_f64(vec![1e308], &[]).unwrap() };
        x.softplus_options(1., f64::INFINITY).unwrap().backward_with(&seed);
        assert_eq!(x.grad().unwrap().to_vec_f64(), vec![f64::INFINITY]);
    }
}

#[test]
fn softplus_options_typed_threshold_and_nonfinite_branches() {
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::from_vec(vec![f32::NEG_INFINITY, -2., 0., 2., f32::INFINITY, f32::NAN], &[2, 3]).unwrap()
            .to_dtype(dtype).requires_grad_(true).unwrap();
        let y = x.softplus_options(-1., 0.).unwrap();
        let values = y.to_vec_f64();
        assert_eq!(values[0], f64::NEG_INFINITY);
        assert_eq!(values[1], -2.);
        assert!((values[2] + 2.0f64.ln()).abs() < 1e-7);
        assert!((values[3] + (-2.0f64).exp().ln_1p()).abs() < 1e-7);
        assert_eq!(values[4], 0.);
        assert!(values[4].is_sign_negative() && values[5].is_nan());
        y.sum().backward();
        let values = x.grad().unwrap().to_vec_f64();
        assert_eq!(&values[..3], &[1., 1., 0.5]);
        assert!((values[3] - 0.11920292202211755).abs() < 1e-7);
        assert_eq!(values[4], 0.);
        assert!(values[5].is_nan());
    }
    let x = Tensor::from_vec(vec![1., -1.], &[2]).unwrap().requires_grad_(true).unwrap();
    x.softplus_options(0., -1.).unwrap().sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![1., 1.]);
}

#[test]
fn softplus_options_gradcheck_empty_and_invalid_metadata() {
    let x = Tensor::from_vec(vec![-0.7, 0.3, 1.8, 2.4], &[2, 2]).unwrap();
    grad_check(&[x], |inputs| inputs[0].softplus_options(0.75, 1.).unwrap().sum());
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::zeros(&[0, 8]).to_dtype(dtype).requires_grad_(true).unwrap();
        let view = x.narrow(1, 4, 4).unwrap();
        let output = view.softplus_options(f64::NAN, f64::NAN).unwrap();
        assert_eq!(output.shape(), &[0, 4]);
        output.sum().backward();
        assert_eq!(x.grad().unwrap().shape(), &[0, 8]);
    }
    assert!(Tensor::from_vec_i64(vec![1], &[1]).unwrap().softplus().is_err());
    assert!(ferro_core::capture(|| Tensor::scalar(2.).softplus()).is_err());
    assert!(Tensor::zeros(&[0]).softplus_options(1e100, 20.).is_err());
    let huge = Tensor::scalar(1.).expand(&[isize::MAX as usize / 4 + 1]).unwrap();
    assert!(huge.softplus().is_err());
}
