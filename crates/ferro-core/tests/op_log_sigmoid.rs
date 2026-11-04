use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn log_sigmoid_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let got = a.log_sigmoid().unwrap().to_vec();
    let expect = |x: f32| (1.0 / (1.0 + (-x).exp())).ln();
    assert!((got[0] - expect(0.0)).abs() < 1e-5);
    assert!((got[1] - expect(1.0)).abs() < 1e-5);
    assert!((got[2] - expect(-1.0)).abs() < 1e-5);
}

#[test]
fn log_sigmoid_large_input_no_underflow() {
    let a = Tensor::from_vec(vec![-50.0, 50.0], &[2]).unwrap();
    let got = a.log_sigmoid().unwrap().to_vec();
    assert!(got[0].is_finite());
    assert!(got[1].is_finite());
    assert!((got[0] - (-50.0)).abs() < 1e-3);
    assert!((got[1] - 0.0).abs() < 1e-5);
}

#[test]
fn log_sigmoid_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, -1.5], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].log_sigmoid().unwrap().sum());
}

#[test]
fn typed_log_sigmoid_retains_tails_signed_zero_and_staged_adjoints() {
    let x = Tensor::from_vec_f64(vec![-30., 30., 100., f64::INFINITY, f64::NEG_INFINITY], &[5]).unwrap().requires_grad_(true).unwrap();
    let output = x.log_sigmoid().unwrap();
    let values = output.to_vec_f64();
    assert!(values[0] < -30. && values[1] < 0. && values[2] < 0.);
    assert_eq!(values[3], 0.);
    assert!(!values[3].is_sign_negative());
    assert_eq!(values[4], f64::NEG_INFINITY);
    output.sum().backward();
    let gradients = x.grad().unwrap().to_vec_f64();
    assert!(gradients[0] < 1. && gradients[1] > 0. && gradients[2] > 0.);
    assert_eq!(&gradients[3..], &[0., 1.]);
    let tiny = f32::from_bits(1);
    let x = Tensor::scalar(0.).requires_grad_(true).unwrap();
    x.log_sigmoid().unwrap().backward_with(&Tensor::scalar(-tiny));
    assert_eq!(x.grad().unwrap().to_vec()[0].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn typed_log_sigmoid_contiguous_output_empty_and_capacity() {
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::from_vec(vec![0.; 6], &[2, 3]).unwrap().to_dtype(dtype).transpose(0, 1).unwrap();
        assert_eq!(x.log_sigmoid().unwrap().strides(), &[2, 1]);
        let x = Tensor::zeros(&[0, 8]).to_dtype(dtype).requires_grad_(true).unwrap();
        let output = x.narrow(1, 4, 4).unwrap().log_sigmoid().unwrap();
        assert_eq!(output.shape(), &[0, 4]);
        output.sum().backward();
        assert_eq!(x.grad().unwrap().shape(), &[0, 8]);
        let huge = Tensor::scalar(1.).to_dtype(dtype).expand(&[isize::MAX as usize / 8 + 1]).unwrap();
        assert!(huge.log_sigmoid().is_err());
    }
    assert!(ferro_core::capture(|| Tensor::scalar(1.).log_sigmoid()).is_err());
    assert!(Tensor::from_vec_i64(vec![1], &[1]).unwrap().log_sigmoid().is_err());
}
