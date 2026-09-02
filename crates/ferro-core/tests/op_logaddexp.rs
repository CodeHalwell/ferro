use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn logaddexp_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![0.0, 2.0, 1.0], &[3]).unwrap();
    let got = a.logaddexp(&b).unwrap().to_vec();
    assert!((got[0] - 2.0f32.ln()).abs() < 1e-5);
    assert!((got[1] - (1.0f32.exp() + 2.0f32.exp()).ln()).abs() < 1e-5);
    assert!((got[2] - (2.0f32.exp() + 1.0f32.exp()).ln()).abs() < 1e-5);
}

#[test]
fn logaddexp_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![0.5, -0.5, 1.5], &[3]).unwrap();
    let got = a.logaddexp(&b).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    let av = a.to_vec();
    let bv = vec![0.5f32, -0.5, 1.5, 0.5, -0.5, 1.5];
    for i in 0..6 {
        let expected = (av[i].exp() + bv[i].exp()).ln();
        assert!((got.to_vec()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn logaddexp_large_magnitude() {
    let a = Tensor::from_vec(vec![1000.0, -1000.0, 500.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![1000.0, -1000.0, 500.1], &[3]).unwrap();
    let got = a.logaddexp(&b).unwrap().to_vec();
    assert!(got.iter().all(|v| v.is_finite()));
    assert!((got[0] - (1000.0 + 2.0f32.ln())).abs() < 1e-3);
    assert!((got[1] - (-1000.0 + 2.0f32.ln())).abs() < 1e-3);
    let expected2 = 500.1 + (1.0 + (-0.1f32).exp()).ln();
    assert!((got[2] - expected2).abs() < 1e-3);
}

#[test]
fn logaddexp_equal_infinities() {
    // The stable m + ln(exp(a-m) + exp(b-m)) form computes inf - inf = NaN
    // when both operands (and so m) are the same infinity; the op must
    // special-case this and return that infinity directly.
    let a = Tensor::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY], &[3]).unwrap();
    let b = Tensor::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY, 3.0], &[3]).unwrap();
    let got = a.logaddexp(&b).unwrap().to_vec();
    assert_eq!(got[0], f32::INFINITY);
    assert_eq!(got[1], f32::NEG_INFINITY);
    assert_eq!(got[2], f32::INFINITY);
}

#[test]
fn logaddexp_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, 2.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![-0.4, 0.9, -1.1, 1.8], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].logaddexp(&t[1]).unwrap().sum());
}
