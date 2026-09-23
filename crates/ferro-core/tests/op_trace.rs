use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn trace_values() {
    let sq = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let got = sq.trace().unwrap();
    assert_eq!(got.shape(), &[] as &[usize]);
    assert!((got.item() - 5.0).abs() < 1e-6);
    // Non-square: only the min(n, m) leading diagonal entries count.
    let wide = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!((wide.trace().unwrap().item() - 6.0).abs() < 1e-6);
    let tall = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    assert!((tall.trace().unwrap().item() - 5.0).abs() < 1e-6);
}

#[test]
fn trace_rejects_non_matrices() {
    assert!(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap().trace().is_err());
    assert!(Tensor::scalar(1.0).trace().is_err());
    assert!(Tensor::from_vec(vec![1.0; 8], &[2, 2, 2]).unwrap().trace().is_err());
}

#[test]
fn trace_grad() {
    let a = Tensor::from_vec(vec![0.3, -1.2, 2.0, 0.7, 1.1, -0.4], &[2, 3]).unwrap();
    grad_check(&[a], |t| t[0].trace().unwrap());
}
