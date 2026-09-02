use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn pairwise_distance_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 6.0, 3.0, 3.0, 5.0, 4.0], &[2, 3]).unwrap();
    let got = a.pairwise_distance(&b, 2.0, 0.0).unwrap().to_vec();
    assert!((got[0] - 5.0).abs() < 1e-5);
    assert!((got[1] - 41.0f32.sqrt()).abs() < 1e-5);
}

#[test]
fn pairwise_distance_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, -1.0, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.4, 3.2, -2.5, 1.1], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].pairwise_distance(&t[1], 2.0, 0.01).unwrap().sum());
}
