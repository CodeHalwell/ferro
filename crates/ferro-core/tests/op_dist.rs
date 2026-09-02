use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn dist_values_p2() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 6.0, 3.0], &[3]).unwrap();
    let got = a.dist(&b, 2.0).unwrap().item();
    assert!((got - 5.0).abs() < 1e-5);
}

#[test]
fn dist_values_p1() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![0.0, 5.0, 1.0], &[3]).unwrap();
    let got = a.dist(&b, 1.0).unwrap().item();
    assert!((got - 6.0).abs() < 1e-5);
}

#[test]
fn dist_grad() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 0.5, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.3, -1.1, 1.7, 1.2], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].dist(&t[1], 2.0).unwrap());
}
