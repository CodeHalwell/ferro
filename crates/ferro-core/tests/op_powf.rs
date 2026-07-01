use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn powf_values() {
    let a = Tensor::from_vec(vec![2.0, 4.0], &[2]).unwrap();
    let cubed = a.powf(3.0).to_vec();
    assert!((cubed[0] - 8.0).abs() < 1e-5);
    let root = a.powf(0.5).to_vec();
    assert!((root[1] - 2.0).abs() < 1e-5);
}

#[test]
fn powf_grad() {
    let a = Tensor::from_vec(vec![0.5, 1.5, 2.0, 3.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].powf(3.0).sum());

    let b = Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0], &[2, 2]).unwrap();
    grad_check(&[b], |t| t[0].powf(2.5).sum());
}
