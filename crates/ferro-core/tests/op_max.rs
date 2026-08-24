use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn max_value() {
    let a = Tensor::from_vec(vec![1.0, 5.0, 3.0], &[3]).unwrap();
    assert!((a.max().unwrap().item() - 5.0).abs() < 1e-6);
}

#[test]
fn max_nan_propagates() {
    let a = Tensor::from_vec(vec![1.0, f32::NAN, 0.5], &[3]).unwrap();
    assert!(a.max().unwrap().item().is_nan());
}

#[test]
fn max_all_nan() {
    let a = Tensor::from_vec(vec![f32::NAN, f32::NAN], &[2]).unwrap();
    assert!(a.max().unwrap().item().is_nan());
}

#[test]
fn max_empty_is_err() {
    let a = Tensor::from_vec(vec![], &[0]).unwrap();
    assert!(a.max().is_err());
}

#[test]
fn max_grad() {
    let a = Tensor::from_vec(vec![0.2, -1.0, 1.7, 0.5], &[4]).unwrap();
    grad_check(&[a], |t| t[0].max().unwrap());
}

#[test]
fn max_ties_split_gradient() {
    let x = Tensor::from_vec(vec![1.0, 0.5, 1.0, 1.0], &[4])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    x.max().unwrap().backward();
    assert_eq!(
        x.grad().unwrap().to_vec(),
        vec![1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0]
    );
}

#[test]
fn max_nan_result_routes_gradient_to_nans() {
    let x = Tensor::from_vec(vec![1.0, f32::NAN], &[2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let m = x.max().unwrap();
    assert!(m.item().is_nan());
    m.backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![0.0, 1.0]);

    let y = Tensor::from_vec(vec![f32::NAN, 2.0, f32::NAN], &[3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    y.max().unwrap().backward();
    assert_eq!(y.grad().unwrap().to_vec(), vec![0.5, 0.0, 0.5]);
}
