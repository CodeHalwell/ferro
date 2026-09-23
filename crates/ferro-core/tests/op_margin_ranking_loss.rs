use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn margin_ranking_loss_values() {
    let x1 = Tensor::from_vec(vec![1.0, 2.0, 0.5, -1.0], &[4]).unwrap();
    let x2 = Tensor::from_vec(vec![0.5, 3.0, 0.5, 1.0], &[4]).unwrap();
    let target = Tensor::from_vec(vec![1.0, -1.0, 1.0, 1.0], &[4]).unwrap();
    let got = x1.margin_ranking_loss(&x2, &target, 0.2).unwrap();
    assert_eq!(got.shape(), &[] as &[usize]);
    // losses = [max(0,-0.3), max(0,-0.8), 0.2, 2.2]; mean = 2.4/4 = 0.6
    assert!((got.item() - 0.6).abs() < 1e-5);
}

#[test]
fn margin_ranking_loss_shape_mismatch_is_err() {
    let x1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let x2 = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let target = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();
    assert!(x1.margin_ranking_loss(&x2, &target, 0.5).is_err());
}

#[test]
fn margin_ranking_loss_grad() {
    // Active/inactive mix with no element at the hinge.
    let x1 = Tensor::from_vec(vec![0.3, -0.2, 2.5, 4.0], &[2, 2]).unwrap();
    let x2 = Tensor::from_vec(vec![1.0, -1.0, 1.0, -1.0], &[2, 2]).unwrap();
    let target = Tensor::from_vec(vec![1.0, -1.0, 1.0, -1.0], &[2, 2]).unwrap();
    grad_check(&[x1, x2, target], |t| t[0].margin_ranking_loss(&t[1], &t[2], 0.5).unwrap());
}
