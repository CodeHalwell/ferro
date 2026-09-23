use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn triplet_margin_loss_values() {
    let a = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap();
    let pos = Tensor::from_vec(vec![3.0, 4.0, 1.0, 0.0], &[2, 2]).unwrap();
    let neg = Tensor::from_vec(vec![6.0, 8.0, 0.0, 0.5], &[2, 2]).unwrap();
    let got = a.triplet_margin_loss(&pos, &neg, 1.0, 2.0, 1e-6).unwrap();
    assert_eq!(got.shape(), &[] as &[usize]);
    // row0: max(0, 5 - 10 + 1) = 0; row1: max(0, 1 - 0.5 + 1) = 1.5; mean 0.75
    assert!((got.item() - 0.75).abs() < 1e-4);
    // p = 1 changes row0's distances (7 vs 14) but not the active row.
    let got1 = a.triplet_margin_loss(&pos, &neg, 1.0, 1.0, 1e-6).unwrap();
    assert!((got1.item() - 0.75).abs() < 1e-4);
    // A larger margin activates row0: max(0, 5 - 10 + 6) = 1; mean (1 + 6.5)/2
    let got2 = a.triplet_margin_loss(&pos, &neg, 6.0, 2.0, 1e-6).unwrap();
    assert!((got2.item() - 3.75).abs() < 1e-4);
}

#[test]
fn triplet_margin_loss_shape_mismatch_is_err() {
    let a = Tensor::from_vec(vec![0.0, 0.0], &[1, 2]).unwrap();
    let pos = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    assert!(a.triplet_margin_loss(&pos, &a, 1.0, 2.0, 1e-6).is_err());
}

#[test]
fn triplet_margin_loss_grad() {
    // Row 0 is inactive and row 1 active; no row sits at the hinge or at zero distance.
    let a = Tensor::from_vec(vec![0.2, -0.4, 1.1, 0.7], &[2, 2]).unwrap();
    let pos = Tensor::from_vec(vec![0.5, 0.1, -0.3, 1.2], &[2, 2]).unwrap();
    let neg = Tensor::from_vec(vec![-0.6, 0.9, 1.4, 0.2], &[2, 2]).unwrap();
    grad_check(&[a, pos, neg], |t| t[0].triplet_margin_loss(&t[1], &t[2], 0.5, 2.0, 1e-6).unwrap());
}
