use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn masked_fill_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let m = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[2, 3]).unwrap();
    assert_eq!(a.masked_fill(&m, -9.0).unwrap().to_vec(), vec![-9.0, 2.0, -9.0, 4.0, 5.0, -9.0]);
    // Broadcast i64 mask over rows (the causal/padding-mask shape).
    let row = Tensor::from_vec_i64(vec![0, 0, 1], &[3]).unwrap();
    let out = a.masked_fill(&row, f32::NEG_INFINITY).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.to_vec(), vec![1.0, 2.0, f32::NEG_INFINITY, 4.0, 5.0, f32::NEG_INFINITY]);
}

#[test]
fn masked_fill_errors() {
    let a = Tensor::zeros(&[2, 3]);
    assert!(a.masked_fill(&Tensor::zeros(&[2, 2]), 0.0).is_err());
    assert!(a.masked_fill(&Tensor::zeros(&[4, 2, 3]), 0.0).is_err());
    let i = Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap();
    assert!(i.masked_fill(&Tensor::zeros(&[2]), 0.0).is_err());
}

#[test]
fn masked_fill_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.0, 2.0, 1.5, -0.25, 0.75], &[2, 3]).unwrap();
    let m = Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).unwrap();
    let w = Tensor::from_vec(vec![0.3, 0.7, -0.2, 1.1, 0.4, -0.6], &[2, 3]).unwrap();
    grad_check(&[a], move |t| t[0].masked_fill(&m, 3.0).unwrap().mul(&w).unwrap().sum());
}
