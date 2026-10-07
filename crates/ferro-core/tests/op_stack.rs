use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn stack_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    let s0 = Tensor::stack(&[a.clone(), b.clone()], 0).unwrap();
    assert_eq!(s0.shape(), &[2, 3]);
    assert_eq!(s0.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let s1 = Tensor::stack(&[a, b], 1).unwrap();
    assert_eq!(s1.shape(), &[3, 2]);
    assert_eq!(s1.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let i = Tensor::from_vec_i64(vec![7, 8], &[2]).unwrap();
    assert_eq!(Tensor::stack(&[i.clone(), i], 0).unwrap().to_vec_i64(), vec![7, 8, 7, 8]);
}

#[test]
fn stack_errors() {
    let a = Tensor::zeros(&[2, 2]);
    assert!(Tensor::stack(&[], 0).is_err());
    assert!(Tensor::stack(&[a.clone(), Tensor::zeros(&[2, 3])], 0).is_err());
    assert!(Tensor::stack(&[a.clone(), a], 3).is_err());
}

#[test]
fn stack_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.0, 2.0, 1.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.3, 0.7, -0.2, 1.1], &[2, 2]).unwrap();
    let w = Tensor::from_vec((0..8).map(|i| i as f32 * 0.3 - 1.0).collect(), &[2, 2, 2]).unwrap();
    grad_check(&[a, b], move |t| Tensor::stack(t, 1).unwrap().mul(&w).unwrap().sum());
}
