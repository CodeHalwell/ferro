use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn max_pool_values() {
    let a = Tensor::from_vec((1..=16).map(|v| v as f32).collect(), &[1, 1, 4, 4]).unwrap();
    let p = a.max_pool2d(2, 2).unwrap();
    assert_eq!(p.shape(), &[1, 1, 2, 2]);
    assert_eq!(p.to_vec(), vec![6.0, 8.0, 14.0, 16.0]);

    let b = Tensor::from_vec(vec![1.0, 5.0, 2.0, 3.0, 4.0, 6.0, 9.0, 8.0, 7.0], &[1, 1, 3, 3]).unwrap();
    let q = b.max_pool2d(2, 1).unwrap();
    assert_eq!(q.shape(), &[1, 1, 2, 2]);
    assert_eq!(q.to_vec(), vec![5.0, 6.0, 9.0, 8.0]);
}

#[test]
fn max_pool_errors() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(a.max_pool2d(2, 2).is_err());

    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
    assert!(b.max_pool2d(3, 1).is_err());
}

#[test]
fn max_pool_grad() {
    // Unique, well-separated values so the argmax is stable under fd eps.
    let vals = vec![0.3, 1.2, -0.5, 2.1, 0.9, -1.3, 1.6, 0.1, -0.8, 2.4, 0.6, -0.2, 1.9, 0.4, -1.1, 0.8];

    let a = Tensor::from_vec(vals.clone(), &[1, 1, 4, 4]).unwrap();
    let w = Tensor::from_vec(vec![0.5, -1.0, 2.0, 3.0], &[1, 1, 2, 2]).unwrap();
    grad_check(&[a], move |t| t[0].max_pool2d(2, 2).unwrap().mul(&w).unwrap().sum());

    // Overlapping windows: one input element can win several windows, so the
    // backward scatter must ADD contributions rather than overwrite.
    let b = Tensor::from_vec(vals, &[1, 1, 4, 4]).unwrap();
    let w2 = Tensor::from_vec(vec![0.5, -1.0, 2.0, 3.0, -0.5, 1.5, 2.5, -2.0, 1.0], &[1, 1, 3, 3]).unwrap();
    grad_check(&[b], move |t| t[0].max_pool2d(2, 1).unwrap().mul(&w2).unwrap().sum());
}
