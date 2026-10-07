use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn expand_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let e = a.expand(&[2, 3, 2]).unwrap();
    assert_eq!(e.shape(), &[2, 3, 2]);
    assert_eq!(e.to_vec(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    assert_eq!(e._storage_ptr(), a._storage_ptr());
    assert_eq!(a.expand(&[3, 1]).unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
    let r = e.reshape(&[12]).unwrap();
    assert_eq!(r.to_vec(), e.to_vec());
}

#[test]
fn expand_errors() {
    let a = Tensor::zeros(&[3, 2]);
    assert!(a.expand(&[3, 4]).is_err());
    assert!(a.expand(&[2]).is_err());
    assert!(a.expand(&[2, 2, 2]).is_err());
}

#[test]
fn expand_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.0, 2.0], &[3, 1]).unwrap();
    let w = Tensor::from_vec((0..12).map(|i| i as f32 * 0.2 - 1.0).collect(), &[2, 3, 2]).unwrap();
    grad_check(&[a], move |t| t[0].expand(&[2, 3, 2]).unwrap().mul(&w).unwrap().sum());
}
