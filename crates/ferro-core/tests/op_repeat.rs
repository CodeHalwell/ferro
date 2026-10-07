use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn repeat_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let r = a.repeat(&[2, 3]).unwrap();
    assert_eq!(r.shape(), &[4, 6]);
    assert_eq!(&r.to_vec()[..12], &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
    assert_eq!(&r.to_vec()[12..], &r.to_vec()[..12]);
    let p = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap().repeat(&[2, 1, 2]).unwrap();
    assert_eq!(p.shape(), &[2, 1, 4]);
    assert_eq!(p.to_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    assert_eq!(a.repeat(&[0, 1]).unwrap().shape(), &[0, 2]);
    let i = Tensor::from_vec_i64(vec![5, 6], &[2]).unwrap();
    assert_eq!(i.repeat(&[2]).unwrap().to_vec_i64(), vec![5, 6, 5, 6]);
}

#[test]
fn repeat_errors() {
    let a = Tensor::zeros(&[2, 2]);
    assert!(a.repeat(&[2]).is_err());
    assert!(a.repeat(&[usize::MAX, 2]).is_err());
}

#[test]
fn repeat_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.0, 2.0, 1.5], &[2, 2]).unwrap();
    let w = Tensor::from_vec((0..24).map(|i| (i as f32 * 0.7).sin()).collect(), &[2, 2, 6]).unwrap();
    grad_check(&[a], move |t| t[0].repeat(&[2, 1, 3]).unwrap().mul(&w).unwrap().sum());
}
