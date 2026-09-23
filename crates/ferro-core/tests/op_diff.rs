use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn diff_values() {
    let a = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4]).unwrap();
    let y = a.diff(0).unwrap();
    assert_eq!(y.shape(), &[3]);
    assert_eq!(y.to_vec(), vec![3.0, 5.0, 7.0]);

    let m = Tensor::from_vec(vec![1.0, 2.0, 4.0, 0.0, -1.0, 3.0], &[2, 3]).unwrap();
    let d1 = m.diff(1).unwrap();
    assert_eq!(d1.shape(), &[2, 2]);
    assert_eq!(d1.to_vec(), vec![1.0, 2.0, -1.0, 4.0]);
    let d0 = m.diff(0).unwrap();
    assert_eq!(d0.shape(), &[1, 3]);
    assert_eq!(d0.to_vec(), vec![-1.0, -3.0, -1.0]);
}

#[test]
fn diff_rejects_bad_dims() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(a.diff(1).is_err());
    let single = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    assert!(single.diff(0).is_err());
    assert!(Tensor::scalar(1.0).diff(0).is_err());
}

#[test]
fn diff_grad() {
    let a = Tensor::from_vec(vec![0.3, -1.2, 2.0, 0.7, 1.1, -0.4], &[2, 3]).unwrap();
    grad_check(&[a], |t| {
        let y = t[0].diff(1).unwrap();
        let c = Tensor::from_vec((0..4).map(|i| 0.15 + 0.27 * i as f32).collect::<Vec<_>>(), &[2, 2]).unwrap();
        y.mul(&c).unwrap().sum()
    });
}
