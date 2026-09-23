use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn flatten_values() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let a = Tensor::from_vec(data.clone(), &[2, 3, 4]).unwrap();
    let y = a.flatten(1, 2).unwrap();
    assert_eq!(y.shape(), &[2, 12]);
    assert_eq!(y.to_vec(), data);
    assert_eq!(a.flatten(0, 2).unwrap().shape(), &[24]);
    assert_eq!(a.flatten(0, 1).unwrap().shape(), &[6, 4]);
    assert_eq!(a.flatten(1, 1).unwrap().shape(), &[2, 3, 4]);
    assert_eq!(Tensor::scalar(3.0).flatten(0, 0).unwrap().shape(), &[1]);
}

#[test]
fn flatten_rejects_bad_ranges() {
    let a = Tensor::from_vec(vec![0.0; 24], &[2, 3, 4]).unwrap();
    assert!(a.flatten(2, 1).is_err());
    assert!(a.flatten(0, 3).is_err());
    assert!(a.flatten(3, 3).is_err());
}

#[test]
fn flatten_grad() {
    let a = Tensor::from_vec(vec![0.3, -1.2, 2.0, 0.7, 1.1, -0.4], &[2, 3]).unwrap();
    grad_check(&[a], |t| {
        let y = t[0].flatten(0, 1).unwrap();
        let c = Tensor::from_vec((0..6).map(|i| 0.15 + 0.27 * i as f32).collect::<Vec<_>>(), &[6]).unwrap();
        y.mul(&c).unwrap().sum()
    });
}
