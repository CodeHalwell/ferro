use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn unsqueeze_values() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::from_vec(data.clone(), &[2, 3]).unwrap();
    for (dim, shape) in [(0, [1, 2, 3]), (1, [2, 1, 3]), (2, [2, 3, 1])] {
        let out = a.unsqueeze(dim).unwrap();
        assert_eq!(out.shape(), &shape);
        assert_eq!(out.to_vec(), data);
    }
}

#[test]
fn squeeze_values() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::from_vec(data.clone(), &[1, 2, 3]).unwrap();
    let out = a.squeeze(0).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.to_vec(), data);

    let b = Tensor::from_vec(data.clone(), &[2, 1, 3]).unwrap();
    let out = b.squeeze(1).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.to_vec(), data);

    assert!(b.squeeze(0).is_err());
    assert!(b.squeeze(3).is_err());
    assert!(b.unsqueeze(4).is_err());
}

#[test]
fn squeeze_roundtrip_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.5, 2.0, 3.0, -0.25, 1.0], &[2, 3]).unwrap();
    let w = Tensor::from_vec(vec![1.5, -2.0, 0.5, 1.0, 2.5, -1.0], &[2, 3]).unwrap();
    grad_check(&[a], |t| t[0].unsqueeze(0).unwrap().squeeze(0).unwrap().mul(&w).unwrap().sum());
}
