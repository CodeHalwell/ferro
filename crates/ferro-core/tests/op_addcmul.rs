use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn addcmul_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let t1 = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap();
    let t2 = Tensor::from_vec(vec![5.0, 6.0, 7.0], &[3]).unwrap();
    let got = a.addcmul(&t1, &t2, 0.5).unwrap().to_vec();
    assert!((got[0] - 6.0).abs() < 1e-5);
    assert!((got[1] - 11.0).abs() < 1e-5);
    assert!((got[2] - 17.0).abs() < 1e-5);
}

#[test]
fn addcmul_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let t1 = Tensor::from_vec(vec![1.0, 0.0, -1.0], &[3]).unwrap();
    let t2 = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 3]).unwrap();
    let got = a.addcmul(&t1, &t2, 2.0).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    let av = a.to_vec();
    let t1v = vec![1.0f32, 0.0, -1.0, 1.0, 0.0, -1.0];
    let t2v = t2.to_vec();
    let gv = got.to_vec();
    for i in 0..6 {
        let expected = av[i] + 2.0 * (t1v[i] * t2v[i]);
        assert!((gv[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn addcmul_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, 2.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![-0.4, 0.9, -1.1, 1.8], &[2, 2]).unwrap();
    let c = Tensor::from_vec(vec![0.6, -1.3, 0.8, -0.2], &[2, 2]).unwrap();
    grad_check(&[a, b, c], |t| t[0].addcmul(&t[1], &t[2], 0.5).unwrap().sum());
}
