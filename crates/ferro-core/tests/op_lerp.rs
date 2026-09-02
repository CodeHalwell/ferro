use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn lerp_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![3.0, 6.0, 9.0], &[3]).unwrap();
    let got = a.lerp(&b, 0.25).unwrap().to_vec();
    assert!((got[0] - 1.5).abs() < 1e-5);
    assert!((got[1] - 3.0).abs() < 1e-5);
    assert!((got[2] - 4.5).abs() < 1e-5);
}

#[test]
fn lerp_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![0.5, -0.5, 1.5], &[3]).unwrap();
    let got = a.lerp(&b, 0.4).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    let av = a.to_vec();
    let bv = vec![0.5f32, -0.5, 1.5, 0.5, -0.5, 1.5];
    let gv = got.to_vec();
    for i in 0..6 {
        let expected = av[i] + 0.4 * (bv[i] - av[i]);
        assert!((gv[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn lerp_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, 2.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![-0.4, 0.9, -1.1, 1.8], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].lerp(&t[1], 0.3).unwrap().sum());
}
