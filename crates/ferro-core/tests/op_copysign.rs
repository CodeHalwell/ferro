use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn copysign_values() {
    let a = Tensor::from_vec(vec![3.0, -3.0, 5.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 1.0, -1.0], &[3]).unwrap();
    let got = a.copysign(&b).unwrap().to_vec();
    assert!((got[0] - 3.0).abs() < 1e-5);
    assert!((got[1] - 3.0).abs() < 1e-5);
    assert!((got[2] - -5.0).abs() < 1e-5);
}

#[test]
fn copysign_broadcast() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![-1.0, 1.0, -1.0], &[3]).unwrap();
    let got = a.copysign(&b).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    let av = a.to_vec();
    let bv = vec![-1.0f32, 1.0, -1.0, -1.0, 1.0, -1.0];
    let gv = got.to_vec();
    for i in 0..6 {
        assert!((gv[i] - av[i].copysign(bv[i])).abs() < 1e-5);
    }
}

#[test]
fn copysign_grad() {
    let a = Tensor::from_vec(vec![1.5, -2.5, 3.5, -0.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![2.0, 1.0, -1.0, -2.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].copysign(&t[1]).unwrap().sum());
}
