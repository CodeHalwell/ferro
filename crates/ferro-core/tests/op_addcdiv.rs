use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn addcdiv_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let t1 = Tensor::from_vec(vec![4.0, 9.0, 16.0], &[3]).unwrap();
    let t2 = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap();
    let got = a.addcdiv(&t1, &t2, 0.5).unwrap().to_vec();
    assert!((got[0] - 2.0).abs() < 1e-5);
    assert!((got[1] - 3.5).abs() < 1e-5);
    assert!((got[2] - 5.0).abs() < 1e-5);
}

#[test]
fn addcdiv_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let t1 = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0], &[2, 3]).unwrap();
    let t2 = Tensor::from_vec(vec![2.0, 4.0, 6.0], &[3]).unwrap();
    let got = a.addcdiv(&t1, &t2, 0.5).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    let av = a.to_vec();
    let t1v = t1.to_vec();
    let t2v = t2.to_vec();
    let gv = got.to_vec();
    for i in 0..6 {
        let expected = av[i] + 0.5 * (t1v[i] / t2v[i % 3]);
        assert!((gv[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn addcdiv_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, -2.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.1, -0.4, 0.9, -1.3], &[2, 2]).unwrap();
    let c = Tensor::from_vec(vec![2.0, 3.0, -2.5, 4.0], &[2, 2]).unwrap();
    grad_check(&[a, b, c], |t| t[0].addcdiv(&t[1], &t[2], 0.5).unwrap().sum());
}
