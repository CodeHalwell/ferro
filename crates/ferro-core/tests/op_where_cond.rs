use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn where_values() {
    let cond = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[4]).unwrap();
    let a = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
    let b = Tensor::from_vec(vec![-1.0, -2.0, -3.0, -4.0], &[4]).unwrap();
    let got = Tensor::where_cond(&cond, &a, &b).unwrap();
    assert_eq!(got.to_vec(), vec![10.0, -2.0, 30.0, -4.0]);
}

#[test]
fn where_values_broadcast() {
    // cond [4] broadcast against a [2,4] and a one-element b; nonzero = true.
    let cond = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], &[4]).unwrap();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
    let b = Tensor::from_vec(vec![9.0], &[1]).unwrap();
    let got = Tensor::where_cond(&cond, &a, &b).unwrap();
    assert_eq!(got.shape(), &[2, 4]);
    assert_eq!(got.to_vec(), vec![1.0, 9.0, 9.0, 4.0, 5.0, 9.0, 9.0, 8.0]);
}

#[test]
fn where_grad() {
    let mask = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[4]).unwrap();
    let w = Tensor::from_vec(vec![0.5, -1.0, 2.0, 3.0], &[4]).unwrap();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let b = Tensor::from_vec(vec![-1.0, -2.0, -3.0, -4.0], &[4]).unwrap();
    grad_check(&[a, b], |t| {
        Tensor::where_cond(&mask, &t[0], &t[1])
            .unwrap()
            .mul(&w)
            .unwrap()
            .sum()
    });
}

#[test]
fn where_grad_broadcast() {
    // a [2,4], b [1]: exercises unbroadcast of both gradients.
    let mask = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[4]).unwrap();
    let w = Tensor::from_vec(vec![0.5, -1.0, 2.0, 3.0, -0.5, 1.5, -2.0, 0.25], &[2, 4]).unwrap();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
    let b = Tensor::from_vec(vec![0.5], &[1]).unwrap();
    grad_check(&[a, b], |t| {
        Tensor::where_cond(&mask, &t[0], &t[1])
            .unwrap()
            .mul(&w)
            .unwrap()
            .sum()
    });
}
