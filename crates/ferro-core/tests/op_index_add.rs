use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn index_add_values() {
    let a = Tensor::zeros(&[3, 2]);
    let idx = Tensor::from_vec_i64(vec![2, 0, 2], &[3]).unwrap();
    let src = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let out = a.index_add(0, &idx, &src).unwrap();
    assert_eq!(out.to_vec(), vec![3.0, 4.0, 0.0, 0.0, 6.0, 8.0]);

    let b = Tensor::ones(&[2, 3]);
    let idx1 = Tensor::from_vec_i64(vec![1], &[1]).unwrap();
    let col = Tensor::from_vec(vec![10.0, 20.0], &[2, 1]).unwrap();
    assert_eq!(b.index_add(1, &idx1, &col).unwrap().to_vec(), vec![1.0, 11.0, 1.0, 1.0, 21.0, 1.0]);
}

#[test]
fn index_add_errors() {
    let a = Tensor::zeros(&[3, 2]);
    let src = Tensor::zeros(&[2, 2]);
    let idx = Tensor::from_vec_i64(vec![0, 1], &[2]).unwrap();
    assert!(a.index_add(2, &idx, &src).is_err());
    assert!(a.index_add(0, &Tensor::from_vec_i64(vec![0, 3], &[2]).unwrap(), &src).is_err());
    assert!(a.index_add(0, &Tensor::from_vec_i64(vec![0, -1], &[2]).unwrap(), &src).is_err());
    assert!(a.index_add(0, &Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap(), &src).is_err());
    assert!(a.index_add(0, &idx, &Tensor::zeros(&[3, 2])).is_err());
    assert!(a.index_add(0, &Tensor::from_vec_i64(vec![0, 1], &[1, 2]).unwrap(), &src).is_err());
}

#[test]
fn index_add_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.0, 2.0, 1.5, -0.25, 0.75], &[3, 2]).unwrap();
    let src = Tensor::from_vec(vec![0.3, 0.7, -0.2, 1.1, 0.4, -0.6], &[3, 2]).unwrap();
    let idx = Tensor::from_vec_i64(vec![2, 0, 2], &[3]).unwrap();
    let w = Tensor::from_vec(vec![1.0, -2.0, 0.5, 3.0, -1.5, 0.25], &[3, 2]).unwrap();
    grad_check(&[a, src], move |t| t[0].index_add(0, &idx, &t[1]).unwrap().mul(&w).unwrap().sum());
}
