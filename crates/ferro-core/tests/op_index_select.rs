use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn index_select_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

    let s = a.index_select(0, &[2, 0]).unwrap();
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.to_vec(), vec![5.0, 6.0, 1.0, 2.0]);

    let c = a.index_select(1, &[1]).unwrap();
    assert_eq!(c.shape(), &[3, 1]);
    assert_eq!(c.to_vec(), vec![2.0, 4.0, 6.0]);

    let d = a.index_select(0, &[1, 1]).unwrap();
    assert_eq!(d.shape(), &[2, 2]);
    assert_eq!(d.to_vec(), vec![3.0, 4.0, 3.0, 4.0]);
}

#[test]
fn index_select_errors() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    assert!(a.index_select(2, &[0]).is_err());
    assert!(a.index_select(0, &[0, 3]).is_err());
}

#[test]
fn index_select_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let w = Tensor::from_vec(vec![0.5, -1.0, 2.0, 3.0], &[2, 2]).unwrap();
    grad_check(&[a], move |t| t[0].index_select(0, &[2, 0]).unwrap().mul(&w).unwrap().sum());

    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let w2 = Tensor::from_vec(vec![0.5, -1.0, 2.0, 3.0], &[2, 2]).unwrap();
    grad_check(&[b], move |t| t[0].index_select(0, &[1, 1]).unwrap().mul(&w2).unwrap().sum());
}
