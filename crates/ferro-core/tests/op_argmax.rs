use ferro_core::Tensor;

#[test]
fn argmax_argmin_values() {
    let a = Tensor::from_vec(vec![1.0, 5.0, 3.0, 4.0, 0.0, 2.0], &[2, 3]).unwrap();
    let am = a.argmax(1, false).unwrap();
    assert_eq!(am.shape(), &[2]);
    assert_eq!(am.to_vec_i64(), vec![1, 0]);
    assert_eq!(a.argmin(1, false).unwrap().to_vec_i64(), vec![0, 1]);
    assert_eq!(a.argmax(0, false).unwrap().to_vec_i64(), vec![1, 0, 0]);
}

#[test]
fn argmax_keepdim_shape() {
    let a = Tensor::from_vec(vec![1.0, 5.0, 3.0, 4.0, 0.0, 2.0], &[2, 3]).unwrap();
    let am = a.argmax(1, true).unwrap();
    assert_eq!(am.shape(), &[2, 1]);
    assert_eq!(am.to_vec_i64(), vec![1, 0]);
}

#[test]
fn argmax_ties_take_lowest_index_and_nan_wins() {
    let a = Tensor::from_vec(vec![2.0, 2.0, 1.0], &[3]).unwrap();
    assert_eq!(a.argmax(0, false).unwrap().to_vec_i64(), vec![0]);
    let b = Tensor::from_vec(vec![1.0, f32::NAN, 9.0], &[3]).unwrap();
    assert_eq!(b.argmax(0, false).unwrap().to_vec_i64(), vec![1]);
    assert_eq!(b.argmin(0, false).unwrap().to_vec_i64(), vec![1]);
}

#[test]
fn argmax_rejects_bad_dim() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    assert!(a.argmax(1, false).is_err());
}
