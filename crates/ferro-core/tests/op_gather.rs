use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn gather_values_torch_doc_example() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let idx = Tensor::from_vec_i64(vec![0, 0, 1, 0], &[2, 2]).unwrap();
    let got = t.gather(1, &idx).unwrap();
    assert_eq!(got.shape(), &[2, 2]);
    assert_eq!(got.to_vec(), vec![1.0, 1.0, 4.0, 3.0]);
}

#[test]
fn gather_dim0_and_narrower_index() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let idx = Tensor::from_vec_i64(vec![2, 0], &[1, 2]).unwrap();
    assert_eq!(t.gather(0, &idx).unwrap().to_vec(), vec![5.0, 2.0]);
}

#[test]
fn gather_rejects_bad_inputs() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let out_of_range = Tensor::from_vec_i64(vec![0, 2], &[1, 2]).unwrap();
    assert!(t.gather(1, &out_of_range).is_err());
    let wrong_rank = Tensor::from_vec_i64(vec![0, 1], &[2]).unwrap();
    assert!(t.gather(1, &wrong_rank).is_err());
    let too_wide = Tensor::from_vec_i64(vec![0, 0, 0], &[1, 3]).unwrap();
    assert!(t.gather(0, &too_wide).is_err());
    let float_idx = Tensor::from_vec(vec![0.0, 1.0], &[1, 2]).unwrap();
    assert!(t.gather(1, &float_idx).is_err());
}

#[test]
fn gather_grad_accumulates_duplicates() {
    let t = Tensor::from_vec(vec![0.7, -1.1, 0.4, 1.3, -0.6, 0.2], &[2, 3]).unwrap();
    let idx = Tensor::from_vec_i64(vec![1, 1, 0, 2], &[2, 2]).unwrap();
    grad_check(&[t], |x| x[0].gather(1, &idx).unwrap().mul(&x[0].gather(1, &idx).unwrap()).unwrap().sum());
}
