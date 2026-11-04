use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::ops_ext::embedding_options::embedding_with_options as embedding;

#[test]
fn ordinary_lookup_matches_finite_differences() {
    let weight = Tensor::from_vec(vec![0.2, -0.4, 0.7, 1.1, -0.3, 0.8], &[3, 2]).unwrap();
    let ids = Tensor::from_vec_i64(vec![2, 0, 2, 1], &[2, 2]).unwrap();
    ferro_core::testkit::grad_check(&[weight], |values| embedding(&values[0], &ids, None, None, 2., false).unwrap().square().unwrap().sum());
}

#[test]
fn padding_and_frequency_are_independent_of_weight_values() {
    let weight = Tensor::from_vec_f64(vec![1., 2., 3., 4., 5., 6.], &[3, 2]).unwrap().requires_grad_(true).unwrap();
    let ids = Tensor::from_vec_i64(vec![0, 1, 1, 2, 2, 2], &[2, 3]).unwrap();
    let output = embedding(&weight, &ids, Some(0), None, 2., true).unwrap();
    assert_eq!(output.shape(), &[2, 3, 2]);
    assert_eq!(&output.to_vec_f64()[..2], &[1., 2.]);
    output.sum().backward();
    assert_eq!(weight.grad().unwrap().to_vec_f64(), vec![0., 0., 1., 1., 1., 1.]);
}

#[test]
fn renorm_is_inplace_no_grad_and_multiple_forwards_remain_differentiable() {
    let weight = Tensor::from_vec(vec![3., 4., 0., 2.], &[2, 2]).unwrap().requires_grad_(true).unwrap();
    let ids = Tensor::from_vec_i64(vec![0], &[1]).unwrap();
    let storage = weight._storage_ptr();
    let a = embedding(&weight, &ids, None, Some(1.), 2., false).unwrap();
    let b = embedding(&weight, &ids, None, Some(1.), 2., false).unwrap();
    assert_eq!(weight._storage_ptr(), storage);
    assert_eq!(weight._version(), 2);
    let values = weight.to_vec();
    assert!((values[0] - 0.6).abs() < 1e-6 && (values[1] - 0.8).abs() < 1e-6);
    assert_eq!(&values[2..], &[0., 2.]);
    a.sum().add(&b.sum()).unwrap().backward();
    assert_eq!(weight.grad().unwrap().to_vec(), vec![2., 2., 0., 0.]);
}

#[test]
fn saved_indices_are_checked_but_weight_mutation_is_allowed() {
    let weight = Tensor::ones(&[3, 2]).requires_grad_(true).unwrap();
    let ids = Tensor::from_vec_i64(vec![0, 1], &[2]).unwrap();
    let output = embedding(&weight, &ids, None, None, 2., false).unwrap();
    with_grad_enabled(false, || weight.copy_leaf_from(&Tensor::zeros(&[3, 2])).unwrap());
    output.sum().backward();
    assert_eq!(weight.grad().unwrap().to_vec(), vec![1., 1., 1., 1., 0., 0.]);
    let output = embedding(&weight, &ids, None, None, 2., false).unwrap();
    ids.copy_from(&Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap()).unwrap();
    assert!(output.vjp_wrt(&[&weight], &Tensor::ones(&[2, 2]), false).is_err());
}

#[test]
fn zero_width_lookup_does_not_expand_indices_or_loop_over_virtual_rows() {
    let count = 1_000_000_000_000usize;
    let weight = Tensor::from_vec_f64(vec![], &[count, 0]).unwrap().requires_grad_(true).unwrap();
    let ids = Tensor::from_vec_i64(vec![1], &[1]).unwrap().expand(&[count]).unwrap();
    let output = embedding(&weight, &ids, None, Some(1.), 2., true).unwrap();
    assert_eq!(output.shape(), &[count, 0]);
    assert_eq!(output.dtype(), DType::F64);
    assert_eq!(weight._version(), 1);
    output.sum().backward();
    assert_eq!(weight.grad().unwrap().shape(), &[count, 0]);
}

#[test]
fn validation_and_capture_reject_before_weight_changes() {
    let weight = Tensor::ones(&[3, 2]);
    let ids = Tensor::from_vec_i64(vec![-1], &[1]).unwrap();
    assert!(embedding(&weight, &ids, None, Some(1.), 2., false).is_err());
    assert_eq!(weight._version(), 0);
    assert_eq!(weight.to_vec(), vec![1.; 6]);
    assert!(embedding(&weight, &Tensor::zeros(&[1]), None, None, 2., false).is_err());
    assert!(embedding(&weight, &Tensor::from_vec_i64(vec![0], &[1]).unwrap(), Some(3), None, 2., false).is_err());
    ferro_core::capture(|| {
        assert!(embedding(&weight, &Tensor::from_vec_i64(vec![0], &[1]).unwrap(), None, Some(1.), 2., false).is_err());
    });
    assert_eq!(weight._version(), 0);
}
