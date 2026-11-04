use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::params::Param;

#[test]
fn shared_parameters_have_independent_leaf_flags_and_gradients() {
    let source = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    let parameter = Param::from_shared(&source, true).unwrap();
    let leaf = parameter.tensor();
    assert_eq!(source._storage_ptr(), leaf._storage_ptr());
    assert_ne!(source.id(), leaf.id());
    leaf.square().unwrap().sum().backward();
    assert_eq!(leaf.grad().unwrap().to_vec(), vec![2., 2.]);
    assert!(source.grad().is_none());
    with_grad_enabled(false, || source.copy_leaf_from(&Tensor::from_vec(vec![3., 4.], &[2]).unwrap()).unwrap());
    assert_eq!(parameter.tensor().to_vec(), vec![3., 4.]);
}

#[test]
fn frozen_shared_leaf_and_ordinary_owning_parameters_remain_distinct_contracts() {
    let source = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    let frozen = Param::from_shared(&source, false).unwrap();
    let owning = Param::new(source.clone());
    assert!(frozen.is_trainable());
    assert!(!frozen.tensor().requires_grad());
    assert!(source.requires_grad());
    assert_eq!(source._storage_ptr(), frozen.tensor()._storage_ptr());
    assert_ne!(source._storage_ptr(), owning.tensor()._storage_ptr());
    assert!(!frozen.tensor().sum().requires_grad());
}

#[test]
fn unsupported_shared_destinations_reject_without_changing_source() {
    let source = Tensor::ones(&[2, 3]);
    assert!(Param::from_shared(&source.transpose(0, 1).unwrap(), true).is_err());
    assert!(Param::from_shared(&source.narrow(0, 0, 1).unwrap(), true).is_err());
    assert!(Param::from_shared(&source.to_dtype(DType::I64), true).is_err());
    assert!(!source.requires_grad());
    assert_eq!(source.to_vec(), vec![1.; 6]);
}
