use ferro_core::{checkpoint::Checkpoint, nn::Module, DType, Param, Result, Tensor};

fn tensor(values: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec_f64(values.to_vec(), shape).unwrap()
}

#[test]
fn owning_float_parameters_preserve_exact_values_and_independent_flags() {
    let values = [1.0000000000000002, 1e-200, 1e200, -0.0];
    let source = tensor(&values, &[2, 2]);
    let parameter = Param::try_new(source.clone()).unwrap();
    let leaf = parameter.tensor();
    assert!(!source.requires_grad());
    assert!(leaf.requires_grad());
    assert_ne!(source.id(), leaf.id());
    assert_ne!(source._storage_ptr(), leaf._storage_ptr());
    assert_eq!(leaf.dtype(), DType::F64);
    assert_eq!(leaf.to_vec_f64().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    leaf.sum().backward();
    assert_eq!(leaf.grad().unwrap().dtype(), DType::F64);
    assert_eq!(leaf.grad().unwrap().to_vec_f64(), vec![1.; 4]);
    assert!(source.grad().is_none());
    let old = Tensor::ones(&[2]);
    assert!(Param::new(old.clone()).tensor().requires_grad());
    assert!(!old.requires_grad());
}

#[test]
fn shared_and_exact_leaf_slots_keep_distinct_ownership_contracts() {
    let source = tensor(&[1.0000000000000002, 2.], &[2]).requires_grad_(true).unwrap();
    let exact = Param::from_leaf(&source).unwrap();
    let shared = Param::from_shared(&source, false).unwrap();
    assert_eq!(exact.tensor().id(), source.id());
    assert_ne!(shared.tensor().id(), source.id());
    assert_eq!(shared.tensor()._storage_ptr(), source._storage_ptr());
    assert!(!shared.tensor().requires_grad());
    assert!(source.requires_grad());
    source.sum().backward();
    assert_eq!(exact.grad().unwrap().id(), source.grad().unwrap().id());
    assert!(shared.grad().is_none());
    ferro_core::autograd::with_grad_enabled(false, || shared.tensor().copy_leaf_from(&tensor(&[3., 4.], &[2]))).unwrap();
    assert_eq!(source.to_vec_f64(), vec![3., 4.]);
    let strided = tensor(&[1., 2., 3., 4.], &[2, 2]).transpose(0, 1).unwrap();
    assert!(Param::from_leaf(&strided).is_err());
    assert!(Param::from_shared(&strided, true).is_err());
    assert_eq!(Param::try_new(strided).unwrap().tensor().to_vec_f64(), vec![1., 3., 2., 4.]);
}

#[test]
fn replacement_is_fallible_atomic_and_preserves_parameter_slot_identity() {
    let parameter = Param::new(Tensor::ones(&[2]));
    parameter.set_trainable(false);
    let identity = parameter.identity();
    let old = parameter.tensor();
    assert!(parameter.try_set(Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap()).is_err());
    let live = tensor(&[2., 3.], &[2]).requires_grad_(true).unwrap();
    assert!(parameter.try_set(live.square().unwrap()).is_err());
    let huge = tensor(&[1.], &[1]).expand(&[isize::MAX as usize / 8 + 1]).unwrap();
    assert!(parameter.try_set(huge).is_err());
    assert_eq!(parameter.tensor().id(), old.id());
    assert_eq!(parameter.tensor()._storage_ptr(), old._storage_ptr());
    let source = tensor(&[1.0000000000000002, 1e-200], &[2]);
    parameter.try_set(source.clone()).unwrap();
    assert_eq!(parameter.identity(), identity);
    assert!(!parameter.is_trainable());
    assert!(!source.requires_grad());
    assert_ne!(parameter.tensor().id(), old.id());
    assert_eq!(parameter.tensor().to_vec_f64(), source.to_vec_f64());
    let current = parameter.tensor().id();
    assert!(ferro_core::capture(|| parameter.try_set(source.clone()).map(|()| source.clone())).is_err());
    assert_eq!(parameter.tensor().id(), current);
}

struct Pair { weight: Param, bias: Param }
impl Module for Pair {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight.tensor())?.add(&self.bias.tensor())
    }
    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![("weight".into(), self.weight.clone()), ("bias".into(), self.bias.clone())]
    }
}

#[test]
fn float64_module_checkpoint_restores_bytes_without_replacing_live_leaves() {
    let module = Pair { weight: Param::new(tensor(&[1.0000000000000002, 1e-200], &[2, 1])), bias: Param::new(tensor(&[-0.0], &[1])) };
    let state = Checkpoint::from_module(7, &module);
    let identity = module.weight.tensor().id();
    let storage = module.weight.tensor()._storage_ptr();
    ferro_core::autograd::with_grad_enabled(false, || module.weight.tensor().copy_leaf_from(&tensor(&[9., 8.], &[2, 1]))).unwrap();
    ferro_core::autograd::with_grad_enabled(false, || module.bias.tensor().copy_leaf_from(&tensor(&[7.], &[1]))).unwrap();
    let mut invalid = state.clone();
    invalid.tensors[1].1 = Tensor::zeros(&[1]);
    assert!(invalid.load_into_module(&module).is_err());
    assert_eq!(module.weight.tensor().to_vec_f64(), vec![9., 8.]);
    state.load_into_module(&module).unwrap();
    assert_eq!(module.weight.tensor().id(), identity);
    assert_eq!(module.weight.tensor()._storage_ptr(), storage);
    assert_eq!(module.weight.tensor().to_vec_f64(), vec![1.0000000000000002, 1e-200]);
    assert_eq!(module.bias.tensor().to_vec_f64()[0].to_bits(), (-0.0f64).to_bits());
    let input = tensor(&[3., 4.], &[1, 2]).requires_grad_(true).unwrap();
    module.forward(&input).unwrap().sum().backward();
    assert_eq!(input.grad().unwrap().to_vec_f64(), vec![1.0000000000000002, 1e-200]);
    assert_eq!(module.weight.grad().unwrap().to_vec_f64(), vec![3., 4.]);
    assert_eq!(module.bias.grad().unwrap().to_vec_f64(), vec![1.]);
}
