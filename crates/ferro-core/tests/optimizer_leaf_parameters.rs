use ferro_core::{DType, Param, Tensor, optim::{Adaptive, Adam, AdamW, OptimizerState, Sgd}};

#[test]
fn exact_leaf_construction_preserves_storage_gradient_slot_and_independent_flags() {
    for requires_grad in [false, true] {
        let tensor = Tensor::from_vec(vec![1., 2.], &[2]).unwrap().requires_grad_(requires_grad).unwrap();
        let gradient = Tensor::from_vec(vec![2., 3.], &[2]).unwrap();
        tensor.set_leaf_grad(Some(gradient.clone())).unwrap();
        let parameter = Param::from_leaf(&tensor).unwrap();
        assert_eq!(parameter.tensor().id(), tensor.id());
        assert_eq!(parameter.tensor()._storage_ptr(), tensor._storage_ptr());
        assert_eq!(parameter.grad().unwrap().id(), gradient.id());
        assert_eq!(tensor.requires_grad(), requires_grad);
        assert!(parameter.is_trainable());
        let mut optimizer = Sgd::new(vec![parameter.clone()], 0.1);
        optimizer.step();
        assert_eq!(tensor.to_vec(), vec![0.8, 1.7]);
        assert_eq!(parameter.tensor().id(), tensor.id());
        assert_eq!(tensor.requires_grad(), requires_grad);
        assert!(tensor.grad().is_none(), "raw Rust steps retain their consuming contract");
        tensor.set_leaf_grad(Some(gradient)).unwrap();
        parameter.set_trainable(false);
        assert!(!parameter.is_trainable());
        assert!(tensor.grad().is_none());
        assert_eq!(tensor.requires_grad(), requires_grad);
    }
    for shape in [&[][..], &[0][..], &[0, 3][..]] {
        let tensor = Tensor::zeros(shape);
        assert_eq!(Param::from_leaf(&tensor).unwrap().tensor().id(), tensor.id());
    }
}

#[test]
fn nonleaf_typed_and_partial_layout_inputs_reject_without_mutation() {
    let tensor = Tensor::ones(&[2, 3]).requires_grad_(true).unwrap();
    let gradient = Tensor::full(&[2, 3], 2.);
    tensor.set_leaf_grad(Some(gradient.clone())).unwrap();
    for invalid in [tensor.square().unwrap(), tensor.transpose(0, 1).unwrap().detach(),
        tensor.narrow(0, 0, 1).unwrap().detach(), tensor.to_dtype(DType::Bool).detach(),
        Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap()] {
        let before = (invalid.id(), invalid._version(), invalid.to_vec_f64());
        assert!(Param::from_leaf(&invalid).is_err());
        assert_eq!((invalid.id(), invalid._version(), invalid.to_vec_f64()), before);
    }
    assert_eq!(tensor.grad().unwrap().id(), gradient.id());
    assert_eq!(tensor.to_vec(), vec![1.; 6]);
    assert!(tensor.requires_grad());
}

#[test]
fn conflicting_leaf_wrappers_reject_before_native_state_or_parameter_updates() {
    let tensor = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    let gradient = Tensor::full(&[2], 2.);
    tensor.set_leaf_grad(Some(gradient.clone())).unwrap();
    let first = Param::from_leaf(&tensor).unwrap();
    let second = Param::from_leaf(&tensor).unwrap();
    assert_ne!(first.identity(), second.identity());
    let mut optimizer = Adaptive::new(vec![first.clone(), second], &[5., 0.1, 0.9, 1e-8, 0., 0., 0., 0.]).unwrap();
    let before: Vec<_> = optimizer.snapshot().into_iter().map(|(key, value)| (key, value.to_vec())).collect();
    assert!(optimizer.step().is_err());
    assert_eq!(optimizer.snapshot().into_iter().map(|(key, value)| (key, value.to_vec())).collect::<Vec<_>>(), before);
    assert_eq!(tensor.to_vec(), vec![1.; 2]);
    assert_eq!(tensor.grad().unwrap().id(), gradient.id());
    let mut tied = Adaptive::new(vec![first.clone(), first], &[5., 0.1, 0.9, 1e-8, 0., 0., 0., 0.]).unwrap();
    assert_eq!(tied.state_parameters().unwrap().len(), 1);
    tied.step().unwrap();
    assert_ne!(tensor.to_vec(), vec![1.; 2]);
}

#[test]
fn legacy_fused_updates_materialize_gradient_views_without_replacing_the_parameter_leaf() {
    fn verify<O: OptimizerState>(a: &Param, b: &Param, mut actual: O, mut control: O, step: fn(&mut O)) {
        let held = a.tensor();
        for iteration in 0..4 {
            let source = Tensor::from_vec((0..6).map(|i| 0.1 + (i + iteration) as f32 * 0.07).collect(), &[3, 2]).unwrap();
            let gradient = if iteration % 2 == 0 { source.narrow(0, 1, 2).unwrap() }
                else { source.narrow(0, 0, 2).unwrap().transpose(0, 1).unwrap() };
            a.tensor().set_leaf_grad(Some(gradient.clone())).unwrap();
            b.tensor().set_leaf_grad(Some(Tensor::from_vec(gradient.to_vec(), &[2, 2]).unwrap())).unwrap();
            step(&mut actual); step(&mut control);
            assert_eq!(a.tensor().id(), held.id());
            assert_eq!(a.tensor()._storage_ptr(), held._storage_ptr());
            assert_eq!(held.to_vec(), b.tensor().to_vec());
            assert!(!held.requires_grad());
            for ((name, value), (key, expected)) in actual.snapshot().into_iter().zip(control.snapshot()) {
                assert_eq!(name, key); assert_eq!(value.to_vec(), expected.to_vec());
            }
        }
    }
    for family in 0..3 {
        let tensor = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
        let a = Param::from_leaf(&tensor).unwrap(); let b = Param::new(tensor);
        match family {
            0 => verify(&a, &b, Sgd::new(vec![a.clone()], 0.1).with_momentum(0.5), Sgd::new(vec![b.clone()], 0.1).with_momentum(0.5), Sgd::step),
            1 => verify(&a, &b, Adam::new(vec![a.clone()], 0.1), Adam::new(vec![b.clone()], 0.1), Adam::step),
            _ => verify(&a, &b, AdamW::new(vec![a.clone()], 0.1), AdamW::new(vec![b.clone()], 0.1), AdamW::step),
        }
    }
}

#[test]
fn active_forward_level_rejects_before_configuration_or_native_optimizer_state_changes() {
    let tensor = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    let parameter = Param::from_leaf(&tensor).unwrap();
    let gradient = Tensor::full(&[2], 2.); tensor.set_leaf_grad(Some(gradient.clone())).unwrap();
    let mut optimizer = Adaptive::new(vec![parameter.clone()], &[5., 0.1, 0.9, 1e-8, 0., 0., 0., 0.]).unwrap();
    let before: Vec<_> = optimizer.snapshot().into_iter().map(|(key, value)| (key, value.to_vec())).collect();
    let primal = Tensor::ones(&[1]); let tangent = Tensor::ones(&[1]);
    {
        let _level = ferro_core::forward_ad::DualLevel::new(&[&primal], &[&tangent]).unwrap();
        assert!(ferro_core::optim::validate_gradient_storage(&[parameter]).is_err());
        assert!(optimizer.step().is_err());
    }
    assert_eq!(tensor.to_vec(), vec![1.; 2]); assert_eq!(tensor.grad().unwrap().id(), gradient.id());
    assert_eq!(optimizer.snapshot().into_iter().map(|(key, value)| (key, value.to_vec())).collect::<Vec<_>>(), before);
    optimizer.step().unwrap(); assert_ne!(tensor.to_vec(), vec![1.; 2]);
}
