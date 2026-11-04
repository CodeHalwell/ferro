use ferro_core::autograd::with_grad_enabled;
use ferro_core::{DType, Tensor};

fn leaf(dtype: DType) -> Tensor {
    Tensor::from_vec_f64(vec![1., 2., 3., 4., 5., 6.], &[1, 2, 3]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap()
}

fn transform(case: usize, x: &Tensor) -> Tensor {
    match case {
        0 => x.reshape(&[6]).unwrap(),
        1 => x.transpose(1, 2).unwrap(),
        2 => x.permute(&[2, 0, 1]).unwrap(),
        3 => x.unsqueeze(1).unwrap(),
        4 => x.squeeze(0).unwrap(),
        5 => x.expand(&[4, 2, 3]).unwrap(),
        6 => x.slice(2, 1, 3, 1).unwrap(),
        7 => x.sum(),
        8 => x.mean(),
        9 => x.sum_dim(2, false).unwrap(),
        10 => x.mean_dim(2, false).unwrap(),
        11 => x.transpose(1, 2).unwrap().reshape(&[6]).unwrap(),
        _ => unreachable!(),
    }
}

fn expected(case: usize) -> Vec<f64> {
    match case {
        5 => vec![4.; 6],
        6 => vec![0., 1., 1., 0., 1., 1.],
        8 => vec![1. / 6.; 6],
        10 => vec![1. / 3.; 6],
        _ => vec![1.; 6],
    }
}

fn close(actual: &Tensor, expected: &[f64]) {
    assert_eq!(actual.numel(), expected.len());
    for (a, b) in actual.to_vec_f64().into_iter().zip(expected) { assert!((a - b).abs() < 1e-6, "{a} != {b}"); }
}

#[test]
fn metadata_only_adjoints_allow_real_input_mutation_in_both_engines() {
    for dtype in [DType::F32, DType::F64] { for case in 0..12 {
        let x = leaf(dtype);
        let identity = x.id();
        let output = transform(case, &x).sum();
        with_grad_enabled(false, || x.copy_leaf_from(&Tensor::full(x.shape(), 9.).to_dtype(dtype))).unwrap();
        let gradient = output.grad_wrt(&[&x], false).unwrap().remove(0);
        close(&gradient, &expected(case));
        assert!(x.grad().is_none());
        output.backward();
        close(&x.grad().unwrap(), &expected(case));
        assert_eq!(x.id(), identity);
        assert_eq!(x._version(), 1);
    } }
}

#[test]
fn graph_building_adjoints_keep_live_seed_without_saving_unused_input_values() {
    for case in 0..12 {
        let x = leaf(DType::F32);
        let output = transform(case, &x).sum();
        let seed = Tensor::scalar(0.25).requires_grad_(true).unwrap();
        with_grad_enabled(false, || x.copy_leaf_from(&Tensor::full(x.shape(), 9.))).unwrap();
        let gradient = output.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
        close(&gradient, &expected(case).iter().map(|v| v * 0.25).collect::<Vec<_>>());
        with_grad_enabled(false, || x.copy_leaf_from(&Tensor::full(x.shape(), 11.))).unwrap();
        let second = gradient.sum().grad_wrt(&[&seed, &x], false).unwrap();
        close(&second[0], &[expected(case).iter().sum()]);
        close(&second[1], &[0.; 6]);
        assert!(seed.grad().is_none() && x.grad().is_none());
    }
    let x = leaf(DType::F32);
    let selected = x.slice(2, 1, 3, 1).unwrap();
    let seed = Tensor::ones(selected.shape()).requires_grad_(true).unwrap();
    let adjoint = selected.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
    with_grad_enabled(false, || seed.copy_leaf_from(&Tensor::full(seed.shape(), 7.))).unwrap();
    close(&adjoint.sum().grad_wrt(&[&seed], false).unwrap()[0], &[1.; 4]);
}

#[test]
fn value_dependent_consumers_and_saved_outputs_still_reject_modified_storage() {
    for dtype in [DType::F32, DType::F64] {
        let x = leaf(dtype);
        let view = x.transpose(1, 2).unwrap();
        let loss = view.mul(&view).unwrap().sum();
        with_grad_enabled(false, || x.copy_leaf_from(&Tensor::full(x.shape(), 9.).to_dtype(dtype))).unwrap();
        assert!(loss.grad_wrt(&[&x], false).is_err());
        assert!(x.grad().is_none());

        let x = leaf(dtype);
        let nonlinear = x.mul(&x).unwrap().reshape(&[6]).unwrap().sum();
        with_grad_enabled(false, || x.copy_leaf_from(&Tensor::full(x.shape(), 9.).to_dtype(dtype))).unwrap();
        assert!(nonlinear.grad_wrt(&[&x], false).is_err());

        let x = leaf(dtype);
        let output = x.exp();
        output.detach().copy_from(&Tensor::zeros(output.shape()).to_dtype(dtype)).unwrap();
        assert!(output.sum().grad_wrt(&[&x], false).is_err());
    }
}

#[test]
fn unknown_custom_and_optional_rules_retain_conservative_input_version_checks() {
    let x = leaf(DType::F32);
    let shape = x.shape().to_vec();
    let custom = Tensor::scalar(0.).record_fn(vec![x.clone()], move |_| vec![Tensor::ones(&shape)]);
    let optional = Tensor::scalar(0.).record_fn_optional(vec![x.clone()], |_| vec![None]);
    with_grad_enabled(false, || x.copy_leaf_from(&Tensor::full(x.shape(), 9.))).unwrap();
    assert!(custom.grad_wrt(&[&x], false).is_err());
    assert!(optional.grad_wrt(&[&x], false).is_err());
    assert!(x.grad().is_none());
}
