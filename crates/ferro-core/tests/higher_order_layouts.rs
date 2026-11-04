use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use ferro_core::{capture, graph::CompiledChain, Tensor};

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() <= 2e-5 * (1. + b.abs()), "{a} != {b}"); }
}

#[test]
fn shared_strided_slice_cat_stack_support_four_derivatives() {
    let data = [-0.3f32, 0.2, 0.4, -0.5, 0.7, 0.6];
    let x = Tensor::from_vec(data.to_vec(), &[6]).unwrap().requires_grad_(true).unwrap();
    let selected = x.slice(0, 1, 6, 2).unwrap();
    let joined = Tensor::cat(&[selected.clone(), selected], 0).unwrap();
    let stacked = Tensor::stack(&[joined.clone(), joined], 0).unwrap();
    let mut loss = stacked.powf(4.).sum();
    for order in 1..=4 {
        let gradient = loss.grad_wrt(&[&x], order < 4).unwrap().remove(0);
        let expected = data.iter().enumerate().map(|(i, &value)| {
            if i % 2 == 0 { 0. } else { match order { 1 => 16. * value.powi(3), 2 => 48. * value.powi(2), 3 => 96. * value, _ => 96. } }
        }).collect::<Vec<_>>();
        close(&gradient.to_vec(), &expected);
        loss = gradient.sum();
    }
    assert!(x.grad().is_none());
}

#[test]
fn layout_vjp_retains_live_noncontiguous_seed_and_shared_input_contributions() {
    let x = Tensor::ones(&[2, 6]).requires_grad_(true).unwrap();
    let selected = x.slice(1, 1, 6, 2).unwrap();
    let output = Tensor::cat(&[selected.clone(), selected], 0).unwrap();
    let base = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), &[3, 4]).unwrap().requires_grad_(true).unwrap();
    let seed = base.transpose(0, 1).unwrap();
    let gradient = output.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
    close(&gradient.to_vec(), &[0., 4., 0., 12., 0., 20., 0., 6., 0., 14., 0., 22.]);
    close(&gradient.sum().grad_wrt(&[&base], false).unwrap()[0].to_vec(), &[1.; 12]);
    assert!(x.grad().is_none() && base.grad().is_none());
}

#[test]
fn generic_replay_executes_cat_slice_and_slice_adjoint_from_current_inputs() {
    let x = Tensor::from_vec(vec![0.2, 0.4, 0.6, 0.8], &[4]).unwrap().requires_grad_(true).unwrap();
    let selected = x.slice(0, 1, 4, 2).unwrap();
    let output = Tensor::cat(&[selected.clone(), selected], 0).unwrap().powf(3.).sum();
    let gradient = output.grad_wrt(&[&x], true).unwrap().remove(0);
    let compiled = CompiledChain::compile(&gradient).unwrap();
    close(&compiled.replay().unwrap().to_vec(), &[0., 0.96, 0., 3.84]);
    with_grad_enabled(false, || x.copy_leaf_from(&Tensor::from_vec(vec![0.1, 0.3, 0.5, 0.7], &[4]).unwrap())).unwrap();
    close(&compiled.replay().unwrap().to_vec(), &[0., 0.54, 0., 2.94]);
    assert!(compiled.prepare_static().is_err());

    let source = Tensor::from_vec(vec![1., 2., 3., 4.], &[4]).unwrap();
    let captured = capture(|| {
        let part = source.slice(0, 0, 4, 2)?;
        Tensor::cat(&[part.clone(), part], 0)
    }).unwrap();
    let chain = CompiledChain::compile(&captured).unwrap();
    source.copy_from(&Tensor::from_vec(vec![5., 6., 7., 8.], &[4]).unwrap()).unwrap();
    close(&chain.replay().unwrap().to_vec(), &[5., 7., 5., 7.]);
}

#[test]
fn slice_adjoint_handles_empty_shapes_and_rejects_impossible_materialization() {
    let empty = Tensor::zeros(&[0, 7]).requires_grad_(true).unwrap();
    let output = empty.slice(1, 1, 7, 2).unwrap();
    let seed = Tensor::zeros(&[0, 3]).requires_grad_(true).unwrap();
    let gradient = output.vjp_wrt(&[&empty], &seed, true).unwrap().remove(0);
    assert_eq!(gradient.shape(), &[0, 7]);
    assert!(gradient.requires_grad());
    assert_eq!(gradient.sum().grad_wrt(&[&seed], false).unwrap()[0].shape(), &[0, 3]);

    let huge = Tensor::ones(&[1]).expand(&[isize::MAX as usize / 4 + 1]).unwrap().requires_grad_(true).unwrap();
    let tiny = huge.narrow(0, 0, 1).unwrap();
    assert!(tiny.vjp_wrt(&[&huge], &Tensor::ones(&[1]), true).is_err());
    assert!(Tensor::cat(&[huge.clone()], 0).is_err());
    assert!(is_grad_enabled());
    assert!(huge.grad().is_none() && empty.grad().is_none() && seed.grad().is_none());

    let giant_empty = Tensor::zeros(&[usize::MAX, 0]).requires_grad_(true).unwrap();
    Tensor::cat(&[giant_empty.clone(), giant_empty.clone()], 1).unwrap().sum().backward();
    assert_eq!(giant_empty.grad().unwrap().shape(), &[usize::MAX, 0]);
}

#[test]
fn empty_slices_preserve_normalized_offsets_and_reject_offset_overflow() {
    let x = Tensor::from_vec(vec![0., 1., 2., 3., 4.], &[5]).unwrap().requires_grad_(true).unwrap();
    for (start, end, expected) in [(3, 3, 3), (4, 2, 4), (99, 100, 5), (99, 0, 5)] {
        let view = x.slice(0, start, end, 1).unwrap();
        assert_eq!(view.shape(), &[0]);
        assert_eq!(view.storage_offset(), expected);
        assert!(view.to_vec().is_empty());
        let seed = Tensor::zeros(&[0]).requires_grad_(true).unwrap();
        let gradient = view.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
        assert!(gradient.requires_grad());
        assert_eq!(gradient.to_vec(), vec![0.; 5]);
        assert_eq!(gradient.sum().grad_wrt(&[&seed], false).unwrap()[0].shape(), &[0]);
    }
    let strided = x.slice(0, 1, 5, 2).unwrap().narrow(0, 1, 0).unwrap();
    assert_eq!(strided.storage_offset(), 3);
    assert_eq!(strided.strides(), &[2]);
    let edge = Tensor::zeros(&[usize::MAX, 0]).slice(0, usize::MAX, usize::MAX, 1).unwrap();
    assert_eq!(edge.storage_offset(), usize::MAX);
    assert!(edge.to_vec().is_empty());
    assert!(edge.reshape(&[1, 0]).unwrap().slice(0, 1, 1, 1).is_err());
    assert!(x.grad().is_none());
}
