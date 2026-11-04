use ferro_core::{capture, DType, Tensor};
use ferro_core::graph::CompiledChain;

#[test]
fn strided_unflatten_retains_storage_and_exact_typed_values() {
    for dtype in [DType::F32, DType::F64, DType::I64, DType::Bool] {
        let base = Tensor::from_vec_f64((0..24).map(|x| x as f64).collect(), &[4, 6]).unwrap().to_dtype_autograd(dtype, false).unwrap();
        let input = base.transpose(0, 1).unwrap().narrow(0, 1, 4).unwrap();
        let view = input.unflatten(-1, &[2, -1]).unwrap();
        assert_eq!(view.shape(), &[4, 2, 2]);
        assert_eq!(view.dtype(), dtype);
        assert_eq!(view._storage_ptr(), base._storage_ptr());
        assert_eq!(view.to_vec(), input.to_vec());
        let replacement = Tensor::ones(&[4, 6]).to_dtype_autograd(dtype, false).unwrap();
        base.copy_from(&replacement).unwrap();
        assert_eq!(view.to_vec(), vec![1.; 16]);
    }
}

#[test]
fn floating_unflatten_gradients_and_higher_order_reshape_rule() {
    for dtype in [DType::F32, DType::F64] {
        let input = Tensor::from_vec_f64((0..12).map(|i| 0.1 + i as f64 / 20.).collect(), &[3, 4]).unwrap().to_dtype_autograd(dtype, false).unwrap().requires_grad_(true).unwrap();
        let view = input.transpose(0, 1).unwrap().unflatten(0, &[2, 2]).unwrap();
        view.square().unwrap().sum().backward();
        assert_eq!(input.grad().unwrap().dtype(), dtype);
        for (got, expected) in input.grad().unwrap().to_vec().iter().zip(input.to_vec()) {
            assert!((got - 2. * expected).abs() < 1e-6);
        }
    }
    let input = Tensor::from_vec(vec![0.2; 12], &[3, 4]).unwrap();
    ferro_core::testkit::grad_check_strict(&[input.clone()], |inputs| inputs[0].transpose(0, 1).unwrap().unflatten(0, &[2, 2]).unwrap().square().unwrap().sum());
    let x = input.requires_grad_(true).unwrap();
    let y = x.transpose(0, 1).unwrap().unflatten(0, &[2, 2]).unwrap();
    let first = y.mul(&y).unwrap().mul(&y).unwrap().sum().grad_wrt(&[&x], true).unwrap().remove(0);
    let second = first.sum().grad_wrt(&[&x], true).unwrap().remove(0);
    let third = second.sum().grad_wrt(&[&x], false).unwrap().remove(0);
    assert_eq!(third.to_vec(), vec![6.; 12]);
}

#[test]
fn inference_capture_replays_unflatten_after_source_update() {
    let input = Tensor::from_vec((0..12).map(|i| i as f32).collect(), &[3, 4]).unwrap();
    let result = capture(|| input.transpose(0, 1).unwrap().unflatten(0, &[2, 2]).unwrap().neg());
    let graph = CompiledChain::compile(&result).unwrap();
    assert_eq!(graph.replay().unwrap().to_vec(), result.to_vec());
    input.copy_from(&Tensor::ones(&[3, 4])).unwrap();
    assert_eq!(graph.replay().unwrap().to_vec(), vec![-1.; 12]);
}

#[test]
fn empty_inferred_and_broadcast_views_validate_metadata_without_allocating() {
    let empty = Tensor::from_vec(vec![], &[3, 0, 2]).unwrap().transpose(0, 2).unwrap();
    assert_eq!(empty.unflatten(1, &[-1, 2]).unwrap().shape(), &[2, 0, 2, 3]);
    assert!(empty.unflatten(1, &[-1, 0]).is_err());
    let huge = Tensor::scalar(1.).expand(&[1usize << 60]).unwrap();
    let view = huge.unflatten(0, &[1 << 30, 1 << 30]).unwrap();
    assert_eq!(view.shape(), &[1usize << 30, 1usize << 30]);
    assert_eq!(view._storage_ptr(), huge._storage_ptr());
    for sizes in [&[][..], &[-1, -1][..], &[-2, 3][..], &[2, 2][..]] {
        assert!(Tensor::ones(&[6]).unflatten(0, sizes).is_err());
    }
    assert!(Tensor::scalar(1.).unflatten(0, &[1]).is_err());
    assert!(Tensor::ones(&[6]).unflatten(1, &[2, 3]).is_err());
    assert!(Tensor::from_vec(vec![], &[0]).unwrap().unflatten(0, &[0, isize::MAX, 2]).is_err());
}
