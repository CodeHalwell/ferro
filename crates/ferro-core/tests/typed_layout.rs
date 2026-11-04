use ferro_core::autograd::with_grad_enabled;
use ferro_core::{capture, DType, Tensor};

fn double(values: &[f64], shape: &[usize]) -> Tensor { Tensor::from_vec_f64(values.to_vec(), shape).unwrap() }

#[test]
fn detached_alias_shares_storage_version_but_not_history_or_slots() {
    let x = double(&[2., 3.], &[2]).requires_grad_(true).unwrap();
    let y = x.square().unwrap().sum();
    let alias = x.detach();
    assert!(!alias.requires_grad());
    assert_ne!(alias.id(), x.id());
    alias.copy_from(&double(&[5., 7.], &[2])).unwrap();
    assert_eq!(x.to_vec_f64(), vec![5., 7.]);
    assert!(y.grad_wrt(&[&x], false).is_err());
    assert!(x.grad().is_none() && alias.grad().is_none());
    let leaf = alias.requires_grad_(true).unwrap();
    leaf.square().unwrap().sum().backward();
    assert_eq!(leaf.grad().unwrap().to_vec_f64(), vec![10., 14.]);
    assert!(x.grad().is_none() && alias.grad().is_none());
}

#[test]
fn owning_clone_does_not_save_parent_values_but_consumers_save_clone_values() {
    for dtype in [DType::F32, DType::F64] {
        let x = double(&[2., 3.], &[2]).to_dtype(dtype).requires_grad_(true).unwrap();
        let copied = x.clone_tensor(true).unwrap();
        let loss = copied.square().unwrap().sum();
        with_grad_enabled(false, || x.copy_leaf_from(&double(&[5., 7.], &[2]).to_dtype(dtype))).unwrap();
        assert_eq!(copied.to_vec_f64(), vec![2., 3.]);
        assert_eq!(loss.grad_wrt(&[&x], false).unwrap()[0].to_vec_f64(), vec![4., 6.]);
        loss.backward();
        assert_eq!(x.grad().unwrap().to_vec_f64(), vec![4., 6.]);
        let stale = copied.square().unwrap().sum();
        copied.detach().copy_from(&double(&[8., 9.], &[2]).to_dtype(dtype)).unwrap();
        assert!(stale.grad_wrt(&[&x], false).is_err());
        assert_eq!(x.to_vec_f64(), vec![5., 7.]);
    }
    let x = Tensor::from_vec(vec![2., 3.], &[2]).unwrap().requires_grad_(true).unwrap();
    let copied = x.clone_tensor(true).unwrap();
    let y = copied.mul(&copied).unwrap().sum();
    let g = y.grad_wrt(&[&x], true).unwrap().remove(0);
    assert_eq!(g.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec(), vec![2., 2.]);
}

#[test]
fn clone_layouts_are_dense_preserving_or_contiguous_with_exact_typed_values() {
    let base = Tensor::from_vec_i64(vec![i64::MIN, i64::MAX, (1 << 53) + 1, 4, 5, 6], &[2, 3]).unwrap();
    let transposed = base.transpose(0, 1).unwrap();
    let preserve = transposed.clone_tensor(true).unwrap();
    let packed = transposed.clone_tensor(false).unwrap();
    assert_eq!(preserve.to_vec_i64(), transposed.to_vec_i64());
    assert_eq!(preserve.strides(), transposed.strides());
    assert_eq!(packed.strides(), &[2, 1]);
    assert!(!preserve.is_contiguous());
    assert!(packed.is_contiguous());
    assert_eq!(base.contiguous().unwrap().id(), base.id());
    assert_ne!(transposed.contiguous().unwrap().id(), transposed.id());
    let narrowed = base.narrow(0, 1, 1).unwrap();
    assert_eq!(narrowed.storage_offset(), 3);
    assert_eq!(narrowed.clone_tensor(true).unwrap().storage_offset(), 0);
    let gaps = Tensor::from_vec_i64((0..12).collect(), &[2, 6]).unwrap().slice(1, 0, 6, 2).unwrap();
    assert_eq!(gaps.clone_tensor(true).unwrap().strides(), &[3, 1]);
    let broadcast = double(&[1.0000000000000002], &[1]).expand(&[2, 3]).unwrap();
    assert_eq!(broadcast.clone_tensor(true).unwrap().strides(), &[3, 1]);
    assert_eq!(broadcast.clone_tensor(true).unwrap().to_vec_f64(), vec![1.0000000000000002; 6]);
    let empty = double(&[], &[2, 0, 3]);
    assert_eq!(empty.strides(), &[3, 3, 1]);
    let view = empty.transpose(0, 2).unwrap();
    assert!(view.is_contiguous());
    assert_eq!(view.contiguous().unwrap().id(), view.id());
    assert_eq!(view.clone_tensor(true).unwrap().strides(), view.strides());
    let bits = Tensor::from_vec_f16_bits(vec![0x7c01, 1, 0x8000, 0xffff], &[2, 2]).unwrap().transpose(0, 1).unwrap();
    assert_eq!(bits.clone_tensor(true).unwrap().to_vec_f16_bits().unwrap(), bits.to_vec_f16_bits().unwrap());
    let huge = double(&[1.], &[1]).expand(&[isize::MAX as usize / 8 + 1]).unwrap();
    assert!(huge.clone_tensor(true).is_err());
}

#[test]
fn empty_offset_views_never_read_past_storage() {
    for dtype in [DType::F32, DType::F64, DType::I64, DType::Bool, DType::F16, DType::BF16] {
        let base = Tensor::zeros(&[0, 8]).to_dtype(dtype);
        let view = base.narrow(1, 4, 4).unwrap();
        assert_eq!(view.storage_offset(), 4);
        assert!(view.is_contiguous());
        assert!(view.to_vec().is_empty());
        assert!(view.to_vec_f64().is_empty());
        assert!(view.to_vec_i64().is_empty());
        assert!(view.to_vec_bool().is_empty());
        assert!(view.detach_copy().to_vec().is_empty());
        for preserve in [true, false] {
            let copied = view.clone_tensor(preserve).unwrap();
            assert_eq!(copied.dtype(), dtype);
            assert_eq!(copied.shape(), &[0, 4]);
            assert_eq!(copied.storage_offset(), 0);
            assert!(copied.to_vec().is_empty());
        }
        if matches!(dtype, DType::F32 | DType::F64) {
            let leaf = base.requires_grad_(true).unwrap();
            let view = leaf.narrow(1, 4, 4).unwrap();
            let weights = Tensor::ones(&[4, 2]).to_dtype(dtype).requires_grad_(true).unwrap();
            let out = view.sigmoid().mul(&view).unwrap().add(&Tensor::ones(&[4]).to_dtype(dtype)).unwrap();
            assert_eq!(out.shape(), &[0, 4]);
            assert_eq!(view.sum().to_vec_f64(), vec![0.0]);
            assert!(view.detach_copy().to_vec_f64().is_empty());
            let direct = view.reshape(&[0, 4]).unwrap().matmul(&weights).unwrap().sum();
            out.matmul(&weights).unwrap().sum().add(&direct).unwrap().backward();
            assert_eq!(leaf.grad().unwrap().shape(), &[0, 8]);
            assert_eq!(weights.grad().unwrap().to_vec_f64(), vec![0.0; 8]);
            let destination = Tensor::zeros(&[0, 4]).to_dtype(dtype);
            destination.copy_from(&view.detach()).unwrap();
            if dtype == DType::F32 { destination.add_(&view.detach()).unwrap(); }
        }
        assert_eq!(view.storage_offset(), 4);
    }
}

#[test]
fn same_shape_reshape_retains_all_typed_layouts_and_shared_versions() {
    for dtype in [DType::F32, DType::F64, DType::I64, DType::Bool, DType::F16, DType::BF16] {
        let base = Tensor::from_vec((0..24).map(|n| n as f32).collect(), &[4, 6]).unwrap().to_dtype(dtype);
        let views = [base.transpose(0, 1).unwrap(), base.slice(1, 1, 6, 2).unwrap(),
            base.narrow(0, 0, 1).unwrap().expand(&[3, 6]).unwrap(),
            base.narrow(0, 0, 0).unwrap().narrow(1, 3, 2).unwrap()];
        let aliases = views.iter().map(|view| view.reshape(view.shape()).unwrap()).collect::<Vec<_>>();
        for (view, alias) in views.iter().zip(&aliases) {
            assert_ne!(alias.id(), view.id());
            assert_eq!(alias.dtype(), dtype);
            assert_eq!(alias.strides(), view.strides());
            assert_eq!(alias.storage_offset(), view.storage_offset());
            assert_eq!(alias._storage_ptr(), base._storage_ptr());
            assert_eq!(alias.to_vec_f64(), view.to_vec_f64());
        }
        base.copy_from(&Tensor::zeros(base.shape()).to_dtype(dtype)).unwrap();
        for (view, alias) in views.iter().zip(&aliases) {
            assert_eq!(alias._version(), base._version());
            assert_eq!(alias.to_vec_f64(), vec![0.; view.numel()]);
        }
    }
    let huge = double(&[1.], &[1]).expand(&[isize::MAX as usize / 8 + 1]).unwrap();
    let alias = huge.reshape(huge.shape()).unwrap();
    assert_eq!(alias._storage_ptr(), huge._storage_ptr());
    assert_eq!(alias.strides(), &[0]);
}

#[test]
fn same_shape_reshape_preserves_autograd_and_forward_replay() {
    for dtype in [DType::F32, DType::F64] {
        let base = double(&[1., 2., 3., 4., 5., 6.], &[2, 3]).to_dtype(dtype).requires_grad_(true).unwrap();
        let view = base.transpose(0, 1).unwrap();
        let alias = view.reshape(view.shape()).unwrap();
        alias.square().unwrap().sum().backward();
        assert_eq!(base.grad().unwrap().to_vec_f64(), vec![2., 4., 6., 8., 10., 12.]);
        let frozen = with_grad_enabled(false, || view.reshape(view.shape())).unwrap();
        assert!(frozen.requires_grad());
        assert!(!ferro_core::graph::_dbg_op_present(&frozen));
        let stale = alias.square().unwrap().sum();
        with_grad_enabled(false, || base.copy_leaf_from(&Tensor::zeros(base.shape()).to_dtype(dtype))).unwrap();
        assert!(stale.grad_wrt(&[&base], false).is_err());
        let source = double(&[1., 2., 3., 4., 5., 6.], &[2, 3]).to_dtype(dtype);
        let view = source.transpose(0, 1).unwrap();
        let captured = capture(|| view.reshape(view.shape())).unwrap();
        let compiled = ferro_core::graph::CompiledChain::compile(&captured).unwrap();
        source.copy_from(&Tensor::ones(source.shape()).to_dtype(dtype)).unwrap();
        let replay = compiled.replay().unwrap();
        assert_eq!(replay.strides(), view.strides());
        assert_eq!(replay.to_vec_f64(), vec![1.; 6]);
    }
}

#[test]
fn clone_forward_replay_produces_real_independent_copies() {
    for dtype in [DType::F32, DType::F64] {
        let source = double(&[2., 3.], &[2]).to_dtype(dtype);
        let cloned = capture(|| source.clone_tensor(true)).unwrap();
        let chain = ferro_core::graph::CompiledChain::compile(&cloned).unwrap();
        source.copy_from(&double(&[4., 5.], &[2]).to_dtype(dtype)).unwrap();
        let first = chain.replay().unwrap();
        assert_eq!(first.to_vec_f64(), vec![4., 5.]);
        source.copy_from(&double(&[6., 7.], &[2]).to_dtype(dtype)).unwrap();
        assert_eq!(first.to_vec_f64(), vec![4., 5.]);
        let second = chain.replay().unwrap();
        assert_eq!(second.to_vec_f64(), vec![6., 7.]);
        first.copy_from(&double(&[8., 9.], &[2]).to_dtype(dtype)).unwrap();
        assert_eq!(second.to_vec_f64(), vec![6., 7.]);
        assert_eq!(source.to_vec_f64(), vec![6., 7.]);
        assert!(chain.prepare_static().is_err());
    }
}

#[test]
fn no_grad_views_retain_flags_without_edges_and_explicit_leaf_enables_edges() {
    let x = double(&[1., 2., 3., 4.], &[2, 2]).requires_grad_(true).unwrap();
    let views = with_grad_enabled(false, || {
        let views = vec![x.reshape(&[4]).unwrap(), x.transpose(0, 1).unwrap(), x.unsqueeze(0).unwrap(),
                         x.permute(&[1, 0]).unwrap(), x.narrow(0, 1, 1).unwrap(), x.expand(&[2, 2]).unwrap()];
        assert!(!x.clone_tensor(true).unwrap().requires_grad());
        assert!(!x.transpose(0, 1).unwrap().reshape(&[4]).unwrap().requires_grad());
        assert_eq!(x.contiguous().unwrap().id(), x.id());
        views
    });
    for view in views {
        assert!(view.requires_grad());
        assert!(!ferro_core::graph::_dbg_op_present(&view));
        assert!(view.grad_wrt(&[&view], false).is_err());
        let loss = view.sum();
        loss.backward();
        assert!(view.grad().is_none() && x.grad().is_none());
        assert!(loss.vjp_wrt_optional(&[&view, &x], &double(&[1.], &[]), false).unwrap().iter().all(Option::is_none));
        let enabled = view.requires_grad_(true).unwrap();
        enabled.sum().backward();
        assert_eq!(enabled.grad().unwrap().to_vec_f64(), vec![1.; enabled.numel()]);
        assert!(x.grad().is_none());
    }
}

#[test]
fn internal_view_vjp_retains_live_cotangent_flags_without_adding_edges() {
    let x = double(&[1., 2., 3., 4.], &[2, 2]).requires_grad_(true).unwrap();
    for output in [x.reshape(&[4]).unwrap(), x.transpose(0, 1).unwrap(), x.unsqueeze(0).unwrap(), x.clone_tensor(true).unwrap()] {
        let seed = Tensor::from_vec_f64(vec![1.; output.numel()], output.shape()).unwrap().requires_grad_(true).unwrap();
        let result = output.vjp_wrt_preserve(&[&x], &seed).unwrap().remove(0);
        assert!(result.requires_grad());
        assert!(!ferro_core::graph::_dbg_op_present(&result));
        assert!(!output.vjp_wrt(&[&x], &seed, false).unwrap()[0].requires_grad());
        if result.id() != seed.id() {
            result.sum().backward();
            assert!(seed.grad().is_none() && result.grad().is_none());
        }
    }
}

#[test]
fn saved_outputs_reject_alias_mutations_but_input_only_ops_allow_output_edits() {
    let operations: &[fn(&Tensor) -> Tensor] = &[|x| x.exp(), |x| x.exp2().unwrap(), |x| x.expm1().unwrap(),
        |x| x.sigmoid(), |x| x.tanh(), |x| x.sqrt(), |x| x.reciprocal().unwrap(), |x| x.relu(),
        |x| x.softmax(0).unwrap(), |x| x.log_softmax(0).unwrap(), |x| x.max().unwrap()];
    for dtype in [DType::F32, DType::F64] { for operation in operations {
        let x = double(&[0.5, 1.5], &[2]).to_dtype(dtype).requires_grad_(true).unwrap();
        let output = operation(&x);
        output.detach().copy_from(&Tensor::zeros(output.shape()).to_dtype(dtype)).unwrap();
        assert!(output.sum().grad_wrt(&[&x], false).is_err());
        assert!(x.grad().is_none());
    } }
    let x = Tensor::from_vec(vec![2., 3.], &[2]).unwrap().requires_grad_(true).unwrap();
    let output = x.square().unwrap();
    output.detach().copy_from(&Tensor::zeros(&[2])).unwrap();
    output.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![4., 6.]);
}
