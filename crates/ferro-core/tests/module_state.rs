use ferro_core::{Tensor, nn::{Module, Sequential}, modules::{BatchNorm, Dropout}, checkpoint::Checkpoint};

#[test]
fn dropout_checkpoint_restores_stream_and_mode() {
    let d = Dropout::new(0.4).with_seed(u64::MAX - 5);
    let x = Tensor::ones(&[64]);
    d.forward(&x).unwrap();
    let cp = Checkpoint::from_module(1, &d);
    let expected = d.forward(&x).unwrap().to_vec();
    let fresh = Dropout::new(0.4).with_seed(7);
    fresh.set_training(false);
    cp.load_into_module(&fresh).unwrap();
    assert_eq!(fresh.forward(&x).unwrap().to_vec(), expected);
}

#[test]
fn nested_batchnorm_buffers_rank2_rank4_and_eval_resume() {
    for shape in [&[2, 2][..], &[1, 2, 1, 2][..]] {
        let m = Sequential::new(vec![Box::new(ferro_core::modules::ModuleList::new(vec![Box::new(BatchNorm::new(2))]))]);
        let x = Tensor::from_vec(vec![1., 2., 3., 5.], shape).unwrap().requires_grad_(true).unwrap();
        let y = m.forward(&x).unwrap();
        y.sum().backward();
        assert!(x.grad().is_some());
        assert!(m.parameters().iter().all(|p| p.grad().is_some()));
        let buffers = m.named_buffers();
        assert_eq!(buffers.iter().map(|(n,_)| n.as_str()).collect::<Vec<_>>(), vec!["0.0.running_mean", "0.0.running_var"]);
        let saved = buffers[0].1.to_vec();
        m.set_training(false);
        let cp = Checkpoint::from_module(1, &m);
        let expected = m.forward(&x).unwrap().to_vec();
        m.set_training(true);
        m.forward(&x).unwrap();
        assert_ne!(buffers[0].1.to_vec(), saved);
        cp.load_into_module(&m).unwrap();
        assert_eq!(buffers[0].1.to_vec(), saved);
        assert_eq!(m.forward(&x).unwrap().to_vec(), expected);
        assert_eq!(buffers[0].1.to_vec(), saved);
    }
}

use ferro_core::{nn::Linear, optim::{Adam, Sgd, OptimizerState}, rng::Rng};

fn training_model(seed: u64) -> Sequential {
    Sequential::new(vec![Box::new(Linear::new(2, 2, &Rng::new(seed))),
        Box::new(BatchNorm::new(2)), Box::new(Dropout::new(0.25).with_seed(seed)),
        Box::new(Linear::new(2, 1, &Rng::new(seed + 1)))])
}
fn optimizers(m: &dyn Module) -> (Adam, Sgd) {
    let p = m.parameters();
    (Adam::new(p[..4].to_vec(), 0.01), Sgd::new(p[4..].to_vec(), 0.03).with_momentum(0.8))
}
fn step(m: &dyn Module, a: &mut Adam, b: &mut Sgd, rng: &Rng, i: usize) -> (Vec<f32>, Vec<f32>, f32) {
    a.zero_grad(); b.zero_grad();
    let draws: Vec<_> = (0..8).map(|_| rng.normal()).collect();
    let x = Tensor::from_vec(draws.clone(), &[4, 2]).unwrap();
    let out = m.forward(&x).unwrap();
    let loss = out.mul(&out).unwrap().mean();
    let result = (draws, out.to_vec(), loss.item());
    loss.backward(); a.step();
    if i % 2 == 0 { b.step(); }
    a.zero_grad(); b.zero_grad();
    result
}
fn arrays(cp: &Checkpoint) -> Vec<(String, Vec<u32>)> {
    cp.tensors.iter().map(|(n,t)| (n.clone(), t.to_vec().iter().map(|v| v.to_bits()).collect())).collect()
}
#[test]
fn stochastic_two_optimizer_disk_restart_matches_uninterrupted() {
    let m = training_model(10); let rng = Rng::new(32); let (mut a, mut b) = optimizers(&m);
    for i in 0..3 { step(&m, &mut a, &mut b, &rng, i); }
    let cp = Checkpoint::from_training_state(3, &m, &[("a", &a), ("b", &b)], &rng).unwrap();
    let saved = arrays(&cp);
    let expected: Vec<_> = (3..8).map(|i| step(&m, &mut a, &mut b, &rng, i)).collect();
    assert_eq!(arrays(&cp), saved);
    let dir = std::env::temp_dir().join(format!("ferro_module_restart_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    cp.save_to_dir(&dir).unwrap();
    let cp = Checkpoint::load_from_dir(&dir).unwrap();
    let fresh = training_model(900); fresh.set_training(false);
    let fresh_rng = Rng::new(99); let (mut c, mut d) = optimizers(&fresh);
    let identities: Vec<_> = fresh.parameters().iter().map(|p| p.tensor()._storage_ptr()).collect();
    cp.load_training_state_into(&fresh, &mut [("a", &mut c), ("b", &mut d)], &fresh_rng).unwrap();
    assert_eq!(fresh.parameters().iter().map(|p| p.tensor()._storage_ptr()).collect::<Vec<_>>(), identities);
    let actual: Vec<_> = (3..8).map(|i| step(&fresh, &mut c, &mut d, &fresh_rng, i)).collect();
    println!("continuation (input RNG draws, stochastic outputs, loss): {actual:?}");
    assert_eq!(actual, expected);
    let end1 = Checkpoint::from_training_state(8, &m, &[("a", &a), ("b", &b)], &rng).unwrap();
    let end2 = Checkpoint::from_training_state(8, &fresh, &[("a", &c), ("b", &d)], &fresh_rng).unwrap();
    assert_eq!(arrays(&end1), arrays(&end2));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rejected_transaction_does_not_touch_any_live_state() {
    let src = training_model(1); let r = Rng::new(2); let (mut a, mut b) = optimizers(&src);
    step(&src, &mut a, &mut b, &r, 0);
    let good = Checkpoint::from_training_state(1, &src, &[("a", &a), ("b", &b)], &r).unwrap();
    let dst = training_model(7); let dr = Rng::new(8); let (mut c, mut d) = optimizers(&dst);
    step(&dst, &mut c, &mut d, &dr, 0); dst.set_training(false);
    for key in ["optim.b.velocity.1", "optim.b.config", "rng.xorshift128", "scalars.2.training", "buffers.1.running_var"] {
        let mut bad = good.clone();
        let (_, value) = bad.tensors.iter_mut().find(|(n,_)| n == key).unwrap();
        *value = Tensor::scalar(-999.0);
        let before = arrays(&Checkpoint::from_training_state(1, &dst, &[("a", &c), ("b", &d)], &dr).unwrap());
        let handles: Vec<_> = dst.parameters().iter().map(|p| p.tensor()).chain(dst.named_buffers().into_iter().map(|(_,t)|t)).collect();
        let versions: Vec<_> = handles.iter().map(|t| t._version()).collect();
        assert!(bad.load_training_state_into(&dst, &mut [("a", &mut c), ("b", &mut d)], &dr).is_err(), "{key}");
        let after = arrays(&Checkpoint::from_training_state(1, &dst, &[("a", &c), ("b", &d)], &dr).unwrap());
        assert_eq!(before, after, "{key}");
        assert_eq!(handles.iter().map(|t| t._version()).collect::<Vec<_>>(), versions);
    }
    let mut reordered = Adam::new(dst.parameters()[..4].iter().rev().cloned().collect(), 0.1);
    assert!(good.load_training_state_into(&dst, &mut [("a", &mut reordered), ("b", &mut d)], &dr).is_err());
}

#[test]
fn failed_dropout_forward_does_not_advance_counter() {
    let d = Dropout::new(2.0);
    assert!(d.forward(&Tensor::ones(&[10])).is_err());
    assert_eq!(d.rng_offset(), 0);
}

#[test]
fn dropout_exact_counter_overflow_rejects_without_mutation() {
    let d = Dropout::new(0.2);
    let mut state = d.snapshot_scalars();
    state.iter_mut().find(|(n,_)| n == "offset").unwrap().1 = u64::MAX - 1;
    d.validate_scalars(&state).unwrap(); d.commit_scalars(&state);
    assert!(d.forward(&Tensor::ones(&[4])).is_err());
    assert_eq!(d.snapshot_scalars(), state);
}

use ferro_core::params::Param;
struct Pair { left: Param, right: Param }
impl Pair {
    fn new(tied: bool) -> Self {
        let left = Param::new(Tensor::ones(&[1]));
        let right = if tied { left.clone() } else { Param::new(Tensor::ones(&[1])) };
        Self { left, right }
    }
}
impl Module for Pair {
    fn forward(&self, x: &Tensor) -> ferro_core::Result<Tensor> { x.mul(&self.left.tensor())?.add(&self.right.tensor()) }
    fn named_parameters(&self) -> Vec<(String, Param)> { vec![("left".into(), self.left.clone()), ("right".into(), self.right.clone())] }
}
#[test]
fn training_restore_rejects_changed_tie_topology_even_without_optimizer() {
    let tied = Pair::new(true); let untied = Pair::new(false); let rng = Rng::new(1);
    let cp = Checkpoint::from_training_state(0, &tied, &[], &rng).unwrap();
    assert!(cp.load_training_state_into(&untied, &mut [], &rng).is_err());
}

#[test]
fn frozen_tied_parameter_preserves_identity_and_optimizer_state() {
    let m = Pair::new(true); let held = m.left.tensor(); let id = m.left.identity();
    let mut a = Adam::new(m.parameters(), 0.1);
    m.forward(&Tensor::ones(&[1])).unwrap().sum().backward(); a.step();
    m.left.set_trainable(false);
    let previous = arrays(&Checkpoint::new(0).with_optimizer("a", &a).unwrap());
    let values = held.to_vec();
    m.forward(&Tensor::ones(&[1])).unwrap().sum().backward(); a.step();
    assert_eq!(held.to_vec(), values);
    assert_eq!(m.left.identity(), id);
    assert_eq!(arrays(&Checkpoint::new(0).with_optimizer("a", &a).unwrap()), previous);
    let cp = Checkpoint::from_training_state(1, &m, &[("a", &a)], &Rng::new(1)).unwrap();
    let fresh = Pair::new(true); let mut b = Adam::new(fresh.parameters(), 0.01);
    cp.load_training_state_into(&fresh, &mut [("a", &mut b)], &Rng::new(4)).unwrap();
    assert!(!fresh.left.is_trainable()); assert!(!fresh.right.is_trainable());
    fresh.left.set_trainable(true);
    fresh.forward(&Tensor::ones(&[1])).unwrap().sum().backward(); b.step();
    assert_ne!(fresh.left.tensor().to_vec(), values);
}

#[test]
fn legacy_module_state_dict_restores_buffers_without_rebinding() {
    let m = BatchNorm::new(2); let x = Tensor::from_vec(vec![1., 2., 4., 8.], &[2,2]).unwrap();
    m.forward(&x).unwrap();
    let dir = std::env::temp_dir().join(format!("ferro_bn_dict_{}.safetensors", std::process::id()));
    ferro_core::nn::save_module(&dir, &m).unwrap();
    let fresh = BatchNorm::new(2); let held = fresh.parameters()[0].tensor();
    let values = m.named_buffers()[0].1.to_vec();
    ferro_core::nn::load_module(&dir, &fresh).unwrap();
    assert_eq!(fresh.named_buffers()[0].1.to_vec(), values);
    fresh.forward(&x).unwrap().sum().backward();
    assert!(held.grad().is_some(), "load must retain the same autograd leaf");
    std::fs::remove_file(dir).unwrap();
}

#[test]
fn stale_optimizer_buffer_shape_rejects_before_model_commit() {
    let src = Pair::new(true); src.left.set(Tensor::full(&[2], 7.0));
    let a = Adam::new(src.parameters(), 0.01); let r = Rng::new(1);
    let cp = Checkpoint::from_training_state(0, &src, &[("a", &a)], &r).unwrap();
    let dst = Pair::new(true); let mut b = Adam::new(dst.parameters(), 0.02);
    dst.forward(&Tensor::ones(&[1])).unwrap().sum().backward(); b.step();
    dst.left.set(Tensor::full(&[2], 3.0));
    let before = dst.left.tensor().to_vec();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cp.load_training_state_into(&dst, &mut [("a", &mut b)], &r)));
    assert!(result.is_ok(), "malformed destination must return an error, not panic");
    assert!(result.unwrap().is_err());
    assert_eq!(dst.left.tensor().to_vec(), before);
}

#[test]
fn module_restore_rejects_new_version_and_duplicate_keys() {
    let m = BatchNorm::new(2);
    let mut cp = Checkpoint::from_module(0, &m); cp.version = u32::MAX;
    assert!(cp.load_into_module(&m).is_err());
    let mut cp = Checkpoint::from_module(0, &m);
    cp.tensors.push(("scalars.training".into(), Tensor::zeros(&[4])));
    assert!(cp.load_into_module(&m).is_err());
}

#[test]
fn adamw_prepare_abort_and_commit_are_separate() {
    use ferro_core::optim::AdamW;
    let m = Pair::new(true); let mut a = AdamW::new(m.parameters(), 0.1);
    m.forward(&Tensor::ones(&[1])).unwrap().sum().backward(); a.step();
    let original = arrays(&Checkpoint::new(0).with_optimizer("a", &a).unwrap());
    let b = AdamW::new(m.parameters(), 0.3);
    let state = b.snapshot(); let config = b.configuration();
    drop(a.prepare_cpu_restore(&config, &state).unwrap());
    assert_eq!(arrays(&Checkpoint::new(0).with_optimizer("a", &a).unwrap()), original);
    a.prepare_cpu_restore(&config, &state).unwrap()();
    assert_eq!(arrays(&Checkpoint::new(0).with_optimizer("a", &a).unwrap()), arrays(&Checkpoint::new(0).with_optimizer("a", &b).unwrap()));
}

#[test]
fn mixed_child_modes_restore_without_flattening_them() {
    let bn = BatchNorm::new(2); bn.set_training(false);
    let m = Sequential::new(vec![Box::new(bn), Box::new(Dropout::new(0.2))]);
    let cp = Checkpoint::from_module(0, &m);
    let expected = m.snapshot_scalars();
    m.set_training(true);
    cp.load_into_module(&m).unwrap();
    assert_eq!(m.snapshot_scalars(), expected);
    let before = arrays(&Checkpoint::from_module(0, &m));
    let bad = cp.with_tensor("scalars.9.training", Tensor::zeros(&[4]));
    assert!(bad.load_into_module(&m).is_err());
    assert_eq!(arrays(&Checkpoint::from_module(0, &m)), before);
}

#[test]
fn batchnorm_module_matches_functional_and_rejects_singleton_training() {
    for shape in [&[2,2][..], &[1,2,1,2][..]] {
        let bn = BatchNorm::new(2);
        let x = Tensor::from_vec(vec![1., 2., 3., 5.], shape).unwrap();
        let reference = x.batch_norm(&Tensor::ones(&[2]), &Tensor::zeros(&[2]), &Tensor::zeros(&[2]), &Tensor::ones(&[2]), 1e-5, true, 0.1).unwrap();
        assert_eq!(bn.forward(&x).unwrap().to_vec(), reference.output.to_vec());
        assert_eq!(bn.named_buffers()[0].1.to_vec(), reference.running_mean.to_vec());
        assert_eq!(bn.named_buffers()[1].1.to_vec(), reference.running_var.to_vec());
        let previous = arrays(&Checkpoint::from_module(0, &bn));
        assert!(bn.forward(&Tensor::ones(&[1,2])).is_err());
        assert_eq!(arrays(&Checkpoint::from_module(0, &bn)), previous);
        bn.set_training(false);
        assert!(bn.forward(&Tensor::ones(&[1,2])).is_ok());
        assert!(bn.forward(&Tensor::zeros(&[0,2])).is_ok());
    }
}

mod device_compatibility {
    use super::*;
    use std::{any::Any, sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}}};
    use ferro_core::{Device, Result, dispatch::{Backend, DeviceBuffer, UnaryKind, BinaryKind, register_backend}};
    const DEV: Device = Device::Cuda(241);
    static LOCK: Mutex<()> = Mutex::new(());
    static COPIES: AtomicUsize = AtomicUsize::new(0);
    struct Buf(Mutex<Vec<f32>>);
    impl DeviceBuffer for Buf {
        fn device(&self) -> Device { DEV }
        fn len(&self) -> usize { self.0.lock().unwrap_or_else(|e| e.into_inner()).len() }
        fn as_any(&self) -> &dyn Any { self }
    }
    fn cell(b: &dyn DeviceBuffer) -> &Mutex<Vec<f32>> { &b.as_any().downcast_ref::<Buf>().unwrap().0 }
    struct Fake;
    impl Backend for Fake {
        fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
        fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
        fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected compute") }
        fn alloc_from_host(&self, d: &[f32]) -> Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buf(Mutex::new(d.to_vec())))) }
        fn copy_to_host(&self, b: &dyn DeviceBuffer) -> Result<Vec<f32>> { Ok(cell(b).lock().unwrap().clone()) }
        fn copy_into_dev(&self, dst: &dyn DeviceBuffer, src: &dyn DeviceBuffer) -> Result<()> {
            let values = cell(src).lock().unwrap().clone();
            cell(dst).lock().unwrap().copy_from_slice(&values);
            COPIES.fetch_add(1, Ordering::SeqCst); Ok(())
        }
    }
    #[test]
    fn legacy_device_optimizer_restore_survives_cpu_transaction_restriction() {
        let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        register_backend(DEV, Arc::new(Fake));
        let src = Pair::new(false);
        let mut a = ferro_core::optim::AdamW::new(src.parameters(), 0.1);
        src.forward(&Tensor::ones(&[1])).unwrap().sum().backward(); a.step();
        let cp = Checkpoint::from_module_with_optim(1, &src, &a);
        let dst = Pair::new(false);
        for p in dst.parameters() { p.set(p.tensor().to_device(DEV).unwrap()); }
        let mut b = ferro_core::optim::AdamW::new(dst.parameters(), 0.1).capturable();
        assert!(b.is_capturable());
        cp.load_optim_into(&mut b).unwrap();
        COPIES.store(0, Ordering::SeqCst);
        cp.load_optim_into(&mut b).unwrap();
        assert!(COPIES.load(Ordering::SeqCst) > 0, "existing device buffers must receive copy commits");
        assert_eq!(arrays(&Checkpoint::new(0).with_optimizer("a", &a).unwrap()), arrays(&Checkpoint::new(0).with_optimizer("a", &b).unwrap()));
        assert!(dst.parameters().iter().all(|p| p.tensor().device() == DEV));
        let mixed = Pair::new(false); mixed.right.set(mixed.right.tensor().to_device(DEV).unwrap());
        let before = arrays(&Checkpoint::from_module(0, &mixed));
        COPIES.store(0, Ordering::SeqCst);
        assert!(cp.load_into_module(&mixed).is_err());
        assert_eq!(arrays(&Checkpoint::from_module(0, &mixed)), before);
        assert_eq!(COPIES.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn aliased_optimizer_clones_are_rejected_before_transaction_writes() {
    let src = Pair::new(true); let r = Rng::new(1);
    let mut a = Adam::new(src.parameters(), 0.01); let b = Adam::new(src.parameters(), 0.02);
    src.forward(&Tensor::ones(&[1])).unwrap().sum().backward(); a.step();
    let cp = Checkpoint::from_training_state(1, &src, &[("a", &a), ("b", &b)], &r).unwrap();
    let dst = Pair::new(true); let mut c = Adam::new(dst.parameters(), 0.03);
    dst.forward(&Tensor::ones(&[1])).unwrap().sum().backward(); c.step();
    let mut d = c.clone();
    let before = arrays(&Checkpoint::from_module(0, &dst).with_optimizer("c", &c).unwrap().with_optimizer("d", &d).unwrap());
    assert!(cp.load_training_state_into(&dst, &mut [("a", &mut c), ("b", &mut d)], &r).is_err());
    assert_eq!(arrays(&Checkpoint::from_module(0, &dst).with_optimizer("c", &c).unwrap().with_optimizer("d", &d).unwrap()), before);
    assert!(Checkpoint::from_training_state(1, &dst, &[("a", &c), ("b", &d)], &r).is_err());
}
