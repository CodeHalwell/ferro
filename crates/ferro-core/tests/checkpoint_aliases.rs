use ferro_core::{Tensor, nn::Module, params::Param, checkpoint::Checkpoint, optim::Adam, rng::Rng};
use std::cell::Cell;

struct State { p: Param, a: Tensor, b: Tensor, mode: Cell<u64> }
impl State {
    fn new(tied: bool, value: f32) -> Self {
        let a = Tensor::from_vec(vec![value], &[1]).unwrap();
        let b = if tied { a.clone() } else { Tensor::from_vec(vec![value], &[1]).unwrap() };
        Self { p: Param::new(Tensor::ones(&[1])), a, b, mode: Cell::new(1) }
    }
}
impl Module for State {
    fn forward(&self, x: &Tensor) -> ferro_core::Result<Tensor> { x.mul(&self.p.tensor()) }
    fn named_parameters(&self) -> Vec<(String, Param)> { vec![("p".into(), self.p.clone())] }
    fn named_buffers(&self) -> Vec<(String, Tensor)> { vec![("a".into(), self.a.clone()), ("nested.b".into(), self.b.clone())] }
    fn snapshot_scalars(&self) -> Vec<(String, u64)> { vec![("mode".into(), self.mode.get())] }
    fn validate_scalars(&self, s: &[(String, u64)]) -> ferro_core::Result<()> {
        assert_eq!(s.len(), 1); assert_eq!(s[0].0, "mode"); Ok(())
    }
    fn commit_scalars(&self, s: &[(String, u64)]) { self.mode.set(s[0].1); }
}
fn bits(cp: &Checkpoint) -> Vec<(String, Vec<u32>)> {
    cp.tensors.iter().map(|(n,t)| (n.clone(), t.to_vec().iter().map(|v| v.to_bits()).collect())).collect()
}
fn reject_unchanged(src: &State, dst: &State) {
    let r = Rng::new(12); let dr = Rng::new(42);
    let a = Adam::new(src.parameters(), 0.01); let mut b = Adam::new(dst.parameters(), 0.5);
    dst.p.tensor().sum().backward(); b.step(); b.zero_grad();
    dst.mode.set(0); dst.p.set_trainable(false);
    let cp = Checkpoint::from_training_state(0, src, &[("adam", &a)], &r).unwrap();
    let source = bits(&cp);
    let before = bits(&Checkpoint::from_training_state(0, dst, &[("adam", &b)], &dr).unwrap());
    let handles = [dst.p.tensor(), dst.a.clone(), dst.b.clone()];
    let versions: Vec<_> = handles.iter().map(Tensor::_version).collect();
    assert!(cp.load_training_state_into(dst, &mut [("adam", &mut b)], &dr).is_err(), "changed alias topology accepted");
    assert_eq!(before, bits(&Checkpoint::from_training_state(0, dst, &[("adam", &b)], &dr).unwrap()));
    assert_eq!(versions, handles.iter().map(Tensor::_version).collect::<Vec<_>>());
    assert_eq!(source, bits(&cp));
}
#[test]
fn tied_buffers_cannot_split() { reject_unchanged(&State::new(true, 1.), &State::new(false, 9.)); }
#[test]
fn split_buffers_cannot_merge_even_with_equal_values() { reject_unchanged(&State::new(false, 1.), &State::new(true, 9.)); }
#[test]
fn split_buffers_cannot_merge_with_different_values() {
    let mut src = State::new(false, 1.); src.b = Tensor::from_vec(vec![2.], &[1]).unwrap();
    reject_unchanged(&src, &State::new(true, 9.));
}
#[test]
fn parameter_buffer_alias_cannot_split_or_merge() {
    for reverse in [false, true] {
        let mut tied = State::new(false, 1.); tied.a = tied.p.tensor();
        let split = State::new(false, 1.);
        if reverse { reject_unchanged(&split, &tied); } else { reject_unchanged(&tied, &split); }
    }
}
#[test]
fn missing_or_corrupt_storage_schema_rejects_atomically() {
    let src = State::new(true, 1.); let dst = State::new(true, 9.); let r = Rng::new(1);
    let good = Checkpoint::from_training_state(0, &src, &[], &r).unwrap();
    for missing in [true, false] {
        let mut cp = good.clone();
        if missing { cp.tensors.retain(|(n,_)| !n.starts_with("storage.")); }
        else { cp.tensors.iter_mut().find(|(n,_)| n == "storage.buffers.nested.b").unwrap().1 = Tensor::ones(&[4]); }
        let before = bits(&Checkpoint::from_training_state(0, &dst, &[], &r).unwrap());
        let version = dst.a._version();
        assert!(cp.load_training_state_into(&dst, &mut [], &r).is_err());
        assert_eq!(before, bits(&Checkpoint::from_training_state(0, &dst, &[], &r).unwrap()));
        assert_eq!(version, dst.a._version());
    }
}
#[test]
fn matching_cross_parameter_buffer_aliases_resume_optimizer_updates() {
    let mut src = State::new(false, 1.); src.a = src.p.tensor();
    let mut dst = State::new(false, 9.); dst.a = dst.p.tensor();
    let mut a = Adam::new(src.parameters(), 0.01); let mut b = Adam::new(dst.parameters(), 0.5);
    let r = Rng::new(1);
    let cp = Checkpoint::from_training_state(0, &src, &[("adam", &a)], &r).unwrap();
    let saved = bits(&cp);
    cp.load_training_state_into(&dst, &mut [("adam", &mut b)], &r).unwrap();
    src.p.tensor().sum().backward(); a.step();
    dst.p.tensor().sum().backward(); b.step();
    assert_eq!(src.a.to_vec(), dst.a.to_vec());
    assert_eq!(dst.a._storage_ptr(), dst.p.tensor()._storage_ptr());
    assert_eq!(saved, bits(&cp));
}
#[test]
fn unsupported_target_views_reject_before_any_commit() {
    let mut src = State::new(true, 1.);
    src.a = Tensor::ones(&[2, 2]); src.b = src.a.clone();
    let mut dst = State::new(true, 9.);
    dst.a = Tensor::from_vec(vec![9., 8., 7., 6.], &[2, 2]).unwrap();
    dst.b = dst.a.transpose(0, 1).unwrap(); dst.mode.set(0); dst.p.set_trainable(false);
    let a = Adam::new(src.parameters(), 0.01); let mut b = Adam::new(dst.parameters(), 0.5);
    let r = Rng::new(1); let dr = Rng::new(9);
    let cp = Checkpoint::from_training_state(0, &src, &[("adam", &a)], &r).unwrap();
    let before = bits(&Checkpoint::from_module(0, &dst).with_optimizer("adam", &b).unwrap().with_rng(&dr));
    let version = dst.a._version();
    assert!(cp.load_training_state_into(&dst, &mut [("adam", &mut b)], &dr).is_err());
    assert_eq!(before, bits(&Checkpoint::from_module(0, &dst).with_optimizer("adam", &b).unwrap().with_rng(&dr)));
    assert_eq!(version, dst.a._version()); assert!(!dst.p.is_trainable());
}
#[test]
fn unsupported_source_views_are_rejected() {
    let base = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
    for view in [base.transpose(0, 1).unwrap(), base.reshape(&[4]).unwrap()] {
        let mut src = State::new(false, 1.);
        src.a = base.clone(); src.b = view;
        assert_eq!(src.a._storage_ptr(), src.b._storage_ptr());
        assert!(Checkpoint::from_training_state(0, &src, &[], &Rng::new(1)).is_err(), "distinct aliased views accepted");
    }
}
#[test]
fn matching_tied_storage_restores_and_remains_tied() {
    let src = State::new(true, 1.); let dst = State::new(true, 9.); let r = Rng::new(2);
    let cp = Checkpoint::from_training_state(0, &src, &[], &r).unwrap();
    cp.load_training_state_into(&dst, &mut [], &r).unwrap();
    assert_eq!(dst.a._storage_ptr(), dst.b._storage_ptr());
    dst.a.copy_from(&Tensor::from_vec(vec![7.], &[1]).unwrap()).unwrap();
    assert_eq!(dst.b.to_vec(), vec![7.]);
    assert_eq!(src.a.to_vec(), vec![1.]);
}
