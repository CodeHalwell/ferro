use ferro_core::{Tensor, Param, Result, nn::Module, checkpoint::Checkpoint};

struct IncompleteModule { buffer: Tensor }
impl Module for IncompleteModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
    fn named_parameters(&self) -> Vec<(String, Param)> { Vec::new() }
    fn named_buffers(&self) -> Vec<(String, Tensor)> { vec![("buffer".into(), self.buffer.clone())] }
    fn snapshot_scalars(&self) -> Vec<(String, u64)> { vec![("counter".into(), 7)] }
}

#[test]
fn inherited_scalar_validation_rejects_missing_custom_protocol() {
    let source = IncompleteModule { buffer: Tensor::scalar(9.) };
    let target = IncompleteModule { buffer: Tensor::scalar(2.) };
    let mut cp = Checkpoint::from_module(0, &source);
    assert!(cp.load_into_module(&target).is_err());
    cp.tensors.retain(|(n,_)| !n.starts_with("scalars."));
    assert!(cp.load_into_module(&target).is_err(), "missing scalar state bypassed inherited validator");
    assert_eq!(target.buffer.item(), 2.);
}

#[test]
fn every_missing_or_duplicate_training_key_preserves_live_state() {
    use ferro_core::{nn::{Sequential, Linear}, modules::{BatchNorm, Dropout}, optim::Adam, Rng};
    let model = Sequential::new(vec![Box::new(Linear::new(2, 2, &Rng::new(5))),
        Box::new(BatchNorm::new(2)), Box::new(Dropout::new(0.3).with_seed(77))]);
    let rng = Rng::new(8);
    let mut opt = Adam::new(model.parameters(), 0.01);
    let x = Tensor::from_vec(vec![0.2, 0.7, -0.3, 0.8], &[2,2]).unwrap();
    model.forward(&x).unwrap().sum().backward(); opt.step(); opt.zero_grad();
    let good = Checkpoint::from_training_state(1, &model, &[("adam", &opt)], &rng).unwrap();
    let snapshot = |cp: &Checkpoint| cp.tensors.iter().map(|(n,t)|
        (n.clone(), t.to_vec().iter().map(|v| v.to_bits()).collect::<Vec<_>>())).collect::<Vec<_>>();
    let before = snapshot(&good);
    let handles: Vec<_> = model.parameters().iter().map(|p| p.tensor()).chain(model.named_buffers().into_iter().map(|(_,t)|t)).collect();
    let versions: Vec<_> = handles.iter().map(|t| t._version()).collect();
    for index in 0..good.tensors.len() {
        for duplicate in [false, true] {
            let mut bad = Checkpoint::from_training_state(1, &model, &[("adam", &opt)], &rng).unwrap();
            if duplicate { bad.tensors.push(bad.tensors[index].clone()); } else { bad.tensors.remove(index); }
            assert!(bad.load_training_state_into(&model, &mut [("adam", &mut opt)], &rng).is_err(), "key {} duplicate={duplicate}", good.tensors[index].0);
            let after = Checkpoint::from_training_state(1, &model, &[("adam", &opt)], &rng).unwrap();
            assert_eq!(snapshot(&after), before);
            assert_eq!(handles.iter().map(|t| t._version()).collect::<Vec<_>>(), versions);
        }
    }
}
