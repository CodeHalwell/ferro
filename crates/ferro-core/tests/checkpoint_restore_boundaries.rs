use std::cell::Cell;
use ferro_core::{checkpoint::Checkpoint, forward_ad::DualLevel, nn::Module, optim::{OptimizerState, Sgd}, rng::Rng, Param, Result, Tensor};

struct Empty(Cell<u64>);
impl Module for Empty {
    fn forward(&self, input: &Tensor) -> Result<Tensor> { Ok(input.clone()) }
    fn named_parameters(&self) -> Vec<(String, Param)> { Vec::new() }
    fn snapshot_scalars(&self) -> Vec<(String, u64)> { vec![("mode".into(), self.0.get())] }
    fn validate_scalars(&self, values: &[(String, u64)]) -> Result<()> {
        if values.len() != 1 || values[0].0 != "mode" {
            return Err(ferro_core::Error::Format { op: "empty", msg: "invalid mode schema".into() });
        }
        Ok(())
    }
    fn commit_scalars(&self, values: &[(String, u64)]) { self.0.set(values[0].1); }
}

#[test]
fn empty_restore_entries_reject_forward_ad_before_scalar_rng_or_optimizer_configuration_changes() {
    let source = Empty(Cell::new(1)); let target = Empty(Cell::new(7));
    let rng = Rng::new(11); let previous = rng.state();
    let source_rng = Rng::new(7);
    let source_opt = Sgd::new(vec![], 0.3); let mut target_opt = Sgd::new(vec![], 0.7);
    let config = target_opt.configuration();
    let named = Checkpoint::new(0);
    let module = Checkpoint::from_module(0, &source);
    let training = Checkpoint::from_training_state(0, &source, &[], &source_rng).unwrap();
    let random = Checkpoint::new(0).with_rng(&source_rng);
    let optimizer = Checkpoint::new(0).with_optimizer("main", &source_opt).unwrap();
    let legacy = Checkpoint::from_module_with_optim(0, &source, &source_opt);
    let primal = Tensor::ones(&[1]); let tangent = Tensor::ones(&[1]);
    {
        let _level = DualLevel::new(&[&primal], &[&tangent]).unwrap();
        assert!(named.load_named_state_into(&[]).is_err());
        assert!(module.load_into_module(&target).is_err());
        assert!(training.load_training_state_into(&target, &mut [], &rng).is_err());
        assert!(random.load_rng_into(&rng).is_err());
        assert!(optimizer.load_named_optim_into("main", &mut target_opt).is_err());
        assert!(legacy.load_optim_into(&mut target_opt).is_err());
        assert_eq!(target.0.get(), 7); assert_eq!(rng.state(), previous);
        assert_eq!(target_opt.configuration(), config);
    }
    training.load_training_state_into(&target, &mut [], &rng).unwrap();
    assert_eq!(target.0.get(), 1); assert_eq!(rng.state(), source_rng.state());
}

#[test]
fn empty_restore_entries_reject_capture_before_scalar_or_rng_changes() {
    let source = Empty(Cell::new(1)); let target = Empty(Cell::new(7));
    let rng = Rng::new(11); let previous = rng.state();
    let saved = Checkpoint::from_training_state(0, &source, &[], &Rng::new(7)).unwrap();
    assert!(ferro_core::capture(|| saved.load_training_state_into(&target, &mut [], &rng).map(|()| Tensor::scalar(1.))).is_err());
    assert!(ferro_core::capture(|| Checkpoint::new(0).load_named_state_into(&[]).map(|()| Tensor::scalar(1.))).is_err());
    assert_eq!(target.0.get(), 7); assert_eq!(rng.state(), previous);
}
