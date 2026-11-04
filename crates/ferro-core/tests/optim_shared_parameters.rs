use ferro_core::{optim::{Adaptive, Adafactor, Adam, AdamW, Lbfgs, OptimizerState, Sgd}, Param, Tensor};

fn shared(values: Vec<f32>) -> (Tensor, Vec<Param>) {
    let data = Tensor::from_vec(values, &[2, 2]).unwrap();
    let parameters = (0..2).map(|_| Param::from_shared(&data, true).unwrap()).collect();
    (data, parameters)
}
fn gradient(parameter: &Param, value: f32) {
    parameter.tensor().set_leaf_grad(Some(Tensor::full(&[2, 2], value))).unwrap();
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) { assert!((actual-expected).abs() <= tolerance, "{actual} != {expected}"); }
}

#[test]
fn shared_rmsprop_and_adafactor_match_actual_torch_regressions() {
    let (data, params) = shared(vec![1.0; 4]);
    gradient(&params[0], 2.0); gradient(&params[1], 3.0);
    Adaptive::new(params, &[5., 0.1, 0.9, 1e-8, 0., 0., 0., 0.]).unwrap().step().unwrap();
    close(&data.to_vec(), &[0.3675444424; 4], 2e-7);
    let (data, params) = shared(vec![1.0; 4]);
    gradient(&params[0], 2.0); gradient(&params[1], 3.0);
    Adafactor::new(params, &[14., 0.01, -0.8, -1., 0.001, 1., 0., 0.]).unwrap().step().unwrap();
    close(&data.to_vec(), &[0.9801000357; 4], 1e-7);
}

#[test]
fn every_adaptive_family_preserves_order_and_independent_moments_for_shared_leaves() {
    let configs = [vec![5., 0.03, 0.8, 1e-6, 0.1, 0.4, 1., 0.], vec![6., 0.03, 0.02, 0.1, 0.2, 1e-6, 0.],
        vec![7., 0.03, 0.7, 0.9, 1e-6, 0.1, 0.], vec![8., 0.7, 0.8, 1e-5, 0.1, 0.],
        vec![9., 0.03, 0.03, 0.7, 2., 0.1, 0.], vec![10., 0.03, 0.5, 1.2, 0.001, 0.1, 0.],
        vec![11., 0.03, 0.7, 0.9, 1e-6, 0.1, 0., 0.], vec![12., 0.03, 0.7, 0.9, 1e-6, 0.1, 0.004, 1., 0.]];
    for config in configs {
        let (data, params) = shared(vec![0.7, -1.2, 0.4, 0.2]);
        let (control, separate) = shared(data.to_vec());
        let mut together = Adaptive::new(params.clone(), &config).unwrap();
        let mut individual: Vec<_> = separate.iter().map(|p| Adaptive::new(vec![p.clone()], &config).unwrap()).collect();
        for step in 0..12 {
            for i in 0..2 { if (step+i)%5 != 1 { gradient(&params[i], (step as f32-4.)*0.03+i as f32*0.1); gradient(&separate[i], (step as f32-4.)*0.03+i as f32*0.1); } }
            together.step().unwrap(); for optimizer in &mut individual { optimizer.step().unwrap(); }
            assert_eq!(data.to_vec(), control.to_vec(), "family {} step {step}", config[0]);
            let snapshot = together.snapshot();
            for (i,optimizer) in individual.iter().enumerate() { for (name,value) in optimizer.snapshot() {
                let key = name.strip_suffix(".0").unwrap().to_owned()+&format!(".{i}");
                assert_eq!(snapshot.iter().find(|(name,_)| name == &key).unwrap().1.to_vec(), value.to_vec());
            } }
        }
    }
}

#[test]
fn shared_lbfgs_fixed_and_strong_wolfe_trials_match_actual_torch() {
    for line_search in [false, true] {
        let (data, params) = shared(vec![0.7, -1.2, 0.4, 0.2]);
        let mut config = vec![13., 0.2, 3., 5., 0., 0., 4.]; if line_search { config.push(1.); }
        let mut optimizer = Lbfgs::new(params.clone(), &config).unwrap();
        let mut calls = Vec::new();
        optimizer.step(|| {
            for parameter in &params { parameter.zero_grad(); }
            let a = params[0].tensor().sub(&Tensor::scalar(0.3)).unwrap();
            let b = params[1].tensor().add(&Tensor::scalar(0.2)).unwrap();
            let loss = a.mul(&a).unwrap().sum().add(&b.mul(&b).unwrap().sum().mul(&Tensor::scalar(0.7)).unwrap()).unwrap();
            loss.backward(); calls.push(loss.item()); Ok(loss.item())
        }).unwrap();
        if line_search {
            close(&data.to_vec(), &[0.63883394, -1.06935418, 0.36912006, 0.18931079], 5e-6);
            close(&calls, &[4.06099987, 3.48488951, 3.38482809, 3.36578703, 3.36209965, 3.36138487], 6e-6);
        } else {
            close(&data.to_vec(), &[0.41337854, -0.58779883, 0.25529793, 0.14991085], 5e-6);
            close(&calls, &[4.06099987, 3.48488951, 2.26782346], 6e-6);
        }
    }
}

#[test]
fn active_gradient_storage_aliases_reject_before_native_updates() {
    fn state(optimizer: &dyn OptimizerState) -> Vec<(String, Vec<u32>)> {
        optimizer.snapshot().into_iter().map(|(name,tensor)| (name,tensor.to_vec().into_iter().map(f32::to_bits).collect())).collect()
    }
    for family in 0..5 {
        let (data, params) = shared(vec![1.0; 4]);
        gradient(&params[0], 2.0);
        let alias = data.transpose(0, 1).unwrap().detach();
        params[1].tensor().set_leaf_grad(Some(alias.clone())).unwrap();
        let pointers: Vec<_> = params.iter().map(|p| (p.tensor().id(), p.tensor()._storage_ptr(), p.tensor()._version())).collect();
        macro_rules! rejected {
            ($optimizer:expr, $fallible:expr) => {{
                let mut optimizer = $optimizer; let before = state(&optimizer);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $fallible(&mut optimizer)));
                assert!(result.is_err() || result.unwrap(), "family {family}");
                assert_eq!(state(&optimizer), before);
            }};
        }
        match family {
            0 => rejected!(Sgd::new(params.clone(), 0.1).with_momentum(0.5), |o: &mut Sgd| { o.step(); false }),
            1 => rejected!(Adam::new(params.clone(), 0.1), |o: &mut Adam| { o.step(); false }),
            2 => rejected!(AdamW::new(params.clone(), 0.1), |o: &mut AdamW| { o.step(); false }),
            3 => rejected!(Adaptive::new(params.clone(), &[5., 0.1, 0.9, 1e-8, 0., 0., 0., 0.]).unwrap(), |o: &mut Adaptive| o.step().is_err()),
            _ => rejected!(Adafactor::new(params.clone(), &[14., 0.01, -0.8, -1., 0.001, 1., 0., 0.]).unwrap(), |o: &mut Adafactor| o.step().is_err()),
        }
        assert_eq!(data.to_vec(), vec![1.; 4]);
        assert_eq!(params.iter().map(|p| (p.tensor().id(), p.tensor()._storage_ptr(), p.tensor()._version())).collect::<Vec<_>>(), pointers);
        assert_eq!(params[1].tensor().grad().unwrap().id(), alias.id());
        params[1].set_trainable(false);
        params[1].tensor().set_leaf_grad(Some(alias)).unwrap();
        assert!(ferro_core::optim::validate_gradient_storage(&params).is_ok());
    }
}

#[test]
fn strong_wolfe_rejected_closure_gradient_alias_restores_trial_baseline() {
    let (data, params) = shared(vec![0.7, -1.2, 0.4, 0.2]);
    let original = data.to_vec();
    let mut optimizer = Lbfgs::new(params.clone(), &[13., 0.2, 3., 5., 0., 0., 4., 1.]).unwrap();
    let mut calls = 0;
    let error = optimizer.step(|| {
        calls += 1;
        for parameter in &params { parameter.zero_grad(); }
        let loss = params[0].tensor().mul(&params[0].tensor()).unwrap().sum()
            .add(&params[1].tensor().sum()).unwrap();
        loss.backward();
        if calls == 2 { params[1].tensor().set_leaf_grad(Some(data.detach())).unwrap(); }
        Ok(loss.item())
    }).unwrap_err();
    assert!(error.to_string().contains("share parameter storage"));
    assert_eq!(calls, 2);
    assert_eq!(data.to_vec(), original);
}
