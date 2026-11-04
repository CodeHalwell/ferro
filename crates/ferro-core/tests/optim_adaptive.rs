use ferro_core::optim::{Adaptive, OptimizerState};
use ferro_core::{capture, Param, Tensor};

fn configs() -> Vec<Vec<f32>> {
    vec![vec![5., 0.031, 0.8, 0.001, 0.02, 0.4, 1., 0.],
         vec![6., 0.04, 0.03, 0.02, 0.15, 0.001, 1.],
         vec![7., 0.025, 0.7, 0.8, 0.001, 0.02, 0.],
         vec![8., 0.8, 0.75, 0.001, 0.02, 1.],
         vec![9., 0.03, 0.02, 0.65, 3., 0.01, 0.],
         vec![10., 0.03, 0.5, 1.3, 0.005, 0.08, 1.],
         vec![11., 0.025, 0.7, 0.8, 0.001, 0.02, 1., 0.],
         vec![12., 0.025, 0.7, 0.8, 0.001, 0.02, 0.02, 0., 1.]]
}
fn state(opt: &Adaptive) -> Vec<(String, Vec<usize>, Vec<u32>)> {
    opt.snapshot().into_iter().map(|(name,t)| (name,t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect()
}
fn gradient(p: &Param, values: &[f32]) {
    p.tensor().backward_with(&Tensor::from_vec(values.to_vec(), p.tensor().shape()).unwrap());
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a,b) in actual.iter().zip(expected) { assert!((a-b).abs() <= 2e-6 + 2e-6*b.abs(), "{actual:?} != {expected:?}"); }
}

#[test]
fn native_updates_and_every_moment_match_torch_twenty_step_fixtures() {
    // PyTorch 2.13.0 CPU, foreach=False, f32-rounded group settings, fixed
    // external gradients, skips at 2/8/14 and an lr change at step 10.
    let expected: Vec<(Vec<f32>, Vec<(&str,Vec<f32>)>)> = vec![
        (vec![0.70628023,-1.3490602,0.118493035], vec![("square_avg",vec![0.045588203,0.0005297936,0.28818005]),("momentum_buffer",vec![1.6320744,-1.0270034,1.2046981]),("grad_avg",vec![0.09059431,-0.008505745,0.23247717])]),
        (vec![0.7307135,-1.1268438,0.45811686], vec![("sum",vec![1.0967749,0.2867842,4.9808974])]),
        (vec![0.7736537,-1.2886556,0.30643526], vec![("exp_avg",vec![0.11510544,-0.010881413,0.25469878]),("exp_inf",vec![0.26660287,0.01449879,0.63224983])]),
        (vec![0.76267004,-0.88877225,0.6949011], vec![("square_avg",vec![0.03942447,0.0014894052,0.28683013]),("acc_delta",vec![0.0026016384,0.00057764095,0.007614318])]),
        (vec![0.73844767,-1.206616,0.31095496], vec![("ax",vec![0.747624,-1.2088162,0.33777848]),("eta",vec![0.017928755]),("mu",vec![0.071428575])]),
        (vec![0.9713465,0.025609732,0.43049997], vec![("prev",vec![-0.25,-0.0125,-0.625]),("step_size",vec![0.036201056,0.08,0.005])]),
        (vec![0.7424832,-1.281541,0.32515225], vec![("exp_avg",vec![0.099404715,0.015332542,0.24822132]),("exp_avg_sq",vec![0.042985655,0.0008087763,0.28628197])]),
        (vec![0.7524515,-0.9847011,0.5065257], vec![("mu_product",vec![2.0086093e-8]),("exp_avg",vec![-0.08472516,-0.035663318,-0.23852877]),("exp_avg_sq",vec![0.04100763,0.002030366,0.28209195])]),
    ];
    for (config,(weights,moments)) in configs().into_iter().zip(expected) {
        let p = Param::new(Tensor::from_vec(vec![0.75,-1.2,0.4], &[3]).unwrap());
        let mut opt = Adaptive::new(vec![p.clone()], &config).unwrap();
        let id = (p.tensor().id(), p.tensor()._storage_ptr());
        for step in 0..20 {
            if step == 10 { opt.set_lr(config[1]*0.6).unwrap(); }
            if step % 6 != 2 { gradient(&p,&[(step % 7) as f32*0.125-0.375,0.25/(step+1) as f32,if step%2==0 {-0.375} else {0.625}]); }
            opt.step().unwrap();
            assert!(p.grad().is_none());
        }
        assert_eq!((p.tensor().id(),p.tensor()._storage_ptr()),id);
        assert_eq!(p.tensor()._version(),17);
        close(&p.tensor().to_vec(), &weights);
        let snapshot = opt.snapshot();
        assert_eq!(snapshot.iter().find(|(n,_)| n=="t.0").unwrap().1.item(),17.);
        for (name, values) in moments { close(&snapshot.iter().find(|(n,_)| n==&format!("{name}.0")).unwrap().1.to_vec(), &values); }
    }
}

#[test]
fn owning_state_and_exact_resume_preserve_skips_and_freezing() {
    for config in configs() {
        let p = Param::new(Tensor::scalar(0.7));
        let q = Param::new(Tensor::scalar(-1.2));
        let mut opt = Adaptive::new(vec![p.clone(),q.clone(),p.clone()], &config).unwrap();
        assert_eq!(opt.state_parameters().unwrap().len(),2);
        for i in 0..4 { gradient(&p,&[0.3+i as f32*0.1]); if i%2==0 { gradient(&q,&[-0.4]); } opt.step().unwrap(); }
        let saved = opt.snapshot(); let original = state(&opt);
        let mut cloned = opt.clone();
        for (a,b) in opt.state_buffers().unwrap().iter().zip(cloned.state_buffers().unwrap()) { assert_ne!(a._storage_ptr(),b._storage_ptr()); }
        gradient(&p,&[0.2]); cloned.step().unwrap(); assert_eq!(state(&opt),original);
        let a = Param::new(p.tensor()); let b = Param::new(q.tensor());
        let mut restored = Adaptive::new(vec![a.clone(),b.clone()], &config).unwrap();
        restored.restore(&saved).unwrap();
        let ptrs: Vec<_> = opt.state_buffers().unwrap().iter().map(Tensor::_storage_ptr).collect();
        for i in 0..7 {
            q.set_trainable(i != 2); b.set_trainable(i != 2);
            gradient(&p,&[i as f32*0.13-0.4]); gradient(&a,&[i as f32*0.13-0.4]);
            if i%3!=1 { gradient(&q,&[0.4-i as f32*0.07]); gradient(&b,&[0.4-i as f32*0.07]); }
            opt.step().unwrap(); restored.step().unwrap();
            assert_eq!(state(&opt),state(&restored));
            assert_eq!(p.tensor().to_vec(),a.tensor().to_vec()); assert_eq!(q.tensor().to_vec(),b.tensor().to_vec());
        }
        assert_eq!(opt.state_buffers().unwrap().iter().map(Tensor::_storage_ptr).collect::<Vec<_>>(),ptrs);
        assert_eq!(saved.iter().map(|(n,t)| (n.clone(),t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect::<Vec<_>>(),original);
    }
}

#[test]
fn malformed_late_state_and_configuration_leave_every_live_value_unchanged() {
    for config in configs() {
        let params: Vec<_> = [1.,-0.5].into_iter().map(|v| Param::new(Tensor::scalar(v))).collect();
        let mut opt = Adaptive::new(params.clone(), &config).unwrap();
        for p in &params { gradient(p,&[0.3]); } opt.step().unwrap();
        let before = state(&opt); let versions: Vec<_> = opt.state_buffers().unwrap().iter().map(Tensor::_version).collect();
        let good = opt.snapshot();
        let mut bad = good.clone(); bad.last_mut().unwrap().1 = Tensor::zeros(&[2]);
        assert!(opt.prepare_cpu_restore(&config,&bad).is_err());
        let mut invalid = config.clone(); invalid[1] = f32::NAN;
        assert!(opt.prepare_cpu_restore(&invalid,&good).is_err());
        let mut duplicate = good.clone(); duplicate.push(good[0].clone());
        assert!(opt.restore(&duplicate).is_err());
        let mut invalid_counter = good.clone(); invalid_counter.iter_mut().find(|(n,_)|n=="t.1").unwrap().1 = Tensor::scalar(0.5);
        assert!(opt.restore(&invalid_counter).is_err());
        assert_eq!(state(&opt),before); assert_eq!(opt.configuration(),config);
        assert_eq!(opt.state_buffers().unwrap().iter().map(Tensor::_version).collect::<Vec<_>>(),versions);
        let token = opt.prepare_cpu_restore(&config,&good).unwrap(); drop(token);
        assert_eq!(state(&opt),before);
        opt.restore(&good).unwrap();
        assert_eq!(state(&opt),before);
        assert_eq!(opt.state_buffers().unwrap().iter().map(Tensor::_version).collect::<Vec<_>>(),versions.iter().map(|v|v+1).collect::<Vec<_>>());
    }
}

#[test]
fn counter_capacity_and_capture_fail_before_any_step_mutation() {
    for config in configs() {
        let params: Vec<_> = [1.,2.].into_iter().map(|v|Param::new(Tensor::scalar(v))).collect();
        let mut opt = Adaptive::new(params.clone(),&config).unwrap();
        let mut saved = opt.snapshot();
        saved.iter_mut().find(|(n,_)|n=="t.1").unwrap().1 = Tensor::scalar((1 << 24) as f32);
        if [9.,10.].contains(&config[0]) { saved.iter_mut().find(|(n,_)|n=="initialized.1").unwrap().1 = Tensor::scalar(1.); }
        opt.restore(&saved).unwrap();
        for p in &params { gradient(p,&[0.5]); }
        let before = state(&opt);
        assert!(opt.step().is_err()); assert_eq!(state(&opt),before);
        assert_eq!(params.iter().map(|p|p.tensor().item()).collect::<Vec<_>>(),vec![1.,2.]);
        assert!(params.iter().all(|p|p.grad().is_some()));
        assert!(capture(||opt.step()).is_err()); assert_eq!(state(&opt),before);
    }
}

#[test]
fn initialization_empty_slots_and_zero_epsilon_have_explicit_semantics() {
    for mut config in configs() {
        let mut empty = Adaptive::new(vec![],&config).unwrap();
        empty.step().unwrap(); assert!(empty.snapshot().is_empty()); empty.restore(&[]).unwrap();
        let p = Param::new(Tensor::zeros(&[0]));
        let mut opt = Adaptive::new(vec![p.clone()],&config).unwrap();
        gradient(&p,&[]); opt.step().unwrap();
        assert_eq!(opt.snapshot().iter().find(|(n,_)|n=="t.0").unwrap().1.item(),1.);
        let p = Param::new(Tensor::scalar(1.));
        let opt = Adaptive::new(vec![p],&config).unwrap();
        if config[0]==6. { assert_eq!(opt.state_buffers().unwrap()[0].item(),config[4]); }
        else { assert!(opt.state_buffers().unwrap().is_empty()); }
        if ![9.,10.].contains(&config[0]) {
            let epsilon = match config[0] as u8 {5|8=>3,6=>5,7|11|12=>4,_=>unreachable!()}; config[epsilon]=0.;
            let p = Param::new(Tensor::scalar(1.)); let mut opt = Adaptive::new(vec![p.clone()],&config).unwrap();
            gradient(&p,&[0.4]); opt.step().unwrap(); assert!(p.tensor().item().is_finite());
        }
    }
}

#[test]
fn rprop_step_zero_state_preserves_custom_step_sizes_and_lazy_learning_rates() {
    let config = [10.,0.1,0.5,1.2,0.001,1.,0.];
    let p = Param::new(Tensor::scalar(1.));
    let mut opt = Adaptive::new(vec![p.clone()],&config).unwrap();
    let lazy = opt.snapshot();
    opt.restore(&lazy).unwrap(); opt.set_lr(0.2).unwrap();
    gradient(&p,&[0.5]); opt.step().unwrap(); close(&p.tensor().to_vec(),&[0.8]);
    let q = Param::new(Tensor::scalar(1.));
    let mut initialized = Adaptive::new(vec![q.clone()],&config).unwrap();
    let mut custom = lazy;
    custom.iter_mut().find(|(n,_)|n=="initialized.0").unwrap().1 = Tensor::scalar(1.);
    custom.iter_mut().find(|(n,_)|n=="step_size.0").unwrap().1 = Tensor::full(&[1],0.4);
    initialized.restore(&custom).unwrap();
    gradient(&q,&[0.5]); initialized.step().unwrap(); close(&q.tensor().to_vec(),&[0.6]);
}

#[test]
fn asgd_step_zero_state_distinguishes_lazy_and_initialized_scalars() {
    let config = [9.,0.1,0.,0.,0.,0.,0.];
    let p = Param::new(Tensor::scalar(1.));
    let mut opt = Adaptive::new(vec![p.clone()],&config).unwrap();
    let lazy = opt.snapshot();
    opt.restore(&lazy).unwrap(); opt.set_lr(0.2).unwrap();
    gradient(&p,&[0.5]); opt.step().unwrap();
    close(&p.tensor().to_vec(),&[0.9]);
    let q = Param::new(Tensor::scalar(1.));
    let mut initialized = Adaptive::new(vec![q.clone()],&config).unwrap();
    let mut saved = lazy;
    for (key,value) in [("initialized.0",1.),("eta.0",0.4),("mu.0",0.25),("ax.0",2.)] {
        let tensor = &mut saved.iter_mut().find(|(n,_)|n==key).unwrap().1;
        *tensor = Tensor::full(tensor.shape(),value);
    }
    initialized.restore(&saved).unwrap();
    gradient(&q,&[0.5]); initialized.step().unwrap();
    close(&q.tensor().to_vec(),&[0.8]);
    close(&initialized.snapshot().iter().find(|(n,_)|n=="ax.0").unwrap().1.to_vec(),&[1.7]);
    let before = state(&initialized);
    let mut bad = initialized.snapshot();
    bad.iter_mut().find(|(n,_)|n=="initialized.0").unwrap().1 = Tensor::scalar(0.);
    assert!(initialized.restore(&bad).is_err()); assert_eq!(state(&initialized),before);
}
