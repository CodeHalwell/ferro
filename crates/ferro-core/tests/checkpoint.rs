use ferro_core::checkpoint::{Checkpoint, FORMAT_VERSION};
use ferro_core::modules::ModuleList;
use ferro_core::nn::{Linear, Module, RmsNorm};
use ferro_core::params::Param;
use ferro_core::rng::Rng;
use ferro_core::tensor::Tensor;

fn bit_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn sample_model(rng: &Rng) -> ModuleList {
    let layers: Vec<Box<dyn Module>> =
        vec![Box::new(Linear::new(3, 4, rng)), Box::new(RmsNorm::new(4))];
    ModuleList::new(layers)
}

#[test]
fn round_trip_is_bit_exact() {
    let dir = std::env::temp_dir().join(format!("ferro_ckpt_rt_{}", std::process::id()));
    let rng = Rng::new(7);
    let model = sample_model(&rng);
    let mut cp = Checkpoint::from_module(42, &model).with_rng_seed(7);
    cp.tensors.push((
        "optim.momentum.0.weight".into(),
        Tensor::from_vec(vec![0.25f32, -1.5e-8, f32::NAN, 3.0], &[4]).unwrap(),
    ));
    cp.save_to_dir(&dir).unwrap();

    let loaded = Checkpoint::load_from_dir(&dir).unwrap();
    assert_eq!(loaded.version, FORMAT_VERSION);
    assert_eq!(loaded.step, 42);
    assert_eq!(loaded.rng_seed, Some(7));
    for ((name, t), (lname, lt)) in cp.tensors.iter().zip(&loaded.tensors) {
        assert_eq!(name, lname);
        assert_eq!(t.shape(), lt.shape());
        assert!(
            bit_eq(&t.to_vec(), &lt.to_vec()),
            "{name} changed across save/load"
        );
    }
    // NaN must survive the file format too (bit pattern preserved).
    assert!(loaded.f32_buffer("optim.momentum.0.weight").unwrap()[2].is_nan());

    let fresh = sample_model(&Rng::new(999));
    let params_only = Checkpoint {
        tensors: cp
            .tensors
            .iter()
            .filter(|(n, _)| n.starts_with("model."))
            .cloned()
            .collect(),
        ..loaded.clone()
    };
    params_only.load_into_module(&fresh).unwrap();
    for (name, t) in &cp.tensors {
        if !name.starts_with("model.") {
            continue;
        }
        let short = name.strip_prefix("model.").unwrap();
        let now = fresh
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == short)
            .unwrap()
            .1
            .tensor();
        assert!(
            bit_eq(&t.to_vec(), &now.to_vec()),
            "{short} not restored bit-exactly"
        );
    }

    // Strictness: a checkpoint missing a parameter, or carrying an extra one.
    let mut bad = loaded.clone();
    bad.tensors.retain(|(n, _)| !n.contains("weight"));
    assert!(bad.load_into_module(&fresh).is_err());
    let mut extra = loaded.clone();
    extra
        .tensors
        .push(("model.bogus".into(), Tensor::scalar(1.0)));
    assert!(extra.load_into_module(&fresh).is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

/// The OptimizerState path end-to-end: real Adam moments saved through
/// `from_module_with_optim`, round-tripped via save_to_dir/load_from_dir,
/// restored with `load_optim_into`, then proven to keep training identically.
#[test]
fn optimizer_state_round_trips_and_resumes_bit_exactly() {
    use ferro_core::optim::{AdamW, OptimizerState};

    let dir = std::env::temp_dir().join(format!("ferro_ckpt_optim_{}", std::process::id()));
    let mk_model = || -> ModuleList {
        let mut rng = Rng::new(11);
        ModuleList::new(vec![Box::new(Linear::new(2, 2, &mut rng))])
    };
    let batch = || Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5], &[3, 2]).unwrap();

    let train_step = |model: &ModuleList, opt: &mut AdamW| -> f32 {
        let x = batch();
        let target = Tensor::from_vec(vec![0.3f32, -0.6, 0.9, -0.1, 0.4, 0.8], &[3, 2]).unwrap();
        let out = model.forward(&x).unwrap();
        let loss = out
            .sub(&target)
            .unwrap()
            .mul(&out.sub(&target).unwrap())
            .unwrap()
            .mean();
        let l = loss.item();
        opt.zero_grad();
        loss.backward();
        opt.step();
        l
    };

    // Uninterrupted reference run.
    let model_ref = mk_model();
    let mut opt_ref = AdamW::new(model_ref.parameters(), 0.05).with_weight_decay(0.01);
    let ref_losses: Vec<f32> = (0..10)
        .map(|_| train_step(&model_ref, &mut opt_ref))
        .collect();

    // Interrupted at step 4: save params + moments, restore into fresh state.
    let model_a = mk_model();
    let mut opt_a = AdamW::new(model_a.parameters(), 0.05).with_weight_decay(0.01);
    let mut losses = Vec::new();
    for _ in 0..4 {
        losses.push(train_step(&model_a, &mut opt_a));
    }
    Checkpoint::from_module_with_optim(4, &model_a, &opt_a)
        .with_rng_seed(123)
        .with_rng_offset(4096)
        .save_to_dir(&dir)
        .unwrap();
    let cp = Checkpoint::load_from_dir(&dir).unwrap();
    assert_eq!(cp.rng_seed, Some(123));
    assert_eq!(cp.rng_offset, Some(4096));

    // Strictness first: an extra optim tensor must be rejected, and a
    // mismatched optimizer type (Sgd-shaped state) cannot be loaded.
    let mut extra = cp.clone();
    extra
        .tensors
        .push(("optim.bogus".into(), Tensor::scalar(1.0)));
    let model_b = mk_model();
    let mut opt_b = AdamW::new(model_b.parameters(), 0.05).with_weight_decay(0.01);
    assert!(extra.load_optim_into(&mut opt_b).is_err());

    let mut missing = cp.clone();
    missing.tensors.retain(|(n, _)| !n.starts_with("optim.v."));
    assert!(missing.load_optim_into(&mut opt_b).is_err());

    // Real restore, then continue and compare bitwise against the reference.
    cp.load_into_module(&model_b).unwrap();
    cp.load_optim_into(&mut opt_b).unwrap();
    for _ in 4..10 {
        losses.push(train_step(&model_b, &mut opt_b));
    }
    for (i, (a, b)) in losses.iter().zip(&ref_losses).enumerate() {
        assert!(
            bit_eq(std::slice::from_ref(a), std::slice::from_ref(b)),
            "step {i}: {a} vs {b}"
        );
    }
    // Moments themselves are identical after re-convergence.
    assert_eq!(
        opt_b.snapshot().len(),
        opt_ref.snapshot().len() + 0,
        "same array count"
    );
    for ((n, t), (rn, rt)) in opt_b.snapshot().iter().zip(opt_ref.snapshot()) {
        assert_eq!(n.as_str(), rn);
        assert!(bit_eq(&t.to_vec(), &rt.to_vec()), "{n} diverged");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Dropout is counter-based Philox keyed by explicit (seed, offset), so a
/// resumed run is bit-exact through stochastic training provided the offset
/// counter is checkpointed alongside the seed. This is that proof.
#[test]
fn resumed_run_with_dropout_matches_uninterrupted_bitwise() {
    use ferro_core::optim::AdamW;

    let dir = std::env::temp_dir().join(format!("ferro_ckpt_dropout_{}", std::process::id()));
    const SEED: u64 = 5;
    // Each step consumes a disjoint 1024-wide slice of the Philox stream.
    let offset_for = |step: u64| step * 1024;

    let mk = |seed: u64| -> (ModuleList, AdamW) {
        let mut rng = Rng::new(seed);
        let model = ModuleList::new(vec![Box::new(Linear::new(2, 1, &mut rng))]);
        let opt = AdamW::new(model.parameters(), 0.05).with_weight_decay(0.01);
        (model, opt)
    };
    let x = Tensor::from_vec(
        vec![0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, 1.1, -0.4],
        &[4, 2],
    )
    .unwrap();
    let y = Tensor::from_vec(vec![0.3f32, -0.6, 0.9, 0.2], &[4, 1]).unwrap();
    let step_loss = |model: &ModuleList, opt: &mut AdamW, step: u64| -> f32 {
        let dropped = x.dropout(0.25, true, SEED, offset_for(step)).unwrap();
        let pred = dropped
            .matmul(&model.named_parameters()[0].1.tensor())
            .unwrap()
            .add(&model.named_parameters()[1].1.tensor())
            .unwrap();
        let loss = pred
            .sub(&y)
            .unwrap()
            .mul(&pred.sub(&y).unwrap())
            .unwrap()
            .mean();
        let l = loss.item();
        opt.zero_grad();
        loss.backward();
        opt.step();
        l
    };

    let (ref_model, mut ref_opt) = mk(11);
    let ref_losses: Vec<f32> = (0..10)
        .map(|s| step_loss(&ref_model, &mut ref_opt, s))
        .collect();

    let (part_model, mut part_opt) = mk(11);
    let mut losses = Vec::new();
    for s in 0..5u64 {
        losses.push(step_loss(&part_model, &mut part_opt, s));
    }
    Checkpoint::from_module_with_optim(5, &part_model, &part_opt)
        .with_rng_seed(SEED)
        .with_rng_offset(offset_for(5))
        .save_to_dir(&dir)
        .unwrap();

    let cp = Checkpoint::load_from_dir(&dir).unwrap();
    let (resumed_model, mut resumed_opt) = mk(999); // different init seed on purpose
    cp.load_into_module(&resumed_model).unwrap();
    cp.load_optim_into(&mut resumed_opt).unwrap();
    for s in 5..10u64 {
        losses.push(step_loss(&resumed_model, &mut resumed_opt, s));
    }
    for (i, (a, b)) in losses.iter().zip(&ref_losses).enumerate() {
        assert!(
            bit_eq(std::slice::from_ref(a), std::slice::from_ref(b)),
            "step {i}: {a} vs {b}"
        );
    }
    assert!(bit_eq(
        &resumed_model.named_parameters()[0].1.tensor().to_vec(),
        &ref_model.named_parameters()[0].1.tensor().to_vec()
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn atomic_write_leaves_no_temp_files_and_newer_version_is_rejected() {
    let dir = std::env::temp_dir().join(format!("ferro_ckpt_atomic_{}", std::process::id()));
    let model = sample_model(&Rng::new(1));
    Checkpoint::from_module(0, &model)
        .save_to_dir(&dir)
        .unwrap();
    let leftovers: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        leftovers
            .iter()
            .all(|n| n.ends_with(".json") || n.ends_with(".safetensors")),
        "temp files left behind: {leftovers:?}"
    );

    // A future-format sidecar must be rejected, not silently misread.
    std::fs::write(
        dir.join("checkpoint.json"),
        "{\n  \"version\": 99,\n  \"step\": 1\n}\n",
    )
    .unwrap();
    match Checkpoint::load_from_dir(&dir) {
        Err(ferro_core::Error::Unsupported { .. }) => {}
        other => panic!(
            "expected Unsupported, got {:?}",
            other.map(|c| (c.version, c.step))
        ),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Momentum-SGD over a tiny 1-D regression, with the update rule spelled out so
/// the test owns every piece of state a resume needs: parameters, velocity
/// buffer, and step count.
struct Run {
    w: Param,
    b: Param,
    vw: Vec<f32>,
    vb: Vec<f32>,
    step: u64,
}

impl Run {
    fn new(seed: u64) -> Run {
        let rng = Rng::new(seed);
        let w: Vec<f32> = (0..2).map(|_| rng.normal()).collect();
        Run {
            w: Param::new(Tensor::from_vec(w, &[2, 1]).unwrap()),
            b: Param::new(Tensor::zeros(&[1])),
            vw: vec![0.0; 2],
            vb: vec![0.0],
            step: 0,
        }
    }

    fn batch() -> (Tensor, Tensor) {
        let x = Tensor::from_vec(
            vec![[1.0, -2.0], [0.5, 0.25], [-1.5, 1.0]].concat(),
            &[3, 2],
        )
        .unwrap();
        // Fixed teacher: y = 0.6*x0 - 0.4*x1 + 0.1
        let rows: [[f32; 2]; 3] = [[1.0, -2.0], [0.5, 0.25], [-1.5, 1.0]];
        let y = Tensor::from_vec(
            rows.iter().map(|r| 0.6 * r[0] - 0.4 * r[1] + 0.1).collect(),
            &[3, 1],
        )
        .unwrap();
        (x, y)
    }

    /// One training step at lr=0.3, momentum=0.9; returns the loss value.
    fn train_step(&mut self, x: &Tensor, y: &Tensor) -> f32 {
        self.w.zero_grad();
        self.b.zero_grad();
        let pred = x
            .matmul(&self.w.tensor())
            .unwrap()
            .add(&self.b.tensor())
            .unwrap();
        let diff = pred.sub(y).unwrap();
        let loss = diff.mul(&diff).unwrap().mean();
        let l = loss.to_vec()[0];
        loss.backward();
        let (lr, mom) = (0.3f32, 0.9f32);
        let gw = self.w.grad().unwrap().to_vec();
        let gb = self.b.grad().unwrap().to_vec();
        let mut nw = self.w.tensor().to_vec();
        for j in 0..nw.len() {
            self.vw[j] = mom * self.vw[j] + gw[j];
            nw[j] -= lr * self.vw[j];
        }
        let mut nb = self.b.tensor().to_vec();
        self.vb[0] = mom * self.vb[0] + gb[0];
        nb[0] -= lr * self.vb[0];
        self.w.set(Tensor::from_vec(nw, &[2, 1]).unwrap());
        self.b.set(Tensor::from_vec(nb, &[1]).unwrap());
        self.step += 1;
        l
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint::new(self.step)
            .with_rng_seed(123)
            .with_tensor("model.w", self.w.tensor())
            .with_tensor("model.b", self.b.tensor())
            .with_tensor(
                "optim.velocity_w",
                Tensor::from_vec(self.vw.clone(), &[2]).unwrap(),
            )
            .with_tensor("optim.velocity_b", Tensor::scalar(self.vb[0]))
    }

    fn restore(cp: &Checkpoint) -> Run {
        Run {
            w: Param::new(cp.tensor("model.w").unwrap().clone()),
            b: Param::new(cp.tensor("model.b").unwrap().clone()),
            vw: cp.f32_buffer("optim.velocity_w").unwrap(),
            vb: vec![cp.f32_buffer("optim.velocity_b").unwrap()[0]],
            step: cp.step,
        }
    }
}

#[test]
fn resumed_run_matches_uninterrupted_loss_trajectory_bitwise() {
    let dir = std::env::temp_dir().join(format!("ferro_ckpt_resume_{}", std::process::id()));
    let (x, y) = Run::batch();

    // Uninterrupted reference: 12 steps.
    let mut full = Run::new(123);
    let ref_losses: Vec<f32> = (0..12).map(|_| full.train_step(&x, &y)).collect();

    // Interrupted run: 5 steps, checkpoint, restore into a fresh Run, continue.
    let mut part = Run::new(123);
    let mut losses = Vec::new();
    for _ in 0..5 {
        losses.push(part.train_step(&x, &y));
    }
    part.checkpoint().save_to_dir(&dir).unwrap();
    let cp = Checkpoint::load_from_dir(&dir).unwrap();
    assert_eq!(cp.step, 5);
    let mut resumed = Run::restore(&cp);
    while resumed.step < 12 {
        losses.push(resumed.train_step(&x, &y));
    }

    assert_eq!(losses.len(), ref_losses.len());
    for (i, (a, b)) in losses.iter().zip(&ref_losses).enumerate() {
        assert!(
            bit_eq(std::slice::from_ref(a), std::slice::from_ref(b)),
            "step {i}: {a} vs {b}"
        );
    }
    assert!(bit_eq(
        &full.w.tensor().to_vec(),
        &resumed.w.tensor().to_vec()
    ));
    assert!(bit_eq(
        &full.b.tensor().to_vec(),
        &resumed.b.tensor().to_vec()
    ));
    // And it is actually training: loss went down materially.
    assert!(*ref_losses.last().unwrap() < *ref_losses.first().unwrap() * 0.5);
    let _ = std::fs::remove_dir_all(&dir);
}
