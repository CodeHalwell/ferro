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
