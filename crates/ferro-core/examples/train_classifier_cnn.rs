//! End-to-end CNN training proof on synthetic, linearly-separable image
//! classes, using only public ferro-core API: modules::Conv2D / BatchNorm,
//! ops_ext max_pool2d, nn::Linear + cross_entropy_indices, optim::AdamW,
//! data::DataLoader and checkpoint::Checkpoint.
//!
//! The task: two classes of 1x8x8 images. Class 0 has a bright 3x3 block in
//! the top-left corner; class 1 has it in the bottom-right; everything else is
//! N(0,1) noise. The network must learn to localize the block.
//!
//! Run:
//!   cargo run -p ferro-core --example train_classifier_cnn
//!   cargo run -p ferro-core --example train_classifier_cnn -- --steps 300 --out target/cnn_ckpt

use ferro_core::checkpoint::Checkpoint;
use ferro_core::data::{DataLoader, TensorDataset};
use ferro_core::dtype::DType;
use ferro_core::modules::BatchNorm;
use ferro_core::nn::{cross_entropy_indices, Init, Linear, Module};
use ferro_core::optim::AdamW;

use ferro_core::params::Param;
use ferro_core::rng::Rng;
use ferro_core::tensor::Tensor;
use ferro_core::Result;

const IMG: usize = 8;
const CLASSES: usize = 2;

/// Bias-free 2-D convolution wrapping ops_ext::conv2d directly.
///
/// modules::Conv2D is avoided because its NCHW bias add broadcasts a [c_out]
/// bias against the last dim, which only resolves when c_out equals the image
/// width; any other geometry errors out (core limitation, see
/// docs/TRAINING_GATE.md). BatchNorm downstream supplies the shift instead.
struct Conv {
    w: Param,
    stride: usize,
    padding: usize,
}

impl Conv {
    fn new(in_c: usize, out_c: usize, k: usize, stride: usize, padding: usize, rng: &Rng) -> Conv {
        let fan_in = in_c * k * k;
        Conv {
            w: Param::new(Init::Kaiming.fill(rng, &[out_c, in_c, k, k], fan_in, fan_in)),
            stride,
            padding,
        }
    }
}

impl Module for Conv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv2d(&self.w.tensor(), self.stride, self.padding)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![("weight".into(), self.w.clone())]
    }
}

struct CnnNet {
    conv1: Conv,
    conv2: Conv,
    bn: BatchNorm,
    fc1: Linear,
    fc2: Linear,
}

impl CnnNet {
    fn new(rng: &Rng) -> CnnNet {
        // 8x8 -> conv pad 1 -> pool -> 4x4 -> conv pad 1 -> pool -> 2x2.
        CnnNet {
            conv1: Conv::new(1, 4, 3, 1, 1, rng),
            conv2: Conv::new(4, 8, 3, 1, 1, rng),
            bn: BatchNorm::new(8 * 2 * 2),
            fc1: Linear::with_init(8 * 2 * 2, 32, rng, Init::Xavier),
            fc2: Linear::with_init(32, CLASSES, rng, Init::Xavier),
        }
    }
}

impl Module for CnnNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(x)?.relu().max_pool2d(2, 2)?.relu();
        let h = self.conv2.forward(&h)?.relu().max_pool2d(2, 2)?;
        let n = h.shape()[0];
        let flat = h.reshape(&[n, 32])?;
        let h2 = self.bn.forward(&flat)?;
        let h3 = self.fc1.forward(&h2)?.relu();
        self.fc2.forward(&h3)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut out = Vec::new();
        for (prefix, m) in [
            ("conv1", &self.conv1 as &dyn Module),
            ("conv2", &self.conv2),
            ("bn", &self.bn),
            ("fc1", &self.fc1),
            ("fc2", &self.fc2),
        ] {
            out.extend(
                m.named_parameters()
                    .into_iter()
                    .map(|(n, p)| (format!("{prefix}.{n}"), p)),
            );
        }
        out
    }
}

/// Bright 3x3 block at a corner plus iid N(0,1) noise elsewhere.
fn make_data(n: usize, rng: &Rng) -> (Tensor, Tensor) {
    let mut xs = Vec::with_capacity(n * IMG * IMG);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let class = i % CLASSES;
        ys.push(class as f32);
        for row in 0..IMG {
            for col in 0..IMG {
                let in_block_row = row < 3 || row >= IMG - 3;
                let in_block_col = col < 3 || col >= IMG - 3;
                let bright = match class {
                    0 => row < 3 && col < 3,
                    _ => row >= IMG - 3 && col >= IMG - 3,
                };
                let v = if bright && in_block_row && in_block_col {
                    3.0 + rng.normal() * 0.3
                } else {
                    rng.normal()
                };
                xs.push(v);
            }
        }
    }
    (
        Tensor::from_vec(xs, &[n, 1, IMG, IMG]).expect("data shape"),
        Tensor::from_vec(ys, &[n]).expect("labels shape"),
    )
}

fn accuracy(model: &CnnNet, xs: &Tensor, ys: &Tensor) -> Result<f32> {
    let logits = model.forward(xs)?;
    let pred = logits.argmax(1, false)?;
    let truth = ys.to_dtype(DType::I64);
    let p = pred.to_vec_i64();
    let t = truth.to_vec_i64();
    let hit = p.iter().zip(&t).filter(|(a, b)| a == b).count();
    Ok(hit as f32 / t.len() as f32)
}

fn main() -> Result<()> {
    let mut steps = 300usize;
    let mut out = "target/cnn_ckpt".to_string();
    let mut resume: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--steps" => steps = it.next().and_then(|v| v.parse().ok()).unwrap_or(steps),
            "--out" => out = it.next().unwrap_or_else(|| out.clone()),
            "--resume" => resume = it.next(),
            other => panic!("unknown argument {other:?}"),
        }
    }

    let seed = 11u64;
    let rng = Rng::new(seed);
    let (train_x, train_y) = make_data(320, &rng);
    let eval_rng = Rng::new(seed + 1);
    let (eval_x, eval_y) = make_data(128, &eval_rng);

    let ds = std::sync::Arc::new(TensorDataset::new(train_x.clone(), train_y.clone())?);
    let loader = DataLoader::new(ds, 16).shuffle(seed).drop_last(true);

    let model = CnnNet::new(&rng);
    if let Some(dir) = &resume {
        let cp = Checkpoint::load_from_dir(dir)?;
        cp.load_into_module(&model)?;
        println!("resumed from {dir} at step {}", cp.step);
    }
    let mut opt = AdamW::new(model.parameters(), 5e-3).with_max_grad_norm(1.0);
    ferro_core::nn::train(&model);

    println!("train={:?} eval={:?}", train_x.shape(), eval_x.shape());
    let print_every = 20usize;
    for (step, batch) in loader.iter().enumerate() {
        let (x, y_f32) = batch?;
        let targets = y_f32.reshape(&[y_f32.numel()])?.to_dtype(DType::I64);
        let logits = model.forward(&x)?;
        let loss = cross_entropy_indices(&logits, &targets)?;
        opt.zero_grad();
        loss.backward();
        opt.step();
        if (step + 1) % print_every == 0 || step + 1 == steps {
            ferro_core::nn::eval(&model);
            let acc = accuracy(&model, &eval_x, &eval_y)?;
            ferro_core::nn::train(&model);
            println!(
                "step {}: loss {:.4} eval_acc {:.3}",
                step + 1,
                loss.item(),
                acc
            );
        }
        if step + 1 == steps {
            break;
        }
    }

    ferro_core::nn::eval(&model);
    let final_loss = {
        let targets = train_y.to_dtype(DType::I64);
        cross_entropy_indices(&model.forward(&train_x)?, &targets)?.item()
    };
    let acc = accuracy(&model, &eval_x, &eval_y)?;
    Checkpoint::from_module(steps as u64, &model)
        .with_rng_seed(seed)
        .save_to_dir(&out)?;
    println!("final: train_loss {final_loss:.4} eval_acc {acc:.3} checkpoint in {out}");
    assert!(
        acc > 0.95,
        "CNN failed to separate the synthetic classes: {acc}"
    );
    Ok(())
}
