//! End-to-end training proof: a tiny GPT-2-style character-level decoder
//! trained with the public ferro-core API only (Embedding, Linear, autograd
//! ops, AdamW, DataLoader, Checkpoint). The transformer blocks are assembled
//! here from primitives - causal self-attention is masked softmax over
//! q @ k^T / sqrt(d) via nn::scaled_dot_product_attention.
//!
//! Run:
//!   cargo run -p ferro-core --example train_gpt2_small
//!   cargo run -p ferro-core --example train_gpt2_small -- --steps 400 --ckpt-every 100 --out target/gpt2_ckpt
//! Resume (continues from the saved step):
//!   cargo run -p ferro-core --example train_gpt2_small -- --resume target/gpt2_ckpt

use ferro_core::checkpoint::Checkpoint;
use ferro_core::data::{DataLoader, TensorDataset};
use ferro_core::dtype::DType;
use ferro_core::nn::{
    cross_entropy_indices, scaled_dot_product_attention, Embedding, Init, LayerNorm, Linear, Module,
};
use ferro_core::optim::AdamW;
use ferro_core::params::Param;
use ferro_core::rng::Rng;
use ferro_core::tensor::Tensor;
use ferro_core::Result;

const CORPUS: &str = concat!(
    "the quick brown fox jumps over the lazy dog. ",
    "pack my box with five dozen liquor jugs. ",
    "how vexingly quick daft zebras jump. ",
    "sphinx of black quartz judge my vow. ",
    "the five boxing wizards jump quickly. ",
    "bright vixens jump dozy fowl quack. ",
    "quick zephyrs blow vexing daft jim. ",
    "two driven jocks help fax my big quiz. ",
);

const SEQ: usize = 16;
const DIM: usize = 48;
const HEADS: usize = 3;
const N_BLOCKS: usize = 2;

struct Block {
    ln1: LayerNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    ln2: LayerNorm,
    up: Linear,
    down: Linear,
}

impl Block {
    fn new(rng: &Rng) -> Block {
        Block {
            ln1: LayerNorm::new(DIM),
            wq: Linear::with_init(DIM, DIM, rng, Init::Xavier),
            wk: Linear::with_init(DIM, DIM, rng, Init::Xavier),
            wv: Linear::with_init(DIM, DIM, rng, Init::Xavier),
            wo: Linear::with_init(DIM, DIM, rng, Init::Xavier),
            ln2: LayerNorm::new(DIM),
            up: Linear::with_init(DIM, 4 * DIM, rng, Init::Xavier),
            down: Linear::with_init(4 * DIM, DIM, rng, Init::Xavier),
        }
    }

    /// [b*s, d] -> [b*h, s, d/h]: project through the layer, then split heads.
    fn heads(&self, x_flat: &Tensor, layer: &Linear, b: usize, s: usize) -> Result<Tensor> {
        let hd = DIM / HEADS;
        let p = layer.forward(x_flat)?;
        p.reshape(&[b, s, HEADS, hd])?
            .transpose(1, 2)?
            .reshape(&[b * HEADS, s, hd])
    }
}

impl Module for Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 || x.shape()[2] != DIM {
            return Err(ferro_core::Error::InvalidShape {
                op: "block",
                msg: format!("expected [batch, {SEQ}, {DIM}], got {:?}", x.shape()),
            });
        }
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let flat = x.reshape(&[b * s, DIM])?;
        let h = x.add(&self.ln1.forward(&flat)?.reshape(x.shape())?)?;
        let hf = h.reshape(&[b * s, DIM])?;
        let q = self.heads(&hf, &self.wq, b, s)?;
        let k = self.heads(&hf, &self.wk, b, s)?;
        let v = self.heads(&hf, &self.wv, b, s)?;
        let attn = scaled_dot_product_attention(&q, &k, &v, true)?;
        let hd = DIM / HEADS;
        let merged = attn
            .reshape(&[b, HEADS, s, hd])?
            .transpose(1, 2)?
            .reshape(&[b * s, DIM])?;
        let h2 = h.add(&self.wo.forward(&merged)?.reshape(x.shape())?)?;
        let f = self.ln2.forward(&h2.reshape(&[b * s, DIM])?)?;
        let mlp = self.down.forward(&self.up.forward(&f)?.gelu())?;
        h2.add(&mlp.reshape(x.shape())?)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut out = Vec::new();
        for (prefix, m) in [
            ("ln1", &self.ln1 as &dyn Module),
            ("wq", &self.wq),
            ("wk", &self.wk),
            ("wv", &self.wv),
            ("wo", &self.wo),
            ("ln2", &self.ln2),
            ("up", &self.up),
            ("down", &self.down),
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

/// Gelu MLP is composed inline in Block::forward (up projection then gelu).

struct TinyGpt2 {
    tok_embed: Embedding,
    pos_embed: Embedding,
    blocks: Vec<Block>,
    final_ln: LayerNorm,
    head: Linear,
}

impl TinyGpt2 {
    fn new(vocab: usize, rng: &Rng) -> TinyGpt2 {
        TinyGpt2 {
            tok_embed: Embedding::new(vocab, DIM, rng),
            pos_embed: Embedding::new(SEQ, DIM, rng),
            blocks: (0..N_BLOCKS).map(|_| Block::new(rng)).collect(),
            final_ln: LayerNorm::new(DIM),
            head: Linear::with_init(DIM, vocab, rng, Init::Normal(0.02)),
        }
    }

    fn logits(&self, ids: &Tensor) -> Result<Tensor> {
        let shape = ids.shape().to_vec();
        let b = shape[0];
        let pos_ids = Tensor::arange(SEQ as i64);
        let pos = self.pos_embed.forward(&pos_ids)?;
        let x = self.tok_embed.forward(ids)?.add(&pos)?;
        let mut h = x;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let flat = self.final_ln.forward(&h.reshape(&[b * SEQ, DIM])?)?;
        self.head.forward(&flat)
    }
}

impl Module for TinyGpt2 {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        self.logits(ids)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut out = vec![
            (
                "tok_embed.weight".into(),
                self.tok_embed.named_parameters()[0].1.clone(),
            ),
            (
                "pos_embed.weight".into(),
                self.pos_embed.named_parameters()[0].1.clone(),
            ),
        ];
        for (i, b) in self.blocks.iter().enumerate() {
            out.extend(
                b.named_parameters()
                    .into_iter()
                    .map(|(n, p)| (format!("blocks.{i}.{n}"), p)),
            );
        }
        out.extend(
            self.final_ln
                .named_parameters()
                .into_iter()
                .map(|(n, p)| (format!("final_ln.{n}"), p)),
        );
        out.extend(
            self.head
                .named_parameters()
                .into_iter()
                .map(|(n, p)| (format!("head.{n}"), p)),
        );
        out
    }
}

struct Args {
    steps: usize,
    ckpt_every: usize,
    out: String,
    resume: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        steps: 300,
        ckpt_every: 100,
        out: "target/gpt2_ckpt".into(),
        resume: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--steps" => a.steps = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.steps),
            "--ckpt-every" => {
                a.ckpt_every = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(a.ckpt_every)
            }
            "--out" => a.out = it.next().unwrap_or_else(|| a.out.clone()),
            "--resume" => a.resume = it.next(),
            other => panic!("unknown argument {other:?}"),
        }
    }
    a
}

fn main() -> Result<()> {
    let args = parse_args();
    let chars: Vec<char> = {
        let mut c: Vec<char> = CORPUS
            .chars()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        c.sort();
        c
    };
    let vocab = chars.len();

    // Shifted windows: input is text[i .. i+SEQ], target is text[i+1 .. i+SEQ+1].
    let ids: Vec<f32> = CORPUS
        .chars()
        .map(|ch| chars.iter().position(|&c| c == ch).unwrap() as f32)
        .collect();
    let n_total = ids.len() - SEQ - 1;
    let mut xs = Vec::with_capacity(n_total * SEQ);
    let mut ys = Vec::with_capacity(n_total * SEQ);
    for i in 0..n_total {
        xs.extend_from_slice(&ids[i..i + SEQ]);
        ys.extend_from_slice(&ids[i + 1..i + 1 + SEQ]);
    }
    let n_windows = n_total;
    let ds = std::sync::Arc::new(TensorDataset::new(
        Tensor::from_vec(xs, &[n_windows, SEQ])?,
        Tensor::from_vec(ys, &[n_windows, SEQ])?,
    )?);
    let loader = DataLoader::new(ds.clone(), 16)
        .shuffle(1234)
        .drop_last(true);
    let steps_per_epoch = loader.len();

    let seed = 7u64;
    let mut start_step = 0u64;
    let rng = Rng::new(seed);
    let model = TinyGpt2::new(vocab, &rng);
    let mut opt = AdamW::new(model.parameters(), 3e-3)
        .with_weight_decay(0.01)
        .with_max_grad_norm(1.0);

    if let Some(dir) = &args.resume {
        let cp = Checkpoint::load_from_dir(dir)?;
        cp.load_into_module(&model)?;
        start_step = cp.step;
        println!("resumed from {dir} at step {start_step}");
    }

    println!(
        "vocab={vocab} params={} windows={n_windows} steps/epoch={steps_per_epoch}",
        model.named_parameters().len()
    );
    ferro_core::nn::train(&model);

    let mut step = start_step;
    let mut loss_acc = 0.0f32;
    let mut loss_n = 0usize;
    let print_every = 20usize;
    while step < start_step + args.steps as u64 {
        for batch in loader.iter() {
            if step >= start_step + args.steps as u64 {
                break;
            }
            let (x_f32, y_f32) = batch?;
            let x = x_f32.to_dtype(DType::I64);
            let targets = y_f32.reshape(&[y_f32.numel()])?.to_dtype(DType::I64);
            let logits = model.logits(&x)?;
            let loss = cross_entropy_indices(&logits, &targets)?;
            opt.zero_grad();
            loss.backward();
            opt.step();
            loss_acc += loss.item();
            loss_n += 1;
            step += 1;
            if step % print_every as u64 == 0 || step == start_step + args.steps as u64 {
                println!("step {step}: loss {:.4}", loss_acc / loss_n as f32);
                loss_acc = 0.0;
                loss_n = 0;
            }
            if args.ckpt_every > 0 && step % args.ckpt_every as u64 == 0 {
                Checkpoint::from_module(step, &model)
                    .with_rng_seed(seed)
                    .save_to_dir(&args.out)?;
            }
        }
    }
    Checkpoint::from_module(step, &model)
        .with_rng_seed(seed)
        .save_to_dir(&args.out)?;
    println!("done at step {step}; checkpoint in {}", args.out);

    // Greedy sample from the trained model to show it learned the corpus.
    ferro_core::nn::eval(&model);
    let mut ctx: Vec<i64> = ids[..SEQ].iter().map(|&v| v as i64).collect();
    let mut out = String::new();
    for _ in 0..40 {
        let x = Tensor::from_vec_i64(ctx[(ctx.len() - SEQ)..].to_vec(), &[1, SEQ])?;
        let logits = model.logits(&x)?;
        let last = logits.index_select(0, &[SEQ - 1])?.reshape(&[vocab])?;
        let next = last.argmax(0, false)?.item() as usize;
        ctx.push(next as i64);
        out.push(chars[next]);
    }
    println!("sample: {out}");
    Ok(())
}
