//! Char-level transformer LM, end to end in one binary: train on an embedded
//! text, save a safetensors state dict, reload it into a fresh model, and
//! greedy-generate the text back. This exercises the whole M3 pipeline
//! (embedding -> RoPE'd causal attention blocks -> LM head, then
//! save_module/load_module) on the optimized CPU backend.
//!
//! Run: cargo run --release -p ferro-fastcpu --example char_lm

use ferro_core::error::Result;
use ferro_core::nn::{cross_entropy_indices, load_module, save_module, Embedding, Linear, Module, RmsNorm, TransformerBlock};
use ferro_core::optim::AdamW;
use ferro_core::{Param, Rng, Tensor};

const TEXT: &str = "the quick brown fox jumps over the lazy dog while the sly red fox naps. ";
const DIM: usize = 32;
const HEADS: usize = 4;
const DEPTH: usize = 2;

struct CharLm {
    emb: Embedding,
    blocks: Vec<TransformerBlock>,
    norm: RmsNorm,
    head: Linear,
}

impl CharLm {
    fn new(vocab: usize, seed: u64) -> CharLm {
        let rng = Rng::new(seed);
        CharLm {
            emb: Embedding::new(vocab, DIM, &rng),
            blocks: (0..DEPTH).map(|_| TransformerBlock::new(DIM, HEADS, &rng).unwrap()).collect(),
            norm: RmsNorm::new(DIM),
            head: Linear::new(DIM, vocab, &rng),
        }
    }
}

impl Module for CharLm {
    /// `[1, s]` ids -> `[s, vocab]` next-char logits.
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let mut h = self.emb.forward(ids)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        let h = self.norm.forward(&h)?.reshape(&[ids.numel(), DIM])?;
        self.head.forward(&h)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut out: Vec<(String, Param)> =
            self.emb.named_parameters().into_iter().map(|(n, p)| (format!("emb.{n}"), p)).collect();
        for (i, b) in self.blocks.iter().enumerate() {
            out.extend(b.named_parameters().into_iter().map(|(n, p)| (format!("blocks.{i}.{n}"), p)));
        }
        out.extend(self.norm.named_parameters().into_iter().map(|(n, p)| (format!("norm.{n}"), p)));
        out.extend(self.head.named_parameters().into_iter().map(|(n, p)| (format!("head.{n}"), p)));
        out
    }
}

fn main() {
    ferro_fastcpu::install();

    let chars: Vec<char> = {
        let mut c: Vec<char> = TEXT.chars().collect();
        c.sort_unstable();
        c.dedup();
        c
    };
    let encode = |s: &str| -> Vec<i64> {
        s.chars().map(|ch| chars.iter().position(|&c| c == ch).unwrap() as i64).collect()
    };
    let ids = encode(TEXT);
    let n = ids.len() - 1;
    let input = Tensor::from_vec_i64(ids[..n].to_vec(), &[1, n]).unwrap();
    let targets = Tensor::from_vec_i64(ids[1..].to_vec(), &[n]).unwrap();

    let model = CharLm::new(chars.len(), 7);
    println!("training a {DEPTH}-block dim-{DIM} char LM on {} chars, vocab {}", TEXT.len(), chars.len());
    let mut opt = AdamW::new(model.parameters(), 0.01).with_weight_decay(0.0);
    let mut loss_val = f32::NAN;
    for step in 0..=800 {
        let loss = cross_entropy_indices(&model.forward(&input).unwrap(), &targets).unwrap();
        loss_val = loss.item();
        if step % 100 == 0 {
            println!("step {step:4}  loss {loss_val:.4}");
        }
        if loss_val < 0.02 {
            println!("step {step:4}  loss {loss_val:.4}  (early stop)");
            break;
        }
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    let mut path = std::env::temp_dir();
    path.push(format!("ferro_char_lm_{}.safetensors", std::process::id()));
    save_module(&path, &model).unwrap();
    let fresh = CharLm::new(chars.len(), 999);
    load_module(&path, &fresh).unwrap();
    std::fs::remove_file(&path).unwrap();
    println!("saved {} tensors and reloaded them into a fresh model", model.named_parameters().len());

    // Greedy-generate the text back from its first character, using the
    // RELOADED model so a save/load defect cannot hide.
    let mut ctx = vec![ids[0]];
    while ctx.len() < ids.len() {
        let logits = fresh.forward(&Tensor::from_vec_i64(ctx.clone(), &[1, ctx.len()]).unwrap()).unwrap();
        let next = logits.argmax(1, false).unwrap().to_vec_i64()[ctx.len() - 1];
        ctx.push(next);
    }
    let generated: String = ctx.iter().map(|&i| chars[i as usize]).collect();
    println!("seed      : {:?}", &TEXT[..1]);
    println!("generated : {generated:?}");
    let correct = generated.chars().zip(TEXT.chars()).filter(|(a, b)| a == b).count();
    println!("match     : {correct}/{} chars (final loss {loss_val:.4})", TEXT.len());
    assert_eq!(generated, TEXT, "reloaded model failed to reproduce its training text");
    println!("REGENERATED THE TRAINING TEXT FROM A RELOADED STATE DICT");
}
