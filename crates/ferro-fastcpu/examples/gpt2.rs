//! GPT-2 (124M) inference, end to end: load OpenAI's real safetensors weights
//! and the real GPT-2 byte-level BPE vocab, then greedy-generate a continuation
//! of a prompt. This is milestone M3 - a real published LLM running on ferro's
//! own engine, no torch in the loop.
//!
//! The model is the canonical GPT-2: token + learned positional embeddings, a
//! stack of pre-LayerNorm transformer blocks (fused QKV `c_attn`, tanh-approx
//! GELU MLP), a final LayerNorm, and a tied LM head (logits = h @ wte^T). The
//! HF Conv1D weights are stored `[in, out]`, which is exactly ferro's `Linear`
//! layout, so weights load with no transposes.
//!
//! Weights are NOT in the repo. Point FERRO_GPT2_DIR at a directory holding
//! `model.safetensors`, `vocab.json`, and `merges.txt` from
//! https://huggingface.co/openai-community/gpt2 :
//!
//!   FERRO_GPT2_DIR=/path/to/gpt2 \
//!     cargo run --release -p ferro-fastcpu --example gpt2 -- "The meaning of life is"
//!
//! Everything runs on the optimized CPU backend at batch 1.

use ferro_core::error::Result;
use ferro_core::nn::scaled_dot_product_attention;
use ferro_core::Tensor;
use ferro_tokenizer::Bpe;
use std::collections::HashMap;
use std::path::PathBuf;

const N_LAYER: usize = 12;
const N_HEAD: usize = 12;
const DIM: usize = 768;
const HEAD_DIM: usize = DIM / N_HEAD;
const EPS: f32 = 1e-5;
const MAX_POS: usize = 1024;

/// The state dict, addressed by HF tensor name. Every weight the forward pass
/// needs is looked up here by its checkpoint name, so the mapping IS the code.
struct Weights(HashMap<String, Tensor>);

impl Weights {
    fn get(&self, name: &str) -> &Tensor {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("checkpoint is missing tensor {name:?}"))
    }
}

/// LayerNorm over the last dim of a 2-D `[seq, dim]` tensor with affine
/// weight/bias, matching torch's `F.layer_norm`.
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let mu = x.mean_dim(1, true)?;
    let centered = x.sub(&mu)?;
    let var = centered.mul(&centered)?.mean_dim(1, true)?;
    let eps = Tensor::scalar(EPS);
    let norm = centered.div(&var.add(&eps)?.sqrt())?;
    norm.mul(w)?.add(b)
}

/// `y = x @ w + b` for a `[seq, in]` input and HF-layout `[in, out]` weight.
fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.matmul(w)?.add(b)
}

/// Split heads: `[seq, dim] -> [n_head, seq, head_dim]` so each head is an
/// independent batch element for scaled_dot_product_attention.
fn split_heads(x: &Tensor, seq: usize) -> Result<Tensor> {
    x.reshape(&[seq, N_HEAD, HEAD_DIM])?.transpose(0, 1)
}

/// One pre-norm GPT-2 block on a `[seq, dim]` residual stream.
fn block(w: &Weights, i: usize, x: &Tensor, seq: usize) -> Result<Tensor> {
    let p = |s: &str| format!("h.{i}.{s}");

    // Attention sublayer: ln_1 -> fused QKV -> per-head SDPA -> c_proj.
    let a = layer_norm(x, w.get(&p("ln_1.weight")), w.get(&p("ln_1.bias")))?;
    let qkv = linear(
        &a,
        w.get(&p("attn.c_attn.weight")),
        w.get(&p("attn.c_attn.bias")),
    )?; // [seq, 3*dim]
    let q = qkv.index_select(1, &(0..DIM).collect::<Vec<_>>())?;
    let k = qkv.index_select(1, &(DIM..2 * DIM).collect::<Vec<_>>())?;
    let v = qkv.index_select(1, &(2 * DIM..3 * DIM).collect::<Vec<_>>())?;
    let q = split_heads(&q, seq)?;
    let k = split_heads(&k, seq)?;
    let v = split_heads(&v, seq)?;
    let attn = scaled_dot_product_attention(&q, &k, &v, true)?; // [n_head, seq, head_dim]
    let merged = attn.transpose(0, 1)?.reshape(&[seq, DIM])?;
    let attn_out = linear(
        &merged,
        w.get(&p("attn.c_proj.weight")),
        w.get(&p("attn.c_proj.bias")),
    )?;
    let x = x.add(&attn_out)?;

    // MLP sublayer: ln_2 -> c_fc -> gelu(tanh) -> c_proj.
    let m = layer_norm(&x, w.get(&p("ln_2.weight")), w.get(&p("ln_2.bias")))?;
    let h = linear(&m, w.get(&p("mlp.c_fc.weight")), w.get(&p("mlp.c_fc.bias")))?;
    let h = h.gelu();
    let h = linear(
        &h,
        w.get(&p("mlp.c_proj.weight")),
        w.get(&p("mlp.c_proj.bias")),
    )?;
    x.add(&h)
}

/// Forward the whole model over a token id sequence, returning `[seq, vocab]`
/// logits. Runs at batch 1 in `[seq, dim]` (ferro's LayerNorm is 2-D).
fn forward(w: &Weights, ids: &[i64]) -> Result<Tensor> {
    let seq = ids.len();
    assert!(seq <= MAX_POS, "sequence length {seq} exceeds {MAX_POS}");

    let wte = w.get("wte.weight"); // [vocab, dim]
    let wpe = w.get("wpe.weight"); // [max_pos, dim]
    let tok_idx: Vec<usize> = ids.iter().map(|&t| t as usize).collect();
    let pos_idx: Vec<usize> = (0..seq).collect();
    let tok = wte.index_select(0, &tok_idx)?; // [seq, dim]
    let pos = wpe.index_select(0, &pos_idx)?; // [seq, dim]
    let mut x = tok.add(&pos)?;

    for i in 0..N_LAYER {
        x = block(w, i, &x, seq)?;
    }
    let x = layer_norm(&x, w.get("ln_f.weight"), w.get("ln_f.bias"))?;

    // Tied LM head: logits = x @ wte^T. wte is [vocab, dim] -> [dim, vocab].
    x.matmul(&wte.transpose(0, 1)?)
}

fn main() {
    ferro_fastcpu::install();

    let dir = PathBuf::from(std::env::var("FERRO_GPT2_DIR").expect(
        "set FERRO_GPT2_DIR to a directory holding model.safetensors, vocab.json, merges.txt \
         (from https://huggingface.co/openai-community/gpt2)",
    ));
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The meaning of life is".to_string());
    let max_new: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let tok = Bpe::from_files(dir.join("vocab.json"), dir.join("merges.txt"))
        .expect("failed to load GPT-2 tokenizer");
    println!(
        "loaded GPT-2 tokenizer: {} tokens in vocab",
        tok.vocab_size()
    );

    let raw = ferro_core::safetensors::load_safetensors(dir.join("model.safetensors"))
        .expect("failed to load model.safetensors");
    let n = raw.len();
    let weights = Weights(raw.into_iter().collect());
    println!("loaded {n} weight tensors from model.safetensors");

    let mut ids: Vec<i64> = tok
        .encode(&prompt)
        .expect("encode failed")
        .into_iter()
        .map(|t| t as i64)
        .collect();
    let prompt_len = ids.len();
    println!("\nprompt: {prompt:?}  ({prompt_len} tokens)");
    print!("output: {prompt}");
    use std::io::Write;
    std::io::stdout().flush().ok();

    for _ in 0..max_new {
        let logits = forward(&weights, &ids).expect("forward failed");
        let seq = ids.len();
        // Greedy: argmax over the last position's row.
        let next = logits.argmax(1, false).expect("argmax failed").to_vec_i64()[seq - 1];
        // Optional numeric cross-check hook: on the first step, dump the last
        // row of logits so an external oracle (HF transformers) can diff them.
        if std::env::var("FERRO_DUMP_LOGITS").is_ok() && ids.len() == prompt_len {
            let vocab = logits.shape()[1];
            let row: Vec<f32> = logits.to_vec()[(seq - 1) * vocab..seq * vocab].to_vec();
            let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write("ferro_logits.bin", &bytes).expect("dump failed");
            eprintln!("dumped {} logits to ferro_logits.bin", row.len());
        }
        ids.push(next);
        let piece = tok.decode(&[next as u32]).unwrap_or_default();
        print!("{piece}");
        std::io::stdout().flush().ok();
        if ids.len() >= MAX_POS {
            break;
        }
    }
    println!(
        "\n\ngenerated {} tokens greedily on ferro's CPU backend",
        ids.len() - prompt_len
    );
}
