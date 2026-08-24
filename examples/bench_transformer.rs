//! Tier-1 throughput benchmark: a small transformer block (embed -> causal
//! MHA -> Gelu MLP -> norms) trained with AdamW on synthetic data.
//!
//! The block is assembled from public modules (RmsNorm, MultiHeadAttention,
//! Linear) rather than TransformerBlock so that per-stage wall-clock timing is
//! possible without private access; the composition matches
//! TransformerBlock::forward exactly (pre-norm, RoPE 10000, 4x MLP width).
//!
//! Run (from repo root):
//!   cargo run --release -p ferro-fastcpu --example bench_transformer
//! or compile standalone against the release rlibs:
//!   rustc --edition 2021 -C opt-level=3 examples/bench_transformer.rs \
//!     --extern ferro_core=target/release/libferro_core.rlib \
//!     --extern ferro_fastcpu=target/release/libferro_fastcpu.rlib
//!
//! CLI: --batch 8 --seq 128 --d-model 256 --heads 4 --vocab 1024
//!      --warmup 100 --steps 500 --device cpu|cuda [--profile]

use ferro_core::nn::{cross_entropy_indices, Embedding, Linear, Module, MultiHeadAttention, RmsNorm};
use ferro_core::optim::AdamW;
use ferro_core::{Device, Rng, Tensor};

struct Args {
    batch: usize,
    seq: usize,
    d_model: usize,
    heads: usize,
    vocab: usize,
    warmup: usize,
    steps: usize,
    device: Device,
    profile: bool,
}

fn parse_device(s: &str) -> Device {
    match s.trim() {
        "cuda" => Device::Cuda(0),
        _ => Device::Cpu,
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        batch: 8,
        seq: 128,
        d_model: 256,
        heads: 4,
        vocab: 1024,
        warmup: 100,
        steps: 500,
        device: Device::Cpu,
        profile: false,
    };
    let mut it = std::env::args().skip(1);
    fn next(it: &mut impl Iterator<Item = String>) -> String {
        it.next().unwrap_or_default()
    }
    while let Some(k) = it.next() {
        match k.as_str() {
            "--batch" => a.batch = next(&mut it).parse().unwrap_or(a.batch),
            "--seq" => a.seq = next(&mut it).parse().unwrap_or(a.seq),
            "--d-model" => a.d_model = next(&mut it).parse().unwrap_or(a.d_model),
            "--heads" => a.heads = next(&mut it).parse().unwrap_or(a.heads),
            "--vocab" => a.vocab = next(&mut it).parse().unwrap_or(a.vocab),
            "--warmup" => a.warmup = next(&mut it).parse().unwrap_or(a.warmup),
            "--steps" => a.steps = next(&mut it).parse().unwrap_or(a.steps),
            "--device" => a.device = parse_device(&next(&mut it)),
            "--profile" => a.profile = true,
            other => eprintln!("unknown arg {other}"),
        }
    }
    a
}

fn percentile(xs: &[f64], p: f64) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (((sorted.len() as f64) * p / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

struct Block {
    norm1: RmsNorm,
    attn: MultiHeadAttention,
    norm2: RmsNorm,
    up: Linear,
    down: Linear,
}

impl Block {
    fn new(d_model: usize, heads: usize, rng: &Rng) -> Result<Block, ferro_core::Error> {
        Ok(Block {
            norm1: RmsNorm::new(d_model),
            attn: MultiHeadAttention::new(d_model, heads, true, rng)?.with_rope(10000.0),
            norm2: RmsNorm::new(d_model),
            up: Linear::new(d_model, 4 * d_model, rng),
            down: Linear::new(4 * d_model, d_model, rng),
        })
    }

    fn parameters(&self) -> impl Iterator<Item = ferro_core::Param> {
        [
            self.norm1.named_parameters(),
            self.attn.named_parameters(),
            self.norm2.named_parameters(),
            self.up.named_parameters(),
            self.down.named_parameters(),
        ]
        .into_iter()
        .flatten()
        .map(|(_, p)| p)
    }
}

const STAGES: [&str; 5] = ["embed_fwd", "attn_fwd", "mlp_fwd", "loss_bwd", "optim"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Route core matmuls through ferro-fastcpu's blocked AVX2 kernel.
    ferro_fastcpu::install();
    let args = parse_args();

    // CUDA is opt-in and may fail if the driver/backend is unavailable; fall
    // back to CPU so the harness always produces numbers.
    let device = if matches!(args.device, Device::Cuda(_)) {
        #[cfg(feature = "cuda")]
        {
            match ferro_cuda::install(0) {
                Ok(()) => Device::Cuda(0),
                Err(e) => {
                    eprintln!("cuda install failed ({e}); falling back to cpu");
                    Device::Cpu
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("cuda feature not compiled; falling back to cpu");
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    // Probe: some backends lack I64 transfers yet; fall back to cpu rather
    // than panic so the harness always produces numbers.
    let device = match Tensor::from_vec_i64(vec![0, 1], &[2]).unwrap().to_device(device) {
        Ok(_) => device,
        Err(e) => {
            eprintln!("i64 tensors not supported on {device} ({e}); falling back to cpu");
            Device::Cpu
        }
    };
    println!("device: {device}");

    let rng = Rng::new(42);
    let emb = Embedding::new(args.vocab, args.d_model, &rng);
    let block = Block::new(args.d_model, args.heads, &rng)?;
    let head = Linear::new(args.d_model, args.vocab, &rng);

    let mut params = emb.parameters();
    params.extend(block.parameters());
    params.extend(head.parameters());
    // Move every parameter to the target device before the optimizer takes
    // ownership; ops then stay device-resident end to end.
    for p in &mut params {
        let t = p.tensor().to_device(device)?;
        p.set(t);
    }
    let n_params: usize = params.iter().map(|p| p.tensor().numel()).sum();
    let mut opt = AdamW::new(params, 1e-4);

    // Synthetic data: random token ids + shifted targets, fixed for all steps.
    let n_tokens = args.batch * args.seq;
    let ids_data: Vec<i64> =
        (0..n_tokens).map(|i| (i * 2654435761 % args.vocab) as i64).collect();
    let tgt_data: Vec<i64> = ids_data.iter().skip(1).copied().chain(std::iter::once(ids_data[0])).collect();
    let ids = Tensor::from_vec_i64(ids_data, &[args.batch, args.seq]).unwrap().to_device(device)?;
    let targets = Tensor::from_vec_i64(tgt_data, &[n_tokens]).unwrap().to_device(device)?;

    let tokens_per_step = n_tokens as f64;

    // One full training step, timed as a unit for the throughput number.
    let run_step = |opt: &mut AdamW| -> Result<(), ferro_core::Error> {
        let h = block.attn_fwd(&block.norm1.forward(&emb.forward(&ids)?)?)?;
        let h = block.mlp_fwd(&h)?;
        let logits = head.forward(&h.reshape(&[n_tokens, args.d_model])?)?;
        // No loss.item(): no host sync in the timed region (matches the torch
        // loop); backward already consumes the graph.
        let loss = cross_entropy_indices(&logits, &targets)?;
        opt.zero_grad();
        loss.backward();
        opt.step();
        Ok(())
    };

    for _ in 0..args.warmup {
        run_step(&mut opt)?;
    }
    let mut step_ms = Vec::with_capacity(args.steps);
    for _ in 0..args.steps {
        let t = std::time::Instant::now();
        run_step(&mut opt)?;
        step_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let total_s: f64 = step_ms.iter().sum::<f64>() / 1000.0;
    let tps = tokens_per_step * args.steps as f64 / total_s;
    let mean = total_s / args.steps as f64;

    println!(
        "config: batch={} seq={} d_model={} heads={} vocab={} params={}",
        args.batch, args.seq, args.d_model, args.heads, args.vocab, n_params
    );
    println!(
        "steps: warmup={} timed={} total_time={:.3}s",
        args.warmup, args.steps, total_s
    );
    println!("throughput: {:.0} tokens/sec", tps);
    println!(
        "step time ms: mean={:.2} p50={:.2} p90={:.2} p99={:.2}",
        mean * 1000.0,
        percentile(&step_ms, 50.0),
        percentile(&step_ms, 90.0),
        percentile(&step_ms, 99.0)
    );

    if args.profile {
        // Per-stage wall clock over the same step, N iterations. Stages sum to
        // slightly more than the whole-step time (timer overhead per stage).
        let iters = args.steps;
        for _ in 0..args.warmup.min(50) {
            prof_step(&emb, &block, &head, &ids, &targets, &mut opt, &mut [0f64; 5])?;
        }
        let mut acc = vec![0f64; STAGES.len()];
        for _ in 0..iters {
            prof_step(&emb, &block, &head, &ids, &targets, &mut opt, &mut acc)?;
        }
        println!(
            "per-stage mean ms over {iters} iters ({}):",
            STAGES.join("+")
        );
        for (i, name) in STAGES.iter().enumerate() {
            let frac = if mean > 0.0 { acc[i] / iters as f64 / (mean * 1000.0) * 100.0 } else { 0.0 };
            println!("  {name}: {:.2} ms ({:.1}% of mean step)", acc[i] / iters as f64, frac);
        }
    }
    Ok(())
}

impl Block {
    fn attn_fwd(&self, x: &Tensor) -> Result<Tensor, ferro_core::Error> {
        x.add(&self.attn.forward(&self.norm1.forward(x)?)?)
    }
    fn mlp_fwd(&self, x: &Tensor) -> Result<Tensor, ferro_core::Error> {
        let shape = x.shape().to_vec();
        let d = shape[2];
        let normed = self.norm2.forward(x)?;
        let flatn = normed.reshape(&[shape[0] * shape[1], d])?;
        let out = self.down.forward(&self.up.forward(&flatn)?.gelu())?;
        out.reshape(&shape)?.add(x)
    }
}

// Stage indices: 0 embed fwd, 1 attn fwd, 2 mlp fwd, 3 loss+bwd, 4 optim.
// acc is indexed by stage and accumulates elapsed ms.
#[allow(clippy::too_many_arguments)]
fn prof_step(
    emb: &Embedding,
    block: &Block,
    head: &Linear,
    ids: &Tensor,
    targets: &Tensor,
    opt: &mut AdamW,
    acc: &mut [f64],
) -> Result<(), ferro_core::Error> {
    let t = std::time::Instant::now();
    let e = emb.forward(ids)?;
    acc[0] += t.elapsed().as_secs_f64() * 1000.0;

    let t = std::time::Instant::now();
    let h = block.attn_fwd(&e)?;
    acc[1] += t.elapsed().as_secs_f64() * 1000.0;

    let t = std::time::Instant::now();
    let h = block.mlp_fwd(&h)?;
    acc[2] += t.elapsed().as_secs_f64() * 1000.0;

    let d = h.shape()[2];
    let t = std::time::Instant::now();
    let logits = head.forward(&h.reshape(&[h.shape()[0] * h.shape()[1], d])?)?;
    let loss = cross_entropy_indices(&logits, targets)?;
    opt.zero_grad();
    loss.backward();
    acc[3] += t.elapsed().as_secs_f64() * 1000.0;

    let t = std::time::Instant::now();
    opt.step();
    acc[4] += t.elapsed().as_secs_f64() * 1000.0;
    Ok(())
}
