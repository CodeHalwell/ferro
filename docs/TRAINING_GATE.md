# Training gate: end-to-end proof programs

Two runnable training programs prove the ferro-core stack end to end on CPU,
using only the existing public API (nn/modules layers, ops_ext autograd ops,
optim::AdamW, data::DataLoader, checkpoint::Checkpoint). Neither program
contains engine code; everything they need the engine must already provide.

## What each program proves

### `crates/ferro-core/examples/train_gpt2_small.rs`

A GPT-2-style character-level decoder: token embedding + learned positional
embedding, 2 pre-norm blocks of causal multi-head self-attention (masked
softmax over q @ k^T / sqrt(d) via `nn::scaled_dot_product_attention`) plus a
Gelu MLP, final LayerNorm and an LM head. Trained with AdamW (weight decay,
global-norm grad clipping) over shifted windows of an embedded pangram corpus.

Proves: Embedding lookup gradients, 3-D reshape/transpose plumbing, causal
attention forward+backward through bmm/softmax/mask-add, cross-entropy on
flattened [batch*seq] targets, DataLoader batching/shuffling, checkpoint save,
and resume continuation.

Observed run (seed 7, CPU): loss falls from ln(vocab) ~3.33 to 0.64 by step
300, and keeps falling after resume (0.95 -> 0.56 between steps 200-320 in a
separate process). Greedy samples become pangram-like.

### `crates/ferro-core/examples/train_classifier_cnn.rs`

A conv/pool/norm classifier on synthetic separable classes: two classes of
1x8x8 images distinguished by which corner holds a bright 3x3 block under
N(0,1) noise. Two bias-free convolutions (im2col conv2d, Kaiming init) with
ReLU + 2x2 max pooling, BatchNorm, then a two-layer MLP head, trained with
AdamW through the DataLoader.

Proves: Conv2D forward/backward, max_pool2d gradients, BatchNorm train-mode
stats + gradients, argmax-based accuracy, and that the whole chain reaches
perfect accuracy on held-out data.

Observed run (seed 11, CPU): loss 0.0108 at step 20, final eval accuracy
100% (the program asserts > 95%).

## Commands to reproduce

```
cargo run -p ferro-core --example train_gpt2_small
cargo run -p ferro-core --example train_classifier_cnn

# options
cargo run -p ferro-core --example train_gpt2_small -- --steps 400 --ckpt-every 100 --out target/gpt2_ckpt
cargo run -p ferro-core --example train_gpt2_small -- --resume target/gpt2_ckpt   # continue from saved step
cargo run -p ferro-core --example train_classifier_cnn -- --steps 300 --out target/cnn_ckpt
```

Both are deterministic given their fixed seeds; loss trajectories above should
reproduce exactly on any CPU.

## Known core limitation surfaced by these programs

`modules::Conv2D` adds its `[c_out]` bias directly against NCHW output.
Broadcasting aligns trailing dims, so this only resolves when `c_out == image
width`; e.g. `[16, 4, 8, 8] + [4]` fails with ShapeMismatch while
`[2, 4, 4, 4] + [4]` happens to work. The CNN example therefore uses its own
bias-free conv wrapper around `Tensor::conv2d` (BatchNorm supplies the shift).
Fix belongs in core: either broadcast the bias as [1, c, 1, 1] or make add
handle full right-aligned broadcasting.

Resume semantics note: `Checkpoint` restores model parameters and the global
step; optimizer moment buffers are not restored because `Sgd`/`AdamW` keep
them private (see the scope note in checkpoint.rs). Resumed runs therefore
warm-restart Adam moments; the loss trajectory still continues smoothly.

## Checklist toward the >=50% torch-throughput gate

Done:
- [x] Autograd-correct modules compose into a real architecture (transformer,
      convnet) and train end to end on CPU.
- [x] Loss decreases across runs; checkpoints round-trip and resume works.
- [x] Both programs compile warning-free and run green on CPU.

Remaining once `benchmarks/` lands:
- [ ] Benchmark harness measuring tokens/sec for the transformer example and
      images/sec for the CNN, with a torch baseline on identical workload
      shapes (same batch, seq, dims, dtype f32).
- [ ] ferro-fastcpu wired as the default backend for these workloads and the
      gap measured per op class (matmul, softmax, layernorm, conv).
- [ ] Profile top offenders; candidate fixes: fused ops (fused_ops.rs),
      strided fastpaths instead of materializing transposes, threaded GEMM.
- [ ] Gate check: transformer tokens/sec >= 50% of torch on the same machine;
      record numbers and hardware in benchmarks/README.
- [ ] Optional GPU parity: same programs run unchanged on the cuda backend via
      Device selection, throughput recorded.
