"""PyTorch twin of examples/bench_transformer.rs: identical transformer block
(embed -> causal MHA -> Gelu MLP -> norms, LM head) trained with AdamW on the
same synthetic data, same steps. Prints the same metrics so numbers can go
side by side in benchmarks/README.md.

Run:
    benchmarks/.venv/Scripts/python.exe examples/bench_torch.py [same CLI args]
"""

import argparse
import time

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d_model, heads, vocab):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True, bias=False)
        self.norm2 = nn.RMSNorm(d_model)
        self.up = nn.Linear(d_model, 4 * d_model)
        self.down = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        s = x.shape[1]
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), 1)
        h = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask, need_weights=False)[0]
        mlp = self.down(nn.functional.gelu(self.up(self.norm2(h))))
        return h + mlp


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p / 100), len(xs) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"cuda unavailable ({torch.cuda.get_device_name(0) if False else 'no driver'}); falling back to cpu")
        device = "cpu"
    print(f"device: {device}")

    torch.manual_seed(42)
    emb = nn.Embedding(args.vocab, args.d_model).to(device)
    block = Block(args.d_model, args.heads, args.vocab).to(device)
    head = nn.Linear(args.d_model, args.vocab).to(device)
    params = list(emb.parameters()) + list(block.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    n_params = sum(p.numel() for p in params)

    n_tokens = args.batch * args.seq
    ids = torch.tensor([(i * 2654435761 % args.vocab) for i in range(n_tokens)],
                       dtype=torch.long, device=device).reshape(args.batch, args.seq)
    targets = torch.roll(ids.flatten(), -1)

    def step():
        logits = head(block(emb(ids)).reshape(n_tokens, args.d_model))
        return nn.functional.cross_entropy(logits, targets)

    for _ in range(args.warmup):
        opt.zero_grad(); step().backward(); opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    step_ms = []
    for _ in range(args.steps):
        t = time.perf_counter()
        opt.zero_grad(); step().backward(); opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t) * 1000)

    total_s = sum(step_ms) / 1000
    print(f"config: batch={args.batch} seq={args.seq} d_model={args.d_model} "
          f"heads={args.heads} vocab={args.vocab} params={n_params}")
    print(f"steps: warmup={args.warmup} timed={args.steps} total_time={total_s:.3f}s")
    print(f"throughput: {n_tokens * args.steps / total_s:.0f} tokens/sec")
    print(f"step time ms: mean={total_s * 1000 / len(step_ms):.2f} "
          f"p50={percentile(step_ms, 50):.2f} p90={percentile(step_ms, 90):.2f} "
          f"p99={percentile(step_ms, 99):.2f}")


if __name__ == "__main__":
    main()
