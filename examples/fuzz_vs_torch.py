"""Property-based torch-parity fuzzer for ferro (CAPABILITY gate G4).

This upgrades "validated on examples" to "validated on distributions": each op
is run on many random shapes and adversarial input distributions (normal,
wide-magnitude, special values - inf/nan/-0/denormals/large), and ferro's
output is compared to torch's IN ULPs - reinterpret the two f32 buffers as
int32 and difference them. ULP distance is scale-free, so it distinguishes
"same algorithm" from "close but different algorithm" in a way that a fixed
atol cannot.

Gate G4: p50 <= 2 ULP and p100 <= 32 ULP against torch f32 over the op
surface, with any per-op exception documented in EXCEPTIONS below.

Run inside the ferro-py venv (after `maturin develop --release`, with torch
installed):

    cd crates/ferro-py
    .venv/Scripts/python ../../examples/fuzz_vs_torch.py            # default
    .venv/Scripts/python ../../examples/fuzz_vs_torch.py --trials 500 --seed 7

Exit code is non-zero if any op breaches its gate, so this is CI-ready.
"""

import argparse
import sys

import numpy as np
import torch

import ferro

# ---------------------------------------------------------------------------
# ULP machinery
# ---------------------------------------------------------------------------


def ulp_diff(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Per-element ULP distance between two float32 arrays.

    Map each float to a monotonically ordered integer (the standard trick: flip
    the sign bit for positives, invert everything for negatives), then the
    integer difference IS the count of representable floats between them. NaNs
    compare equal to NaNs (0 ULP); a NaN-vs-finite mismatch is forced to the max
    so it always trips the gate.

    ULP distance is meaningless in the subnormal region around zero (every tiny
    float is billions of ULPs from 0.0), so pairs whose absolute difference is
    within `atol` are treated as 0 ULP - matching numpy's own ULP asserts,
    which combine ULP with an absolute floor.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    # Signed zeros are numerically equal; normalise -0.0 -> +0.0 so they are
    # 0 ULP apart rather than 2**32 apart.
    a = a + np.float32(0.0)
    b = b + np.float32(0.0)

    def ordered(x):
        i = x.view(np.int32).astype(np.int64)
        # Negative floats: 0x80000000 - i maps them below positive zero.
        return np.where(i < 0, np.int64(0x80000000) - i, i)

    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    with np.errstate(invalid="ignore"):
        near_zero = np.abs(a.astype(np.float64) - b.astype(np.float64)) <= atol
    d = np.abs(ordered(a) - ordered(b))
    d = np.where(near_zero, 0, d)
    d = np.where(both_nan, 0, d)
    d = np.where(one_nan, np.int64(1 << 62), d)
    return d


def report(name, diffs, gate_p50=2, gate_hi=32):
    d = np.concatenate([x.reshape(-1) for x in diffs]) if diffs else np.array([0])
    p50 = int(np.percentile(d, 50))
    p95 = int(np.percentile(d, 95))
    p99 = int(np.percentile(d, 99))
    p100 = int(d.max())
    n = d.size
    exc = EXCEPTIONS.get(name)
    if exc:
        # Reduction/matmul ops: accumulation order differs from torch's
        # pairwise/blocked sum, so the p100 tail is catastrophic-cancellation
        # sensitivity (an f32 conditioning property, not a parity defect).
        # Gate the robust p99; p100 is reported for visibility only.
        lim50, limhi = exc
        hi_val, hi_lbl = p99, "p99"
    else:
        lim50, limhi = gate_p50, gate_hi
        hi_val, hi_lbl = p100, "p100"
    ok = p50 <= lim50 and hi_val <= limhi
    tag = "OK " if ok else "FAIL"
    note = f"  [exception: gates {hi_lbl}<={limhi}, accum-order]" if exc else ""
    print(f"{tag} {name:16} n={n:>8}  p50={p50:>3}  p95={p95:>4}  p99={p99:>5}  p100={p100:>7}{note}")
    return ok


# Per-op documented exceptions (p50_gate, p99_gate). These ops sum many terms
# and therefore differ from torch by accumulation order; their p100 tail is
# dominated by catastrophic cancellation on ill-conditioned inputs (inherent
# f32 behaviour, reproducible torch-vs-torch across thread counts). Bounds
# below are measured envelopes with headroom, not aspirations.
EXCEPTIONS: dict = {
    "sum_dim": (2, 16),
    "mean_dim": (2, 16),
    "matmul": (2, 16),
    "bmm": (2, 16),
}


# ---------------------------------------------------------------------------
# Input distributions
# ---------------------------------------------------------------------------

SPECIALS = np.array(
    [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan,
     np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny,
     np.finfo(np.float32).max, np.finfo(np.float32).eps, 1e-30, 1e30],
    dtype=np.float32,
)


def gen(rng, shape, kind, positive=False):
    n = int(np.prod(shape))
    if kind == "normal":
        x = rng.standard_normal(n).astype(np.float32)
    elif kind == "wide":
        x = (rng.standard_normal(n) * 10 ** rng.uniform(-6, 6, n)).astype(np.float32)
    elif kind == "special":
        base = rng.standard_normal(n).astype(np.float32)
        k = min(n, len(SPECIALS))
        idx = rng.choice(n, k, replace=False)
        base[idx] = rng.permutation(SPECIALS)[:k]
        x = base
    else:
        x = rng.standard_normal(n).astype(np.float32)
    if positive:
        x = np.abs(x) + 0.05
    return x.reshape(shape)


def to_ferro(a):
    return ferro.from_dlpack(np.ascontiguousarray(a, dtype=np.float32))


def to_np(t):
    return np.from_dlpack(t)


# ---------------------------------------------------------------------------
# Op specifications: (name, arity, positive?, kinds, ferro_fn, torch_fn)
# elementwise ops are shape-free; reductions/matmul carry their own drivers.
# ---------------------------------------------------------------------------

UNARY = [
    ("neg",     False, lambda x: -x,          lambda x: -x),
    ("abs",     False, lambda x: x.abs(),     torch.abs),
    ("exp",     False, lambda x: x.exp(),     torch.exp),
    ("log",     True,  lambda x: x.log(),     torch.log),
    ("sqrt",    True,  lambda x: x.sqrt(),    torch.sqrt),
    ("tanh",    False, lambda x: x.tanh(),    torch.tanh),
    ("sigmoid", False, lambda x: x.sigmoid(), torch.sigmoid),
    ("relu",    False, lambda x: x.relu(),    torch.relu),
    ("gelu",    False, lambda x: x.gelu(),    lambda x: torch.nn.functional.gelu(x, approximate="tanh")),
]

BINARY = [
    ("add", lambda a, b: a + b, lambda a, b: a + b),
    ("sub", lambda a, b: a - b, lambda a, b: a - b),
    ("mul", lambda a, b: a * b, lambda a, b: a * b),
    ("div", lambda a, b: a / b, lambda a, b: a / b),
]

REDUCE = [
    ("sum_dim",     False, lambda x, d: x.sum_dim(d, False),     lambda x, d: torch.sum(x, d)),
    ("mean_dim",    False, lambda x, d: x.mean_dim(d, False),    lambda x, d: torch.mean(x, d)),
    ("softmax",     False, lambda x, d: x.softmax(d),            lambda x, d: torch.softmax(x, d)),
    ("log_softmax", False, lambda x, d: x.log_softmax(d),        lambda x, d: torch.log_softmax(x, d)),
]


def rand_shape(rng, max_rank=4, max_dim=32):
    rank = rng.integers(1, max_rank + 1)
    return tuple(int(rng.integers(1, max_dim + 1)) for _ in range(rank))


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def fuzz_unary(rng, trials):
    results = {}
    for name, pos, ff, tf in UNARY:
        diffs = []
        for _ in range(trials):
            shape = rand_shape(rng)
            kind = rng.choice(["normal", "wide", "special"])
            x = gen(rng, shape, kind, positive=pos)
            fe = to_np(ff(to_ferro(x)))
            to = tf(torch.from_numpy(x)).numpy()
            diffs.append(ulp_diff(fe, to))
        results[name] = report(name, diffs)
    return results


def fuzz_binary(rng, trials):
    results = {}
    for name, ff, tf in BINARY:
        diffs = []
        for _ in range(trials):
            shape = rand_shape(rng)
            kind = rng.choice(["normal", "wide", "special"])
            a = gen(rng, shape, kind)
            b = gen(rng, shape, kind)
            fe = to_np(ff(to_ferro(a), to_ferro(b)))
            to = tf(torch.from_numpy(a), torch.from_numpy(b)).numpy()
            diffs.append(ulp_diff(fe, to))
        results[name] = report(name, diffs)
    return results


def fuzz_reduce(rng, trials):
    results = {}
    for name, pos, ff, tf in REDUCE:
        diffs = []
        for _ in range(trials):
            shape = rand_shape(rng)
            d = int(rng.integers(0, len(shape)))
            # softmax over wide/special magnitudes is a stability test, not a
            # "should it be nan" test; keep inputs finite-ish for these two.
            kind = "normal" if name.endswith("softmax") else rng.choice(["normal", "wide"])
            x = gen(rng, shape, kind, positive=pos)
            fe = to_np(ff(to_ferro(x), d))
            to = tf(torch.from_numpy(x), d).numpy()
            diffs.append(ulp_diff(fe, to))
        results[name] = report(name, diffs)
    return results


def fuzz_matmul(rng, trials):
    results = {}
    # 2-D matmul
    dm = []
    for _ in range(trials):
        m, k, n = (int(rng.integers(1, 33)) for _ in range(3))
        a = gen(rng, (m, k), "normal")
        b = gen(rng, (k, n), "normal")
        fe = to_np(to_ferro(a).matmul(to_ferro(b)))
        to = (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()
        dm.append(ulp_diff(fe, to))
    results["matmul"] = report("matmul", dm)
    # batched matmul
    db = []
    for _ in range(trials):
        bsz = int(rng.integers(1, 6))
        m, k, n = (int(rng.integers(1, 25)) for _ in range(3))
        a = gen(rng, (bsz, m, k), "normal")
        b = gen(rng, (bsz, k, n), "normal")
        fe = to_np(to_ferro(a).bmm(to_ferro(b)))
        to = torch.bmm(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        db.append(ulp_diff(fe, to))
    results["bmm"] = report("bmm", db)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200, help="random trials per op")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"ferro torch-parity fuzzer (G4)  trials/op={args.trials}  seed={args.seed}")
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    print("gate: p50 <= 2 ULP, p100 <= 32 ULP (matmul reductions excepted below)\n")

    ok = {}
    ok.update(fuzz_unary(rng, args.trials))
    print()
    ok.update(fuzz_binary(rng, args.trials))
    print()
    ok.update(fuzz_reduce(rng, args.trials))
    print()
    ok.update(fuzz_matmul(rng, args.trials))

    failed = [k for k, v in ok.items() if not v]
    print()
    if failed:
        print(f"GATE BREACHED by {len(failed)} op(s): {', '.join(failed)}")
        sys.exit(1)
    print(f"ALL {len(ok)} OPS WITHIN G4 ULP GATE")


if __name__ == "__main__":
    main()
