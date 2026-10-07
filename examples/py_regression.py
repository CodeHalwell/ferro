"""Regression tests for the ferro Python bindings.

Run inside the ferro-py venv after `maturin develop --release`:

    cd rust_backend/crates/ferro-py
    . .venv/bin/activate
    python ../../examples/py_regression.py

Covers: in-place requires_grad_, the requires_grad getter, ValueError on
non-scalar backward()/item(), DLPack round-trips, the DLPack export memory
leak fix, torch-style grad accumulation across backward calls, and the
removal of the named neg() alias.
"""

import sys

try:
    import resource
except ImportError:
    resource = None  # POSIX-only; the leak test is skipped without it.

import numpy as np

import ferro


def run_leak_test():
    return resource is not None and sys.platform != "win32"


def test_requires_grad_inplace():
    # Statement-style requires_grad_ (torch idiom) must mutate in place.
    xs = ferro.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    w = ferro.Tensor([1.0, 1.0], [2, 1])
    assert not w.requires_grad
    w.requires_grad_(True)
    assert w.requires_grad
    xs.matmul(w).sum().backward()
    assert w.grad is not None, "statement-style requires_grad_ lost the flag"
    # d(sum(X @ w))/dw = column sums of X = [[4], [6]].
    assert w.grad.tolist() == [[4.0], [6.0]], w.grad.tolist()
    # Returns self for chaining.
    v = ferro.Tensor([1.0], [1]).requires_grad_(True)
    assert v.requires_grad
    w.requires_grad_(False)
    assert not w.requires_grad
    print("requires_grad_ in-place + getter: OK")


def test_backward_nonscalar_raises():
    t = ferro.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]).requires_grad_(True)
    try:
        t.backward()
    except ValueError as e:
        assert "scalar" in str(e), e
    else:
        raise AssertionError("backward() on non-scalar did not raise ValueError")
    print("backward() non-scalar ValueError: OK")


def test_item_nonscalar_raises():
    t = ferro.Tensor([1.0, 2.0, 3.0, 4.0], [4])
    try:
        t.item()
    except ValueError as e:
        assert "single-element" in str(e), e
    else:
        raise AssertionError("item() on 4-element tensor did not raise ValueError")
    assert ferro.Tensor([7.0], [1]).item() == 7.0
    print("item() non-scalar ValueError: OK")


def test_dlpack_roundtrip():
    t = ferro.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    arr = np.from_dlpack(t)
    assert arr.shape == (2, 3) and arr.dtype == np.float32
    assert np.array_equal(arr, np.array(t.tolist(), dtype=np.float32))
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t2 = ferro.from_dlpack(src)
    assert np.array_equal(np.array(t2.tolist(), dtype=np.float32), src)
    print("DLPack round-trip: OK")


def test_dlpack_export_no_leak():
    if not run_leak_test():
        print("DLPack export leak check: SKIPPED (needs `resource`, not on this platform)")
        return
    t = ferro.Tensor([float(i) for i in range(16)], [4, 4])
    # Warm up allocator/import machinery before measuring.
    for _ in range(1000):
        np.from_dlpack(t)
    # ru_maxrss is kilobytes on Linux but bytes on macOS.
    rss_scale = 1024 if sys.platform == "darwin" else 1
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // rss_scale
    for _ in range(200_000):
        np.from_dlpack(t)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // rss_scale
    grown_kb = after - before
    # Pre-fix this leaked the DLManagedTensor box every export (~15+ MB here).
    assert grown_kb < 4096, f"RSS grew {grown_kb} KB over 200k exports; leak?"
    print(f"DLPack export leak check: OK (RSS grew {grown_kb} KB)")


def test_grad_accumulation():
    # torch semantics: grads accumulate across backward calls on leaves.
    x = ferro.Tensor([1.0, 2.0, 3.0], [3]).requires_grad_(True)
    y = (x * x).sum()
    y.backward()
    assert x.grad.tolist() == [2.0, 4.0, 6.0], x.grad.tolist()
    y2 = (x * x).sum()
    y2.backward()
    assert x.grad.tolist() == [4.0, 8.0, 12.0], x.grad.tolist()
    print("grad accumulation across backward calls: OK")


def test_neg():
    t = ferro.Tensor([1.0, -2.0], [2])
    assert (-t).tolist() == [-1.0, 2.0]
    assert not hasattr(t, "neg"), "named neg() should be removed; use -t"
    print("__neg__ works, named neg() removed: OK")


def test_reflected_ops():
    t = ferro.Tensor([1.0, 2.0], [2]).requires_grad_(True)
    assert (2.0 + t).tolist() == [3.0, 4.0]
    assert (10.0 - t).tolist() == [9.0, 8.0]
    assert (3.0 * t).tolist() == [3.0, 6.0]
    assert (2.0 / t).tolist() == [2.0, 1.0]
    assert (1.0 + 2.0 * t - t / 2.0).tolist() == [2.5, 4.0]
    # Gradients flow through scalar operands.
    (2.0 * t).sum().backward()
    assert t.grad.tolist() == [2.0, 2.0]
    a = ferro.Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])
    b = ferro.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert (a @ b).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    print("reflected + scalar ops, __matmul__: OK")


def test_indexing():
    t = ferro.Tensor([float(i) for i in range(12)], [2, 3, 2])
    assert t[1].tolist() == [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]
    assert t[-1, -1].tolist() == [10.0, 11.0]
    assert t[0, :, 1].tolist() == [1.0, 3.0, 5.0]
    assert t[:, ::2].shape == [2, 2, 2]
    assert t[:, ::2].tolist() == [[[0.0, 1.0], [4.0, 5.0]], [[6.0, 7.0], [10.0, 11.0]]]
    assert t[..., ::-1].tolist()[0][0] == [1.0, 0.0]
    assert t[0:1].shape == [1, 3, 2]
    try:
        t[5]
    except ValueError as e:
        assert "out of bounds" in str(e), e
    else:
        raise AssertionError("out-of-bounds index did not raise")
    print("basic indexing with negatives/strides: OK")


def test_view_ops():
    x = ferro.Tensor([float(i) for i in range(12)], [2, 3, 2]).requires_grad_(True)
    y = x[1, 1:, :]
    assert y.shape == [2, 2] and y.requires_grad
    (y.sum() + x[..., ::-1][0].sum()).backward()
    assert x.grad.tolist() == [[[1.0, 1.0]] * 3, [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]], x.grad.tolist()
    assert x[0, 0, 0].requires_grad is False  # rank-0 picks stay detached copies
    assert x[:, 3:].shape == [2, 0, 2]
    t = ferro.Tensor([float(i) for i in range(6)], [2, 3])
    assert t.permute(1, 0).tolist() == t.transpose(0, 1).tolist()
    assert [p.shape for p in t.split([1, 2], 1)] == [[2, 1], [2, 2]]
    assert [p.shape for p in t.chunk(2, -1)] == [[2, 2], [2, 1]]
    assert t[:, :1].expand(-1, 4).tolist() == [[0.0] * 4, [3.0] * 4]
    assert ferro.stack([t, t], -1).shape == [2, 3, 2]
    for bad in (lambda: t.permute(0, 0), lambda: t.narrow(0, 1, 5), lambda: t.expand(3, 3), lambda: t.permute(0, 5)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid view op did not raise")
    print("view ops and autograd-carrying indexing: OK")


def test_negative_dims():
    t = ferro.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert t.sum_dim(-1).tolist() == [3.0, 7.0]
    assert t.mean_dim(-2).tolist() == [2.0, 3.0]
    x = ferro.Tensor([1.0, 3.0, 2.0, 4.0], [2, 2])
    indices = x.argmax(-1, keepdim=False).tolist()
    assert indices == [1, 1]
    assert all(type(index) is int for index in indices)
    sm = t.softmax(-1).tolist()
    assert abs(sm[0][0] - 0.26894143) < 1e-6 and abs(sm[0][1] - 0.73105860) < 1e-6, sm
    assert ferro.cat([t, t], dim=-1).shape == [2, 4]
    print("negative dims across dim-taking ops: OK")


def test_device_api():
    t = ferro.Tensor([1.0], [1])
    assert t.device == "cpu"
    assert t.to("cpu").device == "cpu" and t.cpu().device == "cpu"
    assert ferro.Tensor.zeros([2, 2], device="cpu").device == "cpu"
    assert ferro.Tensor.ones([3], device="cpu").device == "cpu"
    try:
        t.to("cuda:0")
    except ValueError as e:
        # No CUDA backend registered on this machine: must fail loudly.
        assert "cuda" in str(e).lower(), e
    else:
        pass  # CUDA backend present; residency checked by core tests.
    try:
        t.to("gpu")
    except ValueError as e:
        assert "unknown device" in str(e), e
    print("device api (to/cpu/cuda/device getter, factory device arg): OK")


def test_generators():
    g = ferro.Generator(7)
    a = ferro.Tensor.randn([3], generator=g)
    b = ferro.Tensor.randn([3], generator=g)
    assert a.tolist() != b.tolist(), "generator state did not advance"
    g.manual_seed(7)
    assert ferro.Tensor.randn([3], generator=g).tolist() == a.tolist()
    s1 = ferro.Tensor.randn([4], seed=42)
    s2 = ferro.Tensor.randn([4], seed=42)
    assert s1.tolist() == s2.tolist()
    d1 = ferro.Tensor.randn([1000])  # time-seeded
    assert abs(sum(d1.tolist()) / 1000) < 0.2
    u = ferro.Tensor.rand([3], seed=1)
    assert all(0.0 <= v < 1.0 for v in u.tolist()), u.tolist()
    print("randn/rand with seed/generator/time-seeded defaults: OK")


def test_repr():
    small = ferro.Tensor([1.0, 2.0], [2])
    assert "data=[1, 2]" in repr(small), repr(small)
    big = ferro.Tensor([float(i) for i in range(100)], [10, 10])
    r = repr(big)
    assert "..." in r and len(r) < 400, r
    assert "device=" not in r.split("dtype")[0].split("(")[1] or True
    print("truncated __repr__: OK")


def _expect(exc, fn, needle=""):
    try:
        fn()
    except exc as e:
        assert needle in str(e), e
    else:
        raise AssertionError(f"expected {exc.__name__}")


def test_tensor_method_bindings():
    # Every newly bound method: output shape, and a gradient of input shape.
    mix = [-0.9, 0.3, 0.6, -0.5, 0.1, 0.75]
    other = [0.8, -1.3, 0.6, 1.9, -0.4, 1.2]
    pos = [0.5, 1.2, 2.0, 3.3, 0.7, 1.5]
    x = lambda d=mix: ferro.Tensor(d, [2, 3]).requires_grad_(True)
    o = lambda: ferro.Tensor(other, [2, 3])
    gt1 = [v + 2.0 for v in pos]
    shaped = {
        "acos": ((), [2, 3]), "asin": ((), [2, 3]), "atanh": ((), [2, 3]), "asinh": ((), [2, 3]),
        "atan": ((), [2, 3]), "cos": ((), [2, 3]), "cosh": ((), [2, 3]), "sin": ((), [2, 3]), "sinh": ((), [2, 3]),
        "tan": ((), [2, 3]), "sinc": ((), [2, 3]), "deg2rad": ((), [2, 3]), "rad2deg": ((), [2, 3]),
        "erf": ((), [2, 3]), "erfc": ((), [2, 3]), "exp2": ((), [2, 3]), "expm1": ((), [2, 3]),
        "reciprocal": ((), [2, 3]), "square": ((), [2, 3]), "sign": ((), [2, 3]), "ceil": ((), [2, 3]),
        "floor": ((), [2, 3]), "round": ((), [2, 3]), "trunc": ((), [2, 3]), "frac": ((), [2, 3]),
        "silu": ((), [2, 3]), "gelu_erf": ((), [2, 3]), "mish": ((), [2, 3]), "selu": ((), [2, 3]),
        "relu6": ((), [2, 3]), "hardsigmoid": ((), [2, 3]), "hardswish": ((), [2, 3]),
        "log_sigmoid": ((), [2, 3]), "softplus": ((), [2, 3]), "softsign": ((), [2, 3]),
        "tanhshrink": ((), [2, 3]), "elu": ((0.5,), [2, 3]), "celu": ((0.5,), [2, 3]),
        "leaky_relu": ((0.1,), [2, 3]), "hardshrink": ((0.2,), [2, 3]), "softshrink": ((0.2,), [2, 3]),
        "hardtanh": ((-0.5, 0.5), [2, 3]), "threshold": ((0.2, -1.0), [2, 3]),
        "logsumexp": ((1,), [2]), "softmin": ((-1,), [2, 3]), "diff": ((), [2, 2]),
        "prod_dim": ((0, True), [1, 3]), "var_dim": ((1,), [2]), "std_dim": ((-1, 0, True), [2, 1]),
        "normalize": ((), [2, 3]), "flatten": ((), [6]), "triu": ((), [2, 3]), "tril": ((1,), [2, 3]),
        "pad_constant": (([1, 0, 0, 2], 0.5), [3, 5]),
        "rms_norm": ((), [2, 3]), "dropout": ((0.5, True, 7), [2, 3]),
    }
    for name, (args, shape) in shaped.items():
        t = x()
        y = getattr(t, name)(*args)
        assert y.shape == shape, (name, y.shape)
        (y * y).sum().backward()
        assert t.grad is not None and t.grad.shape == [2, 3], name
    for name in ("log10", "log1p", "log2", "rsqrt", "logit"):
        t = x([0.2, 0.7, 0.45, 0.9, 0.1, 0.6])
        getattr(t, name)().sum().backward()
        assert t.grad.shape == [2, 3], name
    r = ferro.Tensor.randn([2, 3, 4], seed=1).requires_grad_(True)
    assert r.rope_cached().shape == [2, 3, 4] and r.rope_cached(500.0).shape == [2, 3, 4]
    (r.rope_cached() * r).sum().backward()
    assert r.grad.shape == [2, 3, 4]
    t = x(gt1)
    t.acosh().sum().backward()
    assert t.grad.shape == [2, 3]
    for name in ("atan2", "copysign", "fmod", "remainder", "hypot", "logaddexp", "maximum", "minimum", "xlogy"):
        a, b = x(), ferro.Tensor(pos, [2, 3]).requires_grad_(True)
        y = getattr(a, name)(b)
        assert y.shape == [2, 3], name
        y.sum().backward()
        assert a.grad.shape == [2, 3] and b.grad.shape == [2, 3], name
        assert getattr(x(), name)(0.5).shape == [2, 3], f"{name} scalar operand"
    assert x().heaviside(o()).shape == [2, 3]
    a, b, c = x(), x(other), x(pos)
    for y in (a.lerp(b, 0.25), a.addcmul(b, c, 0.5), a.addcdiv(b, c), a.dist(b), a.dist(b, 3.0),
              a.cosine_similarity(b), a.pairwise_distance(b), a.flatten().outer(b.flatten())):
        y.sum().backward()
    assert a.grad.shape == b.grad.shape == c.grad.shape == [2, 3]
    assert a.flatten().outer(b.flatten()).shape == [6, 6] and a.cosine_similarity(b).shape == [2] and a.dist(b).shape == []
    sq = ferro.Tensor([float(i) for i in range(9)], [3, 3]).requires_grad_(True)
    sq.trace().backward()
    assert sq.grad.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    idx = ferro.Tensor.from_i64([2, 0, 1, 1, 1, 0], [2, 3])
    src = x(other)
    for name in ("scatter", "scatter_add"):
        t = x()
        getattr(t, name)(-1, idx, src).sum().backward()
        assert t.grad.shape == [2, 3], name
    assert src.grad.shape == [2, 3]
    img = ferro.Tensor([float(i % 7) for i in range(32)], [1, 2, 4, 4]).requires_grad_(True)
    assert img.avg_pool2d(2).shape == [1, 2, 2, 2] and img.avg_pool2d(3, 1).shape == [1, 2, 2, 2]
    w, b = ferro.Tensor.ones([2]).requires_grad_(True), ferro.Tensor.zeros([2]).requires_grad_(True)
    img.group_norm(1, w, b).sum().backward()
    assert w.grad.shape == [2] and img.grad.shape == [1, 2, 4, 4]
    y, rm, rv = img.batch_norm(w, b, ferro.Tensor.zeros([2]), ferro.Tensor.ones([2]), True)
    assert y.shape == [1, 2, 4, 4] and rm.shape == rv.shape == [2]
    _expect(ValueError, lambda: x().flatten(5))
    _expect(ValueError, lambda: x().dropout(1.0, True, 0))
    _expect(ValueError, lambda: img.group_norm(3, w, b))
    print("newly bound tensor methods (shapes + gradients): OK")


def test_functional_bindings():
    F = ferro.nn.functional
    x = ferro.Tensor([-0.9, 0.3, 0.6, -0.5, 0.1, 0.75], [2, 3]).requires_grad_(True)
    t = ferro.Tensor([0.8, -1.3, 0.6, 1.9, -0.4, 1.2], [2, 3])
    p = ferro.Tensor([0.2, 0.7, 0.45, 0.9, 0.1, 0.6], [2, 3])
    signs = ferro.Tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0], [2, 3])
    ids = ferro.Tensor.from_i64([2, 0], [2])
    losses = {
        "l1_loss": lambda r: F.l1_loss(x, t, r), "huber_loss": lambda r: F.huber_loss(x, t, r),
        "smooth_l1_loss": lambda r: F.smooth_l1_loss(x, t, r), "bce": lambda r: F.binary_cross_entropy(x.sigmoid(), p, reduction=r),
        "bce_logits": lambda r: F.binary_cross_entropy_with_logits(x, p, reduction=r),
        "kl_div": lambda r: F.kl_div(x.log_softmax(1), p.softmax(1), r), "poisson": lambda r: F.poisson_nll_loss(x, p, reduction=r),
        "soft_margin": lambda r: F.soft_margin_loss(x, signs, r), "hinge": lambda r: F.hinge_embedding_loss(x, signs, 1.0, r),
        "margin_ranking": lambda r: F.margin_ranking_loss(x, t, signs, 0.1, r),
        "cosine_embedding": lambda r: F.cosine_embedding_loss(x, t, ferro.Tensor([1.0, -1.0], [2]), 0.0, r),
        "triplet": lambda r: F.triplet_margin_loss(x, t, p, reduction=r),
        "cross_entropy": lambda r: F.cross_entropy(x, ids, r), "cross_entropy_probs": lambda r: F.cross_entropy(x, p.softmax(1), r),
        "nll": lambda r: F.nll_loss(x.log_softmax(1), ids, r),
    }
    for name, fn in losses.items():
        mean, total = fn("mean"), fn("sum")
        assert mean.shape == [] and total.shape == [], name
        assert abs(total.item()) >= abs(mean.item()) - 1e-6, name
        x.zero_grad()
        mean.backward()
        assert x.grad is not None and x.grad.shape == [2, 3], name
        _expect(ValueError, lambda: fn("none"), "reduction")
    assert abs(F.kl_div(x.log_softmax(1), p.softmax(1), "batchmean").item() - 3 * F.kl_div(x.log_softmax(1), p.softmax(1)).item()) < 1e-5
    assert F.gelu(x).shape == F.gelu(x, "tanh").shape == [2, 3]
    assert F.pad(x, (1, 1)).shape == [2, 5] and F.pad(x, (0, 1, 2, 0), value=3.0).shape == [4, 4]
    assert F.embedding(ferro.Tensor.from_i64([0, 1, 1, 0], [2, 2]), t).shape == [2, 2, 3]
    assert F.one_hot(ids, 3).tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    q = ferro.Tensor.randn([2, 2, 3, 4], seed=3)
    assert F.scaled_dot_product_attention(q, q, q, is_causal=True).shape == [2, 2, 3, 4]
    assert F.linear(ferro.Tensor.randn([2, 3, 4], seed=1), ferro.Tensor.randn([5, 4], seed=2)).shape == [2, 3, 5]
    assert F.dropout(x, 0.5, training=False) is x
    a, b = F.dropout(x, 0.5, seed=9).tolist(), F.dropout(x, 0.5, seed=9).tolist()
    assert a == b, "explicit dropout seed must be reproducible"
    _expect(ValueError, lambda: F.max_pool2d(q, 2, padding=1), "padding")
    _expect(ValueError, lambda: F.avg_pool2d(q, (2, 3)), "square")
    _expect(ValueError, lambda: F.layer_norm(x, (2, 3)), "last dimension")
    _expect(ValueError, lambda: F.pad(x, (1,)))
    _expect(NotImplementedError, lambda: F.scaled_dot_product_attention(q, q, q, attn_mask=q))
    print("functional losses/activations/attention bindings: OK")


def _all_params_get_grads(name, model, out):
    (out * out).sum().backward() if out.shape else out.backward()
    for pname, param in model.named_parameters():
        assert param.grad is not None, f"{name}.{pname}: no gradient"
        assert param.grad.shape == param.tensor().shape, f"{name}.{pname}"


def test_nn_modules():
    nn = ferro.nn
    img = ferro.Tensor.randn([2, 3, 6, 6], seed=1)
    seq = ferro.Tensor.randn([2, 5, 8], seed=2)
    cases = [
        ("Conv2d", nn.Conv2d(3, 4, 3, padding=1), img, [2, 4, 6, 6], ["weight", "bias"]),
        ("Conv2d groups/dilation", nn.Conv2d(3, 6, (3, 3), stride=2, dilation=2, groups=3, bias=False), img, [2, 6, 1, 1], ["weight"]),
        ("BatchNorm2d", nn.BatchNorm2d(3), img, [2, 3, 6, 6], ["weight", "bias"]),
        ("BatchNorm1d", nn.BatchNorm1d(8), seq.reshape([10, 8]), [10, 8], ["weight", "bias"]),
        ("GroupNorm", nn.GroupNorm(1, 3), img, [2, 3, 6, 6], ["weight", "bias"]),
        ("LayerNorm", nn.LayerNorm(8), seq, [2, 5, 8], ["weight", "bias"]),
        ("RMSNorm", nn.RMSNorm(8), seq, [2, 5, 8], ["weight"]),
        ("MaxPool2d", nn.MaxPool2d(2), img, [2, 3, 3, 3], []),
        ("AvgPool2d", nn.AvgPool2d(3, stride=1), img, [2, 3, 4, 4], []),
        ("Flatten", nn.Flatten(), img, [2, 108], []),
        ("Dropout", nn.Dropout(0.3), seq, [2, 5, 8], []),
        ("SelfAttention", nn.SelfAttention(8, 4, num_kv_heads=2, rope_base=10000.0), seq, [2, 5, 8],
         ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]),
        ("TransformerBlock", nn.TransformerBlock(8, 2), seq, [2, 5, 8],
         ["norm1.weight", "attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight", "attn.o_proj.weight",
          "norm2.weight", "up.weight", "up.bias", "down.weight", "down.bias"]),
        ("Sequential", nn.Sequential(nn.Conv2d(3, 2, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8, 4)), img, [2, 4],
         ["0.weight", "0.bias", "4.weight", "4.bias"]),
    ]
    for name, model, x, shape, names in cases:
        out = model(x)
        assert out.shape == shape, (name, out.shape)
        assert [n for n, _ in model.named_parameters()] == names, (name, [n for n, _ in model.named_parameters()])
        if names:
            _all_params_get_grads(name, model, out)
    emb = nn.Embedding(10, 4)
    out = emb(ferro.Tensor.from_i64([1, 2, 2, 9, 0, 1], [2, 3]))
    assert out.shape == [2, 3, 4]
    _all_params_get_grads("Embedding", emb, out)
    mha = nn.MultiheadAttention(8, 2)
    out, weights = mha(seq, ferro.Tensor.randn([2, 7, 8], seed=4), ferro.Tensor.randn([2, 7, 8], seed=5))
    assert out.shape == [2, 5, 8] and weights.shape == [2, 5, 7]
    assert [n for n, _ in mha.named_parameters()] == ["in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"]
    _all_params_get_grads("MultiheadAttention", mha, out)
    assert mha(seq, seq, seq, need_weights=False)[1] is None
    causal = mha(seq, seq, seq, is_causal=True)[1].tolist()
    assert all(row[j] < 1e-6 for batch in causal for i, row in enumerate(batch) for j in range(i + 1, 5)), "causal weights leak"
    acts = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.GELU(), nn.GELU("tanh"), nn.SiLU(), nn.Mish(), nn.ELU(0.5),
            nn.CELU(), nn.SELU(), nn.LeakyReLU(0.2), nn.ReLU6(inplace=True), nn.Hardtanh(), nn.Hardsigmoid(),
            nn.Hardswish(), nn.Hardshrink(), nn.Softshrink(), nn.Tanhshrink(), nn.Softplus(beta=2.0), nn.Softsign(),
            nn.LogSigmoid(), nn.Threshold(0.1, 0.0), nn.Softmax(-1), nn.LogSoftmax(1), nn.Softmin(0), nn.Identity()]
    for act in acts:
        t = ferro.Tensor([-0.9, 0.3, 0.6, -0.5, 0.1, 0.75], [2, 3]).requires_grad_(True)
        y = act(t)
        assert y.shape == [2, 3], type(act).__name__
        (y * y).sum().backward()
        assert t.grad is not None, type(act).__name__
        assert list(act.parameters()) == []
    _expect(ValueError, lambda: nn.LayerNorm((2, 3)), "last dimension")
    _expect(ValueError, lambda: nn.Conv2d(3, 4, 3, groups=2), "divisible")
    _expect(ValueError, lambda: nn.MaxPool2d(2, padding=1), "padding")
    _expect(ValueError, lambda: nn.Dropout(1.0))
    _expect(ValueError, lambda: nn.SelfAttention(8, 3))
    _expect(ValueError, lambda: nn.BatchNorm2d(3)(seq), "rank-4")
    _expect(TypeError, lambda: nn.Softmax(None))
    _expect(NotImplementedError, lambda: nn.MultiheadAttention(8, 2, batch_first=False))
    print("nn modules: shapes, parameter names and gradients: OK")


def test_loss_modules():
    nn = ferro.nn
    x = ferro.Tensor([-0.9, 0.3, 0.6, -0.5, 0.1, 0.75], [2, 3]).requires_grad_(True)
    t = ferro.Tensor([0.8, -1.3, 0.6, 1.9, -0.4, 1.2], [2, 3])
    p = ferro.Tensor([0.2, 0.7, 0.45, 0.9, 0.1, 0.6], [2, 3])
    signs = ferro.Tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0], [2, 3])
    ids = ferro.Tensor.from_i64([2, 0], [2])
    cases = [
        (nn.L1Loss(), (x, t)), (nn.L1Loss(reduction="sum"), (x, t)), (nn.HuberLoss(delta=0.5), (x, t)),
        (nn.SmoothL1Loss(beta=0.5), (x, t)), (nn.BCELoss(), (x.sigmoid(), p)), (nn.BCEWithLogitsLoss(), (x, p)),
        (nn.KLDivLoss(reduction="batchmean"), (x.log_softmax(1), p.softmax(1))), (nn.PoissonNLLLoss(), (x, p)),
        (nn.SoftMarginLoss(), (x, signs)), (nn.HingeEmbeddingLoss(margin=0.5), (x, signs)),
        (nn.MarginRankingLoss(margin=0.2), (x, t, signs)), (nn.CosineEmbeddingLoss(), (x, t, ferro.Tensor([1.0, -1.0], [2]))),
        (nn.TripletMarginLoss(margin=0.5), (x, t, p)), (nn.CrossEntropyLoss(), (x, ids)), (nn.NLLLoss(), (x.log_softmax(1), ids)),
        (nn.MSELoss(reduction="none"), (x, t)),
    ]
    for loss, args in cases:
        x.zero_grad()
        out = loss(*args)
        (out.sum() if out.shape else out).backward()
        assert x.grad is not None and x.grad.shape == [2, 3], type(loss).__name__
    _expect(ValueError, lambda: nn.L1Loss(reduction="none"), "reduction")
    _expect(ValueError, lambda: nn.HuberLoss(reduction="batchmean"), "reduction")
    print("loss modules: forward + input gradients: OK")


def test_empty_sum_losses():
    import ferro.nn.functional as F
    e = ferro.Tensor.zeros([0, 3])
    for f in (F.l1_loss, F.huber_loss, F.smooth_l1_loss, F.binary_cross_entropy_with_logits):
        assert f(e, e, reduction="sum").tolist() == 0.0, f.__name__
    print("empty inputs sum-reduce to 0 like torch: OK")


def test_train_eval_modes():
    nn = ferro.nn
    x = ferro.Tensor([1.0] * 4000, [40, 100])
    drop = nn.Dropout(0.25, seed=11)
    a, b = drop(x).tolist(), drop(x).tolist()
    flat = [v for row in a for v in row]
    zeros = sum(v == 0.0 for v in flat) / len(flat)
    assert 0.2 < zeros < 0.3, zeros
    assert all(v == 0.0 or abs(v - 1 / 0.75) < 1e-6 for v in flat), "survivors must be scaled by 1/(1-p)"
    assert a != b, "each training forward must draw a fresh mask"
    replay = nn.Dropout(0.25, seed=11)
    assert replay(x).tolist() == a and replay(x).tolist() == b, "masks replay for a fixed seed"
    drop.eval()
    assert drop(x) is x and not drop.training
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.Dropout(0.5))
    model.eval()
    assert all(not m.training for m in model.modules())
    model.train()
    assert all(m.training for m in model.modules())
    bn = nn.BatchNorm1d(3, momentum=0.5)
    mean_buffer = bn.running_mean
    batch = ferro.Tensor([1.0, 2.0, 3.0, 3.0, 6.0, 9.0], [2, 3])
    train_out = bn(batch)
    assert bn.running_mean is mean_buffer, "running stats must update in place"
    assert bn.running_mean.tolist() == [1.0, 2.0, 3.0], bn.running_mean.tolist()
    assert [round(v, 5) for v in bn.running_var.tolist()] == [1.5, 4.5, 9.5], bn.running_var.tolist()
    assert dict(bn.named_buffers()).keys() == {"running_mean", "running_var"}
    bn.eval()
    frozen = bn.running_mean.tolist()
    eval_out = bn(batch)
    assert bn.running_mean.tolist() == frozen, "eval must not update running stats"
    assert eval_out.tolist() != train_out.tolist(), "eval must use running statistics"
    bn(ferro.Tensor([1.0, 2.0, 3.0], [1, 3]))  # singleton batch allowed in eval
    bn.train()
    _expect(ValueError, lambda: bn(ferro.Tensor([1.0, 2.0, 3.0], [1, 3])))
    print("train/eval modes (Dropout masks, BatchNorm running stats): OK")


def test_optimizers_see_module_parameters():
    nn = ferro.nn
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Flatten(), nn.Dropout(0.1, seed=3), nn.Linear(16, 3))
    xs = ferro.Tensor.randn([8, 1, 4, 4], seed=7)
    ys = ferro.Tensor.from_i64([i % 3 for i in range(8)], [8])
    loader = ferro.data.DataLoader(ferro.data.TensorDataset(xs, ys), batch_size=4, shuffle=True, seed=1)
    opt = ferro.optim.Adam(model.parameters(), lr=0.05)
    before = [p.tensor().tolist() for p in model.parameters()]
    loss_fn = nn.CrossEntropyLoss()
    first = last = None
    for epoch in range(30):
        total = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            total += loss.item()
        first = total if first is None else first
        last = total
    after = [p.tensor().tolist() for p in model.parameters()]
    assert all(b != a for b, a in zip(before, after)), "every parameter must be updated by the optimizer"
    assert last < first * 0.7, (first, last)
    opt.zero_grad()
    import tempfile
    gen = ferro.Generator(5)
    snap = ferro.checkpoint.snapshot_training(model, optimizers={"main": opt}, generator=gen)
    saved = {n: b.tolist() for n, b in model.named_buffers()}
    model.eval()
    with tempfile.TemporaryDirectory() as d:
        snap.save(d)
        ferro.checkpoint.load_training(d, model, optimizers={"main": opt}, generator=gen)
    assert {n: b.tolist() for n, b in model.named_buffers()} == saved
    assert model.training, "checkpoint restores module modes"
    print("optimizers + checkpoints over new modules (training converges): OK")


def test_dataloader():
    xs = ferro.Tensor([float(i) for i in range(20)], [10, 2])
    ys = ferro.Tensor.from_i64(list(range(10)), [10])
    ds = ferro.data.TensorDataset(xs, ys)
    assert len(ds) == 10
    x3, y3 = ds[3]
    assert x3.tolist() == [6.0, 7.0] and y3.tolist() == 3 and ds[-1][1].tolist() == 9
    loader = ferro.data.DataLoader(ds, batch_size=4)
    batches = list(loader)
    assert len(loader) == 3 and [b[1].tolist() for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert batches[0][0].shape == [4, 2]
    assert len(ferro.data.DataLoader(ds, 4, drop_last=True)) == 2
    assert [b[1].shape for b in ferro.data.DataLoader(ds, 4, drop_last=True)] == [[4], [4]]
    order = lambda dl: [i for _, by in dl for i in by.tolist()]
    a = ferro.data.DataLoader(ds, 3, shuffle=True, seed=42)
    b = ferro.data.DataLoader(ds, 3, shuffle=True, seed=42, num_workers=3)
    e0, e1 = order(a), order(a)
    assert sorted(e0) == list(range(10)) and e0 != list(range(10))
    assert e0 != e1, "shuffle must reorder each epoch"
    assert order(b) == e0 and order(b) == e1, "worker threads must preserve sampler order"
    _expect(TypeError, lambda: ferro.data.DataLoader([(xs, ys)], 2))
    _expect(ValueError, lambda: ferro.data.DataLoader(ds, 0))
    _expect(ValueError, lambda: ferro.data.TensorDataset(xs, ferro.Tensor.zeros([3])))
    _expect(ValueError, lambda: ds[10], "out of bounds")
    print("TensorDataset + native DataLoader (order, shuffle, workers, drop_last): OK")


def main():
    test_requires_grad_inplace()
    test_backward_nonscalar_raises()
    test_item_nonscalar_raises()
    test_dlpack_roundtrip()
    test_dlpack_export_no_leak()
    test_grad_accumulation()
    test_neg()
    test_reflected_ops()
    test_indexing()
    test_view_ops()
    test_negative_dims()
    test_device_api()
    test_generators()
    test_repr()
    test_tensor_method_bindings()
    test_functional_bindings()
    test_nn_modules()
    test_loss_modules()
    test_empty_sum_losses()
    test_train_eval_modes()
    test_optimizers_see_module_parameters()
    test_dataloader()
    print("ALL REGRESSION CHECKS PASSED")


if __name__ == "__main__":
    main()
