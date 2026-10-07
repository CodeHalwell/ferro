"""Numerical parity checks for the newly bound ferro ops against torch.

Run inside the ferro-py venv: python examples/ops_vs_torch.py
"""

import math

import torch
import torch.nn.functional as F

import ferro

# Fixed literal inputs (no RNG) shared by ferro and torch.
POS = [0.5, 1.2, 2.0, 3.3, 0.7, 1.5]  # strictly positive, for log/sqrt
MIX = [-1.5, 0.3, 2.0, -0.7, 1.1, -2.4]  # mixed signs, for tanh/abs/clamp
SHAPE = [2, 3]

BMM_A = [0.5, -1.0, 2.0, 1.5, 0.25, -0.75, 1.0, 2.0, -0.5, 0.1, 0.9, -1.2]  # (2,2,3)
BMM_B = [1.0, -0.5, 0.75, 2.0, -1.25, 0.5, 0.2, 1.4, -0.6, 0.8, 1.1, -0.3]  # (2,3,2)


def ft(data, shape, requires_grad=False):
    t = ferro.Tensor(data, shape)
    return t.requires_grad_(True) if requires_grad else t


def tt(data, shape, requires_grad=False):
    t = torch.tensor(data).reshape(shape)
    return t.requires_grad_(True) if requires_grad else t


def check(name, f, t):
    values = f.tolist()
    message = f"{name}: ferro={values} torch={t.tolist()}"
    assert f.shape == list(t.shape), message
    assert t.is_floating_point() or t.dtype == torch.int64, message
    scalar_type = float if t.is_floating_point() else int
    pending = [values]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        else:
            assert type(value) is scalar_type, message
    if t.is_floating_point():
        assert torch.allclose(torch.tensor(values), t, atol=1e-5), message
    else:
        assert values == t.tolist(), message
    print(f"OK {name}")


def check_unary(name, data, ferro_op, torch_op):
    check(f"{name} value", ferro_op(ft(data, SHAPE)), torch_op(tt(data, SHAPE)))
    fx = ft(data, SHAPE, requires_grad=True)
    ferro_op(fx).sum().backward()
    tx = tt(data, SHAPE, requires_grad=True)
    torch_op(tx).sum().backward()
    check(f"{name} grad", fx.grad, tx.grad)


def check_conv2d(name, in_shape, w_shape, stride, padding):
    xin = [((i * 7) % 9 - 4) * 0.25 for i in range(math.prod(in_shape))]
    win = [((i * 5) % 7 - 3) * 0.5 for i in range(math.prod(w_shape))]
    fx = ft(xin, in_shape, requires_grad=True)
    fw = ft(win, w_shape, requires_grad=True)
    fy = fx.conv2d(fw, stride=stride, padding=padding)
    tx = tt(xin, in_shape, requires_grad=True)
    tw = tt(win, w_shape, requires_grad=True)
    ty = F.conv2d(tx, tw, stride=stride, padding=padding)
    check(f"{name} value", fy, ty)
    fy.sum().backward()
    ty.sum().backward()
    check(f"{name} grad input", fx.grad, tx.grad)
    check(f"{name} grad weight", fw.grad, tw.grad)


def torch_rope(x, pos, base=10000.0):
    half = x.shape[-1] // 2
    inv = base ** (-torch.arange(half, dtype=torch.float32) * 2.0 / x.shape[-1])
    theta = pos.float()[:, None] * inv[None, :]
    cos, sin = theta.cos(), theta.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def main():
    check_unary("reshape transpose layout", MIX, lambda x: x.transpose(0, 1).reshape([2, 3]), lambda x: x.transpose(0, 1).reshape(2, 3))
    check_unary("layer_norm", MIX, lambda x: x.layer_norm(), lambda x: F.layer_norm(x, (3,)))
    check_unary("log", POS, lambda x: x.log(), torch.log)
    check_unary("tanh", MIX, lambda x: x.tanh(), torch.tanh)
    check_unary("sqrt", POS, lambda x: x.sqrt(), torch.sqrt)
    check_unary("abs", MIX, lambda x: x.abs(), torch.abs)
    check_unary("pow", POS, lambda x: x.pow(3.0), lambda x: x.pow(3.0))
    check_unary("clamp", MIX, lambda x: x.clamp(-1.0, 1.0), lambda x: x.clamp(-1.0, 1.0))

    # max: global reduction to a scalar; grad hits only the argmax.
    check("max value", ft(MIX, SHAPE).max(), tt(MIX, SHAPE).max())
    fx = ft(MIX, SHAPE, requires_grad=True)
    fx.max().backward()
    tx = tt(MIX, SHAPE, requires_grad=True)
    tx.max().backward()
    check("max grad", fx.grad, tx.grad)

    # sum_dim / mean_dim: values with keepdim both ways, plus one grad each.
    for name, fop, top in [
        ("sum_dim", lambda x, d, k: x.sum_dim(d, k), torch.sum),
        ("mean_dim", lambda x, d, k: x.mean_dim(d, k), torch.mean),
    ]:
        for dim in (0, 1):
            for keep in (False, True):
                f = fop(ft(MIX, SHAPE), dim, keep)
                t = top(tt(MIX, SHAPE), dim=dim, keepdim=keep)
                check(f"{name} value dim={dim} keepdim={keep}", f, t)
        fx = ft(MIX, SHAPE, requires_grad=True)
        fop(fx, 1, False).sum().backward()
        tx = tt(MIX, SHAPE, requires_grad=True)
        top(tx, dim=1).sum().backward()
        check(f"{name} grad", fx.grad, tx.grad)

    # softmax / log_softmax: values on both dims; grad through a weighted sum
    # so the backward is non-trivial (plain .sum() of softmax has zero grad).
    w_data = [0.3, -1.0, 0.5, 2.0, -0.25, 1.5]
    for name, fop, top in [
        ("softmax", lambda x, d: x.softmax(d), torch.softmax),
        ("log_softmax", lambda x, d: x.log_softmax(d), torch.log_softmax),
    ]:
        for dim in (0, 1):
            check(f"{name} value dim={dim}", fop(ft(MIX, SHAPE), dim), top(tt(MIX, SHAPE), dim))
        fx = ft(MIX, SHAPE, requires_grad=True)
        (fop(fx, 1) * ft(w_data, SHAPE)).sum().backward()
        tx = tt(MIX, SHAPE, requires_grad=True)
        (top(tx, 1) * tt(w_data, SHAPE)).sum().backward()
        check(f"{name} grad", fx.grad, tx.grad)

    # bmm: value and grads for both operands.
    check("bmm value", ft(BMM_A, [2, 2, 3]).bmm(ft(BMM_B, [2, 3, 2])), tt(BMM_A, [2, 2, 3]).bmm(tt(BMM_B, [2, 3, 2])))
    fa = ft(BMM_A, [2, 2, 3], requires_grad=True)
    fb = ft(BMM_B, [2, 3, 2], requires_grad=True)
    fa.bmm(fb).sum().backward()
    ta = tt(BMM_A, [2, 2, 3], requires_grad=True)
    tb = tt(BMM_B, [2, 3, 2], requires_grad=True)
    ta.bmm(tb).sum().backward()
    check("bmm grad A", fa.grad, ta.grad)
    check("bmm grad B", fb.grad, tb.grad)

    # cat: dim 0 and 1, value + grads on both inputs. Squaring the output
    # makes the grad per-element (2*value) instead of all-ones.
    cat_b = [0.25, 1.75, -0.5, -3.0, 2.0, 0.75]
    for dim in (0, 1):
        fa = ft(MIX, SHAPE, requires_grad=True)
        fb = ft(cat_b, SHAPE, requires_grad=True)
        fc = ferro.cat([fa, fb], dim)
        ta = tt(MIX, SHAPE, requires_grad=True)
        tb = tt(cat_b, SHAPE, requires_grad=True)
        tc = torch.cat([ta, tb], dim)
        check(f"cat value dim={dim}", fc, tc)
        (fc * fc).sum().backward()
        (tc * tc).sum().backward()
        check(f"cat grad a dim={dim}", fa.grad, ta.grad)
        check(f"cat grad b dim={dim}", fb.grad, tb.grad)

    # index_select: duplicate indices must accumulate grad.
    for dim, idx in [(0, [1, 0, 1]), (1, [2, 2, 0])]:
        fx = ft(MIX, SHAPE, requires_grad=True)
        fy = fx.index_select(dim, idx)
        tx = tt(MIX, SHAPE, requires_grad=True)
        ty = torch.index_select(tx, dim, torch.tensor(idx))
        check(f"index_select value dim={dim}", fy, ty)
        (fy * fy).sum().backward()
        (ty * ty).sum().backward()
        check(f"index_select grad dim={dim}", fx.grad, tx.grad)

    # where: grads route to a where the mask is set, to b elsewhere.
    mask = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    fa = ft(MIX, SHAPE, requires_grad=True)
    fb = ft(POS, SHAPE, requires_grad=True)
    fo = ferro.where(ft(mask, SHAPE), fa, fb)
    ta = tt(MIX, SHAPE, requires_grad=True)
    tb = tt(POS, SHAPE, requires_grad=True)
    to = torch.where(tt(mask, SHAPE).bool(), ta, tb)
    check("where value", fo, to)
    (fo * fo).sum().backward()
    (to * to).sum().backward()
    check("where grad a", fa.grad, ta.grad)
    check("where grad b", fb.grad, tb.grad)

    # squeeze / unsqueeze: shapes, then a grad roundtrip through both.
    assert ft(MIX, SHAPE).unsqueeze(0).shape == [1, 2, 3]
    assert ft(MIX, SHAPE).unsqueeze(2).shape == [2, 3, 1]
    assert ft(MIX, SHAPE).unsqueeze(1).squeeze(1).shape == [2, 3]
    print("OK squeeze/unsqueeze shapes")
    fx = ft(MIX, SHAPE, requires_grad=True)
    (fx.unsqueeze(0).squeeze(0) * ft(w_data, SHAPE)).sum().backward()
    tx = tt(MIX, SHAPE, requires_grad=True)
    (tx.unsqueeze(0).squeeze(0) * tt(w_data, SHAPE)).sum().backward()
    check("squeeze/unsqueeze grad", fx.grad, tx.grad)

    # conv2d: value + input and weight grads vs F.conv2d.
    check_conv2d("conv2d s1 p0", [1, 1, 4, 4], [1, 1, 3, 3], stride=1, padding=0)
    check_conv2d("conv2d s2 p1", [1, 1, 5, 5], [1, 1, 3, 3], stride=2, padding=1)
    check_conv2d("conv2d multichannel", [2, 2, 4, 4], [3, 2, 2, 2], stride=1, padding=0)

    # max_pool2d: (i*5)%32 is a permutation of 0..31, so every value is
    # distinct and the argmax (hence the grad) is unambiguous. k=3 s=1 has
    # overlapping windows, so grads accumulate.
    pool_in = [(((i * 5) % 32) - 16) * 0.5 for i in range(32)]
    for kernel, stride in [(2, 2), (3, 1)]:
        fx = ft(pool_in, [1, 2, 4, 4], requires_grad=True)
        fy = fx.max_pool2d(kernel, stride)
        tx = tt(pool_in, [1, 2, 4, 4], requires_grad=True)
        ty = F.max_pool2d(tx, kernel, stride)
        check(f"max_pool2d value k={kernel} s={stride}", fy, ty)
        fy.sum().backward()
        ty.sum().backward()
        check(f"max_pool2d grad k={kernel} s={stride}", fx.grad, tx.grad)

    # gelu: ferro implements the tanh approximation.
    check_unary("gelu", MIX, lambda x: x.gelu(), lambda x: F.gelu(x, approximate="tanh"))

    # cumsum: values on both dims, grad through a square so it is per-element.
    for dim in (0, 1):
        check(f"cumsum value dim={dim}", ft(MIX, SHAPE).cumsum(dim), tt(MIX, SHAPE).cumsum(dim))
    fx = ft(MIX, SHAPE, requires_grad=True)
    (fx.cumsum(1) * fx.cumsum(1)).sum().backward()
    tx = tt(MIX, SHAPE, requires_grad=True)
    (tx.cumsum(1) * tx.cumsum(1)).sum().backward()
    check("cumsum grad", fx.grad, tx.grad)

    # argmax/argmin: exact I64 values and Python int leaves through tolist.
    for name, fop, top in [
        ("argmax", lambda x, d, k: x.argmax(d, k), torch.argmax),
        ("argmin", lambda x, d, k: x.argmin(d, k), torch.argmin),
    ]:
        for dim in (0, 1):
            for keep in (False, True):
                f = fop(ft(MIX, SHAPE), dim, keep)
                t = top(tt(MIX, SHAPE), dim=dim, keepdim=keep)
                check(f"{name} value dim={dim} keepdim={keep}", f, t)

    # gather: duplicate indices must accumulate grad.
    gidx = [1, 1, 0, 2, 0, 0]
    fx = ft(MIX, SHAPE, requires_grad=True)
    fy = fx.gather(1, ferro.Tensor.from_i64(gidx, SHAPE))
    tx = tt(MIX, SHAPE, requires_grad=True)
    ty = tx.gather(1, tt(gidx, SHAPE).long())
    check("gather value", fy, ty)
    (fy * fy).sum().backward()
    (ty * ty).sum().backward()
    check("gather grad", fx.grad, tx.grad)

    # topk: sorted values and indices, grad routed to the selected positions.
    fv, fi = ft(MIX, SHAPE).topk(2, 1)
    tv, ti = tt(MIX, SHAPE).topk(2, dim=1)
    check("topk values", fv, tv)
    check("topk indices", fi, ti)
    fx = ft(MIX, SHAPE, requires_grad=True)
    fv, _ = fx.topk(2, 1)
    (fv * fv).sum().backward()
    tx = tt(MIX, SHAPE, requires_grad=True)
    tv, _ = tx.topk(2, dim=1)
    (tv * tv).sum().backward()
    check("topk grad", fx.grad, tx.grad)

    # rope: half-split rotation vs an inline torch reference (LLaMA/HF
    # convention: pair j rotates by pos * base^(-2j/d)).
    rope_x = [((i * 7) % 11 - 5) * 0.25 for i in range(24)]
    rope_pos = [0, 1, 5]
    fx = ft(rope_x, [2, 3, 4], requires_grad=True)
    fy = fx.rope(ferro.Tensor.from_i64(rope_pos, [3]))
    tx = tt(rope_x, [2, 3, 4], requires_grad=True)
    ty = torch_rope(tx, torch.tensor(rope_pos))
    check("rope value", fy, ty)
    (fy * fy).sum().backward()
    (ty * ty).sum().backward()
    check("rope grad", fx.grad, tx.grad)

    # Error mapping: core errors surface as ValueError.
    try:
        ft(MIX, SHAPE).sum_dim(99)
        raise AssertionError("sum_dim(99) did not raise")
    except ValueError:
        print("OK sum_dim out-of-range raises ValueError")
    try:
        ferro.Tensor([], [0]).max()
        raise AssertionError("empty max() did not raise")
    except ValueError:
        print("OK empty max raises ValueError")

    extended_ops()
    view_ops()
    extended_losses()
    extended_modules()
    print("ALL OPS MATCH TORCH")


UNIT = [-0.9, 0.3, 0.6, -0.5, 0.1, 0.75]  # inside (-1, 1)
GT1 = [1.5, 2.0, 3.3, 1.2, 4.0, 1.1]  # above 1, for acosh
PROB = [0.2, 0.7, 0.45, 0.9, 0.1, 0.6]  # inside (0, 1)
ROUNDING = [-1.4, 0.3, 2.6, -0.7, 1.1, -2.2]  # no .5 ties
OTHER = [0.8, -1.3, 0.6, 1.9, -0.4, 1.2]  # nonzero, never equal to MIX
SIGNS = [1.0, -1.0, 1.0, 1.0, -1.0, -1.0]


def from_torch(t):
    t = t.detach()
    if t.dtype == torch.int64:
        return ferro.Tensor.from_i64(t.flatten().tolist(), list(t.shape))
    return ferro.Tensor(t.flatten().tolist(), list(t.shape))


def check_fn(name, args, ferro_fn, torch_fn, grad=True):
    """Value plus grads of every float input under a fixed non-uniform seed."""
    fs = [ft(d, s, grad) for d, s in args]
    ts = [tt(d, s, grad) for d, s in args]
    fy, ty = ferro_fn(*fs), torch_fn(*ts)
    check(f"{name} value", fy, ty)
    if not grad:
        return
    seed = [((i * 7) % 5 - 2) * 0.5 + 0.25 for i in range(max(ty.numel(), 1))]
    (fy * ft(seed, list(ty.shape))).sum().backward()
    (ty * tt(seed, list(ty.shape))).sum().backward()
    for i, (f, t) in enumerate(zip(fs, ts)):
        if f.grad is None:
            # An input the output does not depend on: torch may report zeros.
            assert t.grad is None or not t.grad.any(), f"{name} grad {i}: ferro=None torch={t.grad}"
            print(f"OK {name} grad {i} (independent)")
        else:
            check(f"{name} grad {i}", f.grad, t.grad)


def extended_ops():
    import ferro.nn.functional as FF
    s = SHAPE
    unary = [
        ("acos", UNIT, lambda x: x.acos(), torch.acos), ("asin", UNIT, lambda x: x.asin(), torch.asin),
        ("atanh", UNIT, lambda x: x.atanh(), torch.atanh), ("acosh", GT1, lambda x: x.acosh(), torch.acosh),
        ("asinh", MIX, lambda x: x.asinh(), torch.asinh), ("atan", MIX, lambda x: x.atan(), torch.atan),
        ("cos", MIX, lambda x: x.cos(), torch.cos), ("cosh", MIX, lambda x: x.cosh(), torch.cosh),
        ("sin", MIX, lambda x: x.sin(), torch.sin), ("sinh", MIX, lambda x: x.sinh(), torch.sinh),
        ("tan", UNIT, lambda x: x.tan(), torch.tan), ("sinc", MIX, lambda x: x.sinc(), torch.sinc),
        ("deg2rad", MIX, lambda x: x.deg2rad(), torch.deg2rad), ("rad2deg", MIX, lambda x: x.rad2deg(), torch.rad2deg),
        ("erf", MIX, lambda x: x.erf(), torch.erf), ("erfc", MIX, lambda x: x.erfc(), torch.erfc),
        ("exp2", MIX, lambda x: x.exp2(), torch.exp2), ("expm1", MIX, lambda x: x.expm1(), torch.expm1),
        ("log10", POS, lambda x: x.log10(), torch.log10), ("log1p", POS, lambda x: x.log1p(), torch.log1p),
        ("log2", POS, lambda x: x.log2(), torch.log2), ("logit", PROB, lambda x: x.logit(), torch.logit),
        ("reciprocal", MIX, lambda x: x.reciprocal(), torch.reciprocal), ("rsqrt", POS, lambda x: x.rsqrt(), torch.rsqrt),
        ("square", MIX, lambda x: x.square(), torch.square), ("sign", MIX, lambda x: x.sign(), torch.sign),
        ("ceil", ROUNDING, lambda x: x.ceil(), torch.ceil), ("floor", ROUNDING, lambda x: x.floor(), torch.floor),
        ("round", ROUNDING, lambda x: x.round(), torch.round), ("trunc", ROUNDING, lambda x: x.trunc(), torch.trunc),
        ("frac", ROUNDING, lambda x: x.frac(), torch.frac),
        ("silu", MIX, FF.silu, F.silu), ("gelu exact", MIX, FF.gelu, F.gelu),
        ("gelu tanh", MIX, lambda x: FF.gelu(x, "tanh"), lambda x: F.gelu(x, approximate="tanh")),
        ("mish", MIX, FF.mish, F.mish), ("selu", MIX, FF.selu, F.selu), ("relu6", MIX, FF.relu6, F.relu6),
        ("hardsigmoid", MIX, FF.hardsigmoid, F.hardsigmoid), ("hardswish", MIX, FF.hardswish, F.hardswish),
        ("logsigmoid", MIX, FF.logsigmoid, F.logsigmoid), ("softplus", MIX, FF.softplus, F.softplus),
        ("softplus beta=2", MIX, lambda x: FF.softplus(x, beta=2.0), lambda x: F.softplus(x, beta=2.0)),
        ("softsign", MIX, FF.softsign, F.softsign), ("tanhshrink", MIX, FF.tanhshrink, F.tanhshrink),
        ("elu", MIX, lambda x: FF.elu(x, 0.7), lambda x: F.elu(x, 0.7)),
        ("celu", MIX, lambda x: FF.celu(x, 0.7), lambda x: F.celu(x, 0.7)),
        ("leaky_relu", MIX, lambda x: FF.leaky_relu(x, 0.2), lambda x: F.leaky_relu(x, 0.2)),
        ("hardshrink", MIX, FF.hardshrink, F.hardshrink), ("softshrink", MIX, FF.softshrink, F.softshrink),
        ("hardtanh", MIX, FF.hardtanh, F.hardtanh),
        ("threshold", MIX, lambda x: FF.threshold(x, 0.5, -1.0), lambda x: F.threshold(x, 0.5, -1.0)),
    ]
    for name, data, fop, top in unary:
        check_fn(name, [(data, s)], fop, top)
    check_fn("trace", [([float(i % 4) - 1.5 for i in range(9)], [3, 3])], lambda x: x.trace(), torch.trace)

    binary = [
        ("atan2", MIX, OTHER, lambda a, b: a.atan2(b), torch.atan2),
        ("copysign", MIX, OTHER, lambda a, b: a.copysign(b), torch.copysign),
        ("fmod", MIX, OTHER, lambda a, b: a.fmod(b), torch.fmod),
        ("remainder", MIX, OTHER, lambda a, b: a.remainder(b), torch.remainder),
        ("hypot", MIX, OTHER, lambda a, b: a.hypot(b), torch.hypot),
        ("logaddexp", MIX, OTHER, lambda a, b: a.logaddexp(b), torch.logaddexp),
        ("maximum", MIX, OTHER, lambda a, b: a.maximum(b), torch.maximum),
        ("minimum", MIX, OTHER, lambda a, b: a.minimum(b), torch.minimum),
        ("xlogy", MIX, POS, lambda a, b: a.xlogy(b), torch.xlogy),
        ("lerp", MIX, OTHER, lambda a, b: a.lerp(b, 0.3), lambda a, b: torch.lerp(a, b, 0.3)),
        ("dist p=2", MIX, OTHER, lambda a, b: a.dist(b), torch.dist),
        ("dist p=3", MIX, OTHER, lambda a, b: a.dist(b, 3.0), lambda a, b: torch.dist(a, b, 3.0)),
        ("cosine_similarity", MIX, OTHER, FF.cosine_similarity, F.cosine_similarity),
        ("pairwise_distance", MIX, OTHER, FF.pairwise_distance, F.pairwise_distance),
    ]
    for name, a, b, fop, top in binary:
        check_fn(name, [(a, s), (b, s)], fop, top)
    check_fn("heaviside", [(ROUNDING, s), (POS, s)], lambda a, b: a.heaviside(b), torch.heaviside, grad=False)
    check_fn("maximum scalar", [(MIX, s)], lambda a: a.maximum(0.5), lambda a: torch.maximum(a, torch.tensor(0.5)))
    check_fn("outer", [(MIX[:3], [3]), (OTHER[:4], [4])], lambda a, b: a.outer(b), torch.outer)
    check_fn("addcmul", [(MIX, s), (OTHER, s), (POS, s)], lambda a, b, c: a.addcmul(b, c, 0.5), lambda a, b, c: torch.addcmul(a, b, c, value=0.5))
    check_fn("addcdiv", [(MIX, s), (POS, s), (OTHER, s)], lambda a, b, c: a.addcdiv(b, c, 0.5), lambda a, b, c: torch.addcdiv(a, b, c, value=0.5))

    for dim in (0, 1, -1):
        check_fn(f"logsumexp dim={dim}", [(MIX, s)], lambda x: x.logsumexp(dim), lambda x: torch.logsumexp(x, dim))
        check_fn(f"softmin dim={dim}", [(MIX, s)], lambda x: FF.softmin(x, dim), lambda x: F.softmin(x, dim))
        check_fn(f"normalize dim={dim}", [(MIX, s)], lambda x: FF.normalize(x, dim=dim), lambda x: F.normalize(x, dim=dim))
        for keep in (False, True):
            check_fn(f"prod_dim dim={dim} keepdim={keep}", [(MIX, s)], lambda x: x.prod_dim(dim, keep), lambda x: torch.prod(x, dim, keepdim=keep))
            check_fn(f"var_dim dim={dim} keepdim={keep}", [(MIX, s)], lambda x: x.var_dim(dim, keepdim=keep), lambda x: torch.var(x, dim, keepdim=keep))
            check_fn(f"std_dim c=0 dim={dim} keepdim={keep}", [(MIX, s)], lambda x: x.std_dim(dim, 0, keep), lambda x: torch.std(x, dim, correction=0, keepdim=keep))
    check_fn("diff dim=1", [(MIX, s)], lambda x: x.diff(1), lambda x: torch.diff(x, dim=1))
    check_fn("diff dim=0", [(MIX, s)], lambda x: x.diff(0), lambda x: torch.diff(x, dim=0))
    cube = [((i * 5) % 13 - 6) * 0.25 for i in range(12)]
    check_fn("flatten 1", [(cube, [2, 3, 2])], lambda x: x.flatten(1), lambda x: torch.flatten(x, 1))
    check_fn("flatten 0..1", [(cube, [2, 3, 2])], lambda x: x.flatten(0, 1), lambda x: torch.flatten(x, 0, 1))
    sq = [float(i) - 4.0 for i in range(9)]
    for d in (-1, 0, 1):
        check_fn(f"triu {d}", [(sq, [3, 3])], lambda x: x.triu(d), lambda x: torch.triu(x, d))
        check_fn(f"tril {d}", [(sq, [3, 3])], lambda x: x.tril(d), lambda x: torch.tril(x, d))
    check_fn("pad", [(MIX, s)], lambda x: FF.pad(x, (1, 2, 0, 1), value=0.5), lambda x: F.pad(x, (1, 2, 0, 1), value=0.5))
    check_fn("pad last only", [(cube, [2, 3, 2])], lambda x: FF.pad(x, (2, 1)), lambda x: F.pad(x, (2, 1)))

    idx = [2, 0, 1, 1, 2, 0]
    check_fn("scatter", [(MIX, s), (OTHER, s)], lambda x, src: x.scatter(1, ferro.Tensor.from_i64(idx, s), src),
             lambda x, src: x.scatter(1, torch.tensor(idx).reshape(s), src))
    dup = [0, 0, 2, 1, 1, 1]
    check_fn("scatter_add", [(MIX, s), (OTHER, s)], lambda x, src: x.scatter_add(1, ferro.Tensor.from_i64(dup, s), src),
             lambda x, src: x.scatter_add(1, torch.tensor(dup).reshape(s), src))

    pool_in = [(((i * 5) % 32) - 16) * 0.5 for i in range(32)]
    for k, st in [(2, 2), (3, 1), (2, None)]:
        check_fn(f"avg_pool2d k={k} s={st}", [(pool_in, [1, 2, 4, 4])], lambda x: FF.avg_pool2d(x, k, st), lambda x: F.avg_pool2d(x, k, st))
        check_fn(f"max_pool2d k={k} s={st}", [(pool_in, [1, 2, 4, 4])], lambda x: FF.max_pool2d(x, k, st), lambda x: F.max_pool2d(x, k, st))

    w3, b3 = [0.5, -1.2, 2.0], [0.1, 0.2, -0.3]
    check_fn("layer_norm affine", [(MIX, s), (w3, [3]), (b3, [3])], lambda x, w, b: FF.layer_norm(x, 3, w, b), lambda x, w, b: F.layer_norm(x, (3,), w, b))
    check_fn("rms_norm affine", [(MIX, s), (w3, [3])], lambda x, w: FF.rms_norm(x, 3, w, 1e-6), lambda x, w: F.rms_norm(x, (3,), w, 1e-6))
    check_fn("rms_norm default eps", [(MIX, s)], lambda x: FF.rms_norm(x, [3]), lambda x: F.rms_norm(x, (3,)))
    gn = [((i * 7) % 17 - 8) * 0.25 for i in range(32)]
    w4, b4 = [0.5, -1.2, 2.0, 1.1], [0.1, 0.2, -0.3, 0.0]
    check_fn("group_norm", [(gn, [2, 4, 2, 2]), (w4, [4]), (b4, [4])], lambda x, w, b: FF.group_norm(x, 2, w, b), lambda x, w, b: F.group_norm(x, 2, w, b))
    check_fn("group_norm rank2", [(gn[:16], [4, 4])], lambda x: FF.group_norm(x, 2), lambda x: F.group_norm(x, 2))

    for training in (True, False):
        fm, fv = ft([0.1, -0.2, 0.3, 0.0], [4]), ft([1.0, 0.5, 2.0, 1.5], [4])
        tm, tv = tt([0.1, -0.2, 0.3, 0.0], [4]), tt([1.0, 0.5, 2.0, 1.5], [4])
        check_fn(f"batch_norm training={training}", [(gn, [2, 4, 2, 2]), (w4, [4]), (b4, [4])],
                 lambda x, w, b: FF.batch_norm(x, fm, fv, w, b, training, 0.1, 1e-5),
                 lambda x, w, b: F.batch_norm(x, tm, tv, w, b, training, 0.1, 1e-5))
        check(f"batch_norm running_mean training={training}", fm, tm)
        check(f"batch_norm running_var training={training}", fv, tv)

    ids = [[1, 3, 1], [0, 4, 3]]
    table = [((i * 3) % 7 - 3) * 0.5 for i in range(15)]
    check_fn("embedding", [(table, [5, 3])], lambda w: FF.embedding(ferro.Tensor.from_i64(sum(ids, []), [2, 3]), w),
             lambda w: F.embedding(torch.tensor(ids), w))
    check("one_hot", FF.one_hot(ferro.Tensor.from_i64([2, 0, 1], [3]), 4), F.one_hot(torch.tensor([2, 0, 1]), 4).float())

    q = [((i * 7) % 11 - 5) * 0.2 for i in range(48)]
    k = [((i * 5) % 9 - 4) * 0.25 for i in range(48)]
    v = [((i * 3) % 13 - 6) * 0.15 for i in range(48)]
    for causal in (False, True):
        check_fn(f"sdpa rank3 causal={causal}", [(q, [2, 6, 4]), (k, [2, 6, 4]), (v, [2, 6, 4])],
                 lambda a, b, c: FF.scaled_dot_product_attention(a, b, c, is_causal=causal),
                 lambda a, b, c: F.scaled_dot_product_attention(a, b, c, is_causal=causal))
        check_fn(f"sdpa rank4 causal={causal}", [(q, [2, 2, 3, 4]), (k, [2, 2, 3, 4]), (v, [2, 2, 3, 4])],
                 lambda a, b, c: FF.scaled_dot_product_attention(a, b, c, is_causal=causal),
                 lambda a, b, c: F.scaled_dot_product_attention(a, b, c, is_causal=causal))
    check_fn("rope_cached", [(q, [4, 3, 4])], lambda x: x.rope_cached(), lambda x: torch_rope(x, torch.arange(3)))
    check_fn("rope_cached base=500", [(q, [2, 6, 4])], lambda x: x.rope_cached(500.0), lambda x: torch_rope(x, torch.arange(6), 500.0))

    fx = ft(MIX, s)
    assert FF.dropout(fx, 0.5, training=False) is fx
    check("dropout p=0 identity", FF.dropout(fx, 0.0), tt(MIX, s))
    print("OK functional dropout eval/p=0 identity")


def extended_losses():
    import warnings
    import ferro.nn.functional as FF
    warnings.filterwarnings("ignore", message="reduction: 'mean' divides")
    s = SHAPE
    logp = lambda x: x.log_softmax(1)
    cases = [
        ("l1_loss", [(MIX, s), (OTHER, s)], FF.l1_loss, F.l1_loss),
        ("huber_loss", [(MIX, s), (OTHER, s)], lambda a, b, r: FF.huber_loss(a, b, r, 2.0), lambda a, b, reduction: F.huber_loss(a, b, reduction=reduction, delta=2.0)),
        ("smooth_l1_loss", [(MIX, s), (OTHER, s)], lambda a, b, r: FF.smooth_l1_loss(a, b, r, 2.0), lambda a, b, reduction: F.smooth_l1_loss(a, b, reduction=reduction, beta=2.0)),
        ("binary_cross_entropy", [(PROB, s)], lambda a, r: FF.binary_cross_entropy(a, ft(SIGNS, s).relu(), reduction=r),
         lambda a, reduction: F.binary_cross_entropy(a, tt(SIGNS, s).relu(), reduction=reduction)),
        ("binary_cross_entropy_with_logits", [(MIX, s), (PROB, s)], lambda a, b, r: FF.binary_cross_entropy_with_logits(a, b, reduction=r), F.binary_cross_entropy_with_logits),
        ("poisson_nll_loss", [(MIX, s)], lambda a, r: FF.poisson_nll_loss(a, ft([1.0, 0.0, 3.0, 2.0, 1.0, 4.0], s), reduction=r),
         lambda a, reduction: F.poisson_nll_loss(a, tt([1.0, 0.0, 3.0, 2.0, 1.0, 4.0], s), reduction=reduction)),
        ("soft_margin_loss", [(MIX, s)], lambda a, r: FF.soft_margin_loss(a, ft(SIGNS, s), r), lambda a, reduction: F.soft_margin_loss(a, tt(SIGNS, s), reduction=reduction)),
        ("hinge_embedding_loss", [(POS, s)], lambda a, r: FF.hinge_embedding_loss(a, ft(SIGNS, s), 1.0, r),
         lambda a, reduction: F.hinge_embedding_loss(a, tt(SIGNS, s), 1.0, reduction=reduction)),
        ("margin_ranking_loss", [(MIX, s), (OTHER, s)], lambda a, b, r: FF.margin_ranking_loss(a, b, ft(SIGNS, s), 0.1, r),
         lambda a, b, reduction: F.margin_ranking_loss(a, b, tt(SIGNS, s), 0.1, reduction=reduction)),
        ("cosine_embedding_loss", [(MIX, s), (OTHER, s)], lambda a, b, r: FF.cosine_embedding_loss(a, b, ft([1.0, -1.0], [2]), -0.9, r),
         lambda a, b, reduction: F.cosine_embedding_loss(a, b, tt([1.0, -1.0], [2]), -0.9, reduction=reduction)),
        ("triplet_margin_loss", [(MIX, s), (OTHER, s), (ROUNDING, s)], lambda a, b, c, r: FF.triplet_margin_loss(a, b, c, 2.0, reduction=r),
         lambda a, b, c, reduction: F.triplet_margin_loss(a, b, c, 2.0, reduction=reduction)),
        ("cross_entropy indices", [(MIX, s)], lambda a, r: FF.cross_entropy(a, ferro.Tensor.from_i64([2, 0], [2]), r),
         lambda a, reduction: F.cross_entropy(a, torch.tensor([2, 0]), reduction=reduction)),
        ("cross_entropy probabilities", [(MIX, s)], lambda a, r: FF.cross_entropy(a, ft(OTHER, s).softmax(1), r),
         lambda a, reduction: F.cross_entropy(a, tt(OTHER, s).softmax(1), reduction=reduction)),
        ("nll_loss", [(MIX, s)], lambda a, r: FF.nll_loss(logp(a), ferro.Tensor.from_i64([2, 0], [2]), r),
         lambda a, reduction: F.nll_loss(logp(a), torch.tensor([2, 0]), reduction=reduction)),
    ]
    for reduction in ("mean", "sum"):
        for name, args, fop, top in cases:
            check_fn(f"{name} {reduction}", args, lambda *a: fop(*a, reduction), lambda *a: top(*a, reduction=reduction))
    for reduction in ("mean", "sum", "batchmean"):
        check_fn(f"kl_div {reduction}", [(MIX, s)], lambda a: FF.kl_div(logp(a), ft(OTHER, s).softmax(1), reduction),
                 lambda a: F.kl_div(logp(a), tt(OTHER, s).softmax(1), reduction=reduction))


def load(param, t):
    with ferro.no_grad():
        param.tensor().copy_(from_torch(t))


def check_module(name, fmod, tmod, params, inputs):
    """Copy torch parameters in, then compare output, input grads and param grads."""
    for fp, tp in params:
        load(fp, tp)
    check_fn(name, inputs, fmod, tmod)
    for i, (fp, tp) in enumerate(params):
        check(f"{name} param grad {i}", fp.grad, tp.grad)


def torch_self_attention(x, wq, wk, wv, wo, heads, kv, base, causal):
    b, s, d = x.shape
    hd = d // heads
    q, k, v = (x @ wq.T).view(b, s, heads, hd).transpose(1, 2), (x @ wk.T).view(b, s, kv, hd).transpose(1, 2), (x @ wv.T).view(b, s, kv, hd).transpose(1, 2)
    if base is not None:
        q = torch_rope(q.reshape(b * heads, s, hd), torch.arange(s), base).view(b, heads, s, hd)
        k = torch_rope(k.reshape(b * kv, s, hd), torch.arange(s), base).view(b, kv, s, hd)
    k, v = k.repeat_interleave(heads // kv, dim=1), v.repeat_interleave(heads // kv, dim=1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return o.transpose(1, 2).reshape(b, s, d) @ wo.T


def extended_modules():
    nn = ferro.nn
    torch.manual_seed(0)
    data = lambda n, a=7, m=11, sc=0.25: [((i * a) % m - m // 2) * sc for i in range(n)]

    tm = torch.nn.Conv2d(4, 6, 3, stride=2, padding=1, groups=2)
    fm = nn.Conv2d(4, 6, 3, stride=2, padding=1, groups=2)
    check_module("Conv2d groups/stride/padding", fm, tm, [(fm.weight, tm.weight), (fm.bias, tm.bias)], [(data(200), [2, 4, 5, 5])])
    tm = torch.nn.Conv2d(2, 3, 2, dilation=2, bias=False)
    fm = nn.Conv2d(2, 3, 2, dilation=2, bias=False)
    assert fm.bias is None
    check_module("Conv2d dilation no-bias", fm, tm, [(fm.weight, tm.weight)], [(data(50), [1, 2, 5, 5])])

    for fcls, tcls, shape in [(nn.BatchNorm2d, torch.nn.BatchNorm2d, [2, 3, 2, 2]), (nn.BatchNorm1d, torch.nn.BatchNorm1d, [4, 3])]:
        fm, tm = fcls(3, momentum=0.2), tcls(3, momentum=0.2)
        torch.nn.init.normal_(tm.weight)
        torch.nn.init.normal_(tm.bias)
        n = math.prod(shape)
        for step in range(2):
            check_module(f"{fcls.__name__} train step {step}", fm, tm, [(fm.weight, tm.weight), (fm.bias, tm.bias)], [(data(n, 5 + step, 13), shape)])
            fm.weight.zero_grad(); fm.bias.zero_grad(); tm.zero_grad()
            check(f"{fcls.__name__} running_mean step {step}", fm.running_mean, tm.running_mean)
            check(f"{fcls.__name__} running_var step {step}", fm.running_var, tm.running_var)
        fm.eval(); tm.eval()
        check_module(f"{fcls.__name__} eval", fm, tm, [(fm.weight, tm.weight), (fm.bias, tm.bias)], [(data(n, 3, 7), shape)])

    for name, fm, tm, shape in [
        ("LayerNorm", nn.LayerNorm(4), torch.nn.LayerNorm(4), [3, 4]),
        ("LayerNorm rank3 no-bias", nn.LayerNorm(4, bias=False), torch.nn.LayerNorm(4, bias=False), [2, 3, 4]),
        ("RMSNorm", nn.RMSNorm(4), torch.nn.RMSNorm(4), [2, 3, 4]),
        ("RMSNorm eps", nn.RMSNorm([4], eps=1e-5), torch.nn.RMSNorm([4], eps=1e-5), [3, 4]),
        ("GroupNorm", nn.GroupNorm(2, 4), torch.nn.GroupNorm(2, 4), [2, 4, 2, 2]),
    ]:
        params = [(fp, tp) for (_, fp), tp in zip(fm.named_parameters(), tm.parameters())]
        for _, tp in params:
            torch.nn.init.normal_(tp)
        check_module(name, fm, tm, params, [(data(math.prod(shape)), shape)])
    fm, tm = nn.LayerNorm(4, elementwise_affine=False), torch.nn.LayerNorm(4, elementwise_affine=False)
    assert list(fm.parameters()) == []
    check_module("LayerNorm no affine", fm, tm, [], [(data(12), [3, 4])])

    ids = [[1, 3, 1], [0, 4, 3]]
    fm, tm = nn.Embedding(5, 3), torch.nn.Embedding(5, 3)
    load(fm.weight, tm.weight)
    fy, ty = fm(ferro.Tensor.from_i64(sum(ids, []), [2, 3])), tm(torch.tensor(ids))
    check("Embedding value", fy, ty)
    (fy * fy).sum().backward()
    (ty * ty).sum().backward()
    check("Embedding weight grad", fm.weight.grad, tm.weight.grad)

    pool_in = [(((i * 5) % 32) - 16) * 0.5 for i in range(32)]
    for fm, tm in [(nn.MaxPool2d(2), torch.nn.MaxPool2d(2)), (nn.AvgPool2d(3, 1), torch.nn.AvgPool2d(3, 1)), (nn.MaxPool2d((2, 2), stride=1), torch.nn.MaxPool2d(2, 1))]:
        check_fn(type(fm).__name__, [(pool_in, [1, 2, 4, 4])], fm, tm)

    tm = torch.nn.MultiheadAttention(8, 2, batch_first=True)
    torch.nn.init.normal_(tm.in_proj_bias)
    torch.nn.init.normal_(tm.out_proj.bias)
    fm = nn.MultiheadAttention(8, 2)
    params = [(fm.in_proj_weight, tm.in_proj_weight), (fm.in_proj_bias, tm.in_proj_bias), (fm.out_proj.weight, tm.out_proj.weight), (fm.out_proj.bias, tm.out_proj.bias)]
    inputs = [(data(48, 7, 11, 0.2), [2, 3, 8]), (data(64, 5, 9), [2, 4, 8]), (data(64, 3, 13, 0.15), [2, 4, 8])]
    check_module("MultiheadAttention cross need_weights=False", lambda q, k, v: fm(q, k, v, need_weights=False)[0],
                 lambda q, k, v: tm(q, k, v, need_weights=False)[0], params, inputs)
    for fp, tp in params:
        fp.zero_grad(); tp.grad = None
    check_module("MultiheadAttention weights", lambda q, k, v: fm(q, k, v)[1], lambda q, k, v: tm(q, k, v)[1], params[:2], inputs)

    d, heads = 8, 4
    for kv, base, causal in [(4, None, False), (2, 10000.0, True), (1, 500.0, True)]:
        fm = nn.SelfAttention(d, heads, num_kv_heads=kv, causal=causal, rope_base=base)
        ws = [torch.randn(d, d) * 0.3, torch.randn(kv * d // heads, d) * 0.3, torch.randn(kv * d // heads, d) * 0.3, torch.randn(d, d) * 0.3]
        for w in ws:
            w.requires_grad_(True)
        mods = [fm.q_proj, fm.k_proj, fm.v_proj, fm.o_proj]
        check_module(f"SelfAttention kv={kv} rope={base} causal={causal}", fm,
                     lambda x: torch_self_attention(x, *ws, heads, kv, base, causal),
                     [(m.weight, w) for m, w in zip(mods, ws)], [(data(2 * 5 * d), [2, 5, d])])

    fm = nn.TransformerBlock(d, 2)
    tp = {n: (torch.randn(*p.tensor().shape) * 0.3).requires_grad_(True) for n, p in fm.named_parameters()}

    def torch_block(x):
        a = torch_self_attention(F.rms_norm(x, (d,), tp["norm1.weight"], 1e-5), tp["attn.q_proj.weight"], tp["attn.k_proj.weight"],
                                 tp["attn.v_proj.weight"], tp["attn.o_proj.weight"], 2, 2, 10000.0, True)
        h = x + a
        m = F.rms_norm(h, (d,), tp["norm2.weight"], 1e-5)
        return h + F.linear(F.gelu(F.linear(m, tp["up.weight"], tp["up.bias"]), approximate="tanh"), tp["down.weight"], tp["down.bias"])

    check_module("TransformerBlock", fm, torch_block, [(p, tp[n]) for n, p in fm.named_parameters()], [(data(2 * 5 * d), [2, 5, d])])

    acts = [
        (nn.ReLU(), torch.nn.ReLU()), (nn.Sigmoid(), torch.nn.Sigmoid()), (nn.Tanh(), torch.nn.Tanh()),
        (nn.GELU(), torch.nn.GELU()), (nn.GELU("tanh"), torch.nn.GELU("tanh")), (nn.SiLU(), torch.nn.SiLU()),
        (nn.Mish(), torch.nn.Mish()), (nn.ELU(0.5), torch.nn.ELU(0.5)), (nn.CELU(0.5), torch.nn.CELU(0.5)),
        (nn.SELU(), torch.nn.SELU()), (nn.LeakyReLU(0.1), torch.nn.LeakyReLU(0.1)), (nn.ReLU6(), torch.nn.ReLU6()),
        (nn.Hardtanh(-0.5, 1.5), torch.nn.Hardtanh(-0.5, 1.5)), (nn.Hardsigmoid(), torch.nn.Hardsigmoid()),
        (nn.Hardswish(), torch.nn.Hardswish()), (nn.Hardshrink(0.6), torch.nn.Hardshrink(0.6)),
        (nn.Softshrink(0.6), torch.nn.Softshrink(0.6)), (nn.Tanhshrink(), torch.nn.Tanhshrink()),
        (nn.Softplus(), torch.nn.Softplus()), (nn.Softplus(beta=3.0), torch.nn.Softplus(beta=3.0)),
        (nn.Softsign(), torch.nn.Softsign()), (nn.LogSigmoid(), torch.nn.LogSigmoid()),
        (nn.Threshold(0.4, -2.0), torch.nn.Threshold(0.4, -2.0)), (nn.Softmax(1), torch.nn.Softmax(1)),
        (nn.LogSoftmax(-1), torch.nn.LogSoftmax(-1)), (nn.Softmin(0), torch.nn.Softmin(0)),
        (nn.Flatten(), torch.nn.Flatten()), (nn.Identity(), torch.nn.Identity()),
    ]
    for fm, tm in acts:
        check_fn(f"module {type(tm).__name__}", [(MIX, SHAPE)], fm, tm)

    s = SHAPE
    losses = [
        (nn.L1Loss(), torch.nn.L1Loss(), [(MIX, s), (OTHER, s)]),
        (nn.L1Loss(reduction="sum"), torch.nn.L1Loss(reduction="sum"), [(MIX, s), (OTHER, s)]),
        (nn.HuberLoss(delta=2.0), torch.nn.HuberLoss(delta=2.0), [(MIX, s), (OTHER, s)]),
        (nn.SmoothL1Loss(beta=2.0), torch.nn.SmoothL1Loss(beta=2.0), [(MIX, s), (OTHER, s)]),
        (nn.BCEWithLogitsLoss(), torch.nn.BCEWithLogitsLoss(), [(MIX, s), (PROB, s)]),
        (nn.BCELoss(), torch.nn.BCELoss(), [(PROB, s), ([0.3, 0.8, 0.5, 0.0, 1.0, 0.25], s)]),
        (nn.MarginRankingLoss(margin=0.1), torch.nn.MarginRankingLoss(margin=0.1), [(MIX, s), (OTHER, s), (SIGNS, s)]),
        (nn.TripletMarginLoss(margin=2.0), torch.nn.TripletMarginLoss(margin=2.0), [(MIX, s), (OTHER, s), (ROUNDING, s)]),
    ]
    losses += [
        (nn.L1Loss(reduction="sum"), torch.nn.L1Loss(reduction="sum"), [([0.5, -1.0], [2, 1]), (MIX, s)]),
        (nn.HuberLoss(reduction="sum"), torch.nn.HuberLoss(reduction="sum"), [([0.5, -1.0], [2, 1]), (MIX, s)]),
    ]
    for fm, tm, args in losses:
        check_fn(f"module {type(tm).__name__} {tm.reduction}", args, fm, tm, grad=False)
    check("module CrossEntropyLoss", nn.CrossEntropyLoss()(ft(MIX, s), ferro.Tensor.from_i64([2, 0], [2])),
          torch.nn.CrossEntropyLoss()(tt(MIX, s), torch.tensor([2, 0])))
    check("module KLDivLoss batchmean", nn.KLDivLoss(reduction="batchmean")(ft(MIX, s).log_softmax(1), ft(OTHER, s).softmax(1)),
          torch.nn.KLDivLoss(reduction="batchmean")(tt(MIX, s).log_softmax(1), tt(OTHER, s).softmax(1)))


def view_ops():
    v = [((i * 5) % 11 - 5) * 0.37 + 0.1 for i in range(24)]
    s = [2, 3, 4]
    mask = [1.0 if i % 3 == 0 else 0.0 for i in range(12)]
    cases = [
        ("permute", lambda x: x.permute(2, 0, 1), lambda x: x.permute(2, 0, 1)),
        ("permute neg", lambda x: x.permute(-1, -3, -2), lambda x: x.permute(-1, -3, -2)),
        ("narrow", lambda x: x.narrow(2, 1, 2), lambda x: x.narrow(2, 1, 2)),
        ("narrow neg start", lambda x: x.narrow(-1, -3, 2), lambda x: x.narrow(-1, -3, 2)),
        ("getitem slices", lambda x: x[:, 1:, ::2], lambda x: x[:, 1:, ::2]),
        ("getitem int+slice", lambda x: x[1, :, -3:], lambda x: x[1, :, -3:]),
        ("getitem ellipsis", lambda x: x[..., 1], lambda x: x[..., 1]),
        ("getitem neg step", lambda x: x[:, ::-1], lambda x: x.flip(1)),
        ("split size", lambda x: ferro.cat(x.split(3, 2)[::-1], 2), lambda x: torch.cat(x.split(3, 2)[::-1], 2)),
        ("split sections", lambda x: ferro.cat(x.split([1, 2], 1)[::-1], 1), lambda x: torch.cat(x.split([1, 2], 1)[::-1], 1)),
        ("chunk", lambda x: ferro.cat(x.chunk(3, -1), 0), lambda x: torch.cat(x.chunk(3, -1), 0)),
        ("expand", lambda x: x[:, :1, :].expand(2, 5, 4), lambda x: x[:, :1, :].expand(2, 5, 4)),
        ("expand -1 new dim", lambda x: x[:1].expand(3, -1, -1, -1), lambda x: x[:1].expand(3, -1, -1, -1)),
        ("repeat", lambda x: x.repeat(2, 1, 3), lambda x: x.repeat(2, 1, 3)),
        ("flip", lambda x: x.flip(0, 2), lambda x: x.flip(0, 2)),
        ("roll", lambda x: x.roll(1, 2), lambda x: x.roll(1, 2)),
        ("roll multi", lambda x: x.roll([1, -2], [0, 2]), lambda x: x.roll((1, -2), (0, 2))),
        ("roll flat", lambda x: x.roll(5), lambda x: x.roll(5)),
        ("stack", lambda x: ferro.stack([x, x * 2.0], 1), lambda x: torch.stack([x, x * 2.0], 1)),
        ("stack neg", lambda x: ferro.stack([x, x], -1), lambda x: torch.stack([x, x], -1)),
        ("masked_fill", lambda x: x.masked_fill(ft(mask, [3, 4]), -9.0), lambda x: x.masked_fill(tt(mask, [3, 4]).bool(), -9.0)),
    ]
    for name, f, t in cases:
        check_fn(name, [(v, s)], f, t)
    idx = ferro.Tensor.from_i64([2, 0, 2], [3])
    src = [0.5 * i - 2.0 for i in range(24)]
    check_fn("index_add", [(v, s), (src, [2, 3, 4])],
             lambda x, y: x.index_add(1, idx, y), lambda x, y: x.index_add(1, torch.tensor([2, 0, 2]), y))


if __name__ == "__main__":
    main()
