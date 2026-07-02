"""Numerical parity checks for the newly bound ferro ops against torch.

Run inside the ferro-py venv: python examples/ops_vs_torch.py
"""

import torch

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
    assert torch.allclose(torch.tensor(f.tolist()), t, atol=1e-5), (
        f"{name}: ferro={f.tolist()} torch={t.tolist()}"
    )
    print(f"OK {name}")


def check_unary(name, data, ferro_op, torch_op):
    check(f"{name} value", ferro_op(ft(data, SHAPE)), torch_op(tt(data, SHAPE)))
    fx = ft(data, SHAPE, requires_grad=True)
    ferro_op(fx).sum().backward()
    tx = tt(data, SHAPE, requires_grad=True)
    torch_op(tx).sum().backward()
    check(f"{name} grad", fx.grad, tx.grad)


def main():
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

    print("ALL OPS MATCH TORCH")


if __name__ == "__main__":
    main()
