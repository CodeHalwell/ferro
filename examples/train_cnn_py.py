"""Train a tiny CNN classifier on the Rust backend, end to end from Python.

Synthetic 6x6 single-channel images: class A has a bright 2x2 top-left block,
class B a bright 2x2 bottom-right block, plus small deterministic sin noise.
Model: conv2d(1->4, 3x3, pad 1) -> relu -> max_pool2d(2,2) -> reshape ->
linear -> logits. Cross-entropy composed from log_softmax/mul/sum_dim/mean,
manual SGD on the leaf weights.
"""

import math

import ferro


def make_data(n_per_class=20):
    xs, targets = [], []
    idx = 0
    for _ in range(n_per_class):
        for cls in (0, 1):
            img = [[0.1 * math.sin(3.1 * idx + 6 * r + c) for c in range(6)] for r in range(6)]
            block = (0, 1) if cls == 0 else (4, 5)
            for r in block:
                for c in block:
                    img[r][c] += 1.0
            xs += [v for row in img for v in row]
            targets += [1.0, 0.0] if cls == 0 else [0.0, 1.0]
            idx += 1
    n = 2 * n_per_class
    return ferro.Tensor(xs, [n, 1, 6, 6]), ferro.Tensor(targets, [n, 2]), n


def flatten(nested):
    if not isinstance(nested, list):
        return [nested]
    return [v for item in nested for v in flatten(item)]


def cross_entropy(logits, one_hot):
    return -(logits.log_softmax(1) * one_hot).sum_dim(1).mean()


def sgd_update(params, lr):
    updated = []
    for p in params:
        flat_p = flatten(p.tolist())
        flat_g = flatten(p.grad.tolist())
        new = [pv - lr * gv for pv, gv in zip(flat_p, flat_g)]
        updated.append(ferro.Tensor(new, list(p.shape)).requires_grad_(True))
    return updated


def scaled_randn(shape, seed, scale):
    raw = flatten(ferro.Tensor.randn(shape, seed=seed).tolist())
    return ferro.Tensor([scale * v for v in raw], shape).requires_grad_(True)


def forward(x, conv_w, w, b, n):
    h = x.conv2d(conv_w, stride=1, padding=1).relu().max_pool2d(2, 2)
    return h.reshape([n, 4 * 3 * 3]).matmul(w) + b


def main():
    x, targets, n = make_data()
    conv_w = scaled_randn([4, 1, 3, 3], seed=1, scale=0.5)
    w = scaled_randn([36, 2], seed=2, scale=0.3)
    b = ferro.Tensor.zeros([2]).requires_grad_(True)

    lr = 0.1
    first = last = None
    for epoch in range(150):
        loss = cross_entropy(forward(x, conv_w, w, b, n), targets)
        loss.backward()
        conv_w, w, b = sgd_update([conv_w, w, b], lr)
        if first is None:
            first = loss.item()
        last = loss.item()
        if epoch % 25 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

    probs = forward(x, conv_w, w, b, n).softmax(1).tolist()
    labels = targets.tolist()
    correct = sum((p[0] > p[1]) == (t[0] > t[1]) for p, t in zip(probs, labels))
    acc = correct / n
    print(f"loss: {first:.4f} -> {last:.4f}   train accuracy: {acc:.1%}")
    assert last < first * 0.1, "loss did not converge"
    assert acc >= 0.95, f"accuracy too low: {acc}"
    print("CNN TRAINED ON THE RUST BACKEND")


if __name__ == "__main__":
    main()
