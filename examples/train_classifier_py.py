"""Train a 2-layer MLP classifier on the Rust backend, end to end from Python.

Two interleaved half-moon-ish clusters, cross-entropy composed from bound ops
(log_softmax/mul/sum_dim/mean), manual SGD on the leaf weights.
"""

import math

import ferro


def make_data(n_per_class=40):
    xs, targets = [], []
    for i in range(n_per_class):
        a = i / n_per_class * math.pi
        xs += [math.cos(a) + 0.05 * math.sin(7 * a), math.sin(a) - 0.05 * math.cos(5 * a)]
        targets += [1.0, 0.0]
        xs += [1.0 - math.cos(a) + 0.05 * math.cos(9 * a), 0.3 - math.sin(a) + 0.05 * math.sin(3 * a)]
        targets += [0.0, 1.0]
    n = 2 * n_per_class
    return ferro.Tensor(xs, [n, 2]), ferro.Tensor(targets, [n, 2]), n


def cross_entropy(logits, one_hot):
    return -(logits.log_softmax(1) * one_hot).sum_dim(1).mean()


def sgd_update(params, lr):
    updated = []
    for p in params:
        flat_p = [v for row in as_rows(p.tolist()) for v in row]
        flat_g = [v for row in as_rows(p.grad.tolist()) for v in row]
        new = [pv - lr * gv for pv, gv in zip(flat_p, flat_g)]
        updated.append(ferro.Tensor(new, list(p.shape)).requires_grad_(True))
    return updated


def as_rows(nested):
    return nested if isinstance(nested[0], list) else [nested]


def main():
    x, targets, n = make_data()
    hidden = 16
    w1 = ferro.Tensor.randn([2, hidden], seed=1).requires_grad_(True)
    b1 = ferro.Tensor.zeros([hidden]).requires_grad_(True)
    w2 = ferro.Tensor.randn([hidden, 2], seed=2).requires_grad_(True)
    b2 = ferro.Tensor.zeros([2]).requires_grad_(True)

    lr = 0.5
    first = last = None
    for epoch in range(300):
        logits = (x.matmul(w1) + b1).relu().matmul(w2) + b2
        loss = cross_entropy(logits, targets)
        loss.backward()
        w1, b1, w2, b2 = sgd_update([w1, b1, w2, b2], lr)
        if first is None:
            first = loss.item()
        last = loss.item()
        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

    logits = (x.matmul(w1) + b1).relu().matmul(w2) + b2
    probs = logits.softmax(1).tolist()
    labels = targets.tolist()
    correct = sum((p[0] > p[1]) == (t[0] > t[1]) for p, t in zip(probs, labels))
    acc = correct / n
    print(f"loss: {first:.4f} -> {last:.4f}   train accuracy: {acc:.1%}")
    assert last < first * 0.2, "loss did not converge"
    assert acc >= 0.95, f"accuracy too low: {acc}"
    print("CLASSIFIER TRAINED ON THE RUST BACKEND")


if __name__ == "__main__":
    main()
