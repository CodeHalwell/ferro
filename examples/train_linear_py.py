"""Pure-ferro linear regression demo: manual SGD, no torch.

Fits w in y = X @ w for a known target weight using MSE loss. Prints the loss
each epoch so you can watch it decrease.
"""

import ferro


def make_data(n, d, true_w):
    # Deterministic pseudo-random features, targets from the true weight.
    xs = ferro.Tensor.randn([n, d], seed=0)
    w = ferro.Tensor(true_w, [d, 1])
    ys = xs.matmul(w)
    return xs, ys


def main():
    n, d = 64, 3
    true_w = [2.0, -3.0, 0.5]
    xs, ys = make_data(n, d, true_w)

    w = ferro.Tensor.randn([d, 1], seed=42).requires_grad_(True)
    lr = 0.1

    losses = []
    for epoch in range(100):
        pred = xs.matmul(w)
        err = pred - ys
        loss = (err * err).mean()
        loss.backward()

        grad = w.grad.tolist()
        # SGD step: rebuild the leaf weight from updated values.
        new_w = [row[0] - lr * grad[i][0] for i, row in enumerate(w.tolist())]
        w = ferro.Tensor(new_w, [d, 1]).requires_grad_(True)

        losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 99:
            print(f"epoch {epoch:3d}  loss {loss.item():.6f}")

    print(f"\nlearned w: {[round(v[0], 4) for v in w.tolist()]}")
    print(f"target  w: {true_w}")
    print(f"loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    assert losses[-1] < losses[0], "loss did not decrease"


if __name__ == "__main__":
    main()
