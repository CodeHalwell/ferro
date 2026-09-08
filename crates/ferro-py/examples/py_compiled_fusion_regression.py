"""Compiled fusion regression. Run in the ferro-py venv from the repo root.

CPU and CUDA numeric/shape/graph-boundary tests; real launch counting lives in
core's fusion_exec.rs. CUDA mutation uses the public zero-copy DLPack export,
with both frameworks fenced. Optimizer mutation is covered in Rust on both devices.
"""
import ferro
import numpy as np


def close(a, b):
    assert a.shape == b.shape
    assert a.device == b.device
    np.testing.assert_allclose(a.cpu().tolist(), b.cpu().tolist(), rtol=1e-5, atol=1e-6)


def reject(expr, reason):
    try:
        expr.compile_fused()
    except ValueError as exc:
        assert reason in str(exc), str(exc)
    else:
        raise AssertionError("unsupported graph was silently accepted")


def check(device):
    def leaf(values, shape):
        return ferro.Tensor(values, shape).to(device).requires_grad_(True)

    x = leaf([-2., 3., 4., 5., -6., 7.], [2, 3])
    y = leaf([2., 4., 8.], [3])
    z = leaf([1., 2., 3.], [3])
    build = lambda: (x.relu() - y) / z
    root = build()
    handle = root.compile_fused()
    assert handle.num_steps == 3, "compiled handle must include the first relu"
    assert handle.num_operands == 3
    close(handle.replay(), root)
    close(handle.replay(), handle.replay())
    assert not handle.replay().requires_grad

    for expr in (x * y + z, x * x - x, x.relu() * y + z):
        close(expr.compile_fused().replay(), expr)
    assert (x * x - x).compile_fused().num_operands == 1
    reject(y.relu() * x + z, "seed-expanding")
    reject(x.relu() * y.relu() + z, "side branches")
    reject(x - x.relu(), "side branches")
    reject(x / x.relu(), "side branches")
    u = x.relu()
    reject(u * u, "side branches")
    reject(x.sum().relu(), "not supported")
    reject(x, "no recorded operations")

    if device != "cpu":
        import torch
        ferro.cuda_synchronize()
        aliases = [torch.from_dlpack(t) for t in (x, y, z)]
        old = handle.replay().cpu().tolist()
        for _ in range(3):
            ferro.cuda_synchronize()
            for t in aliases:
                t.add_(0.25)
            torch.cuda.synchronize()
            close(handle.replay(), build())
        assert handle.replay().cpu().tolist() != old
    print(f"OK {device}: full first op, binary-first, right broadcast, duplicates, "
          "noncommutative order, safe graph rejection" +
          (", current upstream leaves" if device != "cpu" else ""))


def main():
    check("cpu")
    if ferro.cuda_is_available():
        ferro.cuda_init(0)
        check("cuda:0")
    else:
        print("CUDA not available; skipped GPU check")
    print("py_compiled_fusion_regression: PASS")


if __name__ == "__main__":
    main()
