"""Regression: Tensor.fuse() must collapse a pointwise chain to one launch and
match the eager result on CPU and (when present) CUDA.

Run: python examples/py_fuse_regression.py  (from repo root, ferro-py venv).
Exits non-zero on any mismatch so it can gate CI.
"""

import sys

import numpy as np


def check(dev, ferro):
    x = ferro.Tensor.randn([512, 512], seed=1, device=dev).requires_grad_(True)
    y = ferro.Tensor.randn([512, 512], seed=2, device=dev).requires_grad_(True)
    z = ferro.Tensor.randn([512, 512], seed=3, device=dev).requires_grad_(True)

    lb, la = (x.relu() * y + z).fusion_launches()
    assert lb == 3, f"{dev}: expected 3 eager launches, got {lb}"
    assert la == 1, f"{dev}: expected 1 fused launch, got {la}"

    xn = np.asarray(x.cpu().tolist(), dtype="float32")
    yn = np.asarray(y.cpu().tolist(), dtype="float32")
    zn = np.asarray(z.cpu().tolist(), dtype="float32")
    ref = np.maximum(xn, 0.0) * yn + zn

    eager = np.asarray((x.relu() * y + z).cpu().tolist(), dtype="float32")
    fused = np.asarray((x.relu() * y + z).fuse().cpu().tolist(), dtype="float32")

    d_ref = float(np.max(np.abs(fused - ref)))
    d_eager = float(np.max(np.abs(fused - eager)))
    tol = 1e-4
    assert d_ref < tol, f"{dev}: fused vs ref {d_ref:.2e} >= {tol}"
    assert d_eager < tol, f"{dev}: fused vs eager {d_eager:.2e} >= {tol}"
    print(f"OK {dev}: launches {lb}->{la}  max|fused-ref|={d_ref:.2e}  max|fused-eager|={d_eager:.2e}")


def main():
    import ferro
    check("cpu", ferro)
    if ferro.cuda_is_available():
        ferro.cuda_init(0)
        check("cuda:0", ferro)
    else:
        print("SKIP cuda:0 (no device)")
    print("ALL FUSE REGRESSION CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
