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

import resource
import sys

import numpy as np

import ferro


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


def main():
    test_requires_grad_inplace()
    test_backward_nonscalar_raises()
    test_item_nonscalar_raises()
    test_dlpack_roundtrip()
    test_dlpack_export_no_leak()
    test_grad_accumulation()
    test_neg()
    print("ALL REGRESSION CHECKS PASSED")


if __name__ == "__main__":
    main()
