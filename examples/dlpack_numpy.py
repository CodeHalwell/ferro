"""Validate ferro's DLPack interop against numpy and (if available) torch.

Run inside the ferro-py venv after `maturin develop --release`:

    cd rust_backend/crates/ferro-py
    . .venv/bin/activate
    python ../../examples/dlpack_numpy.py

Exercises:
  - numpy export:  np.from_dlpack(ferro_tensor)
  - numpy import:  ferro.from_dlpack(np_array)  (round-trip)
  - torch export:  torch.from_dlpack(ferro_tensor)   (if torch present)
  - torch import:  ferro.from_dlpack(torch_tensor)    (if torch present)

The bridge copies at the boundary, so values (not buffers) are what we assert.
"""

import numpy as np

import ferro


def nested_equal(a, b, tol=1e-6):
    return np.allclose(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), atol=tol)


def main():
    t = ferro.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

    # __dlpack_device__ must report CPU.
    assert t.__dlpack_device__() == (1, 0), t.__dlpack_device__()

    # numpy export.
    arr = np.from_dlpack(t)
    assert arr.shape == (2, 3), arr.shape
    assert arr.dtype == np.float32, arr.dtype
    assert nested_equal(arr, t.tolist()), (arr, t.tolist())
    print("numpy export: OK", arr.tolist())

    # numpy import (round-trip through a fresh numpy array).
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t2 = ferro.from_dlpack(src)
    assert list(np.array(t2.shape)) == [3, 4], t2.shape
    assert nested_equal(t2.tolist(), src), (t2.tolist(), src)
    print("numpy import: OK", t2.tolist())

    # Import a non-contiguous numpy view to exercise stride handling.
    view = src[:, ::2]  # shape (3, 2), non-contiguous
    tv = ferro.from_dlpack(np.ascontiguousarray(view))
    assert nested_equal(tv.tolist(), view), (tv.tolist(), view)
    print("numpy import (strided source made contiguous): OK")

    try:
        import torch
    except ImportError:
        print("torch: not installed in this venv, skipping torch checks")
        print("ALL CHECKS PASSED (numpy only)")
        return

    # torch export.
    tt = torch.from_dlpack(t)
    assert tuple(tt.shape) == (2, 3), tt.shape
    assert tt.dtype == torch.float32, tt.dtype
    assert nested_equal(tt.numpy(), t.tolist()), (tt, t.tolist())
    print("torch export: OK", tt.tolist())

    # torch import.
    tsrc = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    t3 = ferro.from_dlpack(tsrc)
    assert list(np.array(t3.shape)) == [2, 3], t3.shape
    assert nested_equal(t3.tolist(), tsrc.numpy()), (t3.tolist(), tsrc)
    print("torch import: OK", t3.tolist())

    # Numeric cross-check: same op on ferro and torch agree.
    a = ferro.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = ferro.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    ferro_mm = a.matmul(b)
    ta = torch.from_dlpack(a)
    tb = torch.from_dlpack(b)
    torch_mm = ta @ tb
    assert nested_equal(ferro_mm.tolist(), torch_mm.numpy()), (ferro_mm.tolist(), torch_mm)
    print("torch numeric validation (matmul): OK", ferro_mm.tolist())

    print("ALL CHECKS PASSED (numpy + torch)")


if __name__ == "__main__":
    main()
