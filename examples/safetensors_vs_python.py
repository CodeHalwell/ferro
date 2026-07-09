"""Cross-validate ferro's safetensors read/write against the reference
Python implementation (the `safetensors` package, via its torch loader).

Run inside the ferro-py venv: python examples/safetensors_vs_python.py
"""

import os
import tempfile

import torch
from safetensors.torch import load_file, save_file

import ferro

F32 = [1.5, -2.25, 0.0, 3.125, -0.5, 42.0]
I64 = [-9007199254740993, -1, 0, 9007199254740993]  # exceed f64-exact ints
SHAPE = [2, 3]


def check(name, ok):
    assert ok, name
    print(f"OK {name}")


def main():
    tmp = tempfile.mkdtemp()

    # ferro writes, the reference implementation reads.
    ours = os.path.join(tmp, "ferro.safetensors")
    ferro.save_safetensors(
        ours,
        {"w": ferro.Tensor(F32, SHAPE), "ids": ferro.Tensor.from_i64(I64, [4])},
    )
    loaded = load_file(ours)
    check("ferro->python names", sorted(loaded) == ["ids", "w"])
    check("ferro->python f32 dtype", loaded["w"].dtype == torch.float32)
    check("ferro->python f32 values", torch.equal(loaded["w"], torch.tensor(F32).reshape(SHAPE)))
    check("ferro->python i64 dtype", loaded["ids"].dtype == torch.int64)
    check("ferro->python i64 values", loaded["ids"].tolist() == I64)

    # The reference implementation writes (with metadata), ferro reads.
    theirs = os.path.join(tmp, "torch.safetensors")
    save_file(
        {
            "w": torch.tensor(F32).reshape(SHAPE),
            "prec": torch.tensor([1e-300, -2.5, 3.75], dtype=torch.float64),
            "ids": torch.tensor(I64),
        },
        theirs,
        metadata={"format": "pt"},
    )
    got = ferro.load_safetensors(theirs)
    check("python->ferro names", sorted(got) == ["ids", "prec", "w"])
    check("python->ferro f32 shape", got["w"].shape == SHAPE)
    check("python->ferro f32 values", got["w"].tolist() == torch.tensor(F32).reshape(SHAPE).tolist())
    # tolist() casts to f32, so compare the f64/i64 payloads through a
    # byte-exact re-save instead of the lossy cast.
    resaved = os.path.join(tmp, "resaved.safetensors")
    ferro.save_safetensors(resaved, dict(got.items()))
    back = load_file(resaved)
    check("python->ferro f64 roundtrip", torch.equal(back["prec"], torch.tensor([1e-300, -2.5, 3.75], dtype=torch.float64)))
    check("python->ferro i64 roundtrip", back["ids"].tolist() == I64)

    # A loaded tensor is a live autograd leaf, not just data.
    x = got["w"].requires_grad_(True)
    (x * x).sum().backward()
    check("loaded tensor grads", x.grad.tolist() == (torch.tensor(F32).reshape(SHAPE) * 2).tolist())

    print("SAFETENSORS MATCHES THE REFERENCE IMPLEMENTATION")


if __name__ == "__main__":
    main()
