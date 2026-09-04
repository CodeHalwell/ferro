"""Regression: Tensor.compile_fused() must plan once, replay to one launch, and
match the eager result on CPU and (when present) CUDA.

Run: python examples/py_compiled_fusion_regression.py  (ferro-py venv, from repo root)
"""
import sys
import ferro


def check(device):
    n = 4096
    x = ferro.Tensor.randn([n], device=device).requires_grad_(True)
    y = ferro.Tensor.randn([n], device=device)
    z = ferro.Tensor.randn([n], device=device)
    expr = (x.relu() * y + z)
    before, after = expr.fusion_launches()
    assert before == 3, f"expected 3 eager launches, got {before}"
    assert after == 1, f"expected 1 fused launch, got {after}"

    handle = expr.compile_fused()
    assert handle.num_steps >= 1, "compiled handle has no steps"

    eager = (x.relu() * y + z)
    rep = handle.replay()
    a = eager.cpu().tolist()
    b = rep.cpu().tolist()
    mad = max(abs(pa - pb) for pa, pb in zip(a, b))
    tol = 0.0 if device == "cpu" else 1e-4
    assert mad <= tol, f"{device}: replay mismatch max|Δ|={mad} > {tol}"

    # Replaying twice must be stable and identical.
    rep2 = handle.replay()
    mad2 = max(abs(pb - pc) for pb, pc in zip(b, rep2.cpu().tolist()))
    assert mad2 == 0.0, f"{device}: replay not deterministic, Δ={mad2}"
    print(f"OK {device}: launches {before}->{after}, steps={handle.num_steps}, "
          f"operands={handle.num_operands}, max|Δ|={mad:.2e}")


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
