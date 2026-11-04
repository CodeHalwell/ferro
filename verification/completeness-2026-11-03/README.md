# Verified candidate32 checkpoint

This checkpoint is prepared for [PR #33](https://github.com/CodeHalwell/ferro/pull/33).
Full PyTorch/PyG parity remains open. References: Torch 2.13.0+cpu and PyG 2.8.1.
The zero-dependency Rust core and single record_fn/Op engine are preserved.

- Python: 1,807 passed; 30 GPU cases unavailable (1,831 executions, 1,837 discovered).
- Native: 1,702 outer runtime passes and two doctests; 19 nested summaries excluded.
- CUDA all-target compilation, DLPack checks and eight standalones pass.
- All 74 focused methods pass across the 13 changed/new test modules.

The 1,133 selected inputs and 124 installed files match the validated artifacts.
The runtime wheel was built from proof1. Before public execution, a test-only
correction normalized two stride assertions to tuple comparisons. Proof2 reuses
the identical wheel under exact native/packaged-source equality; its complete
Python, focused, DLPack and standalone runs pass. No earlier failed public/full
run is claimed. Detailed hashes and commands are retained in manifest.json.

Added: physical-batch Jacobians and indexed AD; shared-output PReLU higher/JVP
rules; ownership-aware empty cotangent accumulation; dependency-ready backward
scheduling by immutable creation sequence; preserve-format unused gradients;
power-zero JVP layout repair; CPU F64 Conv1d/Conv2d with typed training/restart.

No GPU hardware execution is established. Metal is not implemented. Public
batched gradients, additional F64 optimizers/cells/pooling and Hessian follow-ups
remain isolated. Capture, dtype and higher-order coverage is operator-specific.

The prior [candidate31](candidate31/README.md) and earlier checkpoints retain their
reports and failed-attempt provenance. See the
[compatibility ledger](../../docs/PYTORCH_PYG_COMPATIBILITY.md) and
[Rust CUDA research](../../docs/RUST_CUDA.md).
