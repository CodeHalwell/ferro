# Ferro foundations: review corrections and bounded CPU BMM containment

**MERGE BLOCK CLEARED by independently validated SOFTWARE BMM path containment. This is not a claim that the original root cause is resolved, a risk waiver, or authorization to merge/enable auto-merge. Re-review and hosted CI remain separate gates.**

> **Material CPU slowdown:** at [16,128,128,128], production scalar BMM measured **13.8991-13.9863 ms**, versus **0.9193-0.9850 ms** for explicit one-thread packed matmul separately per slab. The comparator is **NOT historical fused BMM**; measurements were limited, sequential, and on a non-isolated host. Ordinary install-only MATMUL-registry users are affected too. Explicit nonbatched `matmul`/`matmul_with_threads` and `FastCpuBackend::matmul` remain packed and unchanged. No speedup or GPU-performance claim.

The historical **343812 ULP** incident (expected `11.969769477844238`, observed `11.641884803771973`) remains **root-cause UNRESOLVED**, tracked in [containment/investigation notes](verification/pr25-fastcpu-containment/README.md). Passing reruns and counterfactual p=90 fingerprints do not explain it. Production BMM now bypasses suspect packed arithmetic entirely, using bounds-checked independent ascending-K scalar products. Direct BMM uses fresh zeroed output; install-only `CpuBackend` retains its pooled aggregate output, with every slab fully overwritten by the independent scalar callback. **Not all wrappers are pool-free**, and no pool defect is diagnosed. Custom backend/kernel registration can override library policy; arbitrary compiler/host-memory/hardware faults are outside this bounded software claim. CUDA resident BMM remains a separate unchanged cuBLAS route.

## Scope and review order

1. **Core foundation/state/AD/CPU architecture**: checked shapes/layouts and typed data; partial-backend correctness, loss adjoints and BatchNorm validation; leaf/version/no-grad contracts and bounded smooth higher-order AD. Rust named buffers/scalars, owning snapshots, strict CPU restore, optimizer state, immutable generation publication and stochastic optimizer-step-boundary continuation. CPU segment/sparse graph algebra, recurrent cells and masked/reset/truncated unroll, spline/KAN basis and rectangular grouped convolution/window adjoints. Shared tensor, module and autograd files are intentionally one coherent commit rather than speculative partial patches.
2. **Fastcpu containment**: independent scalar production BMM and conservative install-only MATMUL callback; retained test-only packed diagnostics and independent all-output oracle. Assertions and tolerances are not relaxed.
3. **Native mixed Python package**: Module/build registration and containers, Linear/MSE, SGD, Trainer/progress, functional AD and native graph/recurrent/basis/convolution wrappers, with regressions and executable public CPU examples.
4. **Documentation/evidence**: bounded verification manifests, historical review reports and explicit publication/evidence policy. This body and [PUBLICATION.md](verification/foundation-wave/pr-readiness/PUBLICATION.md) supersede stale milestone/status prose in earlier planning documents. This is one PR, not completion of the architecture roadmap.

`Module::validate_scalars` rejects both nonempty supplied state and omission when a custom module declares scalars. Custom stateful modules must provide their own restore protocol; public recurrent/basis exports and adversarial tests are included. AI agents assisted implementation, independent scoped reviews, verification and packaging. Human review is still required; no exhaustive security or review approval is claimed.

## Unsupported / follow-up work

- Full Rust training-state snapshots are **not exposed in Python**. Python model/training convenience APIs do not imply checkpoint parity.
- New graph/sparse/recurrent/basis/window/convolution wrapper computations are **CPU-f32 scoped**, not resident CUDA implementations. Legacy host fallback is not GPU residency.
- Whole-model checkpoint transactions are **CPU-only**, require paused training and matching architecture/configuration, and exclude pending gradients, accumulation/update phase, schedulers, data/sampler/prefetch cursors and arbitrary loop state.
- Custom Module/Optimizer implementations remain responsible for pure validators and correct commits. No device rollback, concurrent-reader isolation, panic/OOM rollback or physical power-loss guarantee.
- Higher-order AD supports a declared operator subset, not universal differentiation or all PINN/AE/GAN families. Recurrent unroll is not fused scan or bounded-memory training. Four decreasing-loss fixtures are not all-model-family/generalization/restart certification.
- No performance benchmark, speedup, warning-clean/clippy, sanitizer or Miri certification. Historical root-cause investigation remains open; validated software-path containment clears the documented blocker, not the investigation. There is no causal-resolution ETA.

## Review corrections

`copy_` uses the leaf path only for requires-grad destinations under disabled recording, preserving ordinary CPU/CUDA transfer semantics and existing mutation/version guards. Native convolution defaults match the facade. Loss target negations remain detached, fallible host operations with original device restoration; they avoid redundant copies without adding partial-backend unary requirements. They are **not resident-forward or zero-transfer GPU loss fixes**. `index_select` reuses its already validated output shape. Copilot's two suppressed findings were included in this assessment.

## Latest independently verified evidence

[Independent final review](verification/pr25-final-independent/REVIEW.md), [command exits](verification/pr25-final-independent/final/results.json), [13 integration exits](verification/pr25-final-independent/final/integration13/integration-results.json), [parsed counts](verification/pr25-final-independent/final/parsed-counts.json), [source stability](verification/pr25-final-independent/final/source-stability.json), and [package identity](verification/pr25-final-independent/final/packaging.json).

- Fresh noneditable release wheel built/installed; facade source/wheel/installed bytes and native wheel/installed bytes agree. Frozen source and installed hashes remained unchanged.
- Unchanged integration **13/13 exit 0**. Core **759 parent-harness tests passed / 2 ignored**, plus 9 successful child-process executions: **not 768 unique tests**.
- CUDA **112 parent-harness tests passed**, serial integration and supplemental default-parallel; Python discovery **73 passed**, no skips.
- Binding Rust **15 passed**; Python contracts **17 passed**; architecture **8 passed**; required-CUDA copy regression **3 passed** and native convolution **1 passed**.
- Fastcpu **25 passed / 2 ignored** in default-parallel and release-serial runs; tokenizer **8 passed**. Ignored cases are performance probes, not waived correctness failures.
- Actual microkernel fault injection gives preserved RED 127 vs 128 before direct containment and before install-only callback correction; positive controls cover omission/NaN/infinity/finite corruption. This proves injection sensitivity/path isolation, not natural recurrence or causal diagnosis.
- Existing warnings and expected caught stale-version panics remain; no clippy/sanitizer/Miri or exhaustive security certification.

## Test Plan / reproduction

Literal commands executed during final independent verification include:

```text
python verification/pr25-final-independent/run.py
cargo test -j2 -p ferro-core
cargo test -j2 -p ferro-fastcpu -p ferro-tokenizer
cargo test -j2 -p ferro-cuda
cargo check -j2 -p ferro-cuda --all-targets
cargo test -j2 --manifest-path crates/ferro-py/Cargo.toml --lib
cargo test -j2 -p ferro-fastcpu --release -- --test-threads=1
crates/ferro-py/.venv/Scripts/python.exe -m unittest discover -s crates/ferro-py/tests -v
crates/ferro-py/.venv/Scripts/python.exe verification/python-api-contract/contract.py
crates/ferro-py/.venv/Scripts/python.exe verification/python-architecture-api/test_architecture.py
crates/ferro-py/.venv/Scripts/python.exe crates/ferro-py/tests/test_pr25_copy.py
crates/ferro-py/.venv/Scripts/python.exe crates/ferro-py/tests/test_pr25_convolution.py
git diff --check
```

The first command is archival provenance (local-only fixed-output orchestrator), not a fresh-clone entrypoint. For reproducible full execution use the tracked `python verification/foundation-wave/pr-readiness/verify.py --output-dir verification/foundation-wave/pr-readiness/rerun-02` with the Windows prerequisites in [PUBLICATION.md](verification/foundation-wave/pr-readiness/PUBLICATION.md); always choose a new output directory. Current focused tests and their dependencies are committed. Required GPU verification used `FERRO_REQUIRE_CUDA=1`, explicit NVRTC/cuBLAS DLL directories, Cargo `-j2`; binding tests alone received queried base `PYTHONHOME`.

Selected reports/manifests are published; raw logs, wheels/binaries, prior archives, diagnostic dumps and credentials remain local. References in immutable reports to omitted files are local provenance, not download promises. [Publication policy](verification/foundation-wave/pr-readiness/PUBLICATION.md) and this body supersede historical draft/blocker status without rewriting historical failures. AI agents assisted implementation, independent review, verification and publication; no human approval is implied.
