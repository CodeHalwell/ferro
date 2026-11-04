# Continuing PyTorch/PyG work: verified candidate22b

This local matching-build checkpoint passes every available recorded check.
Full PyTorch/PyG parity remains open. Changes are uncommitted. Earlier green
[candidate21](../candidate21/README.md) and [candidate18](../candidate18/README.md), and
failed [candidate22](../candidate22/README.md), [candidate20](../candidate20/README.md),
[candidate19](../candidate19/README.md) and [candidate17](../candidate17/README.md)
records remain available.

References are PyTorch 2.13.0+cpu and PyG 2.8.1, development dependencies only.
Ferro keeps its zero-dependency Rust core and one record_fn/Op autograd engine.
No NVIDIA or Apple GPU is available. Metal is not implemented.

## Matching-build validation

The noneditable ABI3 wheel uses an isolated copy of 1,004 selected source and
fixture inputs. All 114 installed files (113 Python plus native) match the wheel.
Source ZIP, wheel, isolated inputs and installed files were rehashed at publication.
Both frozen locks use cudarc 0.19.10 with CUDA 12.8 bindings. [manifest.json](manifest.json)
identifies source/wheel/log hashes, versions, commands and results. Raw artifacts
remain under /workspace/work/completeness-2026-11-03.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,552 discovered: 1,522 passed and 30 GPU cases unavailable. |
| Rust workspace | Core 1,290 passed / 2 ignored; fastcpu 39 passed / 2 ignored; CUDA 140 harness passes / 5 mandatory cases filtered; tokenizer 8 harness passes. |
| CUDA doctests | Two intended E0502 borrow errors pass, without E0460 artifact mismatch. |
| CUDA all-target compilation | Passed without a CUDA driver or GPU. |
| Native DLPack | 12 harness passes; one GPU body returns early, three mandatory GPU cases filtered. |
| Eight standalones | Binding, compiled fusion, fuse, operator/Torch, safetensors, 200-trial seed0 fuzz, graph training/restart and API inventory all exit0. |
| Focused repaired/new transforms | All108 methods pass against the rebuilt installed native library, without an overlay. |
| Focused activation/layout and LayerNorm | All93 plus8 methods pass from the frozen tree, without overlays. |

Unittest reports 1,546 executions and 25 skip indicators: 24 individual GPU
skips and one class hiding six cases. Native totals use final outer summaries:
1,477 runtime passes plus two doctests, excluding 18 nested child summaries.
One optional tokenizer body and optional GPU bodies can return early; harness
passes do not establish hardware execution.

## Added and corrected since candidate21

- Native multicolumn jacfwd evaluates the function once, with genuine basis
  columns and explicit RNG policies. Joint functional pullbacks use one union
  traversal. Supported forward transforms allow float64 outputs/directions from
  float32 primals, with typed disconnected zeros and checked result capacities.
  Cast tangents perform actual copies when the primal cast allocated.
- Typed CPU cumsum/cumprod cover bool/int64/float32/float64, exact integer wrap,
  mixed precision and supported live higher/JVP paths. linalg.vander uses genuine
  cumprod. Zero-containing higher cumprod has an explicit 1,048,576 element-work
  bound until a fused rule replaces the current graph construction.
- Where has optional mask/data adjoints and direct JVPs on the existing engine.
  Sigmoid/Tanh backward kernels retain Torch staging and live derivative graphs.
  CPU binary/copy layout planning preserves tested dense physical orders.
  ELU/CELU/SELU gain true float64 and checked scalar-category behavior.
- Data/HeteroData field apply, detach, contiguous, cpu and supported to operations
  preserve tested field ownership, metadata and live graphs. In-place detach
  remains unsupported; callbacks follow sequential mutation/error semantics.

Candidate22 exposed a real forward-output dtype/Cast wrapper gap plus two test
fixture errors. Candidate22b repairs them and reruns full validation. Its public
Hessian implementation and reverse transform dtype limits remain unchanged.
Serial jacrev corresponds to Torch's row-sweep behavior; general batched higher
zero semantics and nested private-frame composition are still under development.

## Remaining scope

Parameters/optimizers remain CPU float32. Full dtype/operator coverage, float64
graph-building VJPs/transform primals, half/complex/FFT, arbitrary indexing/out,
general vmap/compiler/export/distributed support, catalogs and full NVIDIA/Metal
models remain open. Higher/capture support is operator-specific; optional PyG
native sampling backends are unavailable. Stable CPU formulas and portable Rust
math can have documented rounding differences from Torch kernels.

Source23 has advanced with typed indexed selection and graph subgraph extraction;
those changes are excluded from this proof. The name inventory measures presence,
not semantic/device/performance parity. See the [compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md),
[GPU portability](../../../docs/GPU_PORTABILITY.md) and [Rust CUDA research](../../../docs/RUST_CUDA.md).
