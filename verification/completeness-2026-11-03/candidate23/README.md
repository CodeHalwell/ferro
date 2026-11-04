# Continuing PyTorch/PyG work: verified candidate23

This local matching-build checkpoint passes every available recorded check.
Full PyTorch/PyG parity remains open. Changes are uncommitted. Earlier green
[candidate22b](../candidate22b/README.md), [candidate21](../candidate21/README.md) and
[candidate18](../candidate18/README.md), and failed [candidate22](../candidate22/README.md),
[candidate24](../candidate24/README.md), [candidate20](../candidate20/README.md), [candidate19](../candidate19/README.md) and
[candidate17](../candidate17/README.md) records remain available.

References are PyTorch 2.13.0+cpu and PyG 2.8.1, development dependencies only.
Ferro keeps its zero-dependency Rust core and one record_fn/Op autograd engine.
No NVIDIA or Apple GPU is available. Metal is not implemented.

## Matching-build validation

The noneditable ABI3 wheel uses an isolated copy of 1,012 selected source and
fixture inputs. All 116 installed files (115 Python plus native) matched the
wheel when independently rehashed for this publication. The source ZIP and
isolated inputs also matched. The manifest records the verification time; the
global environment can subsequently advance to a newer candidate.
Both frozen locks use cudarc 0.19.10 with CUDA 12.8 bindings. [manifest.json](manifest.json)
identifies source/wheel/log hashes, versions, commands and atomic exit statuses.
Raw artifacts remain under /workspace/work/completeness-2026-11-03.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,574 discovered: 1,544 passed and 30 GPU cases unavailable. |
| Rust workspace | Core 1,310 passed / 2 ignored; fastcpu 39 passed / 2 ignored; CUDA 140 harness passes / 5 mandatory cases filtered; tokenizer 8 harness passes. |
| CUDA doctests | Two intended E0502 borrow errors pass, without E0460 artifact mismatch. |
| CUDA all-target compilation | Passed without a CUDA driver or GPU. |
| Native DLPack | 12 harness passes; one GPU body returns early, three mandatory GPU cases filtered. |
| Eight standalones | Binding, compiled fusion, fuse, operator/Torch, safetensors, 200-trial seed0 fuzz, graph training/restart and API inventory all exit0. |
| Focused installed gates | Take 8, cumulative/extrema 20 and graph 332 methods pass from the frozen tree, without overlays. |

Unittest reports 1,568 executions and 25 skip indicators: 24 individual GPU
skips and one class hiding six cases. Native totals use final outer summaries:
1,497 runtime passes plus two doctests, excluding 18 nested child summaries.
One optional tokenizer body and optional GPU bodies can return early; harness
passes do not establish hardware execution. Focused gates overlap full discovery
and are not added to its unique-method count.

## Added since candidate22b

- Typed CPU take supports int64 indices, negative offsets, scalar/strided inputs,
  repeated selections, live floating gradients and tested higher/JVP paths.
  Indexed gather/scatter adjoints use the existing Op engine and preserve the
  tested saved-index/input dependencies. Explicit out destinations and capture
  remain unsupported; float64 higher-order limits remain explicit.
- CPU cummax/cummin return typed values and int64 indices with tested tie, NaN
  and signed-zero behavior. Genuine indexed adjoints route ordinary, supported
  higher and forward derivatives, including mixed-precision directions.
- Data/HeteroData subgraph and edge_subgraph select aligned typed fields with
  native gradients. Node/edge order and repetitions are retained; endpoints use
  the last selected occurrence of a repeated node. Container/store mappings are
  independent while unchanged graph fields remain aliased. This eager scope
  rejects batched/sampled data, custom multi-space hooks and extra indexed fields
  that require separate remapping.

The [candidate22b report](../candidate22b/README.md) retains the preceding
multicolumn/joint-transform, cumulative product/sum, Where, activation and graph
field work. Public Hessian and reverse transform dtype limits remain unchanged.
Separate array-layout and nested-transform experiments retain their own failed
raw reports; they are not included in this green checkpoint.

## Remaining scope

Parameters/optimizers remain CPU float32. Full dtype/operator coverage, float64
graph-building VJPs/transform primals, half/complex/FFT, arbitrary indexing/out,
general vmap/compiler/export/distributed support, catalogs and full NVIDIA/Metal
models remain open. Higher/capture support is operator-specific; optional PyG
native sampling backends are unavailable. Stable CPU formulas and portable Rust
math can have documented rounding differences from Torch kernels.

Future24 unary layout/backward-mode repairs, typed angle/geometric transforms,
and PReLU/Bilinear/GaussianNLL additions are excluded from this proof. The name
inventory measures presence, not semantic/device/performance parity. See the
[compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md),
[GPU portability](../../../docs/GPU_PORTABILITY.md) and [Rust CUDA research](../../../docs/RUST_CUDA.md).
