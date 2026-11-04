# Continuing PyTorch/PyG work: verified candidate27

The corrected integration passes every available recorded check. Full framework
parity remains open; changes are local and uncommitted. Previous green
[candidate26](../candidate26/README.md) retains its proof. The first candidate27
attempt remains separate failed evidence; this checkpoint uses a fresh build.

References: Torch 2.13.0+cpu and PyG 2.8.1, development dependencies only.
Ferro retains its zero-dependency Rust core and one record_fn/Op AD engine.
No NVIDIA or Apple GPU is available. Metal is not implemented.

## Matching-source validation

Exactly 1,069 selected inputs and all 122 installed files match corrected
candidate27-proof2. Both source ZIPs, frozen trees, wheel payloads, main inputs
and the installed package were rehashed; the manifest records the time.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,703 discovered: 1,673 passed, 30 GPU cases unavailable. |
| Rust workspace | Core 1,405 passed / two ignored; fastcpu 41 passed / two ignored; CUDA 142 harness passes / five mandatory cases filtered; tokenizer eight harness passes. |
| CUDA doctests / all-target compilation | Two intended E0502 borrow errors pass; all targets compile without a driver. |
| Native DLPack | 12 harness passes; one GPU body returns early and three mandatory cases are filtered. |
| Eight standalones | Binding, compiled fusion, fuse, operator/Torch, safetensors, 200-trial seed0 fuzz, graph training/restart and inventory all exit0. |
| Focused integration | All 41 Where, Clamp, power, PReLU, MLP and readout methods pass. |

Unittest reports 1,697 executions and 25 skip indicators: 24 individual GPU
skips and one class hiding six cases. Native totals are 1,596 outer runtime
passes plus two doctests, excluding 18 nested summaries. Optional GPU/tokenizer
bodies can return early; harness passes do not establish hardware execution.
Focused gates overlap full discovery and are not added to its totals.

## Added since candidate26

- Where preserves measured branch/condition/output layouts and genuine optional
  adjoints. Clamp preserves ordered NaN/signed-zero decisions and live bound
  gradients. Saved versions, absent contributions and higher queries stay strict.
- Original scalar powers preserve the Python exponent in the same native Op;
  special exponents use staged square/cube/reciprocal expressions. Portable CPU
  scalar policies do not emulate reference ISA block-dependent narrowing.
- Empty broadcast adjoints use genuine dimension reductions up to 2^20 output
  elements, preserving tested first/higher layouts. Larger virtual results keep
  exact zero values/AD but can differ in strides. PReLU JVP keeps logical axes
  through arithmetic so structural-zero reverse layouts match the reference.
- PyG MLP and MLP-aware DeepSets/readout routes cover copied values, all tested
  gradients, batch-aware norms, COO/CSR routing, dropout and fresh-process state.
  Actual reference global-index rejection behavior and prior schemas are retained.

The failed first attempt exposed two fixture problems and real empty-reduction
and special-power gaps. Corrected tests retain strict versions/layouts and
declared unsupported dtypes; original raw failures remain inspectable.

## Remaining scope

Float64 training, physical-batch Jacobians, shared multi-output/nested rules and
RandomWalkPE are separate experiments. Parameters/optimizers here remain F32.
Full dtype/operator/catalog coverage, F64 transform primals, half/complex/FFT,
general indexing/out, vmap/compiler/export/distributed and complete GPU models
remain open. Name presence does not establish semantics or performance.
See the [compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md),
[GPU portability](../../../docs/GPU_PORTABILITY.md) and
[Rust CUDA research](../../../docs/RUST_CUDA.md).
