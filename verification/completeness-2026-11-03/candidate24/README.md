# Candidate24: retained failing combined build

This first combined build failed Python discovery. It does not replace the
passing [candidate23 checkpoint](../candidate23/README.md). The frozen source ZIP, isolated
tree, wheel and all installed files were rehashed before this record was written.
[manifest.json](manifest.json) records those hashes, commands and atomic statuses.
Raw evidence remains under /workspace/work/completeness-2026-11-03.

The build contains 1,026 selected source/fixture inputs and 118 installed files
(117 Python plus native). References are Torch 2.13.0+cpu and PyG 2.8.1.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,612 discovered: 1,578 passed, four methods failed or errored, 30 GPU cases unavailable. |
| Rust workspace | Core 1,334 passed / two ignored; fastcpu 39 passed / two ignored; CUDA 140 harness passes / five mandatory cases filtered; tokenizer eight harness passes. |
| CUDA doctests / all-target compilation | Two intended borrow-error doctests pass; all targets compile without a driver. |
| Native DLPack | 12 harness passes; one GPU body returns early and three mandatory GPU cases are filtered. |
| Eight standalones | All exit0, including 200-trial seed0 fuzz and graph training/restart. |
| Focused installed scopes | Graph/geometry/angle suite 352 passed; new neural-network catalog 12 passed; LayerNorm eight passed. |

Unittest reports 1,606 executions, one failure indicator, 11 error indicators
and 25 skip indicators. The errors share three unique methods; the failure is
one additional method. Twenty-four skipped GPU methods and one skipped class
containing six cases account for 30 unavailable cases. Native totals exclude
18 nested child summaries and do not establish GPU execution.

All four affected methods are in test_unary_layout.py. Three encounter missing
Tensor.neg()/negative() aliases. The fourth exposes direct JVP tangent
preparation: a direction whose metadata differs from its primal needs a real
differentiable copy with the primal's dtype, layout and storage span. The
original fixtures and failures remain intact while a separate repair is tested.

This build also contains raw unary-layout and saved-output fixes, typed
atan2/acos, seven geometric transforms, PReLU, Bilinear and GaussianNLLLoss.
Passing focused checks overlap full discovery; their counts are not added to
the unique-method total. Full PyTorch/PyG parity remains open. No NVIDIA or
Apple GPU is available, and Metal is not implemented.
