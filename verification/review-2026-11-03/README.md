# Review and graph API continuation, 2026-11-03

Base commit: `5f1ebe9b54742a690cd7cbdd6b17568329b1c701`. The initial checkout
was clean. This report describes uncommitted source changes, not a release or
hosted CI run. Review was bounded to recent GNN work, CUDA gather ownership,
test gating, and the current Rust CUDA integration direction.

## Findings and repairs

1. `gnn::scatter` negated minimum-reduction inputs before validating dtype and
   device. I64 input reproduced a panic. CPU/F32 validation now precedes
   negation; tests cover all unsupported storage dtypes and a sentinel backend
   that forbids compute/download.
2. CUDA row gather validated only the I64 index buffer's device ordinal.
   It now checks its owning stream, including empty buffers. Public
   `index_select_t` now propagates operational/ownership errors; only the explicit
   no-work `GatherUnavailable` capability decline permits a CPU retry.
3. The same fast path passed raw I64 backing storage for sliced/broadcast index
   views, although bounds checks had examined only the logical view. Offset,
   contiguity and full-buffer-length guards keep these views on the existing
   logical-index host path. Counting tests cover offset, stepped, prefix,
   broadcast and empty views, and confirm whole-buffer dispatch still occurs.
4. Legacy CUDA unit/integration tests silently returned when CUDA was required.
   Before the repair, a required-CUDA integration probe returned success without
   a GPU. It and the equivalent unit probe now fail with exit 101. A subprocess
   regression hides devices and verifies the failure diagnostic.
5. Eight native graph fault-injection tests failed on a CPU-only host because
   formatting `DriverError` queried libcuda. Reporting its error code directly
   keeps simulated failures independent of a CUDA installation.

## Continued implementation

`ferro.graph` now exposes immutable `EdgeIndex`, batching, tensor roundtrips,
CPU f32 scatter reductions and `GcnConv`. The layer uses existing Rust tensor
operations and autograd, with registered weight/bias and immutable checkpoint
configuration. No second tensor engine or new dependency was introduced.

Eight Python tests cover a separate dense PyTorch value/gradient oracle with
atol 1e-5 / rtol 1e-4, duplicate edges, isolated nodes, node permutation,
batching, strided inputs, empty graphs, invalid inputs, capture rejection and
saved Adam/model/config/RNG continuation. Continuation is bit-exact within the
same process after restoring fresh objects; a self-loop configuration mismatch
rejects without changing parameters. Topology remains separately persisted input.

The Python implementation and final core gather changes received separate
agent review with no remaining blocking findings. This is a local code review,
not external certification. See [Rust CUDA research](../../docs/RUST_CUDA.md)
for source-backed upstream requirements and the unchanged optional compiler plan.

## Final verification

Linux x86_64, Rust 1.99.1, Python 3.12.14, torch 2.13.0+cpu, cudarc 0.19.10.
No NVIDIA device/driver is exposed. The [manifest](manifest.json) records 495
source/config/lock inputs plus wheel and installed-native hashes. All 20 wheel
Python source/native files match the installed package; generated Python cache
bytecode is excluded from that comparison.

| Command / scope | Result |
|---|---|
| `cargo test --workspace -- --skip required_cuda_ --nocapture` | Exit 0: core 860 passed / 2 ignored; fastcpu 37 passed / 2 ignored; tokenizer 8 harness passes, of which the real-GPT2 body returned early because its fixture was absent. CUDA 131 unit/integration harness passes plus 1 doctest; 5 mandatory GPU tests explicitly filtered. GPU-gated returns are not execution evidence. |
| `cargo check -p ferro-cuda --all-targets` | Exit 0 without a CUDA installation; includes new GPU regressions. |
| `maturin build --manifest-path crates/ferro-py/Cargo.toml --release --locked --out <work>/dist` and noneditable wheel installation | Exit 0, rebuilt after the final production-source changes. |
| `python -m unittest discover -s crates/ferro-py/tests -v` | Exit 0: 126 collected, 101 passed / 25 skipped, including all 8 new graph tests. Existing fresh-process CPU architecture/restart gates also run in this discovery. |
| `python examples/py_regression.py` | Exit 0, all binding/DLPack regression checks passed. |
| `python examples/ops_vs_torch.py` | Exit 0, all operator/reference checks passed. |
| `git diff --check` | Exit 0. |

Counts use each outer test executable's final summary, not nested subprocess
summaries. Do not add earlier targeted runs to these totals. Ignored performance
tests and GPU-gated returns are not passes of their unexecuted bodies.

Local raw logs are retained in `/workspace/work/review-2026-11-03`; their hashes
and final command outcomes are recorded in the manifest. They are not shipped
in this repository. Preserved failures include the initial driver-formatting
failure, the unfiltered run's five mandatory GPU failures, and the first Python
graph test's list/tuple shape assertion mismatch (corrected to the existing
list-valued API). Historical verification directories were not overwritten.

## Remaining limits

- New ownership regressions compile, but no actual CUDA kernel, transfer,
  capture or performance result was obtained. Run the full suite with
  `FERRO_REQUIRE_CUDA=1` on GPU hardware.
- `GatherUnavailable` adds a public exhaustive-enum arm. Custom backends that
  returned `Unsupported` only to request gather fallback must adopt the new
  variant. The new narrow fallback policy is specific to `index_select_t`.
- The new GCN API is first-order CPU f32. General resident GNN training,
  higher-order AD and a fresh-process GCN learning/restart fixture remain open.
- The historical CPU numerical/crash investigations and legacy captured-chain
  lifetime/cleanup hazards in `docs/CUDA_GRAPH_NEXT.md` remain unresolved.
- No Rust-authored GPU kernel, dependency/toolkit migration, benchmark claim,
  commit, push or merge was performed.
