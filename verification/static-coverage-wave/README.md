# Static CUDA coverage wave

## Implemented contracts

- Arbitrary-axis softmax lowers to a collapsed `(outer, inner, axis)` stride copy, existing stable two-pass row softmax, and the inverse `(outer, axis, inner)` stride copy. Last-axis softmax keeps its existing pipeline. Logical output shape stays separate from the physical copy descriptor: rank-17 inputs are covered without imposing the materializer's rank-16 limit on softmax. Every command is captured in the same CUDA graph; no host fallback or retained eager result is used during static replay.
- Expanding pointwise seeds get an explicit `StaticOp::Broadcast` destination before the existing chain. Scalar, row, column and multi-axis expansion are covered, including subtraction/division as the first operation and after a unary operation. Operand order is unchanged. Broadcast destinations never participate in identity-layout aliasing.
- Empty destinations retain descriptor validation but emit no kernel/GEMM command. A wholly empty model owns and launches a real, instantiated zero-node CUDA graph; replay counters and independent empty snapshots work normally.
- Nonempty zero-reduction matmul/BMM emits a captured zero-fill command on every replay. Zero-reduction sum uses the existing sum kernel. Tests dirty both allocator bins and the actual private GEMM output between replays; snapshots are independent.
- Eager SGEMM also skips cuBLAS for zero output dimensions. This was required to construct the public captured empty-matmul graph: the old path called cuBLAS with an invalid leading dimension.
- Existing leaf guards, private scratch ownership, output guards, capture cleanup, graph launch and output snapshot fencing are preserved. No changes were made to Python bindings or the sibling's public Python tests.

## Limits

Static inference still requires whole contiguous resident f32 leaves on the same validated backend, no gradient-requiring leaves, and supported recorded forward operations. Tensor shapes are fixed after preparation; input and weight values can change in place. General layout and explicit expanding-broadcast descriptors retain the existing rank-16 materializer bound. Softmax axis permutation uses rank-3 descriptors regardless of tensor rank. Extents and total output lengths must fit the u32 launch ABI; GEMM extents must fit i32. Checked products, including the product of nonzero dimensions needed by layout strides, must fit usize even for zero-element tensors. Malformed public plans are rejected before allocation. LayerNorm's pre-existing nonzero normalized-width requirement is unchanged.

Rust softmax takes usize axes. Negative-axis normalization belongs to the public Python API and is exercised by the parent's independent binding integration, not claimed by these Rust tests.

## TDD evidence

All CUDA commands used the process environment:

```
PATH=$LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin:$LOCALAPPDATA/Temp/cuda-rt/nvidia/cublas/bin:$PATH
FERRO_REQUIRE_CUDA=1
```

The test-first vertical cycles are retained here:

| Logs | Exact test filter (`cargo test -j2 -p ferro-cuda --test model_static_graph FILTER -- --nocapture`) | Observed RED |
| --- | --- | --- |
| softmax-red.log / softmax-green.log | static_softmax_arbitrary | static softmax requires last dimension |
| broadcast-red.log / broadcast-green.log | static_expanding | expanding static seed is not implemented |
| empty-red.log / empty-static-red.log / empty-green.log | static_empty | eager CUBLAS_STATUS_INVALID_VALUE, then empty static destinations unsupported |
| zero-red.log / zero-green.log | static_zero | invalid or empty static matmul |
| validation-red.log / validation-overflow-red.log / validation-green.log; validation-normalization-red.log / validation-normalization-green.log | empty_static_plans | malformed empty descriptors accepted, including oversized normalization metadata |
| softmax-rank-red.log / softmax-rank-green.log | static_softmax | high-rank input rejected by the inverse layout rank limit |

Supplemental structural and CPU lowering contracts:

```
cargo test -j2 -p ferro-core --test device static_lowering -- --nocapture
cargo test -j2 -p ferro-cuda zero_work_is -- --nocapture
```

`core-lowering.log` verifies the emitted operand slots, broadcast/permutation descriptors, zero-reduction GEMM metadata and zero extra transfers during lowering, using the existing fake backend. It does not claim CUDA execution. `zero-structure.log` verifies a real zero-node graph and one fill node for each zero-reduction GEMM, including dirtying private output before each launch. These supplement the original feature RED/GREEN tests.

## Final verification

```
cargo test -j2 -p ferro-core
cargo test -j2 -p ferro-cuda
```

Both final commands exited 0. Full logs: `core-final.log` and `cuda-final.log`.

- Core: 641 passed, 0 failed, 2 ignored across 168 harness results.
- CUDA: 86 passed, 0 failed, 0 ignored across 9 harness results, including the compile-fail output-lease doctest.
- The model static integration binary contains 11 passing tests, including the five new coverage/validation contracts.
- Existing compiler warnings remain. No benchmark timings were taken and no new performance claims are made.
- Parent owns final binding rebuild and Python integration.

No files were staged, committed or pushed. Earlier uncommitted optimization/DLPack changes and verification artifacts were preserved.
