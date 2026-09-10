# Direct CUDA layout materialization

## Implementation

The core Backend materialize_dev seam and Tensor reshape fast path now use a
bounds-checked CUDA stride copy. Dimensions, strides and offset are scalar kernel
arguments: no per-element host index construction or device index buffer. The
kernel source is cached independently of dimensions. The existing gather-only
backend fallback remains intact.

CUDA supports rank 0 through 16, unsigned strides (including zero/broadcast),
nonzero offsets, and empty outputs. Rank mismatch, excessive rank, element-count
or address overflow, and out-of-buffer addresses return errors before allocation
or launch. Empty layouts permit offset equal to buffer length and perform no
launch. Non-empty counts must fit the existing u32 launch interface. All output
elements are written before returning the uninitialized pooled allocation.

layout_counts() reports successful i64 upload calls, uploaded bytes, and direct
non-empty layout enqueues. It does not count kernel parameter metadata as an
index upload, and does not count later CUDA graph executions as new enqueues.

## Correctness and structural proof

Real CUDA RED: verification/layout-cuda-red-completion.log fails with the missing
materialize_dev implementation (exit 101), not a runtime skip. GREEN:
verification/layout-cuda-green.log contains two real GPU tests, zero failures.
Tests cover transposes, scalar, offset, broadcast, empty and overflow/bounds/rank
cases. A known 2-element i64 upload increments counters by one call and 16 bytes.

All five transformer layout shapes (Q, K, V, transposed K, attention merge) are
validated in eager capture and compiled replay against CPU values. Each pair
measures exactly zero index uploads, zero index bytes and two direct enqueues.
This is actual CUDA backend instrumentation on isolated model layouts, not a
full-model transfer trace or a claim that all model host transfers are zero.
The core counting-backend test also asserts no activation downloads or index
uploads before explicit correctness readback.

The full model benchmark separately passes original, changed and restored input,
and changed/restored Q weight against fresh Torch outputs in both processes.
Maximum absolute error: eager 1.430511474609375e-6, compiled
1.1920928955078125e-6 (rtol=2e-4, atol=2e-5). Layout components are exact.

## Full-model measurements

Same harness, shape and method as TRANSFORMER_PROFILE.md: RTX 3090, Windows WDDM,
FP32, TF32 off, tokens=128, width=256, heads=8, seed=18, 15 warmups and 60 samples.
Two fresh processes ran serially after build/integration; second reverses order.
Medians include a stream fence, not a host output transfer.

| Backend | Run 1 us | Run 2 reversed us |
|---|---:|---:|
| Torch eager | 264.40 | 252.25 |
| Ferro eager | 429.00 | 784.90 |
| Ferro compiled | 368.05 | 369.25 |
| Torch Inductor fullgraph | blocked | blocked |

Historical compiled baseline medians were 923.45-926.10 us in stable runs 2-4
(run 1 was 1075.00 us). The new compiled result is about 2.50x faster than that
historical stable baseline conservatively. This is a historical before/after,
not an interleaved A/B run; combined-tree GELU/sqrt/tanh review fixes are also
present, so do not attribute every microsecond exclusively to layout changes.
Ferro compiled remains slower than Torch eager. Eager shows substantial
run/order sensitivity (429.00 versus 784.90 us); no stable eager speedup headline.
Inductor was actually attempted twice and failed for no working Triton install;
no substituted eager compiler or compiled-Torch speed claim.

Isolated layout medians (Ferro eager / compiled, ranges across two runs):
- Q layout: 30.70-33.70 / 30.60-32.60 us.
- K transpose: 30.20-36.25 / 25.90-31.45 us.
- Attention merge: 30.30-32.50 / 30.90-31.90 us.

Components have independent fences and cannot be summed into full-model cost.
Compiler metadata remains 54 operations, 22 leaves, 44 scheduled runs; these are
not physical CUDA launch counts. Raw samples and all parity/error rows are in
verification/layout-profile/run1.json and run2.json (47 rows each). Original
verification/profile and TRANSFORMER_PROFILE.md are preserved.

## Verification and reproduction

The project .venv binding was rebuilt with maturin develop --release and
CARGO_BUILD_JOBS=2; exit 0, verification/layout-build.log. No compiler retry was
needed. The final combined-tree run_integration.py returned 0 with all 12 command
rows successful (core, fastcpu/tokenizer, real CUDA, CUDA all-target check,
Python capture/compiled/fuse/general/ops/safetensors, 200-trial fuzz, diff check).
Runner selection unit tests also pass. Existing compiler warnings remain; no
claim of warning-free formatting. No commits or pushes.

From repository root in bash:

```bash
export PATH="$LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin:$LOCALAPPDATA/Temp/cuda-rt/nvidia/cublas/bin:$PATH"
export RUST_TEST_THREADS=1
cargo test -j2 -p ferro-cuda --test layout_materialization -- --nocapture
crates/ferro-py/.venv/Scripts/python.exe verification/run_integration.py
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --json verification/layout-profile/run1.json
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --reverse --json verification/layout-profile/run2.json
```

Logs: verification/layout-{cuda-green,integration,runner-tests}.log and
verification/layout-profile-run{1,2}.log. Integration per-command logs and
integration-results.json now describe the combined tree. Layout CUDA tests
explicitly require CUDA rather than silently passing via skip.
