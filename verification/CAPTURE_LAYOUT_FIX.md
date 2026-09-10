# Captured layout input-update fix

## Design and constraints

Captured f32 device transpose/contiguous reshape of graph leaves copy the backing
buffer into a NEW StorageCell while preserving shape, strides and offset. The
recorded forward edge still targets the original leaf, so compiled replay reads
current input values. Retaining the eager root no longer retains storage aliases
of those mutable leaves. Computed inputs already carry mutation-forbidding op
history and are not copied. Noncontiguous reshape that already materialized into
fresh storage also avoids a redundant copy.

No changes to inplace.rs, public uniqueness gates, autograd gates, cell identity,
or version updates. Ordinary device views and detach_copy snapshots continue to
block public writes, even with a compiled capture present. Host view alias
semantics and grad-requiring layout recording are unchanged. Replay suspends
capture and pays no isolation copy. A raw layout replay result can still be an
ordinary alias: callers must release such returned aliases before mutating its
input. Existing ordinary aliases to an input must likewise be released.

Cost: one capture-time full-backing-buffer copy per leaf-backed aliasing device
layout. CUDA copy_dev stays resident; another backend's default copy_dev may use
a host round trip. This is not a claim of zero-copy capture. Internal computed
layout chains keep their existing residency and layout dispatch counts.

Gather fallback now drops the initial read guard after the direct attempt,
builds/uploads layout-only indices unlocked, then reacquires once for gather.
Storage variant/buffer identity are immutable and no pointer escapes this scope.
Device value ordering continues to follow backend stream ordering, not exclusive
Rust locking. A private mock test probes try_write during index upload and gather
without timing, sleeping, or nested guards.

## Verification

All commands used cargo -j2. CUDA PATH:
`$LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin:$LOCALAPPDATA/Temp/cuda-rt/nvidia/cublas/bin:$PATH`

- `cargo test -j2 -p ferro-core --test capture_layout_updates`: transpose and
  contiguous reshape each observed failing on public storage uniqueness before
  their respective changes. Logs: capture-layout-red.log,
  capture-layout-reshape-red.log, capture-layout-transpose-green.log.
- `cargo test -j2 -p ferro-core --lib gather_fallback_unlocks`: observed failing
  on retained source lock before lock-scope edit, then passed.
  Logs: layout-lock-red.log and layout-lock-green.log.
- `cargo test -j2 -p ferro-core`: 629 passed, 0 failed, 2 pre-existing ignored.
  Log: capture-layout-core-full.log. Includes structural one-copy/no-replay-copy,
  stable input tensor/cell identity, alias guards, finite-difference gradients,
  and unchanged host view semantics.
- `cargo test -j2 --manifest-path verification/capture-layout-gpu/Cargo.toml --target-dir target -- --nocapture`:
  real CUDA required via expect, never skips. Passed all six combinations of
  retained/dropped root with transpose, contiguous reshape, and nested
  transpose/reshape multi-consumer graph; each replayed two input updates.
  Also verifies ordinary transpose/reshape/detach alias rejection, captured
  layout backward, and saved-output snapshot protection. Log: capture-layout-gpu.log.
- `cargo test -j2 -p ferro-cuda --test layout_materialization -- --nocapture`:
  both passed, original zero-index-upload/two-layout-enqueue counters preserved.
  Log: capture-layout-cuda-layout.log.
- `cargo test -j2 -p ferro-cuda`: 57 passed, 0 failed, 0 ignored.
  Log: capture-layout-cuda-full.log.
- `git diff --check`: passed (Git reports existing LF/CRLF notices).

Existing unused-variable/dead-code compiler warnings remain. The standalone GPU
verification crate avoids modifying CUDA test files owned by the parallel agent.
Python bindings were not rebuilt by this subagent. No staging, commits, pushes,
or branch changes; pre-existing staged layout changes preserved.
