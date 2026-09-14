# Independent fastcpu second review: unresolved, new omission fingerprint

> Historical scoped report. The independent `omission_probe.rs` is packaged; its executable and other cited forensic scripts/results/logs remain local evidence. Use the portable commands in [PUBLICATION.md](../pr-readiness/PUBLICATION.md). No naturally reproduced fastcpu failure or production fix is claimed.

## Outcome

The historical wrong value is reproduced **bit for bit by omitting exactly the p=90 product** from the reconstructed scalar dot at batch 14, row 14, column 76. This is a counterfactual numerical fingerprint, NOT a reproduction of a fastcpu defect or proof of its mechanism. No production fix, preexisting-bug classification, hardware diagnosis, or acceptance pass is claimed.

Independent Rust (fresh local LCG, no ferro dependency) confirms the Python result:

- A[14,14,90] bits 1047347712, B[14,90,76] bits 1068835684.
- Rounded product bits 1051189367, approximately 0.3278844.
- Normal sequential dot bits 1094681645: 11.969769477844238.
- Dot skipping p=90 bits 1094337833: 11.641884803771973, exactly the historical wrong value.
- Output XOR is 0x5c104 (six differing bits), not a single final-output-bit flip.

`results.json` records targeted counterfactual enumeration, not repeated BMM tests:

| Counterfactual | Cases | Exact matches to historical wrong bits |
|---|---:|---|
| Omit one product | 128 | p=90 only |
| Duplicate one product | 128 | None |
| Omit one nonempty contiguous K interval | 8256 | [90,91) only |
| Flip one bit of one original row/column input | 8192 | A or B at p=90, bit 28 or 29 |
| Flip one accumulator bit before/after any K step | 4128 | None |

The four input-bit matches make that operand tiny enough that its product contributes nothing after f32 rounding. They are alternative mathematical explanations, not evidence that bits flipped in memory. Arbitrary corruption and different arithmetic execution models are not exhaustively covered by these counterfactuals.

## New constraint from the first mismatch

The original log reports the **first** mismatch, not the only mismatch. Both current and archived assertion implementations walk outputs in increasing flat-index order; the archived lines 271-281 are preserved in `final-verification.json`. A persistent missing A entry would also change earlier columns in the same row; a persistent missing B entry would change earlier rows of that column. The standalone Rust probe confirms nonzero differences at earlier coordinates (14,14,0), (14,14,64), (14,0,76), and (14,12,76) when their p=90 contribution is omitted.

Consequently, under intact reference results and the expected tile layout, a permanently zeroed original A/B entry or a whole missing K iteration predicts earlier discrepancies. A persistent packed-A fault across a full tile would affect column 64 before 76; persistent packed-B corruption across the 6-row tile would affect row 12 before 14. A transient/local operand load or multiply/add omission remains compatible. The historical log has no complete mismatch map or intermediate registers, so none of these mechanisms is established.

Conditional on 32 workers (current probes report 32; historical count was not captured), global row 1806 belongs to worker 28, group start 1792, rows_per=64. This is local row 14 in the first MC block, micro row 2, column tile 4/lane 12. p=90 is interior to the single KC block (k=128), not a K tail or pp transition. Packed-A index is 2078; packed-B index is 9644. This directs a future diagnostic toward a specific update rather than more undirected tail stress.

## Unsafe/lifetime audit

Reviewed current source and relevant archived/history source, independently of prior pass counts:

- `ferro-fastcpu/src/lib.rs:117-173`: BMM uses `std::thread::scope`, **not a custom persistent CPU thread pool**. No raw-pointer Send/Sync wrapper, lifetime transmute, erased borrowed job, or custom job destructor exists on this path. Searches across crates found no `unsafe impl Send/Sync` or `transmute` declarations. Dispatch documentation's phrase "thread-pool spin-up" is not evidence of such an implementation.
- `lib.rs:134-141`: `chunks_mut` creates exclusive output partitions; `move` copies scalar g0/group_rows and captures each distinct mutable chunk. Shared A/B are immutable borrows. The scope joins children before out is returned, also on unwinding. There is no user-code forget/join bypass.
- `lib.rs:149-173`: packed A/B are owned Vecs local to each worker, allocated before its loop. Scratch borrows are synchronous and cannot escape `matmul_rows_scratch`. They end before the next group segment reuses scratch, and worker-local Vec destruction follows the final call. Scratch is not taken from the host pool.
- `lib.rs:197-230`: reachable explicit unsafe calls only cross runtime AVX2+FMA guards into target-feature wrappers. No pointer load/store, hand-written ABI, alignment cast, or unchecked indexing appears in the packing/microkernel code. This does not prove compiler/std correctness or homogeneous feature detection on every processor, but no evidence establishes those failures.
- `lib.rs:237-277,283-294,313-351`: packing and writeback are safe slice operations. Every consumed pa/pb lane is written for the current kc, including padding. Accumulators are initialized arrays; pp=0 does not read old output. Distinct worker chunks and synchronous scratch use exclude the proposed application-level shared scratch race under Rust's contracts.
- `ferro-core/src/pool.rs:73-106,117-156`: freelists own their Vecs; `pop` transfers ownership; `give` consumes ownership. RefCell is thread-local. `take_uninit` is an unfortunate name for **valid initialized floats**, not `set_len` over uninitialized memory. Fresh buffers are zeroed; debug recycled buffers are NaN-filled. Vec capacity/length ownership is never rebuilt from pointers.
- `ferro-core/src/tensor.rs:96-105`: final StorageCell drop uses exclusive access and `mem::take` before giving the Vec. Dropping the emptied Storage cannot double-free the transferred buffer. The original test directly owns A, B, want, got Vecs; tensor-storage drop is not used for those outputs. Assigning/dropping a previous result cannot release the current worker scratch or a still-borrowed input under this code.
- `elementwise.rs:386-410`, `dispatch.rs:215` and `661-664`: default reference computes first, then fast BMM. Both may use packed arithmetic because MATMUL is installed. MATMUL is RwLock-protected. No Python/DLPack/CUDA pointer lifetime is on this CPU-only test path. Those unsafe systems are not evidence for this incident.

No concrete application-level UB mechanism was found on the reviewed BMM path. That is a scoped negative audit, not a proof of numerical reliability.

`results.json` independently rechecks archive-vs-HEAD identity and current production-prefix equality for lib.rs, elementwise.rs, pool.rs, dispatch.rs. The BMM-introducing commit 4739328 also already used scoped threads and local owned scratch (before the host output pool). Source age does not establish that an old standalone build reproduces the incident and does not justify calling it preexisting or fixed.

## Tool-supported limits and separate compiler evidence

Only stable x86_64-pc-windows-msvc is installed; current rustc is 1.98.0 (88d9e12ae), LLVM 22.1.8. `cargo miri --version` fails because Miri is unavailable on this toolchain; `rustc -Z help` rejects nightly-only options. Valgrind, clang, clang-cl, and Dr. Memory are not on PATH. No sanitizer/interpreter claim is made, no install/toolchain change was attempted, and no shared Cargo target was touched. A supported nightly/Miri or sanitizer environment requires separately approved setup; Miri scalar execution would not establish native AVX codegen correctness.

`verification/integration-build-access-violation.log:56-59` really records a separate rustc process exiting with STATUS_ACCESS_VIOLATION (0xc0000005) while compiling ferro-py. It is evidence of a compiler-process crash, not evidence of malformed fastcpu machine code or faulty hardware. No dump, matching compiler-binary provenance, or causal link to this BMM failure was obtained.

## Recommendations, not source patches

1. Keep the numerical incident OPEN. The omission fingerprint is concrete progress, not a causal RED/GREEN fix.
2. At the next failure retain a full mismatch-coordinate map and A/B/want/got bits, hashes immediately before each implementation, executable hash, rustc/build flags, actual worker decomposition, and the affected p=90 operands. Current first-row/column diagnostics cannot distinguish a tile-wide pattern from a single-lane transient or corruption before versus during the call. Preserve the first failing artifact before any retry/rebuild.
3. For a robust opt-in diagnostic guard, independently compute f64 dot results with a per-dot f32 forward-error bound and fail closed on breach, dumping evidence. Do not merely compare again with CpuBackend: after install it shares fastcpu matmul. Checking every dot is expensive; sampling is cheaper but cannot guard every output. Bitwise scalar comparison is useful on this fixture but is not a general floating-point-order policy.
4. An explicit independent naive/scalar implementation bypassing MATMUL, packing, and AVX dispatch is a defensible temporary opt-in fallback, with a substantial throughput tradeoff and no claim that the root cause is removed. Merely forcing one worker or default CpuBackend does not isolate shared packed arithmetic.

## Artifacts and commands

Only this directory was created/modified. No production files, commits, GPU runs, repeated BMM reruns, or shared-target rebuilds.

- `forensics.py`, `results.json`: Python counterfactuals, source hashes, tool availability/errors.
- `omission_probe.rs`, `omission_probe.exe` (and compiler sidecar if emitted), `rust-probe.json`: independent Rust LCG replay, assertion results, exact compile/run commands and exits.

From repo root:

```
python verification/foundation-wave/fastcpu-second-review/forensics.py
rustc --edition=2021 verification/foundation-wave/fastcpu-second-review/omission_probe.rs -o verification/foundation-wave/fastcpu-second-review/omission_probe.exe
verification/foundation-wave/fastcpu-second-review/omission_probe.exe
```

Python execution, Rust compilation, and the Rust probe all exited 0. The probe deliberately omits a term to test a hypothesis; it does NOT invoke fastcpu. Rustfmt only suggested formatting for the standalone probe; rustc accepted it without warnings. Sanitizer availability probes failed as recorded rather than being represented as passing tests.
