# PR24 native ownership review

Base: `2adcc52b8493537ec3adff58a173bb7bc99d1297`.
Review source: `review-comments.json`, fetched with
`gh api repos/CodeHalwell/ferro/pulls/24/comments`.

## Finding validity and resolution

Copilot comment 3997749175 is valid. An inner GraphOwner Drop could lose raw
handle records after a failed fence while Flight's subsequent successful fence
released address owners without retaining legacy-capture exclusion. The same
loss existed for bind/destroy errors and uncertain capture status.

The capture session, native graph owner and outer flight now share one sticky
quarantine state. Native owner cleanup runs before address-owner destruction;
Flight checks sticky state after that cleanup, not just its own fence result.
Unresolved native cleanup retains exact remaining handle records, pending upload
stream when applicable, and context. Flight heap-retains the full address,
command, backend, stream and remaining native-owner tuple, retaining one
legacy-capture exclusion slot under the same lock. Published graph Drop uses
the original shared adapter and separates native cleanup from address Drop.

Bind failure stops destruction. Successful destruction clears its owned slot;
failed exec destruction retains exec plus graph, whereas successful exec
followed by failed graph destruction retains only graph. Ambiguous native
destroy errors are not retried: they may be deferred errors, so blindly retrying
could double-destroy. Capture cleanup checks status after an unsuccessful end;
unknown/still-active state is sticky, and does not authorize repeated end calls.
Quarantine registry and existing exclusion locks recover poison. No recursive
Drop owner is stored in the registry.

Native output contracts are unchanged: production out-parameters are null
initialized locally and transferred only after native success. Tests of partial
owned handles model successful creation followed by adapter error/panic, not
unspecified failure-output bits.

## Evidence

All logs are actual cargo output. Required-GPU runs prepend the installed
NVRTC and cuBLAS runtime DLL directories and set `FERRO_REQUIRE_CUDA=1`.
All cargo commands use `-j2`; final CUDA runs use the default parallel harness.

| Evidence | Result and interpretation |
| --- | --- |
| `inner-red.log` | Original policy: expected exclusion 4, got 3 after inner persistent failure then successful outer fence. |
| `inner-green.log` | Same regression passes after shared sticky ownership fix. |
| `status-red.log` | Before status fix: expected exclusion 4, got 3 despite unknown capture state. |
| `destroy-matrix-red.log` | Three independently failing bind/exec/graph regressions with the original destruction policy temporarily restored; plumbing and tests remain. Each loses exclusion (3 instead of 4). This is a targeted sensitivity run, not a pristine-base run. |
| `gpu-red.log` | Temporarily ignoring sticky state in Flight reproduces lost exclusion (0 instead of 1) with actual uploaded graph resources and simulated upload/fence errors. Fix restored immediately. |
| `native-final-green.log` | 26 native-module tests pass, including driver-free exact counters and real-resource injected failure sequences. |
| `cuda-default-parallel-final.log` | Full CUDA suite passes: 68 library tests plus integration/doc harnesses; parsed pass-line sum 112, zero failures. This sum is not claimed to count unique tests because nested harness summaries are included. |
| `cuda-check.log` | `cargo check -j2 -p ferro-cuda --all-targets` passes. |
| `diff-check.log` | `git diff --check` passes (line-ending warnings only). |

Driver-free regressions assert exact handle slots, pending stream, destruction
attempts, no buffer release and exact exclusion counts. They also cover capture
cleanup bind failure, cleanup end still active, cleanup graph destruction error,
partial owned handles, and a poisoned quarantine registry.

Real-resource tests exercise both pointwise and GEMM model preparation with:
- upload adapter error, inner fence skip, cleanup fence skip, outer recovery;
- end adapter error followed by bind skip;
- instantiate adapter error followed by exec-destroy skip;
- end adapter error followed by graph-destroy skip;
- end adapter error followed by capture-status uncertainty.

Each asserts retained leaf ownership and exclusion, rejection of legacy capture,
unchanged event tracking, and successful next unrelated static preparation and
numeric replay. A separate published-graph Drop test poisons the exclusion mutex,
simulates exec-destroy failure, and checks exact retained handles/leaf ownership,
legacy rejection and the next real replay.

These are controlled adapter errors/skipped calls using real CUDA resources,
NOT reproduced native driver faults. Quarantined tuples are deliberately not
reclaimed, including in these tests. The next unrelated static replay succeeding
does not clear prior quarantine or prove recovery from a fatal CUDA context.
Persistent retention can grow until process exit; restart is the reclamation
policy. Fencing while the exclusion lock is held can block preparation. This is
a safety fallback, not an availability guarantee.

`unit-first-green.log` preserves an intermediate run with two old expected-event
lists that assumed binding empty owners or continuing destruction after failed
bind. Those expectations were updated to the safer policy; all final runs pass.
`summary.json` contains programmatically parsed results; `source-hashes.json`
records the exact five CUDA source files tested.

## Scope and handoff

Modified only these CUDA files:
- `crates/ferro-cuda/src/static_graph/native.rs`
- `crates/ferro-cuda/src/static_graph/native/tests.rs`
- `crates/ferro-cuda/src/static_graph/native/gpu_tests.rs`
- `crates/ferro-cuda/src/static_graph.rs` (ownership integration)
- `crates/ferro-cuda/src/static_graph/model.rs` (shared preparation adapter)

Created this evidence directory. Existing untracked evidence was preserved.
No benchmarks, staging, commit, push, merge, or branch change. Existing compiler
warnings remain. Parent owns combined integration, binding rebuild/tests and
other concurrent changes; they are not claimed as completed by this subtask.
