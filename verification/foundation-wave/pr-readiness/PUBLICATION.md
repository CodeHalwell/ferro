# Foundation publication: documented merge blocker cleared by containment

**MERGE BLOCK CLEARED by validated SOFTWARE BMM path containment, not original root-cause resolution.** The authoritative current scope is [PR_BODY.md](PR_BODY.md), with [independent final review](../../pr25-final-independent/REVIEW.md). No merge or auto-merge is authorized. Human/bot re-review and hosted CI are separate from the local recommendation.

**Material CPU cost:** scalar BMM 13.8991-13.9863 ms versus 0.9193-0.9850 ms for sequential explicit one-thread packed matmul per slab at [16,128,128,128]. This is not historical fused BMM, not an isolated-host benchmark, and not a speedup. Install-only registry matmul is also scalar; explicit nonbatched packed APIs remain unchanged. CpuBackend's pooled aggregate BMM output remains pooled and is fully overwritten slab by slab. No pool defect or universal fault immunity is claimed.

The historical 343812-ULP incident remains root-cause UNRESOLVED. Preserve the original failing evidence and investigate independently of containment. Restoring optimized BMM requires causal investigation, renewed all-output oracle/injection/entrypoint validation and explicit review; green repetitions alone do not justify restoring it. See [investigation and containment](../../pr25-fastcpu-containment/README.md).

## Two-commit review packaging

Review corrections cover copy dispatch, native convolution defaults, detached host loss negations and redundant index-select shape construction, with executable Python regressions. The second commit contains conservative CPU BMM containment, its real-microkernel fault/installed-Tensor tests, independent verification manifests and current status documentation. The exact eleven source/test candidates are unchanged from final independent verification; no production edits were made while publishing. Working-tree SHA256 and Git-clean blob identities are distinct on CRLF systems; the publication manifest records both. Historical original proposals remain unchanged and do not describe newly published documentation.

## Evidence policy and historical precedence

Latest combined results: integration 13/13 exit 0; core 759 parent tests plus 9 child executions (not 768 unique), CUDA 112 parent tests, Python 73, binding 15, contracts 17, architecture 8. Full source/package verification is recorded in the independent report and final manifests. Repeated runs are not summed as unique tests. Older foundation/readiness/worker reports are immutable historical snapshots; their draft/blocker or no-publication statements describe their original point in time and are superseded here. The round-2 worker's 768 total includes child executions; use the independent count instead.

Only explicitly selected source, tests, reports and JSON manifests are committed. Raw logs, wheels, executables, PDBs, caches, unrelated archives and secrets are excluded, preserved locally. `verification/pr25-final-independent/run.py` and `pr25-fastcpu-containment/summarize.py` are omitted archival orchestration/collection tools; their references in reports denote local provenance, not runnable fresh-clone commands. The former has a fixed output directory; the latter requires preserved raw logs and overwrites its report. Historical `collect.py` likewise requires local archives. Do not run archival collectors to validate immutable reports. `local-log-hashes.json` records retained raw evidence hashes, without publishing logs. Source manifests may inventory local-only inputs and are not publication allowlists.

## Reproduction with published dependencies

Prerequisites for full Windows verification: Rust/MSVC, project venv at `crates/ferro-py/.venv`, maturin, torch/numpy/safetensors, and real CUDA runtime directories at `%LOCALAPPDATA%/Temp/cuda-rt/nvidia/cuda_nvrtc/bin` and `cublas/bin`. Missing GPU prerequisites fail required checks rather than silently skip. The tracked full verifier builds and installs a noneditable release wheel, sets required CUDA, and uses base PYTHONHOME only for embedded binding tests. Choose a new output directory on every attempt:

```text
python verification/foundation-wave/pr-readiness/verify.py --output-dir verification/foundation-wave/pr-readiness/rerun-02
cargo test -j2 -p ferro-core
cargo test -j2 -p ferro-fastcpu -p ferro-tokenizer
cargo test -j2 -p ferro-fastcpu --release -- --test-threads=1
cargo test -j2 -p ferro-fastcpu --release --test bmm_containment serial_cpu_cost_probe -- --ignored --exact --nocapture --test-threads=1
python verification/pr25-review-round2/run.py copy-rerun-01 @python crates/ferro-py/tests/test_pr25_copy.py
python verification/pr25-review-round2/run.py conv-rerun-01 @python crates/ferro-py/tests/test_pr25_convolution.py
```

The round-2 wrapper uses Torch's DLL directory and requires the freshly installed native wheel; use fresh labels because its log output overwrites the named file. Current tests reproduce current containment, not historical pre-fix RED states. The old `packaging_smoke.py` intentionally pins the old native binary and is historical, not the current-wheel gate. Portable Cargo tests include their checked-in support modules; Python regressions import the installed package, with no untracked helper dependency.

## Scope and remaining work

This remains one foundation PR, not completion of the roadmap. New architecture wrappers are CPU-f32, higher-order AD is partial, Python full training-state snapshots remain unexposed, and CPU checkpoint transactions require paused training/matching configuration with documented state omissions and custom-protocol responsibilities. No GPU architecture residency, all-model-family certification, universal rollback, physical power-loss guarantee or performance certification is implied. AI-assisted implementation/review/verification/publication; no exhaustive audit or human approval claim.
