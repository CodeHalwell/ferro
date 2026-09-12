# PR24 verification tooling review

## Findings and disposition

Full Copilot review bodies (including suppressed comments) and inline comments
were fetched with `gh pr view 24 --json reviews` and
`gh api repos/CodeHalwell/ferro/pulls/24/comments`; saved as `reviews.json` and
`comments.json` in this directory.

- Valid: static performance metadata silently omitted bindings outside one
  Windows venv, and omitted core lowering/other production dependencies.
  `summarize.py` now resolves an active or explicit native extension and rejects
  missing/ambiguous selections. It hashes all tracked/non-ignored untracked
  crate and bench files, verification source/config files, Cargo manifests/lock
  and Cargo/toolchain config. Core graph source is mandatory. The snapshot is
  explicitly current-at-summarization, NOT proof of the build used by old runs.
  Both input and fresh output directories are required; archives are not default
  write targets. External installed dependencies are not hermetically attested.
- Valid: `run_checks.py` omitted Windows CUDA runtime setup. It now validates and
  prepends NVRTC/cuBLAS directories (or explicit runtime root), retains required
  CUDA and serial test settings, and requires a new output directory. `--plan`
  does not load CUDA or run checks and explicitly reports runtime_verified=false.
  A working GPU/driver and a freshly rebuilt binding remain prerequisites.
- Valid: empty-fix collection omitted untracked production source. It now
  snapshots all tracked/non-ignored untracked crate files, including staged
  changes and untracked additions in the patch. Collection does not run tests,
  builds, or refresh old logs. Optional evidence copies are separated and labeled
  unattested against the current source; a new output directory is mandatory.
- Valid: raw-review RED/GREEN aliases were identical current-source commands.
  They are replaced with `current-test`; every phase requires fresh outputs.
  `all` still runs unit/build/eager/snapshot/integration, with isolated child
  output directories. Its integration uses the fresh-output runner rather than
  overwriting continuation evidence. Historical RED was never part of the actual
  five-phase `all` sequence, contrary to the review's wording.
- Partly inaccurate: registry-fix `run.py` has no built-in RED/GREEN phase table;
  it is a generic LABEL COMMAND wrapper with exclusive-create logs. Documentation
  now explains labels are current-source runs, recommends `current-...` labels,
  and states old RED logs cannot be reproduced/validated by relabeling a command.
- Valid: RESULTS.md used a host-specific absolute venv path. It now derives the
  path from the checkout and documents platform/runtime prerequisites. Timing
  tables and verification are explicitly historical layout/guard-stage evidence,
  not final PR24 tree performance. Incomplete historical metadata is disclosed,
  not rewritten to backfill provenance.

## Files changed

- verification/static-perf-wave/{summarize.py,run_checks.py,RESULTS.md}
- verification/static-empty-fix/{collect.py,README.md}
- verification/dlpack-raw-review/{run.py,README.md}
- verification/dlpack-registry-fix/README.md
- verification/pr24-tooling-review/test_tooling.py and this review evidence

Registry runner code itself was left unchanged because its generic command
interface and exclusive-create log behavior did not match the reported duplicate
phase implementation. No crate production/test files were changed here.

## CPU-only verification actually executed

```bash
python -m unittest discover -s verification/pr24-tooling-review -v
python -m unittest discover -s verification -p test_run_integration.py -v
git diff --check -- verification/static-perf-wave verification/static-empty-fix verification/dlpack-raw-review verification/dlpack-registry-fix verification/pr24-tooling-review
```

- `final-tests.log`: 8 tests passed. Covers missing explicit binding, missing and
  ambiguous active binding, core/untracked source hashes, fresh-output refusal,
  unrelated-cwd summary/collector/runner invocation, untracked patch additions,
  required CUDA directory failure before commands, and current-phase naming.
- `integration-runner-tests.log`: 3 existing interpreter-selection tests passed.
- `diff-check.log`: exit 0; only Git LF-to-CRLF notices.
- `binding`, `dependencies`, `untracked`, `runtime`, `collection`, and `phases`
  RED/GREEN logs document CPU fixture TDD failures and fixes, not GPU evidence.

All fixture repositories and synthetic binding/report bytes were temporary and
never loaded as native code or published as performance evidence. Runtime failure
sensitivity patched subprocess execution to prevent launching any Cargo command.
A fixture assertion was corrected to compare resolved Windows paths (8.3 temp
paths differ textually from resolved long paths).

No GPU tests, builds, installs, benchmarks, project staging, commits or pushes
were performed. Existing archived timing JSON, metadata, raw logs and source
snapshots were not rewritten. GPU integration of the updated orchestration is not
claimed; it remains for the owning integration agent. CPU tests prove tooling
behavior, not historical build identity or GPU correctness.
