# PR25 independent final review

## Recommendation: MERGE BLOCK CLEARED via bounded BMM containment

This is an independent local recommendation for head 968b3c687e636f3ced6ee8c0be52bf65749ed3a6 plus the exact source/test changes in final/staging-proposal.json. No PR status, historical blocker wording, commit, push, staging or publication was changed. The parent owns publication. The historical fastcpu numerical incident remains root-cause UNRESOLVED; do not describe it as root-cause-fixed or universally error-proof.

**Material performance cost:** production scalar BMM at [16,128,128,128] measured 13.9863/13.8991 ms versus 0.9193/0.9850 ms for explicit one-thread packed matmul run separately per slab. I independently parsed all four raw sample sets. That comparator is not the old fused BMM implementation; these are not isolated-host measurements or a peak-performance claim. The install-only MATMUL callback is scalar too, so ordinary registry matmul users also pay a throughput cost. FastCpuBackend nonbatched matmul and explicit matmul/matmul_with_threads remain packed.

## Independent findings

- Reviewed actual tracked diff and all four new source/test files, not just worker summaries. No security or correctness blocker found in this change scope. Added-line static security checks had no matches; this is not an exhaustive repository security audit.
- Direct fastcpu BMM visits every output and ascending-K product using fresh zeroed storage, checked dimension products and direct bounds-checked slices. Neither the direct API nor FastCpuBackend's override calls packed arithmetic, dispatch workers or the MATMUL registry. The historical packed BMM implementation/helpers are cfg(test) only.
- CpuBackend's install-only default BMM still uses its existing pooled aggregate output, but each slab is fully overwritten with the independent scalar callback result. Thus this route cannot reenter the suspect packed kernel either. **Do not generalize the direct implementation's fresh-output/no-pool claim to every wrapper:** the default trait wrapper retains pooling, although it performs no candidate acceptance or suspect arithmetic. There is no demonstrated pool defect; arbitrary pool/host-memory/compiler faults are outside this bounded claim.
- Traced Tensor BMM forward and both host backward products through backend.matmul_batch; FastCpuBackend overrides that method. CUDA resident BMM is a separate unchanged cuBLAS route. No additional built-in CPU BMM bypass found. Explicit custom backend/kernel registration can override library policy, as documented.
- Examined preserved RED logs: actual packed microkernel omitted-product test returned 127 instead of 128; install-only route separately returned 127 before its callback correction. Current faults sit inside the real microkernel, not a fabricated backend. Small inline fixtures intentionally exercise thread-local hooks, with positive controls for omission, NaN, infinity and finite corruption. Direct and FastCpuBackend calls stay correct while faults remain armed. The install-only omission test also passes. This is injection sensitivity/isolation proof, not a natural reproduction of the historical cause.
- Original all-output bitwise oracle checks remain intact, including original shape/seeds, K-block/tail, pool-dirtying and concurrent callers. New installed Tensor tests cover empty/K-zero, transposes, all-element values and gradients. Historical-coordinate injection bits are explicitly counterfactual, not causal diagnosis.
- copy_ selects copy_leaf_from only for a requires-grad destination under disabled recording; all other cases retain copy_from. Its underlying storage/layout/history/version safeguards remain unchanged. Required real-GPU tests check CPU/CUDA transfer directions and grad modes, forbidden leaf/nonleaf mutation, stale backward rejection and fresh gradient values.
- Native convolution defaults match facade defaults; native/default and keyword geometry, gradient values and invalid groups pass. Loss negations remain detached fallible host operations with original device restoration, avoiding a new partial-backend unary capability requirement. They are not claimed to make loss forward device-resident. The index-select deletion reuses the previously validated output shape.

## Fresh combined verification

Ran verification/pr25-final-independent/run.py as the sole build/GPU verification owner after confirming no cargo/rustc/maturin process was running. It completed with exit 0. Runtime DLL directories were explicitly prepended, FERRO_REQUIRE_CUDA=1 was set, Rust builds used -j2, and binding unit tests received the queried Python base prefix as PYTHONHOME only for that command.

- Fresh noneditable release wheel built and installed. Every Python facade file matches source, wheel entry and installed file; native extension matches wheel and installed file. Installed facade/native hashes stayed unchanged through final and supplemental tests.
- Existing unchanged 13-command integration: **13/13 exit 0**, including original Rust CPU/CUDA suites, all-target CUDA check, whole Python discovery, original examples/parity/fuzz suites and required GPU capture-layout integration.
- Whole Python discovery: **73 passed**, no skips, including the three new required-CUDA copy tests and native convolution test.
- Core suite: **759 parent-harness tests passed, 2 ignored**, plus **9 successful child-process test executions**. The worker's earlier 768 figure counted child executions as well; it is not 768 unique parent tests.
- fastcpu plus tokenizer integration: **33 passed, 2 ignored** (25 fastcpu, 8 tokenizer). Supplemental fastcpu default-parallel and release-serial runs each passed **25**, with **2 ignored performance probes**.
- CUDA suite: **112 parent-harness tests passed** in both integration serial and supplemental default-parallel runs; each also contains one child-process execution. Do not add repeated runs as unique tests.
- Standalone binding Rust suite, default parallel: **15 passed**.
- Original Python API contracts: **17 passed**. Architecture suite: **8 passed**.
- Standalone repeated copy regression: **3 passed with required CUDA**; convolution: **1 passed**.
- git diff --check passed. Existing compiler warnings and expected caught stale-version panics remain; no test failures were hidden.

## Provenance and staging

final/sources-before.json and sources-after.json freeze the same explicit production/config/test-input set before build and after tests. source-stability.json reports stable=true, changed=[], new_inputs=[], installed_stable=true. Conservative before/after manifests and explicit out-of-scope inventory are retained separately. Supplemental CPU verification rechecked the frozen input and installed hashes unchanged. Generated diagnostics/reports and post-test collectors are bookkeeping, not newly tested production source. No manifest hashes itself.

final/results.json retains every top-level command exit; final/integration13/integration-results.json retains all 13 leaf exits. parsed-counts.json retains parent versus child summaries; supplemental.json retains the independent performance parsing and extra CPU run. All raw logs are under final/. The original RED logs remain in their worker directories.

final/staging-proposal.json lists exact 11 source/test candidates, exact evidence candidates, and every remaining Git-visible untracked path excluded from this task. It is a proposal only, not permission for broad git add. Wheel, PDB, executable, target directory and unrelated archived work must not be staged. This report and final/supplemental.json are post-test evidence candidates; the inventory intentionally does not hash itself.

No production edits were needed by this reviewer. New local files are this report, run.py and the final/ verification artifacts. The source and package used in the successful checks remain the reviewed combined tree.
