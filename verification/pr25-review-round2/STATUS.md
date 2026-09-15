> Historical snapshot: current publication status and corrected parent/child test counts are in [PUBLICATION.md](/verification/foundation-wave/pr-readiness/PUBLICATION.md). The documented merge blocker is cleared by independently validated software BMM containment; historical root cause remains unresolved. Omitted logs/collectors are local provenance.

# PR25 review round 2 - implementation handoff (not publication)

Head fetched: 968b3c687e636f3ced6ee8c0be52bf65749ed3a6, OPEN. All paginated review bodies and inline comments are saved in reviews.json/comments.json, including Copilot's two suppressed comments. No commit, push, review reply, resolve, or merge was performed. Independent check remains required.

## Assessment

- Codex 4009708590 and suppressed Copilot copy finding: fixed native dispatch. Only requires-grad destinations under disabled recording use copy_leaf_from; all others use copy_from with its existing guards. CPU/CUDA transfers both directions now pass in either grad mode. Requiring-grad leaves still reject ordinary enabled-mode writes, nonleaf and cross-device leaf writes remain rejected, stale versions raise on CPU and CUDA, and fresh backward yields exact expected gradients.
- Suppressed native convolution finding: bias=None, stride=[1,1], padding=[0,0], dilation=[1,1], groups=1 now match facade defaults. Native defaults and keyword geometry agree with facade; gradients and invalid groups are tested.
- Copilot 4009648997: BCE detached target derivative now uses fallible raw unary negation on explicitly host-placed input, then restores derivative device. This avoids a redundant host materializing pass for contiguous CPU inputs, without adding a required unary_dev capability to partial backends. Forward and derivative raw_binary already compute on host. Device inputs still transfer; this is NOT a resident-forward or zero-transfer improvement. A directly device-dispatched negation would introduce a new backend capability requirement; deliberately not introduced silently.
- Copilot 4009649060 / 4009649106: Huber/smooth-L1 dx already lives on CPU at this point. raw_unary_k borrows its contiguous CPU slice instead of copying dx.to_vec(), makes a detached negation, then places dt on the original device. No autograd edge is introduced. These were safe cleanup/refactors under already-green value, gradient, partial-backend, saved-version, and backward transfer-counter tests, not newly discovered numerical bugs.
- Copilot 4009649153 / 4009649200 are duplicate comments on index_select.rs. Removed the genuinely redundant inner fast-path out_shape allocation; uses the already overflow-checked shape. The separate public index_select entry still performs its own validation, as required for direct callers. cat.rs has no current review finding and no repeated output shape construction; left unchanged. Both dtype/shape and index/cat gradient suites pass.
- Prior checkpoint finding 4009508125 was fixed/replied on head before this work; not claimed as a new fix here.

## Executed evidence

run.py prepends this venv's Torch/lib DLL directory and sets FERRO_REQUIRE_CUDA=1. @python resolves to crates/ferro-py/.venv/Scripts/python.exe. Logs retain full stdout/stderr. Commands were run from repository root:

```
python verification/pr25-review-round2/run.py copy-red-corrected @python crates/ferro-py/tests/test_pr25_copy.py
python verification/pr25-review-round2/run.py conv-red @python crates/ferro-py/tests/test_pr25_convolution.py
python verification/pr25-review-round2/run.py core-before cargo test -p ferro-core -j2 --test op_bce_with_logits_loss --test op_huber_loss --test op_smooth_l1_loss --test foundation_losses --test op_index_select --test op_cat --test foundation_tensor
python verification/pr25-review-round2/run.py core-green cargo test -p ferro-core -j2 --test op_bce_with_logits_loss --test op_huber_loss --test op_smooth_l1_loss --test foundation_losses --test op_index_select --test op_cat --test foundation_tensor
python verification/pr25-review-round2/run.py build @python -m maturin build --release -j2 --manifest-path crates/ferro-py/Cargo.toml --out verification/pr25-review-round2/wheels
python verification/pr25-review-round2/run.py install @python -m pip install --force-reinstall --no-deps verification/pr25-review-round2/wheels/ferro-0.0.1-cp311-abi3-win_amd64.whl
python verification/pr25-review-round2/run.py copy-green @python crates/ferro-py/tests/test_pr25_copy.py
python verification/pr25-review-round2/run.py conv-green @python crates/ferro-py/tests/test_pr25_convolution.py
python verification/pr25-review-round2/run.py contracts17 @python verification/python-api-contract/contract.py
python verification/pr25-review-round2/run.py grad-mode @python crates/ferro-py/tests/test_grad_mode.py
python verification/pr25-review-round2/run.py core-full cargo test -p ferro-core -j2
python verification/pr25-review-round2/run.py installed @python verification/pr25-review-round2/installed.py
python verification/pr25-review-round2/run.py cuda-check cargo check -p ferro-cuda -j2
python verification/pr25-review-round2/run.py architecture @python verification/python-architecture-api/test_architecture.py
```

Results: corrected copy RED has exactly two cross-device/no-grad errors; convolution RED has missing-defaults TypeError. GREEN: copy 3 tests including required real GPU, convolution 1, contracts 17, grad-mode 7, architecture 8, targeted core 28, full core 768 passed/2 ignored/0 failed (includes doc tests), CUDA check passed. Core foundation_losses includes counting-backend assertions for no backward host transfers and gradchecks. Installed extension hash equals wheel extension hash, saved with source hashes in installed.json. Hashes are handoff snapshots, not a claim of whole-repository source stability during sibling fastcpu work.

## Disclosed limitations / false starts

Initial copy-red.log included a mistaken test expectation that mul's saved CUDA input is always a detached shared snapshot. Inspection showed this graph retains a versioned input handle; corrected test requires stale-backward rejection on both devices. Corrected RED was rerun before dispatch fix.

A proposed Python loss oracle failed because these losses are not bound in Python (losses-gpu.log); the invalid test was removed, not replaced with fabricated coverage. Loss verification uses actual Rust gradient/numerical and counting-backend tests, not a claimed real-CUDA loss run. Real-CUDA validation in this round is the copy regression matrix.

Existing compiler unused-variable/dead-code warnings remain. Automatic patch lint on lib.rs incorrectly defaulted Rust edition for pre-existing C string syntax; release Cargo build succeeded. Expected caught stale-version panics appear in passing Python logs.

The unrelated fastcpu numerical incident remains open and is owned by the sibling worker. Its concurrent changes were not edited or certified here. These local fixes do not by themselves establish PR-wide readiness.
