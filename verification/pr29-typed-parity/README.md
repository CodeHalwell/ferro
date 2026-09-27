# PR29 typed parity follow-up

Hosted run [36318418456](https://github.com/CodeHalwell/ferro/actions/runs/36318418456)
failed at `examples/ops_vs_torch.py`: correct integer `tolist()` output was
compared with an explicitly float-cast torch arg-reduction reference. The original
publication collector ran unit discovery but omitted the standalone operator
acceptance script. This is a harness contract failure, not a native arithmetic
failure or evidence that the unresolved historical crashes are fixed.

The same `Long did not match Float` failure was reproduced locally before editing.
A new comparator regression also demonstrated that `torch.allclose` accepts
adjacent large int64 values incorrectly for this exact-index contract. The fixed
harness compares integer values exactly, checks Python scalar types and shapes,
and retains the original floating tolerances only for floating operations.
Argmax, argmin and topk references no longer cast indices to float. The workflow
script audit also corrected the float index expectation in py_regression and
replaced the stale safetensors lossy-tolist assumption with exact i64/f64 host-read
checks, retaining the original re-save checks. No native implementation changed.

## Executed acceptance commands

From the isolated worktree, the append-only collector `run.py` resolves `python`
to this worktree's `.venv/Scripts/python.exe`. These are the actual invocations:

```text
python verification/pr29-typed-parity/run.py ops-red python examples/ops_vs_torch.py
python verification/pr29-typed-parity/run.py contract-red python crates/ferro-py/tests/test_ops_parity_contract.py -v
python verification/pr29-typed-parity/run.py contract-green python crates/ferro-py/tests/test_ops_parity_contract.py -v
python verification/pr29-typed-parity/run.py ops-green python examples/ops_vs_torch.py
python verification/pr29-typed-parity/run.py binding-regression python examples/py_regression.py
python verification/pr29-typed-parity/run.py python-discovery python -m unittest discover -s crates/ferro-py/tests -v
python verification/pr29-typed-parity/run.py compiled-fusion python crates/ferro-py/examples/py_compiled_fusion_regression.py
python verification/pr29-typed-parity/run.py compiled-fusion-runtime python crates/ferro-py/examples/py_compiled_fusion_regression.py
python verification/pr29-typed-parity/run.py fuse python crates/ferro-py/examples/py_fuse_regression.py
python verification/pr29-typed-parity/run.py safetensors python examples/safetensors_vs_python.py
python verification/pr29-typed-parity/run.py fuzz python examples/fuzz_vs_torch.py --trials 200 --seed 0
python verification/pr29-typed-parity/run.py core cargo test -p ferro-core
python verification/pr29-typed-parity/run.py fastcpu cargo test -p ferro-fastcpu
python verification/pr29-typed-parity/run.py tokenizer cargo test -p ferro-tokenizer
python verification/pr29-typed-parity/run.py cuda-check cargo check -p ferro-cuda --all-targets
python verification/pr29-typed-parity/run.py cuda cargo test -p ferro-cuda
python verification/pr29-typed-parity/run.py python-discovery-runtime python -m unittest discover -s crates/ferro-py/tests -v
```

The RED commands exited 1 as expected. GREEN comparator tests passed (5), the
standalone ops script reached ALL OPS MATCH TORCH, and every workflow acceptance
command above ultimately exited 0. Full discovery ran 119 tests; the first run
skipped one CUDA test, while the runtime-configured run had no skips. The fuzzer
passed all 19 existing operator gates with unchanged tolerances/exceptions.
Core, fastcpu, tokenizer, CUDA tests and all-target CUDA compilation passed.
Ignored performance tests were not run; the optional tokenizer real-vocab body
remains unavailable. No benchmarks were run.

## Environment and failed setup boundary

This is local Windows verification, not an identical hosted Linux environment.
The existing isolated release wheel and installed package hashes remained identical
to `verification/reconciled-publication/installed-after.json`; no wheel rebuild
was needed for test-only changes. Local references remain torch 2.6.0+cu124,
numpy 2.4.6 and safetensors 0.8.0, versus the workflow's CPU torch 2.13.0 and
numpy 2.4.3. Hosted builds and their checks remain a separate acceptance gate.
The Linux-only DLPack resource leak check skips on Windows.

The first compiled-fusion attempt passed CPU checks then failed CUDA initialization
because NVRTC was absent from PATH. Its log and nonzero receipt are preserved.
The collector was then configured with the same local runtime DLL directories as
the publication collector and CUDA_VISIBLE_DEVICES=0 (initially empty), since
ferro detects the installed driver even with an empty Torch device mask. Compiled
fusion and fuse subsequently exercised CPU and CUDA correctness; the final full
discovery also passed. These runtime settings differ from the CPU-only hosted job.

Raw logs/receipts are append-only local artifacts in this new folder and are not
committed; prior evidence is untouched. The collector and this curated report are
shipped. Historical numerical/native-crash questions remain unresolved. No fresh
independent reviewer was available to this one-shot subagent; human review is still
required. AI assisted diagnosis, test changes and verification.
