# CPU architecture restart proofs - 2026-09-19

> Historical broader-checkout run. For the selected PR tree, use [final isolated verification](PR_VERIFICATION.md): 95 Python tests and separate source/build identity. The results below remain scoped to their original manifest.

## Result and scope

The two selected CPU f32 model gates passed at seeds 7, 11 and 43 with
400 optimizer steps each. The acceptance thresholds were declared before
execution in [the ledger](../../docs/CURRENT_STATUS.md): regression held-out
MSE <= 25% of a constant training-mean baseline; classification held-out
accuracy >= 95%. No threshold was changed after execution.

| Seed | Regression held-out MSE | Constant baseline MSE | Error / baseline | Classification accuracy |
|---|---:|---:|---:|---:|
| 7 | 0.0047012763 | 0.6039360672 | 0.7784% | 100% |
| 11 | 0.0032550683 | 0.6039360672 | 0.5390% | 100% |
| 43 | 0.0072796149 | 0.6039360672 | 1.2054% | 100% |

Classification uses two concentric circles with balanced I64 labels and a 50%
class-prior baseline. Regression uses sin(2.5*x) + 0.3*x*x. Train and held-out
coordinates are fixed disjoint grids. These are small synthetic interpolation
proofs, not evidence about arbitrary datasets or architectures.

Every seed compares the input and all four parameter gradients with PyTorch
before updating, at atol=1e-5 and rtol=1e-4. Maximum observed absolute gradient
difference across both fixtures was 2.3841858e-7.

Every seed saves after step 200, continues uninterrupted to step 400, and restores
the saved checkpoint in a fresh Python process into a new compatible model.
The restored optimizer and Generator deliberately start with different settings.
Subsequent losses, final held-out metrics, complete serialized checkpoint state
(except the generation directory name) and eight subsequent normal draws match
bit-for-bit. Serialized state covers parameter values, AdamW moments/counters,
configuration/schema and explicit RNG state. The child reads only the checkpoint,
not the parent's expected results.

The minibatch sequence is reconstructed from fixed data and restored step.
This does not test serialization of DataLoader/prefetch cursors. Snapshots occur
after zero_grad at optimizer-step boundaries. CUDA model training, cross-platform
bitwise reproducibility, general scheduler/accumulation state and complete B7
certification remain open. No independent reviewer or merge approval is claimed.

## Changes

- [Shared public harness](../../examples/training_restart.py) and
  [regression](../../examples/train_restart_regression.py) /
  [classification](../../examples/train_restart_classifier.py) command-line examples.
- [Automated architecture tests](../../crates/ferro-py/tests/test_architecture_restart.py),
  included by the existing CI unittest discovery command without workflow edits.
- Public `ferro.nn.functional.cross_entropy(logits, targets)`: a narrow mean
  loss binding to core's existing cross_entropy_indices implementation. CPU f32
  nonempty [N,C] logits and I64 [N] class IDs only; unsupported shape/dtype/device
  and invalid labels reject. No weighting, smoothing, ignore_index or CUDA claim.
  This adds no engine arithmetic or dependencies.

The new cross-entropy contract test first failed with AttributeError against the
old installed package. After implementation it checks values and gradients
against PyTorch, including large finite logits, and rejects floating labels,
wrong ranks/lengths, empty batches, negative/out-of-range labels and a 2^40 index.
Existing checkpoint schema/alias and late-optimizer failure tests passed in the
full discovery suite; they remain the negative-restore coverage.

## Final verification

[run-02/results.json](run-02/results.json) records 13 commands, all exit 0:

- Noneditable release wheel build/install.
- Both public examples, all three seeds.
- Python discovery: 105 tests, OK, no reported skips.
- Full ferro-core tests and ferro-fastcpu/ferro-tokenizer tests.
- Binding/DLPack regression, operator values/gradients against Torch,
  safetensors parity, compiled fusion, fusion and 200-trial seed-0 fuzz.

The architecture cases are repeated inside discovery, not extra unique seeds.
Fuzz passes its existing G4 rules, including accumulation-order percentile
exceptions; this is not a universal maximum-ULP assertion.
Existing Rust ignored cases and warnings remain; no sanitizer/Miri claim.
Compiled fusion's CPU and cuda:0 paths both passed in the final run. GPU checks
remain supplemental: this verifier does not require CUDA and does not establish
GPU architecture certification or hosted CI success.

[run-02/provenance.json](run-02/provenance.json) records Python 3.11.15,
Torch 2.6.0+cu124, Windows, package paths and hashes. The wheel was built from the
dirty checkout, not claimed to be HEAD alone. Source/wheel/installed Python bytes
and wheel/installed native bytes were compared.
[run-02/stable.json](run-02/stable.json) records 434 selected production/test/
harness/config inputs unchanged through verification, plus stable installed
package hashes. The manifest is Git-visible matching-suffix inputs, not a
complete compiler/environment attestation. Documentation was updated afterward.

Native SHA256: `db8ac4378ff5ef1de0b289331d4bf3e2b061f01db00329201538f30b662b0125`.
Wheel SHA256: `078844cbcc7d18946552b70f18c414c04f49b22080641c9882f38c4310b5e845`.

The first combined attempt, [run-01/results.json](run-01/results.json), stopped
after the compiled-fusion CPU checks because its optional GPU branch could not
find NVRTC. This failure is preserved. The final verifier prepends the installed
Torch DLL directory on Windows, then repeats the final build and suite in a new
directory. No product assertion or tolerance was relaxed.

## Reproduction

With the built package, Torch and NumPy installed:

```text
python examples/train_restart_regression.py
python examples/train_restart_classifier.py
python -m unittest discover -s crates/ferro-py/tests -p test_architecture_restart.py -v
```

Use --seed 7, --seed 11 or --seed 43 on either example for a single case.

For a fresh full build, installation into the selected interpreter environment
and regression run (requires Rust/MSVC on Windows, maturin, Torch, NumPy and
safetensors), choose an unused output directory:

```text
crates/ferro-py/.venv/Scripts/python.exe -B verification/architecture-restart/verify.py --output-dir verification/architecture-restart/run-03
```

The verifier refuses to overwrite an existing output directory. Logs, wheels
and checkpoint temporaries are local evidence; select reports/manifests explicitly
before any publication. No commits, pushes or PR actions were performed.
The pre-existing CPU numerical and illegal-instruction incidents remain open.
