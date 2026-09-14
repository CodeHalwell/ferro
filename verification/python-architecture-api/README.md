# Native Python architecture API verification

> Historical CPU verification report: wheel identity/counts below belong to that earlier run. `verify.py`, `verification.json` and raw logs referenced here are local-only archives. Packaged tests/examples and the current full verifier are listed in [PUBLICATION.md](../foundation-wave/pr-readiness/PUBLICATION.md); its final manifests supersede the earlier wheel.

## Outcome

The installed release mixed-package wheel passes 63 CPU tests: 8 architecture tests, all 17 preserved public API contracts, and 38 existing facade regressions. No GPU execution or performance claims. No core files were edited by this worker.

`verify.py` builds with `maturin build --release -j 2`, installs the wheel using the project interpreter, executes tests/examples, and records source hashes before/after. `verification.json` reports `sources_stable: true`. The environment now imports the installed wheel, not the editable source package. Rebuild or stage the source facade when making subsequent Python edits.

- Wheel SHA256: `6f343469440a3149272f46d4d89f9a8d5475fb3ba117c55b949c2da2eb8f863e`
- Imported `_native.pyd` SHA256: `8db7f4c8fa6fb60e9b38a987a756d54a2f5138815b20e0409c1ec873ab431358`

## API

- `fr.graph.segment_sum/mean/max/softmax(x, ids, num_segments)` operate on axis zero with integer topology. Empty sum/mean segments are zero; max is negative infinity. Max ties share gradients; softmax requires finite inputs.
- `fr.graph.COO(nrows,ncols,rows,cols)` and `CSR(nrows,ncols,row_offsets,col_indices)` are native immutable topology. Methods expose SpMM, SDDMM and conversion; COO also coalesces trainable values. `coo.to_csr()` returns `(csr, order)`: reorder values with `values.index_select(0, order)`. Duplicate edges remain independent until explicitly coalesced.
- `fr.recurrent.rnn_cell/gru_cell/lstm_cell` use native shared tensor parameters. GRU gate order is reset/update/new, reset-after recurrent affine; LSTM order is input/forget/candidate/output. No implicit forget bias.
- `RecurrentState(h, c=None)` exposes hidden/cell tensors and explicit `detach()`. `unroll(kind,input,initial,weight_ih,weight_hh,bias_ih=None,bias_hh=None,*,lengths,reset=None,truncate=None)` returns `UnrollOutput(outputs,state)`. Inputs are time-major; reset is a flat time-major T*B boolean sequence. Padding emits zero without evaluating input, retains state, and ignores padded resets. Truncation cuts history at positive multiples of the interval, not parameter identity.
- `fr.basis.BSplineBasis(knots,degree,outside='zero')` exposes `evaluate`, knots, degree, domain, and num_basis. `outside='error'` rejects out-of-domain input. Knots are fixed. Basis input paths reject higher-order differentiation.
- `fr.basis.kan(x,coefficients,basis)` and `fr.nn.KanLayer(basis,coefficients)` use coefficients shaped `[output,input,num_basis]`; layer creation copies coefficients into a native stable Param, with normal Module registration and SGD updates. There is no hidden base activation or random initialization.
- `fr.nn.functional.conv2d(x,weight,bias=None,stride=1,padding=0,dilation=1,groups=1)` supports scalar/pair geometry, grouped NCHW convolution, and native bias gradients. `unfold2d` and `fold2d` expose rectangular windows; fold sums overlaps.
- Cell functions and `kan` are also exported through `fr.nn.functional`.

All new computation delegates to Rust shared Tensor/autograd operations; there is no NumPy fallback. Segment/sparse/recurrent/basis preserve native CPU-f32 checks. New convolution/window bindings explicitly reject non-CPU tensors, whereas the legacy Tensor.conv2d host-fallback contract is unchanged. CUDA rejection was source-inspected, not GPU-executed. Core validation failures map to ValueError; malformed integer extraction can raise TypeError/OverflowError. No replay/fused-scan/bounded-memory guarantee is added.

## Reproduction

From repository root:

```
python verification/python-architecture-api/verify.py
crates/ferro-py/.venv/Scripts/python.exe verification/python-architecture-api/test_architecture.py
crates/ferro-py/.venv/Scripts/python.exe verification/python-api-contract/contract.py
crates/ferro-py/.venv/Scripts/python.exe verification/python-architecture-api/training_examples.py
```

Architecture tests compare recurrent cells and rectangular grouped convolution/window forward/backward with CPU Torch. Segment/sparse values and adjoints follow native core test fixtures. Spline basis and KAN compare Bernstein values/derivatives. Tests also cover dtype errors, invalid topology, ragged/reset/truncated state, empty sequence, independent coalesced gradients, and parameter identity.

Training examples define ordinary `Module.build()` models without user init boilerplate. Actual first/final MSE:

| Example | First | Final |
|---|---:|---:|
| Graph | 2.9866669178 | 0.0000541346 |
| Sequence delayed-copy TBPTT | 0.1482980251 | 2.220446e-14 |
| KAN | 0.1933799684 | 0.0000028424 |
| Convolution | 0.9720000029 | 1.276772e-9 |

## Caveats and outstanding work

- Preserved facade tests intentionally trigger and catch one native saved-version mutation panic; its diagnostic appears in `facade-regressions.log`, followed by `Ran 38 tests ... OK`.
- Build emits pre-existing unused-variable/dead-code warnings in core/CUDA. File-edit linter also incorrectly used an older Rust edition for existing c-string literals; actual release builds succeeded.
- First verifier attempted unavailable pytest; corrected to stdlib unittest without installing dependencies.
- Full CPU training-state snapshots are not exposed in Python. That is a separate follow-up worker/task; no checkpoint/training/progress implementation was changed here.
