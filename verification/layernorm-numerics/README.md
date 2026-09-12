# Independent LayerNorm numerical validation

Run from the repository root. These commands use the existing bindings venv
under Windows Git Bash; on POSIX replace `Scripts/python.exe` with `bin/python`.
CPU-only helper, oracle, and CLI regression tests do not require a binding rebuild:

```bash
crates/ferro-py/.venv/Scripts/python.exe -W error -m unittest discover -s verification/layernorm-numerics -p 'test_*.py' -v
```

Run the numerical sweep **after the combined binding rebuild**:

```bash
crates/ferro-py/.venv/Scripts/python.exe verification/layernorm-numerics/validate.py --device cuda:0
```

`--eps` must remain finite and strictly positive after f32 rounding. Values such
as `1e-50` (rounds to zero) and `1e40` (overflows to infinity) exit with argparse
status 2 before importing torch/ferro or touching the report. Positive finite f32
values, including subnormals, are accepted. To reproduce rejection without a GPU:

```bash
crates/ferro-py/.venv/Scripts/python.exe verification/layernorm-numerics/validate.py --device cpu --eps 1e-50
crates/ferro-py/.venv/Scripts/python.exe verification/layernorm-numerics/validate.py --device cpu --eps 1e40
```

The CLI regression tests explicitly block torch and ferro imports, so even a
regression cannot start a sweep or initialize CUDA.

A smaller first run (still all affine options, distributions, ranks, and gradients):

```bash
crates/ferro-py/.venv/Scripts/python.exe verification/layernorm-numerics/validate.py --device cuda:0 --seeds 17 --widths 3 33 129 --eps 1e-5 --trials 3 --output verification/layernorm-numerics/smoke.jsonl
```

`--device cpu` explicitly tests CPU bindings instead. No dependency installs or
builds are performed. CUDA availability failure is fatal; there is no silent
CPU substitution. The helpers and `--help` do not import ferro or use CUDA.

## Coverage and independence

- Seeded NumPy input generation, independent of the kernel implementation and
  benchmark fixtures. Seed/width/trial identify a reproducible RNG stream.
- Separate no-affine, weight-only, bias-only, and weight-plus-bias cases. Weights
  are nonuniform signed uniform draws; biases are independent Gaussian draws.
  Width-one affine parameters are necessarily scalar.
- Widths 1, 3, 31, 33, 127, 129, 257, 513, 1025; three trials exercise ranks
  1, 2, 3 (one, five, six rows). Add larger widths with `--widths` as needed.
- Normal, per-element wide-scale, constant, near-constant, and large-offset
  rows; two positive eps values, rounded identically to f32 in both libraries.
- Both inference (no inputs require gradients) and autograd forward paths.
  Backward checks input and every present affine parameter with a random,
  nonuniform upstream vector via `(output * upstream).sum().backward()`.
- Independent DLPack import/export checks precede testing. Producers are kept
  alive and CUDA explicitly synchronized across framework boundaries.
- Output residency is asserted; gradient residency is reported, not asserted.
  This is numerical validation, not proof of fused dispatch or zero-copy import.

## Metrics and acceptance policy

Each JSONL case records p50/p95/p99/p100 absolute, relative, and **strict f32 ULP**
errors for ferro vs device-local torch f32, ferro vs CPU torch f64, and torch f32
vs CPU torch f64. The f64 oracle starts from the exact same f32 inputs. For ULP
only, the f64 result is rounded to f32; absolute and relative comparisons retain
f64 precision. Percentiles use `method="higher"` without integer truncation.
The report retains each distribution/shape/affine/mode separately; it does not
average per-case percentiles or mislabel them as global element percentiles.

Signed zeros share one ordered integer key. Subnormals are **not** flushed.
Both NaNs map to zero ULP, but **any nonfinite result still fails acceptance**
(the generated inputs are finite). A one-sided NaN, unequal infinity, or
infinity/finite pair gets sentinel ULP `2**32 - 1`. Relative errors exclude exact
reference zeros and explicitly count them. ULP percentiles are also split at
`abs(reference) < 1e-5`, without zeroing or hiding that near-zero tail. The worst
ULP index, values, and absolute error are retained, along with input SHA-256.

Only **ferro vs CPU torch f64** determines numerical success. Device-local torch
f32 is not assumed exact for cancellation-sensitive rows; its discrepancies are
reported independently. The `passed` flags on other comparisons are diagnostic.
No universal 1-ULP (or unmeasured p99-ULP) gate is claimed for this reduction.

The all-element mixed-error gate is `abs_error <= atol + rtol * abs(reference)`:

- Ordinary/wide/constant cases: forward `atol=rtol=5e-5`; gradient
  `atol=rtol=3e-4`. These are prospective f32 reduction regression targets, not
  measured maxima. Backward includes additional reductions and cancellation.
- Near-constant/large-offset cases add a normwise conditioning allowance to
  `atol`: `c * 2^-24 * (kappa - 1) * max(1, max(abs(reference)))`, with `c=8`
  forward and `c=32` backward. `kappa` is the largest row
  `max(abs(x)) / sqrt(var64(x) + eps)`, floored at 1. Exactly constant rows use
  1, so degenerate width-one cases cannot gain a large-offset exemption.
  Centering loses absolute precision proportional to the input offset;
  normalization amplifies it by reciprocal standard deviation. The larger
  backward coefficient allows additional reductions. This is a declared
  first-order engineering budget, **not a rigorous universal bound**. Each
  actual allowance and condition number is recorded for review.

The stress allowance is deliberately tensor-normwise and may permit relatively
large errors in a small component. Raw absolute/relative/ULP distributions remain
visible. Do not increase coefficients in response to failures without separately
inspecting a reproducer and documenting the cause. A passing tolerance gate is
not a claim of bitwise parity. An omitted-affine mutation should still be caught
by ordinary cases; an entirely broken normalization is not excused by stress
allowances.

## Artifacts and status

`results.jsonl` is written incrementally with metadata, all case records, and a
final count-checked summary. Exceptions are case failures, not skips. An
interrupted file without a summary is incomplete evidence. Exit status is nonzero
for any failed case or coverage-count mismatch. Rerunning overwrites the selected
output path; choose another `--output` to retain a previous run.

CPU-only helper/oracle tests passed during authoring (four unittest tests in both
the available Python environment and the bindings venv), and CLI help was
exercised. A separate CPU torch-f64 sweep exercised 360 oracle cases and found
all 720 returned output/gradient arrays finite. **No ferro numerical run, GPU
execution, or build was performed during authoring.** Run the commands above
after rebuilding; this
folder does not contain fabricated passing GPU evidence.
