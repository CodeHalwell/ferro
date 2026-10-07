# CPU GEMM incidents: investigation and retirement of the BMM containment

Scope: the two open P0 CPU investigations in docs/CURRENT_STATUS.md (the
historical 343812-ULP BMM discrepancy and the Windows 0xc000001d exit), and
the replacement of fastcpu's GEMM. Host for everything below: Linux x86_64,
4-vCPU Xeon (family 6 model 207, AVX-512F/AVX2/FMA), rustc 1.97.0. The
original incident host (Windows, x86_64-pc-windows-msvc, 32 logical CPUs) was
not available, so neither incident was reproduced. No root cause is claimed.

## 1. The 343812-ULP discrepancy

What was already established (fastcpu-second-review, pr25-fastcpu-containment):
one output, (batch 14, row 14, col 76) of the (16,128,128,128) fixture,
equals the sequential dot with exactly the p=90 product omitted, bit for bit;
earlier coordinates that would share a persistently corrupted operand were
not reported as mismatches; hundreds of reruns on the same host passed.

Mechanisms ruled out here, by source audit of the retired kernel (HEAD
`lib.rs` before this change) plus the tests named:

| Suspect | Finding |
|---|---|
| Stale or uninitialised packing buffer | `pack_b` writes every `kc*NR` lane of every panel it hands the kernel (zero tail included) and `pack_a` every `kc*MR` lane (zero rows included) before each use; buffers are per-call Vecs. A stale read would also be deterministic for a fixed decomposition, and reruns with the 32-worker split passed. |
| Edge tiles (m, n, k not multiples of 6/16/256) | The failing coordinate is interior: k=128 is one K block, p=90 is not a tail, row 14 / col 76 sit in full MR/NR tiles. Edge shapes were and are covered exactly. |
| Accumulating into C without zeroing | `acc` starts at 0 for pp=0 and only reloads C for pp>0; an un-zeroed C would add a whole value, not remove one product. |
| Race on a C tile | Outputs are `chunks_mut` partitions; packing scratch is thread-owned; the only `unsafe` is the target-feature call. Safe Rust excludes a data race on this path. |
| FMA vs non-FMA mismatch | Rust never contracts `a*b + c` into FMA, so the AVX2 and scalar bodies performed identical IEEE operations (which is why they were bitwise equal). A rounding-mode difference cannot remove one product. |
| Deterministic code bug in general | Inputs are a fixed LCG; any deterministic defect would recur on rerun with the same decomposition. It never did. |

What remains consistent with the evidence is a transient fault below the
source level (hardware, OS or a miscompile hit nondeterministically). Note
that the same host also produced the unexplained 0xc000001d exit (section 2)
and a rustc STATUS_ACCESS_VIOLATION (fastcpu-second-review). Three
unexplained process faults on one 32-thread host is the classic signature of
an unstable CPU (for example the 2024 Intel 13th/14th-gen Vmin-shift
instability, whose documented symptoms include 0xc000001d crashes and
compiler crashes). This is a hypothesis, not a finding: the CPU model,
microcode and BIOS of that host were never recorded.

Status: **root cause still unknown; code-level mechanisms ruled out as
above.** The suspect kernel no longer exists (section 3), and the new
`FERRO_GEMM_CHECK=1` guard turns any recurrence into a failing run with
evidence instead of a silent wrong value.

Next evidence to collect on the original host: CPU model and microcode
(`Get-CimInstance Win32_Processor`), a vendor stress/diagnostic run, and
`FERRO_GEMM_CHECK=1 cargo test -p ferro-fastcpu` in a loop.

## 2. The Windows 0xc000001d (STATUS_ILLEGAL_INSTRUCTION) exit

The recorded crash was the debug lib-unittest binary of ferro-fastcpu.
Hypothesis tested: AVX/FMA code reachable without a runtime feature check
(a `#[target_feature]` body inlined into an ungated caller, or a build with
`target-cpu=native`).

- No `.cargo/config*`, `RUSTFLAGS`, `target-cpu` or `target-feature` setting
  exists in the repository, its Cargo manifests or its CI workflow.
- `vexscan_asm.py` (this directory) lists every function containing a
  VEX/EVEX-encoded instruction in `rustc --emit=asm` output and every direct
  call into it. For HEAD before this change, built exactly like the crashing
  binary (`--target x86_64-pc-windows-msvc --profile test`, debug,
  codegen-units=1): only `matmul_rows_avx2`, `matmul_rows_scratch_avx2`,
  `elementwise::unary_chunk_avx2` and `elementwise::binary_chunk_avx2`
  contain VEX code, and each is called only from its dispatcher
  (`matmul_rows`, `matmul_rows_scratch`, `unary_chunk`, `binary_chunk`), all
  of which test `is_x86_feature_detected!("avx2")` and `("fma")` first.
  ferro-core (same target, debug) contains no VEX instruction at all. The same
  scan on the Linux debug test binary gives the same four functions.
- For the new code the scan shows VEX/EVEX only in `gemm::avx2_kernel::<R>`,
  `gemm::avx512_kernel::<R>`, the out-of-line `core::arch` intrinsic shims
  they call (debug builds do not inline those) and the elementwise AVX2
  functions. The kernels are reached only through `Isa::params`, after
  `sgemm_isa` asserts `Isa::supported()`.

Ruled out: ungated SIMD in ferro-fastcpu or ferro-core for the crashing
build configuration. Not ruled out: host CPU instability (section 1), or a
`ud2` from `core::intrinsics::abort` (e.g. Arc refcount overflow; no
plausible path found). Status: **unexplained; not reproduced.**

Reproduce the scan:

```
rustup target add x86_64-pc-windows-msvc
cargo rustc -p ferro-fastcpu --lib --target x86_64-pc-windows-msvc --profile test -- --emit=asm -C codegen-units=1
python3 verification/cpu-gemm-incidents/vexscan_asm.py target/x86_64-pc-windows-msvc/debug/deps/ferro_fastcpu-*.s
```

(The final link fails without link.exe; the .s file is written before that.)

## 3. Retiring the containment

The production BMM containment (unpacked scalar arithmetic, ~14x slower) and
the test-only historical packed kernel are removed. `matmul`, `matmul_batch`,
`install()` and `FastCpuBackend` all use `gemm.rs`, a new explicit-intrinsics
kernel, not the suspect autovectorised one. Its contract is stronger and
fully tested:

- Each output is one fused multiply-add chain in ascending k from +0 on the
  SIMD paths (separate mul/add on the scalar fallback). Blocking never splits
  the chain and threads own disjoint outputs, so results are bitwise
  independent of thread count, blocking and batch split, and the AVX2 and
  AVX-512 kernels agree bitwise.
- `gemm_tests.rs` checks every ISA bit for bit against a plain `f32::mul_add`
  loop over M/N/K edges of every tile and block size, with NaN-poisoned
  outputs and packing scratch; thread counts 1-32 (row and column splits),
  repeated; concurrent callers; specials; batched vs per-slab; the historical
  fixture; and the f64 gamma(k) bound. bmm_parity.rs keeps its independent
  oracle with the fused chain.
- `check.rs` is the independent f64 forward-error check (bound gamma(k) *
  sum|a_p b_p|, valid for any order with or without FMA). The historical
  wrong output is 314x over its bound; a test proves the check flags it.

Numerics therefore differ from the old non-fused results by rounding only
(owner-approved tolerance over bitwise parity with the non-fused reference).
