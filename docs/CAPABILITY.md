# ferro: the capability plan - science, mathematics, computer science

FUTURE.md is the master plan: workstreams, sizes, sequencing, milestones.
This document is the layer underneath it. For each axis along which ferro
can become more capable, it lays out the theory that forces the design, the
concrete design grounded in today's code, and a falsifiable gate that proves
the capability landed. Read FUTURE.md for what and when; read this for how
and why. Cross-references to FUTURE.md workstreams appear as [F.n]; gates
are numbered G1-G14 and collected in section 10.

## 0. Capability, made precise

"Maximise the capability of the library" decomposes into six measurable
axes:

1. Expressiveness: the set of programs ferro differentiates correctly
   (ops x dtypes x view structures x derivative orders).
2. Numerical trustworthiness: the distance between what ferro computes and
   exact real arithmetic, as a provable bound rather than a hope.
3. Throughput: the fraction of the hardware roofline actually attained.
4. Scale: the largest model and batch trainable in a fixed memory budget.
5. Portability: where the engine runs (CPU ISAs, GPUs, wasm, bare metal).
6. Verifiability: the fraction of the above enforced by tests that would
   fail if the claim broke.

Axis 6 is ferro's founding bet and its real differentiator: gradients are
finite-difference-checked, numerics are diffed against torch, structural
claims are proven by counting backend calls. Every design below therefore
ships with its gate; a capability without a test that can fail is a rumor.

Three structural facts about today's core do most of the work in what
follows, so they are worth naming up front:

- Tensors are immutable and Arc-shared, so the autograd graph is a pure SSA
  dataflow graph by construction: no aliasing analysis, no side-effect
  ordering, and recomputation is always semantically safe.
- There is exactly one autograd mechanism (`record_fn`: inputs plus a VJP
  closure with asserted arity and shapes), so engine-level upgrades
  (higher-order, hooks, capture) are changes to one seam, not to every op.
- Dispatch is a named-kernel `Backend` trait per device, so the compiler
  and every performance workstream slot in behind an existing interface.

## 1. Differentiation: from VJP closures to a complete calculus

### 1.1 What the engine already is, in mathematics

Reverse-mode autodiff computes the vector-Jacobian product v -> v^T J_f:
the chain rule, transposed, evaluated outputs-to-inputs. The cheap-gradient
principle (Baur-Strassen; Griewank) says the full gradient of a scalar
function costs a small constant multiple (~3-4x) of one forward evaluation,
independent of the number of inputs - which is why reverse mode, not
forward mode, powers training. `record_fn`'s contract - one cotangent in,
one gradient per input out, in order - is exactly a VJP, and the engine's
arity/shape assertions are the type check on that contract. `unbroadcast`
is the adjoint of `broadcast_to`: broadcasting is a linear map, and summing
over the expanded dims is its transpose.

### 1.2 Higher-order differentiation (create_graph)

The one mathematical incompleteness in the engine: backward closures
compute through detached raw kernels, so gradients carry no graph and
cannot themselves be differentiated. Concretely, `exp` captures
`detach_copy()` of its output and its backward returns `g * y_detached`;
ask for d2/dx2 and the dependence of y on x is invisible - the second
derivative comes out silently wrong (zero contribution), not loudly wrong.

Design:

- `backward_with(cotangent: &Tensor)` for non-scalar roots: seed the
  reverse pass with an explicit cotangent, shape/device-checked against the
  root. Small, unblocks everything else here.
- A thread-local `GradMode { recording: bool, create_graph: bool }`. Under
  create_graph the engine runs backward closures with recording enabled,
  and the gradient-accumulation add in `accumulate_grad` must itself be a
  recorded op (grad = g1 + g2 is a graph node when either term has one).
- The closure rule that makes migration mechanical: express every backward
  in terms of the op's recorded inputs (already captured in the `Op` node)
  and public recorded ops - never in terms of detached snapshots. `exp`'s
  backward becomes recompute-from-input (`g.mul(&x.exp())`) under
  create_graph; purity makes recomputation exactly correct. Ops not yet
  migrated must error loudly under create_graph rather than silently
  detach - a wrong second derivative is the worst failure mode because it
  looks like an answer.

Gate G1: a second-order grad_check - double-backward Hessian-vector
products compared against central differences of the analytic gradient,
plus a torch cross-check in examples/ops_vs_torch.py.

### 1.3 Forward mode, and an oracle stronger than finite differences

Forward mode propagates dual numbers (x, x') at ~2x forward cost, giving
Jacobian-vector products. Two reasons to want it beyond API parity:

- Hessian-vector products the right way: forward-over-reverse
  (Pearlmutter) differentiates <grad f, v> in forward mode - one combined
  pass, O(1) memory beyond the gradient. Reverse-over-reverse via
  create_graph works but costs two graph traversals and more memory. HVPs
  unlock second-order optimizers, curvature diagnostics, and sharpness
  measures.
- The adjoint consistency test: for random probes u, v, check
  <v, JVP(u)> == <VJP(v), u> to rounding precision. Both sides are
  analytic - no truncation error - so it catches transposed-Jacobian and
  wrong-input bugs that finite differences blur over. This belongs in
  testkit next to grad_check and in the fuzzer (section 9).

Design: extend the recording seam with an optional JVP closure per op
(defaulting to "unimplemented, error loudly"), starting with the core op
set; linear ops get their JVP for free (it equals the op itself).

### 1.4 Checkpointing: trading FLOPs for memory, optimally

Reverse mode's memory cost is the live activation set. The theory is
exact: with s checkpoint slots and r allowed re-evaluations, a chain of
length up to C(s+r, s) can be reversed (Griewank-Walther binomial
checkpointing, REVOLVE); the practical sqrt schedule stores O(sqrt(n))
activations for one extra forward pass. ferro's immutability is a genuine
structural advantage here: recomputation cannot be corrupted by mutation,
and once RNG is counter-based (section 7) recomputed dropout masks are
bitwise identical to the originals.

Design: `checkpoint(f, inputs)` runs f without recording, emits one
synthetic op whose backward re-runs f with recording and backpropagates
through the recomputed subgraph.

Gate G2: a k-layer MLP trains with peak live activation bytes O(sqrt(k))
(measured by an allocation-counting backend), loss bitwise equal to the
non-checkpointed run.

### 1.5 Hooks and anomaly mode

Gradient hooks (a closure observing/replacing a tensor's grad during
backward) are the seam for clipping, debugging, and later DDP overlap.
Anomaly mode: when enabled, record per-node provenance at forward time and
scan each backward output for NaN/Inf, reporting the forward op that
produced the offending node. Both are cheap because there is exactly one
engine loop to instrument.

## 2. Expressiveness: views, indexing, dtypes, and the adjoint principle

### 2.1 The adjoint principle as an implementation recipe

For any op that is a linear map L, the backward is the adjoint L^T. This
turns the entire view/indexing family into mechanical work:

    broadcast      <-> sum over expanded dims (unbroadcast)   [exists]
    sum            <-> broadcast                              [exists]
    transpose      <-> transpose                              [exists]
    reshape        <-> inverse reshape                        [exists]
    gather/index   <-> scatter-add                            [index_select,
                                                               embedding]
    concat         <-> split                                  [cat]
    conv           <-> transposed conv (flipped kernels)      [conv2d]
    avg_pool       <-> uniform scatter

The recipe for every new view/indexing op: write the forward as a linear
map, take its adjoint, and let grad_check plus the adjoint test (1.3)
verify the pair mechanically.

### 2.2 as_strided: one primitive, one VJP

Every strided view - slice, narrow, expand, squeeze, transpose, flip - is
an instance of `as_strided(shape, stride, offset)`. Implementing it once
with one VJP (scatter-add through the same strided odometer into a zero
base) covers the whole family; the odometer already exists as `gather` in
tensor.rs. Two consequences worth designing for:

- With strided kernels (5.3), views stop materializing on read and become
  free metadata operations end to end.
- Overlapping views (stride patterns that alias elements) are where torch's
  semantics get treacherous. Reads through overlaps are fine (scatter-add
  accumulates correctly); in-place writes through overlaps are undefined in
  torch. ferro should detect and reject that case outright once mutation
  lands (4.1) - a place where being stricter than torch is simply correct.

### 2.3 Dtype policy: a lattice, applied strictly

Type promotion is a join operation on the lattice i64 < f32 < f64 (with
f16/bf16 slotting below f32 when they land). The design decision is not
the lattice but where it applies: core keeps today's strict rule - float
math is explicit-cast-only, no silent promotion - because silent promotion
is a top source of surprise numerics bugs, and strictness is enforceable
(`DtypeMismatch` already does). The torch-shaped Python shim [F.7]
implements torch's promotion table as data at the binding layer, where
parity is the point. f64 autograd (needed as an oracle, 3.4) requires f64
kernel paths through the same Backend trait - a dtype parameter on the
kernel kinds, not a parallel mechanism.

### 2.4 Generalized contraction: batched matmul and einsum-lite

N-D matmul semantics: (..., m, k) @ (..., k, n) with broadcasting batch
dims, decomposed into reshape/expand + bmm; the backward needs
unbroadcast over batch dims exactly like elementwise ops. einsum-lite
lowers to transpose/reshape/bmm chains; contraction ordering for
multi-factor expressions is the matrix-chain problem (dynamic programming
for chains, NP-hard in general - greedy is fine at the sizes that occur).
The transformer op set [F.2] rides on this: attention is two batched
contractions around a softmax.

## 3. Numerics: error budgets, not vibes

### 3.1 The contract

f32 is round-to-nearest with unit roundoff u = 2^-24 ~ 6e-8, and every
operation satisfies fl(a op b) = (a op b)(1 + d), |d| <= u. Everything in
this section is a corollary of that model (Higham is the reference).

### 3.2 Summation: the least trustworthy kernels in the engine

Naive left-to-right summation of n terms has error bounded by
(n-1) u sum|x_i|: at n = 10^6 the relative bound is ~6e-2. Today every
reduction is naive - `sum`, `mean`, `raw_sum_dim`, the softmax/log_softmax
inner loops, Adam's moment updates. The fixes are standard and cheap:

- Pairwise (blocked-tree) summation as the default reduction kernel:
  error ~ log2(n) u (n = 10^6 -> ~1.2e-6), and the natural SIMD
  implementation - 8/16-way unrolled independent accumulators combined at
  the end - IS a fixed-shape pairwise tree, so the accurate version is
  also the fast version.
- Kahan compensated summation (error 2u + O(n u^2)) as an opt-in kind for
  dot products, norms, and loss reductions.

Determinism corollary: floating-point addition is not associative, so the
reduction tree's shape is part of the numerics contract. Make the tree a
function of the shape only - fixed block boundaries, partials combined in
fixed order - never of `available_parallelism`. The fastcpu matmul already
satisfies this (each output element sweeps k in fixed order regardless of
the row split); keep that invariant as kernels get fancier, and mirror it
on device: per-block tree reductions with a fixed grid combination order.

Gate G3: sum of 10^7 elements with adversarial magnitude spread lands
within 1e-6 relative of the f64 reference, and is bitwise identical across
thread counts 1..N and across runs.

### 3.3 Stability conventions, codified

Already right in the tree: softmax/log_softmax use the max-shift
log-sum-exp trick; LayerNorm variance is two-pass (center, then square),
avoiding the catastrophic cancellation of E[x^2] - E[x]^2; cross_entropy
composes log_softmax rather than log(softmax). Codify the checklist every
new op is reviewed against:

- Never exponentiate before subtracting the running max (the fused
  attention kernel of 5.5 inherits exactly this discipline).
- Prefer expm1/log1p paths for 1+x structures; add them as UnaryKinds so
  backends can implement them natively.
- Document the subgradient chosen at every kink and tie (relu'(0) = 0;
  which index max picks on ties) and grad_check away from kinks - already
  the convention, now stated.
- State where epsilons sit (inside vs outside a sqrt); at low precision
  the two differ materially (see Adam in 8).

### 3.4 The oracle hierarchy

Central differences have error O(eps^2)|f'''| (truncation) plus
O(u|f|/eps) (roundoff); the optimum is eps ~ u^(1/3) ~ 4e-3 for f32 -
precisely testkit's eps, meaning the current checker already sits at the
f32 noise floor and its ~1e-2 tolerances cannot tighten without better
oracles. The ladder, each rung catching what the one below cannot:

1. f32 grad_check (exists; at its theoretical floor).
2. f64 autograd: same graph, f64 kernels (2.3), tolerances drop to ~1e-8;
   catches wrong-formula bugs hiding inside f32 slack.
3. The adjoint test (1.3): exact to rounding, catches transposition bugs.
4. Torch-parity fuzzing [F.2]: property-based shapes/strides/dtypes/special
   values through DLPack, compared in ULPs (reinterpret the bits as
   integers and difference them) - ULP distance is scale-free and
   distinguishes "same algorithm" from "close but different algorithm".

Gate G4: the fuzzer runs in CI over the op surface with p50 <= 2 ULP and
p100 <= 32 ULP against torch f32, documented per-op exceptions listed.

**Status (implemented, `examples/fuzz_vs_torch.py`):** 19 ops fuzzed over
~4-7M element comparisons/op/run across seeds, distributions {normal, wide
10^(-6..6), special: inf/nan/-0/subnormals/max}, random shapes rank 1-4,
including transposed (non-contiguous) DLPack inputs to exercise the stride
importer.
- 11 ops **bit-identical** to torch (p100 = 0): neg, abs, sqrt, relu, add,
  sub, mul, div; and softmax within 8 ULP. exp/log/tanh/sigmoid = 1-4 ULP
  (transcendental, at gate).
- gelu and log_softmax are p99<=2 (effectively bit-identical) but carry a
  near-zero tail: they compose a transcendental / logsumexp, so on inputs
  whose true result rounds toward zero torch flushes to +-0 while ferro keeps
  a ~1e-7 normal (abs diff ~1e-7, ULP-huge). Gated on p99<=32.
- 4 accumulation ops (sum_dim, mean_dim, matmul, bmm): p50<=1, p95<=7,
  p99<=33; p100 (10^4-10^5 ULP) is catastrophic cancellation on
  ill-conditioned inputs - inherent f32, reproducible torch-vs-torch across
  thread counts, NOT a parity defect. Gated on p99<=48 with headroom.

The ULP metric uses offset-binary ordering computed in uint32 (not int64,
which breaks the near-zero sign arithmetic), robust percentiles
(method="higher", never rounding a failing tail into a pass), and flushes
only the subnormal region (|x| < 2^-126) to zero as a documented FTZ
freedom - normal small values keep exact ULP identity, so a genuine
near-zero regression like 1e-7 vs 9e-7 is NOT hidden.

### 3.5 The precision ladder [F.2, F.6]

Formats: f16 (1/5/10) has u = 2^-11 ~ 4.9e-4 and overflows at 65504;
bf16 (1/8/7) has u = 2^-8 ~ 3.9e-3 with f32 range. The mixed-precision
recipe (Micikevicius et al.) follows from arithmetic, not folklore:

- Master weights stay f32 because typical update magnitudes
  |lr * g| / |w| < 2^-11 round to zero when accumulated in f16.
- Matmul/conv accumulate in f32 regardless of storage dtype (cuBLAS
  compute type; the CPU microkernel's accumulators are f32 by
  construction).
- f16 needs dynamic loss scaling: multiply the loss by S, unscale grads
  before the step; on inf/nan skip the step and halve S, double S after N
  clean steps. bf16 skips scaling (range) but keeps f32 master weights for
  the same update-ratio reason.
- Autocast is a per-op table (matmul/conv down-cast; reductions, softmax,
  norms, losses stay f32). Keep the policy as data, mirroring torch's
  tables.
- Stochastic rounding (unbiased: E[round(x)] = x) is the research-grade
  alternative to master weights; the swappable-kernel seam makes it a
  backend experiment rather than core surgery.

## 4. Memory: mutation, allocators, planning

### 4.1 Version counters: the gate to mutation [F.2]

Immutability keeps today's engine simple but forces the optimizer to
reallocate every parameter every step and forbids in-place ops. The safe
path to mutation is torch's `_version` mechanism, which ferro can adopt
exactly: each Storage carries an AtomicU64 version; every recorded op
snapshots (tensor, version-at-save) for operands its backward will read;
backward asserts versions are unchanged before use. That converts the
silent wrong-gradient bug - the worst class - into a loud error. Order
matters: counters and assertions land BEFORE the first in-place op.

### 4.2 The zero-allocation training step

Status 2026-08: 4.1 landed earlier, and the first two mutation consumers
are in - the in-place optimizer step (fused sgd_step/adamw_step kernels;
params and state keep their storage, one launch per param per step, zero
scalar uploads) and in-place gradient accumulation (accumulate_grad adds
into the stored grad when it is provably unshared: sole tensor handle AND
sole storage reference, else the allocating sum). Storage gained a
per-cell RwLock; the cell's variant and buffer identity are immutable, so
exported raw pointers survive mutation.

The HOST buffer pool landed next (pool.rs): thread-local exact-size
freelists fed by StorageCell::drop and drained by the cpu kernels,
constructors, and internal temporaries (which give their buffers back
explicitly, keeping takes and gives balanced). take_uninit's contract -
every element written before any read - is enforced by the whole test
suite: debug builds poison recycled contents with NaN, so a kernel that
misses a slot fails loudly. Gate G5's host half now PASSES:
tests/pool_zero_alloc.rs runs an MLP training step (forward, backward,
Adam) with ZERO pool misses after warmup, at bitwise-identical numerics
to a pool-free run. Still open for full G5: the device caching allocator
(4.3) so a counting device backend shows the same, and strided kernels
(5.3) to remove the remaining broadcast materializations:
- A buffer pool: recycle storage on drop into size-class freelists
  (thread-local fast path). Beyond allocator cost, this dodges page
  zeroing - a fresh vec![0f32; n] takes a page fault per 4 KB on first
  touch, a recycled buffer is warm.
- Strided kernels (5.3) remove broadcast materialization entirely.

Gate G5: with an allocation-counting pool installed, an MLP training step
performs zero new host/device buffer allocations after warmup, on both the
cpu backend and a counting device backend - the residency test's style,
applied to memory.

### 4.3 The device caching allocator [F.4]

cudaMalloc/cudaFree synchronize the device and cost microseconds to
milliseconds; per-op malloc caps GPU utilization regardless of kernel
quality. Design (torch-shaped): size-binned free lists with block
splitting (512 B granularity, separate small/large pools); blocks owned by
the stream that freed them, so same-stream reuse needs no synchronization
and cross-stream reuse is gated on an event; cudaMalloc only on miss; on
OOM, flush the cache and retry once. ferro's immutability removes the
hard part: a buffer is dead exactly when its Arc drops - no
use-after-free through aliased mutation is possible.

Gate G6: zero cudaMalloc calls per training step after warmup (counted by
a wrapping backend), and the step-time benchmark reports the
allocator-on/off gap.

### 4.4 Static memory planning (with the compiler)

Once a step is captured as a graph (6.3), every intermediate's lifetime is
a known interval over a topological clock, and buffer assignment is
offline storage allocation: NP-complete in general, but greedy best-fit by
decreasing size over lifetime intervals is near-optimal in practice. The
payoff is a single arena per step and - critically - stable addresses,
which is exactly what CUDA Graph capture (6.6) requires.

## 5. Throughput: roofline discipline

### 5.1 The model that dictates everything

Attainable throughput = min(peak_flops, AI x bandwidth), with arithmetic
intensity AI = flops/byte (Williams-Waterman-Patterson). A binary
elementwise op does 1 flop per 12 bytes (two reads, one write): AI = 1/12.
At 50 GB/s of DRAM bandwidth that caps at ~4 GFLOP/s against hundreds
available; at 2 TB/s on a GPU, ~170 GFLOP/s against tens of TFLOP/s.
Conclusion 1: no elementwise kernel is ever compute-bound - the only lever
is moving fewer bytes, i.e. fusion (6) and strided execution (5.3).
GEMM's AI grows linearly with tile size once a tile is register/cache
resident. Conclusion 2: matmul and matmul-shaped ops (conv, attention) are
the only places where kernel heroics pay; everything else is a bandwidth
problem. Discipline: every benchmark reports achieved GB/s or GFLOP/s as a
percentage of the machine's measured (not nameplate) roofline, so a
regression names the resource it lost.

### 5.2 CPU GEMM: from 6x16 to ~peak

Where fastcpu stands: a 6x16 register tile (12 ymm accumulators out of
16), k-blocked at KC=256 so the 16 KB B panel stays L1-resident, runtime
AVX2+FMA dispatch, M-split threading. That is the BLIS microkernel plus
the innermost loop of the Goto decomposition. AVX2 peak is 2 FMA ports x
8 lanes x 2 flops = 32 flops/cycle/core. What is missing, in impact order:

- Packing: copy A into contiguous MR-row panels (and B into NR-column
  panels) before the microkernel sweep. Unpacked A reads stride-k floats,
  which thrashes TLB entries and cache sets at large k; packing is O(mk)
  work amortized over n and is the difference between ~60% and ~90% of
  peak at 1024+ sizes.
- The second blocking level (MC x KC panels of A resident in L2 - the full
  BLIS five-loop structure) so large matrices degrade gracefully.
- ISA breadth behind the existing runtime dispatch: AVX-512 (32 zmm
  registers -> larger tiles), NEON for aarch64 - same microkernel shape,
  per-ISA constants.
- 2D thread decomposition (split M and N; M-only starves threads on
  wide-short outputs).

Gate G7: >= 80% of measured peak at 1024^3 f32 on AVX2, single- and
multi-threaded, and within 1.5x of OpenBLAS across the benchmark grid.

### 5.3 Strided and fused elementwise execution

Today every kernel reads materialized contiguous inputs via `to_vec`, so a
broadcast bias physically expands to the output shape before the add. The
fix is standard: kernels iterate the output index space and address each
input through (offset, strides) - a broadcast operand is a stride-0 read
that stays in a register, a transpose feeds the kernel without a copy. The
strided odometer already exists (`gather`); hoist it from materialization
into the kernels, add a SIMD fast path when the innermost dim is
contiguous (the common case), and thread over outer dims. This is also
what makes as_strided views (2.2) free.

Gate G8: bias-add on [4096, 4096] moves ~128 MB (read x, write y, plus
16 KB of bias) instead of today's ~3x that from copy-then-compute,
verified by achieved-GB/s measurement and a zero-intermediate allocation
count.

### 5.4 Convolution: an algorithm ladder, not one kernel

The naive 7-loop direct conv does O(N Cout Cin OH OW KH KW) work with no
data reuse. The ladder, each rung with its math and cost:

- im2col + GEMM: lower input windows into a [N*OH*OW, Cin*KH*KW] matrix
  and reuse the one kernel already near peak. Memory blowup factor KH*KW;
  the correct first rung, an immediate order-of-magnitude win.
- Blocked direct conv: tile (Cout, Cin, spatial) like GEMM to recover
  arithmetic intensity without the im2col copy; the right CPU steady
  state.
- Winograd F(2x2, 3x3) (Lavin & Gray): minimal filtering cuts
  multiplications 2.25x for stride-1 3x3. The catch is numerics: the
  transforms amplify rounding error, and larger tiles amplify more - keep
  it opt-in and fuzz it against direct conv (3.4).
- GPU: implicit GEMM (lowering computed in-kernel, no materialization) or
  pragmatic cuDNN bindings [F.4].

The backward pair costs no new theory (2.1): d_input is a transposed
convolution (correlation with flipped kernels), d_weight a correlation of
input with the cotangent - both lower onto the same GEMM machinery.

Gate G9: ResNet-block conv shapes within 2x of torch CPU eager;
grad_check plus fuzzer green across the stride/pad/dilation/groups grid.

### 5.5 Attention: the marquee fused kernel

Naive attention materializes S = Q K^T: O(N^2) memory, bandwidth-bound.
The entire trick is one identity - the max and normalizer of a
concatenation compose associatively (online softmax):

    m' = max(m, m_t);   l' = l * e^(m - m') + l_t * e^(m_t - m')

so softmax(Q K^T) V is computable tile-by-tile with running (m, l, O) per
query row, never materializing S (FlashAttention). Memory drops to O(N)
and the tiles are GEMM-shaped, so the kernel is compute-bound. Backward
saves only the per-row LSE (m + log l) and recomputes tiles - which is
checkpointing (1.4) applied inside a single op. The same tiling is right
on CPU with L2-resident tiles; this is not a GPU-only design. In ferro
terms: one ops_ext file, a fused forward and custom VJP, grad-checked like
any op.

Gate G10: 8k-token attention runs in O(N) memory (allocation counter
shows no [N, N] buffer), matches the naive composition at <= 2 ULP p50,
and beats it by roughly the bandwidth ratio.

### 5.6 GPU kernel maturity [F.4]

- Reductions: replace correctness-only kernels with two-phase tree
  reductions (warp shuffles within warps, shared memory across, grid
  partials combined in fixed order - which also delivers the determinism
  contract of 3.2 on device).
- Launch overhead: at ~5-10 us per launch, a 1M-element elementwise op is
  launch-bound. Cures: fusion (6) and CUDA Graphs - capture the step's
  launch DAG once, replay at ~single-launch cost; requires stable
  addresses, i.e. 4.4.
- Epilogue fusion via cuBLASLt (bias + activation folded into the GEMM) as
  the cheap fusion rung before the compiler proper.
- TF32 stays opt-in: its 10-bit mantissa silently costs ~3 decimal digits
  on Ampere+ tensor cores, which would break the ULP parity gates. Torch
  reached the same conclusion the hard way.

## 6. The compiler: where the ceiling moves [F.5]

Modern engine performance is decided by capture + fusion, not by the op
library. This is the design; FUTURE.md M5 is the milestone.

### 6.1 Why ferro is unusually well-placed

The graph already exists (record_fn nodes ARE the dataflow graph);
immutability makes it SSA, eliminating the aliasing analysis and mutation
ordering that make torch.compile hard; and ferro-cuda's nvrtc path already
compiles kernels from generated source strings cached by text - codegen
infrastructure in miniature.

### 6.2 The IR, and the one real design tension

ops_ext deliberately has no shared op enum - one file per op is the
horizontal-scaling seam - but a compiler needs semantics. Resolution: ops
optionally declare a semantic descriptor alongside record_fn:

    Elementwise(composition of Unary/BinaryKinds)
  | Reduction(kind, dims)
  | Contraction(m, k, n, batch)
  | Opaque

Anything without a descriptor is Opaque and acts as a correct-by-default
fusion barrier. The one-file seam survives; the compiler sees through the
90% that matters (elementwise chains, reductions, GEMMs).

### 6.3 Capture

A lazy mode where ops append IR nodes carrying shape/dtype/device instead
of executing; meta kernels (shape-only execution, already reserved in the
dispatch design) provide inference. Compiled artifacts cache on
(graph hash, shapes, dtypes, device); guards check the key and fall back
to eager on miss. This is torch.compile's guard model, radically simpler
here: there is no Python bytecode to trace, only ferro's own graph.

### 6.4 Fusion

Legality is trivial under SSA purity; profitability is roofline arithmetic
(5.1): fusing a k-op elementwise chain divides bytes moved by ~k. In
payoff order: (1) greedy producer-consumer pointwise fusion along
single-use edges; (2) reduction prologue/epilogue fusion (elementwise into
the reduction loop: softmax, norms, losses); (3) GEMM epilogue fusion
(bias/activation into the microkernel tail or cuBLASLt); (4) backward
fusion - the backward graph is just more IR, and fusing it is where
compiled training wins live.

### 6.5 Codegen

GPU: emit CUDA C through the existing nvrtc path. CPU, two rungs: (a) an
interpreter over fused expression trees with SIMD primitives - no codegen,
immediate win; (b) cranelift or emitted-Rust JIT for the long term.
Generated reductions inherit the fixed-tree determinism contract (3.2).

### 6.6 Whole-step compilation

Capture forward + backward + optimizer as one closed graph over the
parameters (the in-place step from 4.2 closes the loop), plan memory
statically (4.4), instantiate as a CUDA Graph. This is the configuration
where a small immutable-core engine can genuinely beat eager torch - and
it is reachable only through the prerequisites exactly as sequenced in
section 10.

Gates - G11: a fused chain of k elementwise ops issues exactly one kernel
(counting backend), with measured bytes moved ~1/k of eager. G12: the
captured MLP step beats ferro eager >= 2x (FUTURE M5) with O(1) launches
per step.

## 7. Randomness: counter-based, or determinism dies

The xorshift128+ Rng is right for weight init and wrong as a basis for
op-level randomness. Stateful sequential PRNGs make sampled values depend
on evaluation order and thread count, destroying bitwise reproducibility
(3.2) and checkpoint-recompute equality (1.4: the recomputed dropout mask
must equal the original). The fix is a counter-based RNG - Philox-4x32-10
(Salmon et al.): value = bijective_hash(key, counter), stateless, so
element i of op instance n reads counter (n, i) and gets the same value on
any thread count, any device, any execution order. The same design
underlies torch's CUDA RNG and JAX's keys. Design: keep Rng for init;
route ops (dropout first, which also forces train/eval mode plumbing
[F.6]) through Philox keyed by a per-graph seed and per-op offset.

Gate G13: dropout under checkpoint recompute is bitwise identical to the
original pass, and the mask for a given seed matches across cpu and device
backends.

## 8. Learning dynamics: the science in the training stack [F.6]

- Initialization is variance bookkeeping: for y = Wx with fan_in inputs,
  Var(y) = fan_in * Var(w) * Var(x). Keeping activation variance constant
  across depth forces Var(w) = 1/fan_in (linear/tanh regimes; Glorot
  averages fans) or 2/fan_in (ReLU halves the second moment; He - what
  Linear hardcodes today). Plan: an init registry (kaiming/xavier/
  orthogonal with the standard gain table), because He is wrong for tanh
  networks and output heads.
- AdamW, not Adam-with-L2: under adaptive preconditioning, L2-in-the-
  gradient gets divided by sqrt(v) and stops regularizing exactly the
  high-variance weights it should; decoupled decay (w -= lr * wd * w
  outside the adaptive update) restores the intended penalty (Loshchilov &
  Hutter). Related numerics: eps inside vs outside the sqrt changes
  behavior at low precision - document and test the choice.
- Gradient clipping by global norm: compute the norm with pairwise/
  compensated reduction (3.2), after unscaling (3.5); the inf-check,
  clip, and step must be atomic with respect to the loss scaler.
- Schedulers, decay masks (no decay on norms/biases), weight EMA: small
  and table-driven; they land with the optimizer rewrite that moves state
  onto the parameter's device (4.2).

Gate G14: convergence equivalence - same seed, same data order, ferro vs
torch AdamW on an MNIST-class MLP produces loss curves matching within a
pre-registered noise band; and within ferro, run-to-run curves are
bitwise identical (3.2 + 7).

## 9. Validation as science [F.2, F.8]

The proof program extends the house style - count the calls - to numerics
and performance:

- The torch-parity fuzzer in CI: property-based shapes, strides, dtypes,
  and special values (inf, nan, -0, denormals), ULP metrics (G4). This is
  the single highest-leverage correctness investment: it upgrades
  "validated on examples" to "validated on distributions".
- The oracle ladder per op (3.4): f32 grad_check, f64 grad_check, adjoint
  test, torch diff - each catching what the previous cannot.
- Roofline-annotated benchmarks: criterion suites reporting % of measured
  peak bandwidth/compute per kernel class, tracked with regression alerts
  [F.3, F.8], so a regression names the resource it lost.
- Structural counters as first-class test backends: kernel counts (fusion
  G11, residency - exists), allocation counts (pools G5, checkpointing
  G2, attention G10), launch counts (G12), version assertions (4.1),
  bitwise determinism re-runs (G3, G13, G14).
- End-to-end credibility: FUTURE M3's safetensors transformer inference
  gains a logit-parity gate (<= 1e-4 relative vs torch f32); M4's
  training demo gains G14.

## 10. Dependency structure and the gate ledger

Hard prerequisites (a -> b: a must land first):

    version counters (4.1) -> in-place ops (4.2) -> zero-alloc step (G5)
        -> optimizer state on device (8) -> whole-step capture (6.6)
    backward_with (1.2) -> create_graph (1.2) -> reverse-over-reverse HVP
    JVP seam (1.3) -> forward-over-reverse HVP, adjoint test -> fuzzer
    Philox (7) -> dropout -> checkpointing with dropout (1.4, G2)
    strided kernels (5.3) -> fusion profitability (6.4), free views (2.2)
    semantic descriptors (6.2) -> capture (6.3) -> fusion (6.4) ->
        codegen (6.5) -> memory planning (4.4) -> CUDA graphs (6.6, G12)
    f64 kernels (2.3) -> f64 grad_check oracle (3.4) -> tighter gates for
        every op that follows
    pairwise reductions (3.2) -> determinism contract -> G3, G13, G14

Sequencing intuition: the cheap, high-leverage foundations are the oracle
upgrades (3.2, 3.4), version counters (4.1), Philox (7), and strided
kernels (5.3) - each small, each raising the floor under everything later.
The compiler (6) is the long pole and the largest single win, and it
consumes the rest as prerequisites. FUTURE.md's milestone ordering (M1
GPU validation through M6 differentiators) is unchanged by this document;
this is the technical depth underneath it.

The gates: G1 double-backward gradcheck; G2 O(sqrt(k))-memory
checkpointed training, bitwise-equal loss; G3 accurate + bitwise-stable
reductions; G4 ULP-bounded torch fuzzer in CI; G5 zero-allocation
training step; G6 zero cudaMalloc per step; G7 >= 80%-of-peak CPU GEMM;
G8 roofline-verified strided bias-add; G9 conv within 2x of torch CPU;
G10 O(N)-memory fused attention; G11 one kernel per fused chain; G12
compiled step >= 2x eager with O(1) launches; G13 bitwise-reproducible
dropout across recompute and devices; G14 convergence equivalence vs
torch. Each is falsifiable, most are cheap, and together they turn
"maximally capable" from a slogan into a checklist.

## References

- Baur & Strassen, The Complexity of Partial Derivatives (1983).
- Griewank & Walther, Evaluating Derivatives, 2nd ed. (2008) - reverse
  mode costs, binomial checkpointing / REVOLVE.
- Pearlmutter, Fast Exact Multiplication by the Hessian (1994).
- Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed. (2002).
- Williams, Waterman & Patterson, Roofline: An Insightful Visual
  Performance Model for Multicore Architectures (2009).
- Goto & van de Geijn, Anatomy of High-Performance Matrix Multiplication
  (2008); Van Zee & van de Geijn, BLIS: A Framework for Rapidly
  Instantiating BLAS Functionality (2015).
- Lavin & Gray, Fast Algorithms for Convolutional Neural Networks (2016).
- Milakov & Gimelshein, Online Normalizer Calculation for Softmax (2018);
  Dao, Fu, Ermon, Rudra & Re, FlashAttention: Fast and Memory-Efficient
  Exact Attention with IO-Awareness (2022).
- Salmon, Moraes, Dror & Shaw, Parallel Random Numbers: As Easy as 1, 2, 3
  (2011).
- Micikevicius et al., Mixed Precision Training (2018).
- Loshchilov & Hutter, Decoupled Weight Decay Regularization (2019).
