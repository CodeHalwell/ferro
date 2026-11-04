# Candidate19: retained reference/fixture failures

This isolated build does not replace the green candidate18 checkpoint.
All 970 selected inputs and 111 installed files match their source/wheel hashes
in [manifest.json](manifest.json). Both Cargo locks use cudarc 0.19.10;
the Git-ignored workspace lock is explicitly included in this source ZIP.

Full discovery ran 1,441 cases: 1,414 passed, three methods failed and 30 GPU
cases were unavailable. The 52 failure indicators comprise 50 subcases in one
Softplus test (a tuple/list stride assertion typo) and two Sqrt tests expecting
exact float64 equality with Torch's AVX512 implementation. Numeric Softplus
values/adjoints pass before the typo. The Sqrt difference is one ULP in the
saved output; each library's derivative exactly follows its own saved value.
Source20 repairs the assertions, preserving exact derivative-order, subnormal,
signed-zero and version tests and bounding these ordinary Torch F64 comparisons
by two epsilons. Corrected diagnostic tests pass on the unchanged installed19;
their next matching build is still required.

A separate first frozen linalg83 run failed an explicit-cutoff reference:
Torch's default gelsy nondeterministically reports rank0/1/2 on the same wide
diagonal matrix. An independent Torch-only process reproduces it. The source20
SVD-cutoff fixture selects actual Torch gelsd, while separate default-driver
coverage remains. Original failure and repetition logs are retained; repeated
successes are not substituted for the failed frozen run.

Native core passes 1,209 tests (2 ignored), fastcpu39 (2 ignored), CUDA140
runtime harness cases and tokenizer8 harness cases. Two intended E0502 CUDA
doctests pass; 18 nested child summaries are excluded and five strict GPU cases
filtered. Seven standalone checks, CUDA all-target compilation and native
DLPack's 12 harness cases pass. The binding regression's ReLU6(inplace=True)
constructor relied on the old silently ignored flag. Source21 tests the intended
out-of-place module and separately asserts explicit mutation rejection; this
fixture-only correction passes on installed19. The original frozen standalone
failure remains retained. No GPU execution is established.

Installed graph312, composition77 plus three network probes, training/arithmetic159,
empty-broadcast5 and Cat4 are green. The 83 earlier normalization cases pass;
the new activation/Sqrt fixture findings above remain visible. This is ongoing
CPU work, with complete framework, NVIDIA and Metal parity still open.
