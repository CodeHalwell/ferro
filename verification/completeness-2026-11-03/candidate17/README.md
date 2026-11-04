# Candidate 17: failed full comparison retained

This isolated installed build does not replace the green candidate16 report.
All 951 selected inputs, including 64 aggregation checkpoint fixture files,
and all 106 installed Python/native files match the frozen snapshot. Exact
source, wheel and log hashes are in [manifest.json](manifest.json).

Python discovery ran 1,362 cases: 1,335 passed, three methods failed or errored,
and 30 GPU cases were unavailable (24 individual skips plus one class hiding
six cases). The three methods exposed an opaque square higher-order rule,
an overly early U64-alpha AD rejection plus an incorrect test expecting Torch
to reject forward, and a stale cast-input version assertion. Actual Torch
permits cast-source mutation because the cast adjoint needs no input values.
The corrected cast fixture passes on the unchanged installed candidate17;
square and target-aware alpha repairs require a later matching build.

The Rust workspace passes 1,179 core tests (2 ignored), 140 CUDA harness tests,
39 fastcpu tests (2 ignored), eight tokenizer harness cases and two intended
CUDA E0502 doctests. Five strict GPU cases are filtered. The 18 parsed nested
child summaries are excluded. All eight standalone checks pass; CUDA all-target
compilation and native DLPack's 12 harness cases pass. These are CPU/host
results; no NVIDIA or Apple GPU execution is established.

Focused installed suites also pass all 174 graph, 75 normalization/embedding/
dropout, 72 linear-algebra, 61 metadata/view and 142 training/optimizer methods.
Transform composition passes 69 public methods, 18 direct-network configurations,
18 composed-network configurations and 16 new arithmetic/view integration
configurations. A separately discovered auxiliary no_grad-view leaf-status
mismatch remains next-build work. These focused proofs do not erase the three
full-discovery problems or establish complete PyTorch/PyG parity.
