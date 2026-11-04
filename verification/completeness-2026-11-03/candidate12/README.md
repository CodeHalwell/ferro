# Candidate 12: failed comparisons retained

This isolated installed wheel is an intermediate checkpoint. It is not a green
release or a full PyTorch/PyG parity claim. The latest fully green report remains
candidate 10; implementation and repairs continued afterward.

All 825 selected sources matched the isolated build, and 87 installed Python
files plus the native library matched the wheel. Full Python discovery ran
1,153 methods: 1,124 passed, five methods failed or errored, 24 individual GPU
cases skipped, and a skipped GPU class hid six more cases (1,159 discovered).
The runner reports three failures and 11 error indicators because failed
subcases can share a method, and one method both fails and errors.

The failures preserve two distinct implementation problems: native Jacobi SVD
nonconvergence for a rank-deficient least-squares input, and functional JVP
wrapping of newly created gradient leaves/auxiliary outputs under no_grad.
New JVP test scalar/list assertions and expand argument handling also needed
repair; one old empty-JVP assertion expected the former reverse-contraction
behavior. Their logs remain unmodified here. Subsequent source fixes require
another installed-wheel proof.

The Rust workspace passed 1,112 core, 140 CUDA harness, 39 fastcpu and eight
tokenizer cases, plus two intended E0502 CUDA compile-fail doctests. Four
runtime cases were ignored; five strict GPU tests were filtered. Eighteen
nested child summaries are excluded. All-target CUDA compilation and all eight
standalone Python commands passed. Native DLPack reports 12 harness passes,
including one GPU body that returned early; three mandatory GPU cases filtered.
This machine did not execute NVIDIA or Metal code.

Exact source, installed, wheel hashes and command outcomes are in manifest.json.
The source archive and isolated build remain in the task scratch directory.
