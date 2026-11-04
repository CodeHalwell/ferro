# Candidate20: retained fixture failures

This isolated build is not a green checkpoint. Its exact 981 source/fixture
inputs, 112 installed files (111 Python plus native), wheel and raw evidence
are recorded in [manifest.json](manifest.json). Both frozen Cargo locks use
cudarc 0.19.10. [Candidate21](../candidate21/README.md) reruns the complete validation after
two fixture-only corrections; no production source changed between them.

Full Python discovery found 1,488 cases: 1,457 passed, one method errored, and
30 GPU cases were unavailable. Unittest reports 1,482 executions and 25 skip
indicators: 24 individual skips and one class hiding six tests. The errored
`test_softplus_options_and_primitive_checkpoint_configuration` omitted required
`optimizers` and `generator` arguments to the checkpoint API. The separate
binding regression still constructed `ReLU6(inplace=True)`, despite explicit
rejection of unsupported in-place activations. Its exit status was 1; the
other seven standalone checks exited 0. These original failures are retained.

Candidate21 changes only `crates/ferro-py/tests/test_numpy_scalars.py` and
`examples/py_regression.py`: it supplies the checkpoint arguments, tests the
ordinary ReLU6 module, and separately asserts the unsupported option rejects.
The 979 other frozen inputs and all 111 installed Python files are identical.

Native validation passes: core 1,241 (2 ignored), fastcpu 39 (2 ignored), CUDA
140 runtime harness cases (5 strict GPU cases filtered), and tokenizer 8
harness cases, including one optional early return. The 1,428 runtime total
excludes 18 nested child summaries; two intended E0502 CUDA doctests also pass.
CUDA all-target compilation and native DLPack's 12 harness cases pass. One
DLPack GPU body returns early and three mandatory GPU cases are filtered.
No GPU execution or complete PyTorch/PyG parity is established.

Earlier [candidate18](../candidate18/README.md) is retained as a green historical
checkpoint. Failed [candidate19](../candidate19/README.md) and
[candidate17](../candidate17/README.md) records remain unchanged.
