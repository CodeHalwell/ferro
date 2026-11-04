# Continuation handoff

Current verified source: candidate32, PR #33. Full PyTorch/PyG parity is open.
Validation and exact inputs: verification/completeness-2026-11-03/manifest.json.

Work stopped at the user request to conserve usage. Remaining experiments are
outside the repository in /workspace/work/completeness-2026-11-03:

- batched-autodiff-conv-current31-ready.json: 1,147 inputs, 435 native passes;
  combined installed-wheel/full validation remains pending.
- adaptive-more-f64-ready.json: Adamax/Adadelta/ASGD; native checks pass,
  matching public wheel remains pending; ASGD load/scalar limits documented.
- recurrent-cell-f64-ready.json: typed cells; public wheel remains pending.
- Recurrent F32 gate-transform and F64 pooling experiments remain unfinished.

Resume from these manifests rather than copying entire older source trees.
Preserve current checkpoint changes and exact input hashes. No additional
experiments are asserted as shipped or proven by the current PR.
