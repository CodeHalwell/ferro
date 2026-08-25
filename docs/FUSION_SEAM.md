# Wave 4c: the fusion seam is closed

Status: SHIPPED. The request in this file (chain_dev on dispatch::Backend +
op tags in graph.rs) is implemented and structurally tested. Fused pointwise
chains now flow from user code through core to ferro-cuda's nvrtc generator.

## What landed

1. `dispatch::ChainStepRef` - core-owned fused-step description (Unary /
   Binary / BinaryBc with broadcast dims+strides). Core never depends on
   ferro-cuda's ChainStep type.
2. `dispatch::Backend::chain_dev(steps, inputs)` with a `not_resident`
   default, so backends without a chain generator keep compiling.
3. `OpTag` (dispatch) + `Tensor::record_fn_tagged` + `Op::new_tagged`: every
   kind-routed forward op in ops.rs (add/sub/mul/div/neg/relu/exp/sigmoid)
   and silu records WHICH kernel ran. Composite ops record None and stay
   fusion barriers.
4. `graph::GraphNode.tag`, `Graph.tensors` (captured tensors by node id),
   `FusedChain::resolve(&Graph) -> ExecutableChain` (steps + operands +
   out_shape; second operands deduplicated into kernel slots; broadcast steps
   decomposed against the seed shape), and `FusedChain::run(&ExecutableChain)`:
   one `backend.chain_dev` launch when all operands are resident, else a
   sequential raw-kernel fallback computing identical math.
5. ferro-cuda implements `chain_dev` by converting ChainStepRef ->
   kernels::ChainStep and reusing launch_chain unchanged.

## Proof

tests/fusion_exec.rs: a relu -> add(bias, broadcast) -> silu tape captured on
a counting fake backend resolves to 2 chain steps, saves 2 launches per
plan_fusion's accounting, runs as EXACTLY ONE chain_dev launch with zero
per-op launches, and matches the eager CPU result within 1e-6. Full suite:
331 passed / 0 failed (ferro-core), fastcpu green, ferro-cuda compiles
without CUDA.

## What fusion does NOT do yet (next-wave items)

- Nothing on the ordinary eager path calls plan_fusion/run automatically;
  execution is still opt-in via Graph::capture. Wiring an executor that
  replays captured plans transparently is wave 5 work (it composes with CUDA
  graph capture there).
- Backward VJP links are not yet planned into chains (the FusedChain planner
  walks forward nodes only); each fused forward link should eventually fuse
  its backward links too.
- gelu/other tagged unary candidates in ops_ext can be migrated to
  record_fn_tagged as they are touched.
