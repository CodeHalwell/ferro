use std::collections::{HashMap, HashSet};
use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
    static FUNCTIONAL_VJP: Cell<(bool, bool)> = const { Cell::new((false, false)) };
}

pub(crate) fn functional_vjp() -> bool { FUNCTIONAL_VJP.with(|mode| mode.get().0) }
fn transform_grad_mode() -> bool { FUNCTIONAL_VJP.with(|mode| mode.get().1) }
struct FunctionalVjpGuard((bool, bool));
impl Drop for FunctionalVjpGuard { fn drop(&mut self) { FUNCTIONAL_VJP.with(|mode| mode.set(self.0)); } }

/// Whether ordinary operations record autograd on this thread. Capture is independent.
pub fn is_grad_enabled() -> bool { GRAD_ENABLED.with(Cell::get) }

/// Set this thread's recording mode and return the previous mode for restoration.
pub fn set_grad_enabled(enabled: bool) -> bool { GRAD_ENABLED.with(|mode| mode.replace(enabled)) }

struct GradModeGuard(bool);
impl Drop for GradModeGuard {
    fn drop(&mut self) { set_grad_enabled(self.0); }
}

/// Run with a recording mode, restoring the previous mode even during unwinding.
pub fn with_grad_enabled<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    let _guard = GradModeGuard(set_grad_enabled(enabled));
    run()
}

#[path = "higher_order.rs"]
mod higher_order;
#[path = "forward_rules.rs"]
mod forward_rules;

use crate::dispatch::OpTag;
use crate::dtype::DType;
use crate::tensor::{StorageCell, Tensor};
use std::sync::{Arc, Mutex};

/// Forward executors use current operands, never autograd backward closures.
#[derive(Clone, Debug)]
pub(crate) enum ForwardOp {
    CopyDualLayout { strides: Vec<usize>, offset: usize, storage_len: usize },
    SigmoidBackward, TanhBackward,
    WhereConditional,
    Prelu,
    CumExtrema { dim: usize, maximum: bool },
    IndexedGather { map: crate::ops_ext::indexed::IndexMap, kind: crate::ops_ext::indexed::IndexKind, save_input: bool },
    IndexedScatter { map: crate::ops_ext::indexed::IndexMap, kind: crate::ops_ext::indexed::IndexKind, shape: Vec<usize>, save_input: bool, strides: Option<Vec<usize>> },
    Celu(crate::ops_ext::add_alpha::Alpha),
    Rounding(crate::typed_math::RoundingKind), Frac, Rsqrt, Square,
    ScalarPower(f64),
    RoundedDiv(crate::ops_ext::div_rounding::DivisionRounding), FloorDivide,
    CumSum { dim: usize, dtype: Option<DType>, reverse: bool },
    CumProd { dim: usize, dtype: Option<DType> },
    Prod { dim: Option<usize>, keepdim: bool, dtype: Option<DType> },
    AddAlpha { alpha: crate::ops_ext::add_alpha::Alpha, subtract: bool },
    Sum, Mean, MatMul, Bmm, Reshape(Vec<usize>), View(Vec<usize>), Squeeze(usize), Unsqueeze(usize), Transpose(usize, usize), Clone { preserve_layout: bool },
    Softmax(usize), SumDim(usize, bool), LayerNorm { weight: bool, bias: bool, eps: f32 },
    Cat(usize), Slice { dim: usize, start: usize, end: usize, step: usize },
    Permute(Vec<usize>), Expand(Vec<usize>), Cast(DType), Sin, Cos,
    SliceAdjoint { shape: Vec<usize>, dim: usize, start: usize, end: usize, step: usize },
}

impl ForwardOp {
    pub(crate) fn run(&self, inputs: &[Tensor]) -> crate::Result<Tensor> {
        match self {
            Self::CopyDualLayout { strides, offset, storage_len } => inputs[0].copy_dual_layout(strides,*offset,*storage_len),
            Self::CumExtrema { dim, maximum } => inputs[0].cumulative_extrema(*dim, *maximum).map(|result| result.0),
            Self::Prelu => inputs[0].prelu(&inputs[1]),
            Self::WhereConditional => Tensor::where_cond(&inputs[0], &inputs[1], &inputs[2]),
            Self::IndexedGather { map, kind, save_input } => inputs[0].indexed_gather_kind(&inputs[1], *map, *kind, *save_input),
            Self::IndexedScatter { map, kind, shape, save_input, strides } => Tensor::indexed_scatter_saved(&inputs[0], &inputs[1], shape, *map, *kind, *save_input, strides.as_deref()),
            Self::Celu(alpha) => inputs[0].celu_alpha(*alpha),
            Self::SigmoidBackward => crate::ops_ext::activation_backward::backward(crate::ops_ext::activation_backward::Kind::Sigmoid, &inputs[0], &inputs[1]),
            Self::TanhBackward => crate::ops_ext::activation_backward::backward(crate::ops_ext::activation_backward::Kind::Tanh, &inputs[0], &inputs[1]),
            Self::Sum => Ok(inputs[0].sum()),
            Self::CumSum { dim, dtype, reverse } => inputs[0].cumulative_sum(*dim, *dtype, *reverse),
            Self::CumProd { dim, dtype } => inputs[0].cumprod_dtype(*dim, *dtype),
            Self::Prod { dim, keepdim, dtype } => inputs[0].product(*dim, *keepdim, *dtype),
            Self::RoundedDiv(mode) => inputs[0].div_rounding(&inputs[1], *mode),
            Self::FloorDivide => inputs[0].floor_divide(&inputs[1]),
            Self::Rounding(kind) => crate::typed_math::rounding(&inputs[0], *kind),
            Self::Frac => inputs[0].frac(),
            Self::Rsqrt => inputs[0].rsqrt(),
            Self::Square => inputs[0].square(),
            Self::ScalarPower(power) => inputs[0].pow_scalar(*power),
            Self::AddAlpha { alpha, subtract } => if *subtract { inputs[0].sub_alpha(&inputs[1], *alpha) } else { inputs[0].add_alpha(&inputs[1], *alpha) },
            Self::Mean => Ok(inputs[0].mean()),
            Self::MatMul => inputs[0].matmul(&inputs[1]),
            Self::Bmm => inputs[0].bmm(&inputs[1]),
            Self::Reshape(shape) => inputs[0].reshape(shape),
            Self::View(shape) => inputs[0].view(shape),
            Self::Squeeze(dim) => inputs[0].squeeze(*dim),
            Self::Unsqueeze(dim) => inputs[0].unsqueeze(*dim),
            Self::Transpose(a, b) => inputs[0].transpose(*a, *b),
            Self::Clone { preserve_layout } => inputs[0].clone_tensor(*preserve_layout),
            Self::Softmax(dim) => inputs[0].softmax(*dim),
            Self::SumDim(dim, keep) => inputs[0].sum_dim(*dim, *keep),
            Self::Cat(dim) => Tensor::cat(inputs, *dim),
            Self::Permute(dims) => inputs[0].permute(dims),
            Self::Expand(shape) => inputs[0].expand(shape),
            Self::Cast(dtype) => inputs[0].to_dtype_autograd(*dtype, true),
            Self::Sin => inputs[0].sin(),
            Self::Cos => inputs[0].cos(),
            Self::Slice { dim, start, end, step } => inputs[0].slice(*dim, *start, *end, *step),
            Self::SliceAdjoint { shape, dim, start, end, step } => inputs[0].slice_adjoint(shape, *dim, *start, *end, *step),
            Self::LayerNorm { weight, bias, eps } => inputs[0].layer_norm(weight.then_some(&inputs[1]), bias.then_some(&inputs[2]), *eps),
        }
    }
}

/// The graph node behind every autograd-recorded tensor: the op's inputs plus a
/// vector-Jacobian-product closure returning one gradient per input, in order.
/// Ops that need their own output for backward (exp, sigmoid, softmax) capture
/// a *detached* snapshot in the closure to avoid a reference cycle. Host and
/// materialized device views own fresh storage; whole device snapshots share
/// storage and rely on the public in-place alias gate to prevent mutation.
///
/// `saved_versions` snapshots each value-dependent input storage version.
/// Explicit metadata-only adjoints record None because they read only shapes
/// and the incoming gradient. Other recording seams save every input at record
/// time; both backward engines check those versions before running the callback.
/// Saved output and sibling dependencies remain independently guarded.
pub(crate) struct Op {
    pub(crate) forward: Option<ForwardOp>,
    inputs: Vec<Tensor>,
    // Old edges cannot be created by later flag changes. Public flag-only
    // paths still matter when functional transforms preserve result flags.
    input_edges: Vec<bool>,
    input_flags: Vec<bool>,
    saved_versions: Vec<Option<u64>>,
    saved_values: Mutex<Vec<(Arc<StorageCell>, u64)>>,
    /// Which named kernel this op ran (kind-routed ops only); None for
    /// composite ops. Read by the graph compiler to plan fused execution.
    pub(crate) tag: Option<OpTag>,
    backward: Option<Arc<dyn Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync>>,
    optional: bool,
}

#[derive(Clone, Copy)]
struct FrozenForward<'a> {
    output: &'a Tensor,
    cutoff: usize,
    primals: &'a HashSet<usize>,
    excluded: &'a HashSet<usize>,
}

impl FrozenForward<'_> {
    fn operand(&self, input: &Tensor) -> Tensor {
        if input.id() < self.cutoff || self.primals.contains(&input.id()) { return input.clone(); }
        let mut current = input;
        let mut seen = HashSet::new();
        let view_requires_grad = loop {
            if !seen.insert(current.id()) { break false; }
            if current.id() < self.cutoff { break current.requires_grad(); }
            if self.primals.contains(&current.id()) {
                break current.requires_grad() && !self.excluded.contains(&current.id());
            }
            if current.is_inherited_grad_view() { break current.requires_grad(); }
            let parent = current.0.op.as_ref().and_then(|op| op.inputs.iter().find(|parent| Arc::ptr_eq(&parent.0.storage, &current.0.storage)));
            let Some(parent) = parent else { break false; };
            current = parent;
        };
        let detached = input.detach();
        if view_requires_grad { with_grad_enabled(false, || detached.inherit_no_grad_view(input)) } else { detached }
    }
}

impl Op {
    pub(crate) fn new(
        inputs: Vec<Tensor>,
        backward: Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
    ) -> Op {
        let saved_versions = inputs.iter().map(|t| Some(t.version())).collect();
        let input_edges = inputs.iter().map(|input| input.has_gradient_edge() && crate::forward_ad::records_grad(input)).collect();
        let input_flags = inputs.iter().map(crate::forward_ad::records_grad).collect();
        Op {
            inputs,
            input_edges,
            input_flags,
            saved_versions,
            saved_values: Mutex::new(Vec::new()),
            forward: None,
            tag: None,
            backward: Some(Arc::new(move |g| backward(g).into_iter().map(Some).collect())),
            optional: false,
        }
    }

    pub(crate) fn new_optional(
        inputs: Vec<Tensor>,
        backward: Box<dyn Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync>,
    ) -> Op {
        let saved_versions = inputs.iter().map(|t| Some(t.version())).collect();
        let input_edges = inputs.iter().map(|input| input.has_gradient_edge() && crate::forward_ad::records_grad(input)).collect();
        let input_flags = inputs.iter().map(crate::forward_ad::records_grad).collect();
        Op { inputs, input_edges, input_flags, saved_versions, saved_values: Mutex::new(Vec::new()), forward: None, tag: None, backward: Some(Arc::from(backward)), optional: true }
    }

    /// Like `new` but carrying the kernel tag for fusion planning.
    pub(crate) fn new_tagged(
        inputs: Vec<Tensor>,
        tag: OpTag,
        backward: Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
    ) -> Op {
        let saved_versions = inputs.iter().map(|t| Some(t.version())).collect();
        let input_edges = inputs.iter().map(|input| input.has_gradient_edge() && crate::forward_ad::records_grad(input)).collect();
        let input_flags = inputs.iter().map(crate::forward_ad::records_grad).collect();
        Op {
            inputs,
            input_edges,
            input_flags,
            saved_versions,
            saved_values: Mutex::new(Vec::new()),
            forward: None,
            tag: Some(tag),
            backward: Some(Arc::new(move |g| backward(g).into_iter().map(Some).collect())),
            optional: false,
        }
    }

    pub(crate) fn inference(inputs: Vec<Tensor>, tag: Option<OpTag>) -> Op {
        let input_edges = vec![false; inputs.len()];
        let input_flags = inputs.iter().map(crate::forward_ad::records_grad).collect();
        Op { inputs, input_edges, input_flags, saved_versions: Vec::new(), saved_values: Mutex::new(Vec::new()), tag, forward: None, backward: None, optional: false }
    }

    pub(crate) fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }

    pub(crate) fn rebase_inputs(&self, inputs: Vec<Tensor>, actual_edges: &[bool], actual_flags: &[bool]) -> Op {
        debug_assert_eq!(inputs.len(), self.inputs.len());
        debug_assert!(inputs.iter().zip(&self.inputs).all(|(new, old)| new._storage_ptr() == old._storage_ptr()
            && new.shape() == old.shape() && new.strides() == old.strides() && new.storage_offset() == old.storage_offset()));
        debug_assert_eq!(actual_edges.len(), inputs.len());
        debug_assert_eq!(actual_flags.len(), inputs.len());
        let input_edges = self.input_edges.iter().zip(actual_edges).map(|(old, actual)| *old && *actual).collect();
        let input_flags = self.input_flags.iter().zip(actual_flags).map(|(old, actual)| *old && *actual).collect();
        Op { inputs, input_edges, input_flags, saved_versions: self.saved_versions.clone(),
            saved_values: Mutex::new(self.saved_values.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            forward: self.forward.clone(), tag: self.tag, backward: self.backward.clone(), optional: self.optional }
    }

    /// Forward-time edge eligibility is independent of later leaf flag edits.
    pub(crate) fn input_edges(&self) -> &[bool] { &self.input_edges }

    pub(crate) fn saved_versions(&self) -> &[Option<u64>] {
        &self.saved_versions
    }

    pub(crate) fn save_storage(&self, storage: Arc<StorageCell>, version: u64) {
        if self.backward.is_some() { self.saved_values.lock().unwrap_or_else(|e| e.into_inner()).push((storage, version)); }
    }

    fn validate_saved_values(&self) -> crate::Result<()> {
        for (storage, saved) in self.saved_values.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            if storage.version() != *saved {
                return Err(crate::Error::Unsupported { op: "backward", msg: format!("saved output modified inplace (version {saved} -> {})", storage.version()) });
            }
        }
        Ok(())
    }

    pub(crate) fn omit_saved_input_values(&mut self) {
        self.saved_versions.fill(None);
    }

    pub(crate) fn backward(&self, g: &Tensor, required: &[bool]) -> Vec<Option<Tensor>> {
        self.vjp(g, false, false, None, Some(required), None).expect("inference-only node has no backward")
    }

    fn vjp(&self, g: &Tensor, create_graph: bool, preserve_passthrough: bool, frozen_forward: Option<FrozenForward<'_>>, required: Option<&[bool]>, output: Option<&Tensor>) -> crate::Result<Vec<Option<Tensor>>> {
        let _mode = (!create_graph).then(|| GradModeGuard(set_grad_enabled(false)));
        let backward = self.backward.as_ref().ok_or_else(|| crate::Error::Unsupported {
            op: "vjp_wrt", msg: "inference-only node has no backward".into(),
        })?;
        if matches!(self.forward, Some(ForwardOp::FloorDivide)) { return Err(crate::Error::Unsupported { op: "floor_divide backward", msg: "derivative for floor_divide is not implemented".into() }); }
        if matches!(self.forward, Some(ForwardOp::RoundedDiv(_))) { return crate::ops_ext::div_rounding::zero_vjp(&self.inputs, g, required); }
        if let Some(ForwardOp::Celu(alpha)) = self.forward {
            if required.is_some_and(|mask| !mask[0]) { return Ok(vec![None]); }
            crate::ops_ext::celu::validate_backward(alpha)?;
        }
        if let Some(ForwardOp::AddAlpha { alpha, subtract }) = &self.forward {
            return with_grad_enabled(create_graph, || crate::ops_ext::add_alpha::selected_vjp(&self.inputs, g, *alpha, *subtract, required, create_graph));
        }
        if let Some(ForwardOp::CumExtrema { dim, .. }) = self.forward {
            if self.inputs[0].numel() == 0 {
                let input = &self.inputs[0];
                let gradient = if create_graph || preserve_passthrough { input.clone() } else { input.detach_copy() };
                return Ok(vec![Some(gradient), None]);
            }
            if create_graph { return with_grad_enabled(true, || crate::ops_ext::indexed::gather_vjp(&self.inputs, g, crate::ops_ext::indexed::IndexMap::Axis(dim), crate::ops_ext::indexed::IndexKind::Gather)); }
        }
        if create_graph && matches!(self.forward, Some(ForwardOp::WhereConditional)) {
            return with_grad_enabled(true, || crate::ops_ext::where_cond::graph_vjp(&self.inputs, g));
        }
        if create_graph {
            match &self.forward {
                Some(ForwardOp::IndexedGather { map, kind, .. }) => return with_grad_enabled(true, || crate::ops_ext::indexed::gather_vjp(&self.inputs, g, *map, *kind)),
                Some(ForwardOp::IndexedScatter { map, kind, .. }) => return with_grad_enabled(true, || Ok(vec![Some(g.indexed_gather_kind(&self.inputs[1], *map, *kind, true)?), None])),
                _ => {},
            }
        }
        if !create_graph {
            if functional_vjp() || transform_grad_mode() {
                if let Some(ForwardOp::Prod { dim, keepdim, .. }) = self.forward {
                    return crate::ops_ext::prod::graph_vjp(&self.inputs[0], output, g, dim, keepdim, functional_vjp()).map(|gradient| vec![Some(gradient)]);
                }
            }
            if preserve_passthrough {
                if let Some(ForwardOp::Cat(dim)) = &self.forward {
                    let mut offset = 0;
                    return self.inputs.iter().map(|input| {
                        let gradient = g.narrow(*dim, offset, input.shape()[*dim])?;
                        offset += input.shape()[*dim];
                        Ok(Some(gradient))
                    }).collect();
                }
                let gradient = match &self.forward {
                    Some(ForwardOp::Sum) => Some(g.expand(self.inputs[0].shape())?),
                    Some(ForwardOp::SumDim(dim, keepdim)) => {
                        let restored = if *keepdim { g.clone() } else { g.unsqueeze(*dim)? };
                        Some(restored.expand(self.inputs[0].shape())?)
                    }
                    _ => None,
                };
                if let Some(gradient) = gradient { return Ok(vec![Some(gradient)]); }
            }
            return Ok(backward(g));
        }
        if self.optional {
            let contributions = with_grad_enabled(false, || backward(g));
            if contributions.iter().all(Option::is_none) { return Ok(contributions); }
        }
        let frozen;
        let inputs = if let Some(state) = frozen_forward {
            frozen = self.inputs.iter().map(|input| state.operand(input)).collect::<Vec<_>>();
            &frozen
        } else { &self.inputs };
        // Outputs are borrowed during traversal, never captured by their own
        // Op. Root derivatives read their original output after input edits;
        // frozen function transforms detach that output's reverse history.
        let saved_output = frozen_forward.filter(|state| state.output.id() >= state.cutoff && self.vjp_uses_output()).map(|state| state.output.detach())
            .or_else(|| (matches!(self.forward, Some(ForwardOp::Rsqrt | ForwardOp::CumProd { .. } | ForwardOp::Prod { .. })) || matches!(self.tag, Some(OpTag::Unary(crate::dispatch::UnaryKind::Sqrt | crate::dispatch::UnaryKind::Exp | crate::dispatch::UnaryKind::Tanh | crate::dispatch::UnaryKind::Sigmoid)))).then(|| output.cloned()).flatten());
        self.graph_backward(g, inputs, saved_output.as_ref()).map(|values| values.into_iter().map(Some).collect())
    }

    /// Consume the op, yielding its input tensors. Used by TensorInner's
    /// iterative Drop to unlink deep graphs without recursing.
    pub(crate) fn into_inputs(self) -> Vec<Tensor> {
        self.inputs
    }
}

/// Post-order topological sort via an explicit stack of (node, next child
/// index), so graphs deeper than the native stack cannot overflow it.
fn build_topo(root: &Tensor) -> Vec<Tensor> { build_topo_roots(&[root]) }

fn build_topo_roots(roots: &[&Tensor]) -> Vec<Tensor> {
    let mut seen = HashSet::new();
    let mut topo = Vec::new();
    for &root in roots {
        if !seen.insert(root.id()) { continue; }
        let mut stack = vec![(root.clone(), 0)];
        while let Some((tensor, index)) = stack.pop() {
            if let Some(op) = &tensor.0.op {
                let inputs = op.inputs();
                if index < inputs.len() {
                    let child = inputs[index].clone();
                    stack.push((tensor.clone(), index + 1));
                    if seen.insert(child.id()) { stack.push((child, 0)); }
                    continue;
                }
            }
            topo.push(tensor);
        }
    }
    topo
}

fn validate_vjp_operands(output: &Tensor, inputs: &[&Tensor], seed: Option<&Tensor>, create_graph: bool) -> crate::Result<()> {
    if output.is_inherited_grad_view() { return Err(crate::Error::Unsupported { op: "vjp_wrt", msg: "an inherited view flag has no direct gradient edge; explicitly enable gradients on the view first".into() }); }
    let dtype = if output.dtype() == DType::F64 && !create_graph { DType::F64 } else { DType::F32 };
    for t in std::iter::once(output).chain(seed) {
        if t.dtype() != dtype {
            return Err(crate::Error::DtypeMismatch { op: "vjp_wrt", expected: dtype, got: t.dtype() });
        }
        if t.device() != output.device() {
            return Err(crate::Error::DeviceMismatch { op: "vjp_wrt", lhs: output.device(), rhs: t.device() });
        }
    }
    for &input in inputs {
        if input.dtype() != DType::F32 && (create_graph || input.dtype() != DType::F64) {
            return Err(crate::Error::DtypeMismatch { op: "vjp_wrt", expected: dtype, got: input.dtype() });
        }
        if input.device() != output.device() {
            return Err(crate::Error::DeviceMismatch { op: "vjp_wrt", lhs: output.device(), rhs: input.device() });
        }
    }
    Ok(())
}

impl Tensor {
    /// Structural dependency in the recorded graph, including identity. This
    /// reads metadata only; zero-valued derivatives do not imply independence.
    pub fn depends_on(&self, input: &Tensor) -> bool {
        let mut seen = HashSet::new();
        let mut stack = vec![self.clone()];
        while let Some(tensor) = stack.pop() {
            if tensor.id() == input.id() { return true; }
            if !seen.insert(tensor.id()) { continue; }
            if let Some(op) = &tensor.0.op { stack.extend(op.inputs.iter().zip(&op.input_edges).filter(|(_, edge)| **edge).map(|(input, _)| input.clone())); }
        }
        false
    }

    /// Whether recorded public flags reach a gradient-requiring leaf without
    /// crossing excluded tensors. Later leaf flag edits do not rewrite these
    /// recorded flags. Used by functional transform bookkeeping.
    pub fn has_grad_leaf_except(&self, excluded: &[&Tensor]) -> bool {
        let excluded: HashSet<_> = excluded.iter().map(|t| t.id()).collect();
        let mut seen = HashSet::new();
        let mut stack = vec![(self.clone(), self.requires_grad())];
        while let Some((tensor, required)) = stack.pop() {
            if excluded.contains(&tensor.id()) || !seen.insert(tensor.id()) { continue; }
            if let Some(op) = &tensor.0.op { stack.extend(op.inputs.iter().zip(&op.input_flags).filter(|(_, flag)| **flag).map(|(input, _)| (input.clone(), true))); }
            else if required { return true; }
        }
        false
    }

    /// Current flags of saved operands can create new derivative graphs even
    /// when those operands had no edge in the original forward computation.
    pub fn has_current_grad_leaf_except(&self, excluded: &[&Tensor]) -> bool {
        let excluded: HashSet<_> = excluded.iter().map(|tensor| tensor.id()).collect();
        let mut seen = HashSet::new();
        let mut stack = vec![self.clone()];
        while let Some(tensor) = stack.pop() {
            if excluded.contains(&tensor.id()) || !seen.insert(tensor.id()) { continue; }
            if let Some(op) = &tensor.0.op { stack.extend(op.inputs.iter().cloned()); }
            else if tensor.requires_grad() { return true; }
        }
        false
    }

    /// Functional scalar gradient in input order, without touching `.grad()`.
    /// Unused/non-requiring inputs return detached zeros, which can be differentiated again.
    /// create_graph supports add/sub/mul/div/neg/exp/sigmoid/tanh/sqrt, 2D matmul,
    /// reshape/transpose/sum_dim/sum/mean. Other VJPs return Error::Unsupported, never
    /// silently detached derivatives. Broadcasting and accumulation retain graphs.
    /// Functional differentiation is eager-only and is rejected during capture.
    /// Float64 supports first-order VJPs; create_graph currently requires float32.
    pub fn grad_wrt(&self, inputs: &[&Tensor], create_graph: bool) -> crate::Result<Vec<Tensor>> {
        if crate::capture::is_recording() {
            return Err(crate::Error::Unsupported { op: "grad_wrt", msg: "functional differentiation during capture is unsupported".into() });
        }
        if self.numel() != 1 {
            return Err(crate::Error::InvalidShape { op: "grad_wrt", msg: "requires a single-element output; use vjp_wrt with an explicit seed".into() });
        }
        validate_vjp_operands(self, inputs, None, create_graph)?;
        let seed = if self.dtype() == DType::F64 { crate::typed_math::full_f64(self.shape(), 1.0)? } else { Tensor::full_on(self.shape(), 1.0, self.device())? };
        self.vjp_wrt(inputs, &seed, create_graph)
    }

    /// Functional VJP for selected leaves or intermediates. With create_graph,
    /// the seed is a live differentiable operand (held fixed in this VJP but
    /// available to later derivatives). Otherwise it is detached. Graphs are
    /// retained; returned derivatives never accumulate into `.grad()`.
    /// create_graph controls recording during this VJP even inside no-grad; the
    /// caller's thread-local mode is restored on return, error, or unwinding.
    /// Functional differentiation is eager-only and is rejected during capture.
    /// Float64 supports first-order VJPs; create_graph currently requires float32.
    pub fn vjp_wrt(&self, inputs: &[&Tensor], seed: &Tensor, create_graph: bool) -> crate::Result<Vec<Tensor>> {
        self.vjp_wrt_impl(inputs, seed, create_graph, false, None)
    }

    /// First-order transform seam: unchanged cotangents retain identity/history,
    /// while backward computations still execute with recording disabled.
    #[doc(hidden)]
    pub fn vjp_wrt_preserve(&self, inputs: &[&Tensor], seed: &Tensor) -> crate::Result<Vec<Tensor>> {
        self.vjp_wrt_impl(inputs, seed, false, true, None)
    }

    /// Graph-building pullback for a transform created inside no-grad. Original
    /// intermediates and saved outputs made during that forward are held
    /// constant; selected primals, older captures and the cotangent stay live.
    #[doc(hidden)]
    pub fn vjp_wrt_frozen(&self, inputs: &[&Tensor], seed: &Tensor, cutoff: usize) -> crate::Result<Vec<Tensor>> {
        self.vjp_wrt_frozen_excluding(inputs, seed, cutoff, &[])
    }

    /// `excluded` identifies synthetic transform primals whose original inputs
    /// did not require gradients, so views cannot invent a public grad flag.
    #[doc(hidden)]
    pub fn vjp_wrt_frozen_excluding(&self, inputs: &[&Tensor], seed: &Tensor, cutoff: usize, excluded: &[&Tensor]) -> crate::Result<Vec<Tensor>> {
        if excluded.iter().any(|value| !inputs.iter().any(|input| input.id() == value.id())) {
            return Err(crate::Error::Unsupported { op: "vjp_wrt_frozen", msg: "excluded primals must be selected inputs".into() });
        }
        self.vjp_wrt_impl(inputs, seed, true, false, Some((cutoff, excluded)))
    }

    fn vjp_wrt_impl(&self, inputs: &[&Tensor], seed: &Tensor, create_graph: bool, preserve_passthrough: bool, freeze_created_after: Option<(usize, &[&Tensor])>) -> crate::Result<Vec<Tensor>> {
        self.vjp_wrt_optional_impl(inputs, seed, create_graph, preserve_passthrough, freeze_created_after, false)?.into_iter().zip(inputs).map(|(gradient, input)| {
            match gradient {
                Some(g) => Ok(g),
                None if input.dtype() == DType::F64 => crate::typed_math::full_f64(input.shape(), 0.0),
                None => Tensor::full_on(input.shape(), 0.0, input.device()),
            }
        }).collect()
    }

    /// Internal strict-transform seam: retain the distinction between an
    /// unused derivative and a present derivative whose numerical value is zero.
    /// Tensor grad_wrt/vjp_wrt methods continue materializing unused gradients as zeros.
    #[doc(hidden)]
    pub fn vjp_wrt_optional(&self, inputs: &[&Tensor], seed: &Tensor, create_graph: bool) -> crate::Result<Vec<Option<Tensor>>> {
        self.vjp_wrt_optional_impl(inputs, seed, create_graph, false, None, false)
    }

    /// Function-transform composition on the existing graph-building VJP. The
    /// private caller has already preserved prepared-root duals and will remove
    /// synthetic reverse provenance while retaining the resulting tangents.
    #[doc(hidden)]
    pub fn vjp_wrt_forward(&self, inputs: &[&Tensor], seed: &Tensor) -> crate::Result<Vec<Option<Tensor>>> {
        if !crate::forward_ad::is_active() {
            return Err(crate::Error::Unsupported { op: "vjp_wrt_forward", msg: "forward composition requires an active forward level".into() });
        }
        self.vjp_wrt_optional_impl(inputs, seed, true, false, None, true)
    }

    /// Jointly seeded selected-input VJP on the existing Op graph.
    #[doc(hidden)]
    pub fn vjp_many_wrt_optional(outputs: &[&Tensor], inputs: &[&Tensor], seeds: &[&Tensor], create_graph: bool) -> crate::Result<Vec<Option<Tensor>>> {
        vjp_roots_optional_impl(outputs, inputs, seeds, create_graph, true, None, false, false, false)
    }

    /// Joint transform pullback retaining the existing optional/passthrough/frozen policies.
    #[doc(hidden)]
    pub fn vjp_many_wrt_transform(outputs: &[&Tensor], inputs: &[&Tensor], seeds: &[&Tensor], create_graph: bool, preserve_passthrough: bool, frozen: Option<(usize, &[&Tensor])>) -> crate::Result<Vec<Option<Tensor>>> {
        Self::vjp_many_wrt_transform_policy(outputs, inputs, seeds, create_graph, preserve_passthrough, frozen, false, false)
    }

    /// Invocation metadata is separate from whether derivative operations record.
    #[doc(hidden)]
    pub fn vjp_many_wrt_transform_policy(outputs: &[&Tensor], inputs: &[&Tensor], seeds: &[&Tensor], create_graph: bool, preserve_passthrough: bool, frozen: Option<(usize, &[&Tensor])>, functional: bool, caller_grad_mode: bool) -> crate::Result<Vec<Option<Tensor>>> {
        if let Some((_, excluded)) = frozen {
            if !create_graph || excluded.iter().any(|value| !inputs.iter().any(|input| input.id() == value.id())) {
                return Err(crate::Error::Unsupported { op: "vjp_many_wrt_transform", msg: "frozen pullbacks require create_graph and selected excluded primals".into() });
            }
        }
        vjp_roots_optional_impl(outputs, inputs, seeds, create_graph, preserve_passthrough, frozen, false, functional, caller_grad_mode)
    }

    /// Explicit function-transform policy on the same retained Op traversal.
    #[doc(hidden)]
    pub fn vjp_many_wrt_func(outputs: &[&Tensor], inputs: &[&Tensor], seeds: &[&Tensor], create_graph: bool, preserve_passthrough: bool, frozen: Option<(usize, &[&Tensor])>) -> crate::Result<Vec<Option<Tensor>>> {
        Self::vjp_many_wrt_transform_policy(outputs, inputs, seeds, create_graph, preserve_passthrough, frozen, true, true)
    }

    /// Existing private reverse-frame composition, with all root seeds staged together.
    #[doc(hidden)]
    pub fn vjp_many_wrt_forward(outputs: &[&Tensor], inputs: &[&Tensor], seeds: &[&Tensor]) -> crate::Result<Vec<Option<Tensor>>> {
        if !crate::forward_ad::is_active() {
            return Err(crate::Error::Unsupported { op: "vjp_many_wrt_forward", msg: "forward composition requires an active forward level".into() });
        }
        vjp_roots_optional_impl(outputs, inputs, seeds, true, false, None, true, true, true)
    }

    fn vjp_wrt_optional_impl(&self, inputs: &[&Tensor], seed: &Tensor, create_graph: bool, preserve_passthrough: bool, freeze_created_after: Option<(usize, &[&Tensor])>, allow_forward: bool) -> crate::Result<Vec<Option<Tensor>>> {
        vjp_roots_optional_impl(&[self], inputs, &[seed], create_graph, preserve_passthrough, freeze_created_after, allow_forward, allow_forward, allow_forward)
    }

    /// Reverse-mode autodiff seeded with ones (call on a scalar loss): the
    /// scalar restriction (v = 1) of `backward_with`. Populates `.grad()` on
    /// every leaf/intermediate with `requires_grad = true`. Repeated calls
    /// follow torch's retain_graph semantics: leaf grads accumulate across
    /// calls, interior grads are recomputed from scratch each call.
    pub fn backward(&self) {
        assert!(self.has_gradient_edge(), "the output has no gradient edge; enable gradients before computing it");
        assert!(
            self.numel() == 1,
            "backward() requires a scalar output (single element), got shape {:?}; \
             reduce with .sum() or .mean() first",
            self.shape()
        );
        let seed = if self.dtype() == DType::F64 { crate::typed_math::full_f64(self.shape(), 1.0) } else { Tensor::full_on(self.shape(), 1.0, self.device()) }
            .expect("loss tensor's device backend is registered");
        self.backward_with(&seed);
    }

    /// Reverse-mode autodiff seeded with an explicit cotangent `v`: computes
    /// the vector-Jacobian product v^T J for any (not just scalar) root, so
    /// `backward()` is just the v = 1 case on a scalar. `cotangent` must match
    /// `self`'s shape and floating dtype. Callbacks receive the live cotangent
    /// under disabled grad recording; publication detaches only new public
    /// slots. Differentiating through this pass itself uses the functional
    /// `grad_wrt`/`vjp_wrt` APIs instead.
    /// Same retain_graph semantics as `backward()`.
    pub fn backward_with(&self, cotangent: &Tensor) {
        let _grad_mode = GradModeGuard(set_grad_enabled(false));
        let _functional = FunctionalVjpGuard(FUNCTIONAL_VJP.with(|mode| mode.replace((false, false))));
        assert!(!crate::forward_ad::is_active(), "backward inside an active forward AD level is not supported");
        assert!(self.has_gradient_edge(), "the output has no gradient edge; enable gradients before computing it");
        assert!(
            cotangent.shape() == self.shape(),
            "backward_with() cotangent shape {:?} does not match output shape {:?}",
            cotangent.shape(),
            self.shape()
        );
        assert!(
            cotangent.dtype() == self.dtype() && matches!(self.dtype(), DType::F32 | DType::F64),
            "backward_with() cotangent must match the output floating dtype {}, got {}",
            self.dtype(), cotangent.dtype()
        );
        let topo = build_topo(self);
        // Stage every contribution before publishing slots. A rejected closure
        // or saved-version check must not partially update earlier leaves.
        let mut pending = HashMap::new();
        let seed = if cotangent.device() == self.device() { cotangent.clone() } else {
            cotangent.to_device(self.device()).expect("grad transfer to the tensor's device")
        };
        pending.insert(self.id(), seed);
        for t in topo.iter().rev() {
            let Some(op) = &t.0.op else { continue };
            if op.backward.is_none() { continue; }
            let Some(g) = pending.get(&t.id()) else { continue };
            let inputs = op.inputs();
            op.validate_saved_values().expect("saved output version check failed");
            for (i, (inp, &saved)) in inputs.iter().zip(op.saved_versions()).enumerate() {
                let Some(saved) = saved else { continue };
                let now = inp.version();
                assert!(
                    now == saved,
                    "one of the variables needed for gradient computation has been modified by \
                     an inplace operation: input {i} of the op producing tensor id {} was saved \
                     with version {saved} but is now version {now}",
                    t.id()
                );
            }
            let required = inputs.iter().zip(op.input_edges()).map(|(input, edge)| *edge && input.requires_grad()).collect::<Vec<_>>();
            let grads = op.backward(g, &required);
            assert!(
                grads.len() == inputs.len(),
                "op backward returned {} gradients for {} inputs; record_fn closures \
                 must return one gradient per input",
                grads.len(), inputs.len()
            );
            for (inp, ig) in inputs.iter().zip(&grads) {
                let Some(ig) = ig else { continue };
                assert_eq!(ig.dtype(), inp.dtype(), "gradient dtype must match its tensor");
                assert!(matches!(inp.dtype(), DType::F32 | DType::F64), "only floating tensors have gradients");
                assert!(ig.shape() == inp.shape(), "gradient shape {:?} does not match tensor shape {:?}", ig.shape(), inp.shape());
            }
            for ((inp, edge), ig) in inputs.iter().zip(op.input_edges()).zip(grads) {
                if !edge || !inp.requires_grad() { continue; }
                let Some(ig) = ig else { continue };
                let ig = if ig.device() == inp.device() { ig } else {
                    ig.to_device(inp.device()).expect("grad transfer to the tensor's device")
                };
                let accumulated = Tensor::merge_grad(pending.remove(&inp.id()), ig);
                pending.insert(inp.id(), accumulated);
            }
        }
        Tensor::publish_gradients(&topo, pending).expect("ordinary gradient publication failed");
    }
}

fn vjp_roots_optional_impl(outputs: &[&Tensor], inputs: &[&Tensor], seeds: &[&Tensor], create_graph: bool, preserve_passthrough: bool, freeze_created_after: Option<(usize, &[&Tensor])>, allow_forward: bool, functional: bool, caller_grad_mode: bool) -> crate::Result<Vec<Option<Tensor>>> {
    let _functional = FunctionalVjpGuard(FUNCTIONAL_VJP.with(|mode| mode.replace((functional, caller_grad_mode))));
        if crate::forward_ad::is_active() && !allow_forward {
            return Err(crate::Error::Unsupported { op: "vjp_wrt", msg: "reverse queries inside an active forward AD level are not supported".into() });
        }
        if crate::capture::is_recording() {
            return Err(crate::Error::Unsupported { op: "vjp_wrt", msg: "functional differentiation during capture is unsupported".into() });
        }
        let _grad_mode = GradModeGuard(set_grad_enabled(create_graph));
        use crate::Error;
        if outputs.len() != seeds.len() {
            return Err(Error::Unsupported { op: "vjp_wrt", msg: "one cotangent is required per output".into() });
        }
        for (&output, &seed) in outputs.iter().zip(seeds) {
            if seed.shape() != output.shape() {
                return Err(Error::ShapeMismatch { op: "vjp_wrt", lhs: output.shape().to_vec(), rhs: seed.shape().to_vec() });
            }
            validate_vjp_operands(output, inputs, Some(seed), create_graph)?;
        }
        for &input in inputs {
            if input.dtype() != DType::F32 && (create_graph || input.dtype() != DType::F64) {
                return Err(Error::DtypeMismatch { op: "vjp_wrt", expected: DType::F32, got: input.dtype() });
            }
        }
        let topo = build_topo_roots(outputs);
        let primals = freeze_created_after.map(|_| inputs.iter().map(|input| input.id()).collect::<HashSet<_>>());
        let excluded = freeze_created_after.map(|(_, excluded)| excluded.iter().map(|input| input.id()).collect::<HashSet<_>>());
        let selected: HashSet<_> = inputs.iter().filter(|t| t.requires_grad()).map(|t| t.id()).collect();
        let mut needed = selected;
        for t in &topo {
            if let Some(op) = &t.0.op {
                if op.inputs.iter().zip(&op.input_edges).any(|(i, edge)| *edge && needed.contains(&i.id())) { needed.insert(t.id()); }
            }
        }
        let mut grads: HashMap<usize, Tensor> = HashMap::new();
        for (&output, &seed) in outputs.iter().zip(seeds) {
            if needed.contains(&output.id()) {
                let seed = if create_graph || preserve_passthrough { seed.clone() } else { seed.detach_copy() };
                let value = if let Some(previous) = grads.remove(&output.id()) { previous.add(&seed)? } else { seed };
                grads.insert(output.id(), value);
            }
        }
        for t in topo.iter().rev() {
            let Some(op) = &t.0.op else { continue };
            if !op.inputs.iter().zip(&op.input_edges).any(|(i, edge)| *edge && needed.contains(&i.id()) && i.requires_grad()) { continue; }
            let Some(g) = grads.get(&t.id()) else { continue };
            for (inp, saved) in op.inputs.iter().zip(&op.saved_versions) {
                let Some(saved) = saved else { continue };
                if inp.version() != *saved {
                    return Err(Error::Unsupported { op: "vjp_wrt", msg: format!("saved input {} modified inplace (version {} -> {})", inp.id(), saved, inp.version()) });
                }
            }
            op.validate_saved_values()?;
            let frozen = freeze_created_after.zip(primals.as_ref()).zip(excluded.as_ref()).map(|(((cutoff, _), primals), excluded)| FrozenForward { output: t, cutoff, primals, excluded });
            let required = op.inputs.iter().zip(&op.input_edges).map(|(input, edge)| *edge && input.requires_grad() && needed.contains(&input.id())).collect::<Vec<_>>();
            let contributions = op.vjp(g, create_graph, preserve_passthrough, frozen, Some(&required), Some(t))?;
            if contributions.len() != op.inputs.len() {
                return Err(Error::Unsupported { op: "vjp_wrt", msg: "backward gradient arity does not match inputs".into() });
            }
            for ((inp, edge), contribution) in op.inputs.iter().zip(&op.input_edges).zip(contributions) {
                let Some(contribution) = contribution else { continue };
                if contribution.shape() != inp.shape() {
                    return Err(Error::ShapeMismatch { op: "vjp_wrt backward", lhs: inp.shape().to_vec(), rhs: contribution.shape().to_vec() });
                }
                if contribution.dtype() != inp.dtype() {
                    return Err(Error::DtypeMismatch { op: "vjp_wrt backward", expected: inp.dtype(), got: contribution.dtype() });
                }
                if !edge || !inp.requires_grad() || !needed.contains(&inp.id()) { continue; }
                let contribution = if contribution.device() != inp.device() {
                    if create_graph { return Err(Error::DeviceMismatch { op: "create_graph backward", lhs: inp.device(), rhs: contribution.device() }); }
                    contribution.to_device(inp.device())?
                } else { contribution };
                let value = if let Some(previous) = grads.remove(&inp.id()) { previous.add(&contribution)? } else { contribution };
                grads.insert(inp.id(), if create_graph || preserve_passthrough { value } else { value.detach_copy() });
            }
        }
        Ok(inputs.iter().map(|input| grads.get(&input.id()).map(|g| {
            if create_graph || preserve_passthrough { g.clone() } else { g.detach_copy() }
        })).collect())
}
