//! Scoped forward tangents dispatched by the existing Op recorder.
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use crate::{Device, DType, Error, Result, Tensor};
use crate::autograd::Op;

struct State {
    tangents: HashMap<usize, Vec<Tensor>>,
    columns: usize,
    forbid_random: bool,
    suspended: bool,
    error: Option<Error>,
    functional_cutoff: Option<usize>,
    factory_flags: HashMap<usize, Arc<AtomicBool>>,
    transform_roots: HashMap<usize, Arc<AtomicBool>>,
    reverse_frame: Option<ReverseFrame>,
}

struct ReverseFrame {
    cutoff: usize,
    enabled: bool,
    roots: HashMap<usize, Tensor>,
    root_flags: Vec<usize>,
    owners: HashMap<usize, (Arc<AtomicBool>, Arc<AtomicBool>)>,
    decisions: HashMap<usize, (Vec<bool>, Vec<bool>)>,
    memo: HashMap<usize, Tensor>,
}

impl ReverseFrame {
    fn owner(&self, original: &Arc<AtomicBool>) -> Arc<AtomicBool> {
        self.owners.get(&(Arc::as_ptr(original) as usize)).map(|(_, actual)| actual.clone()).unwrap_or_else(|| original.clone())
    }
    fn public_flag(&self, value: &Tensor) -> bool {
        let (own, base) = value.grad_flag_handles();
        self.owner(&own).load(Ordering::Relaxed) || base.is_some_and(|base| self.owner(&base).load(Ordering::Relaxed))
    }
    fn affected(&self, value: &Tensor) -> bool {
        let (own, base) = value.grad_flag_handles();
        self.owners.contains_key(&(Arc::as_ptr(&own) as usize))
            || base.is_some_and(|base| self.owners.contains_key(&(Arc::as_ptr(&base) as usize)))
    }
    fn unwrap(&mut self, value: &Tensor) -> Tensor {
        let mut stack = vec![(value.clone(), false)];
        while let Some((node, ready)) = stack.pop() {
            if self.memo.contains_key(&node.id()) { continue; }
            if let Some(source) = self.roots.get(&node.id()) { self.memo.insert(node.id(), source.clone()); continue; }
            if !ready {
                stack.push((node.clone(), true));
                if let Some(op) = &node.0.op {
                    for input in op.inputs().iter().rev() { if !self.memo.contains_key(&input.id()) { stack.push((input.clone(), false)); } }
                }
                continue;
            }
            let (old_own, old_base) = node.grad_flag_handles();
            let own = self.owner(&old_own);
            let base = old_base.as_ref().map(|base| self.owner(base));
            let flags_changed = !Arc::ptr_eq(&own, &old_own) || old_base.as_ref().zip(base.as_ref()).is_some_and(|(old, new)| !Arc::ptr_eq(old, new));
            let inputs = node.0.op.as_ref().map(|op| op.inputs().iter().map(|input| self.memo[&input.id()].clone()).collect::<Vec<_>>());
            let inputs_changed = inputs.as_ref().zip(node.0.op.as_ref()).is_some_and(|(inputs, op)| inputs.iter().zip(op.inputs()).any(|(new, old)| new.id() != old.id()));
            let result = if flags_changed || inputs_changed {
                let op = if own.load(Ordering::Relaxed) { node.0.op.as_ref().zip(inputs).map(|(op, inputs)| {
                    let (edges, flags) = self.decisions.get(&node.id()).expect("affected operation has captured actual decisions");
                    op.rebase_inputs(inputs, edges, flags)
                }) } else { None };
                node.rebase_alias(own, base, op)
            } else { node.clone() };
            self.memo.insert(node.id(), result);
        }
        self.memo[&value.id()].clone()
    }
}

/// One reverse function-transform frame inside the current forward level.
/// Logical root flags are local; dropping the frame never mutates user tensors.
#[doc(hidden)]
pub struct ReverseFrameGuard { _thread: PhantomData<Rc<()>> }
impl ReverseFrameGuard {
    pub fn new() -> Result<Self> {
        LEVEL.with(|level| {
            let mut state = level.borrow_mut();
            let state = state.as_mut().filter(|state| !state.suspended).ok_or_else(|| Error::Unsupported {
                op: "forward reverse frame", msg: "reverse transform frame requires an active forward level".into() })?;
            if state.reverse_frame.is_some() { return Err(Error::Unsupported { op: "forward reverse frame", msg: "nested reverse transforms inside a forward level are not supported".into() }); }
            if let Some(error) = &state.error { return Err(error.clone()); }
            state.reverse_frame = Some(ReverseFrame { cutoff: crate::tensor::tensor_id_watermark(), enabled: crate::autograd::is_grad_enabled(),
                roots: HashMap::new(), root_flags: Vec::new(), owners: HashMap::new(), decisions: HashMap::new(), memo: HashMap::new() });
            Ok(Self { _thread: PhantomData })
        })
    }
}
impl Drop for ReverseFrameGuard {
    fn drop(&mut self) { LEVEL.with(|level| { if let Some(state) = level.borrow_mut().as_mut() {
        if let Some(frame) = state.reverse_frame.take() { for identity in frame.root_flags { state.transform_roots.remove(&identity); } }
    } }); }
}

thread_local! { static LEVEL: RefCell<Option<State>> = const { RefCell::new(None) }; }

/// A single eager CPU dual level. Drop restores the prior inactive state.
pub struct DualLevel { _thread: PhantomData<Rc<()>> }

impl DualLevel {
    pub fn new(primals: &[&Tensor], tangents: &[&Tensor]) -> Result<Self> {
        Self::start(primals, tangents, false)
    }

    /// Function-transform wrapping excludes gradient leaves created by the
    /// function itself without changing those user-visible leaf flags.
    #[doc(hidden)]
    pub fn functional(primals: &[&Tensor], tangents: &[&Tensor]) -> Result<Self> {
        Self::start(primals, tangents, true)
    }

    /// A single functional level carrying several independent directions.
    /// The primal is evaluated once; each local Op rule sees one column at a time.
    #[doc(hidden)]
    pub fn functional_columns(primals: &[&Tensor], columns: &[&[&Tensor]]) -> Result<Self> {
        let level = Self::start_columns(primals, columns, true, false)?;
        level.set_randomness_same(false)?;
        Ok(level)
    }

    fn start(primals: &[&Tensor], tangents: &[&Tensor], functional: bool) -> Result<Self> {
        Self::start_columns(primals, &[tangents], functional, true)
    }

    fn start_columns(primals: &[&Tensor], columns: &[&[&Tensor]], functional: bool, normalize: bool) -> Result<Self> {
        let invalid = |msg: &str| Error::Unsupported { op: "forward_ad", msg: msg.into() };
        if crate::capture::is_recording() { return Err(invalid("forward AD cannot run during inference capture")); }
        if LEVEL.with(|level| level.borrow().is_some()) { return Err(invalid("nested forward AD levels are not supported")); }
        if primals.is_empty() || columns.is_empty() || columns.iter().any(|column| column.len() != primals.len()) {
            return Err(invalid("primals and tangent columns must be nonempty matching lists"));
        }
        let mut identities = std::collections::HashSet::new();
        for &primal in primals {
            if primal.device() != Device::Cpu { return Err(invalid("forward AD currently supports CPU tensors")); }
            if primal.dtype() != DType::F32 { return Err(Error::DtypeMismatch { op: "forward_ad", expected: DType::F32, got: primal.dtype() }); }
            if !identities.insert(primal.id()) { return Err(invalid("dual primals must have distinct tensor identities")); }
        }
        for column in columns {
            for (&primal, &tangent) in primals.iter().zip(*column) {
                if tangent.device() != Device::Cpu { return Err(invalid("forward AD currently supports CPU tensors")); }
                if !matches!(tangent.dtype(), DType::F32 | DType::F64) { return Err(Error::DtypeMismatch { op: "forward_ad", expected: DType::F32, got: tangent.dtype() }); }
                if primal.shape() != tangent.shape() { return Err(Error::ShapeMismatch { op: "forward_ad", lhs: primal.shape().to_vec(), rhs: tangent.shape().to_vec() }); }
            }
        }
        let mut values = HashMap::new();
        for (index, primal) in primals.iter().enumerate() {
            let mut tangents = Vec::new();
            tangents.try_reserve_exact(columns.len()).map_err(|_| invalid("tangent column metadata exceeds available capacity"))?;
            for column in columns {
                tangents.push(if normalize { crate::ops_ext::dual_copy::normalize(primal,column[index])? } else { column[index].clone() });
            }
            values.insert(primal.id(), tangents);
        }
        LEVEL.with(|level| *level.borrow_mut() = Some(State { tangents: values, columns: columns.len(), forbid_random: false, suspended: false, error: None,
            functional_cutoff: functional.then(crate::tensor::tensor_id_watermark), factory_flags: HashMap::new(),
            transform_roots: HashMap::new(), reverse_frame: None }));
        Ok(Self { _thread: PhantomData })
    }

    pub fn tangent(&self, output: &Tensor) -> Result<Option<Tensor>> {
        single_column(self.tangents(output)?)
    }

    #[doc(hidden)]
    pub fn tangents(&self, output: &Tensor) -> Result<Option<Vec<Tensor>>> {
        LEVEL.with(|level| {
            let state = level.borrow();
            let state = state.as_ref().expect("live dual level");
            if let Some(error) = &state.error { return Err(error.clone()); }
            Ok(state.tangents.get(&output.id()).cloned())
        })
    }

    #[doc(hidden)]
    pub fn set_randomness_same(&self, same: bool) -> Result<()> {
        LEVEL.with(|level| {
            let mut state = level.borrow_mut();
            let state = state.as_mut().expect("live dual level");
            if let Some(error) = &state.error { return Err(error.clone()); }
            state.forbid_random = !same;
            Ok(())
        })
    }

    /// Unwrap a function result as a primal alias in the caller's grad mode.
    /// Captured auxiliary values retain identity; created/dual values unwrap.
    #[doc(hidden)]
    pub fn primal(&self, value: &Tensor, auxiliary: bool) -> Result<Tensor> {
        let (ignored, wrap, previous) = LEVEL.with(|level| {
            let mut state = level.borrow_mut();
            let state = state.as_mut().expect("live dual level");
            if let Some(error) = &state.error { return Err(error.clone()); }
            let ignored = ignored_factory(state, value);
            let wrap = !auxiliary || state.functional_cutoff.is_some_and(|cutoff| value.id() >= cutoff)
                || state.tangents.contains_key(&value.id());
            let previous = state.suspended;
            state.suspended = true;
            Ok((ignored, wrap, previous))
        })?;
        let _resume = Resume(previous);
        let value = if ignored { value.detach() } else { value.clone() };
        // Auxiliary unwrapping preserves a no-grad view's absent own edge;
        // an ordinary result still unwraps in the caller's recording mode.
        if wrap && auxiliary && value.is_inherited_grad_view() {
            Ok(value.detach().inherit_no_grad_view(&value))
        } else if wrap { value.reshape(value.shape()) } else { Ok(value) }
    }
}

impl Drop for DualLevel { fn drop(&mut self) { LEVEL.with(|level| { level.borrow_mut().take(); }); } }

pub(crate) fn is_functional() -> bool { LEVEL.with(|level| level.borrow().as_ref().is_some_and(|state| state.functional_cutoff.is_some())) }

pub fn is_active() -> bool { LEVEL.with(|level| level.borrow().as_ref().is_some_and(|state| !state.suspended)) }

/// Fail before any random stream advances under a Jacobian error policy.
#[doc(hidden)]
pub fn check_random_operation() -> Result<()> {
    if LEVEL.with(|level| level.borrow().as_ref().is_some_and(|state| state.forbid_random)) {
        return Err(Error::Unsupported { op: "forward_ad randomness", msg: "random operations require randomness='same'; randomness='different' is unsupported".into() });
    }
    Ok(())
}

fn ignored_factory(state: &mut State, input: &Tensor) -> bool {
    let Some(cutoff) = state.functional_cutoff else { return false; };
    let (own, base) = input.grad_flag_handles();
    let own_id = Arc::as_ptr(&own) as usize;
    if state.transform_roots.contains_key(&own_id) || base.as_ref().is_some_and(|base| state.transform_roots.contains_key(&(Arc::as_ptr(base) as usize))) { return false; }
    // Retain owners for the entire level: pointer identity alone permits allocator
    // reuse after a temporary factory dies and could mask an unrelated result.
    if input.id() >= cutoff && input.0.op.is_none() && input.has_gradient_edge() { state.factory_flags.insert(own_id, own); }
    state.factory_flags.contains_key(&own_id) || base.is_some_and(|base| state.factory_flags.contains_key(&(Arc::as_ptr(&base) as usize)))
}

fn check_alias(source: &Tensor, target: &Tensor) -> Result<()> {
    if source.shape() != target.shape() || source.strides() != target.strides()
        || source.storage_offset() != target.storage_offset() || source._storage_ptr() != target._storage_ptr()
        || source.dtype() != target.dtype() || source.device() != target.device() {
        return Err(Error::Unsupported { op: "forward_ad alias", msg: "transform preprocessing must preserve storage, layout, dtype and device".into() });
    }
    Ok(())
}

/// Private function-transform preprocessing: establish a genuine reverse root
/// and preserve its outer dual while leaving public detach semantics unchanged.
#[doc(hidden)]
pub fn prepare_alias(source: &Tensor, target: &Tensor) -> Result<()> {
    check_alias(source, target)?;
    LEVEL.with(|level| {
        let mut state = level.borrow_mut();
        let state = state.as_mut().filter(|state| !state.suspended).ok_or_else(|| Error::Unsupported {
            op: "forward_ad alias", msg: "transform preparation requires an active forward level".into() })?;
        if let Some(error) = &state.error { return Err(error.clone()); }
        let source = if ignored_factory(state, source) { source.detach() } else { source.clone() };
        if let Some(frame) = state.reverse_frame.as_mut() {
            if target.id() < frame.cutoff || !target.has_gradient_edge() { return Err(Error::Unsupported {
                op: "forward_ad alias", msg: "prepared reverse roots must be fresh aliases with their own gradient edge".into() }); }
            let (target_owner, _) = target.grad_flag_handles();
            let (source_owner, source_base) = source.grad_flag_handles();
            let source_owner = if frame.owner(&source_owner).load(Ordering::Relaxed) { frame.owner(&source_owner) }
                else { source_base.as_ref().map(|base| frame.owner(base)).unwrap_or_else(|| frame.owner(&source_owner)) };
            let identity = Arc::as_ptr(&target_owner) as usize;
            if !frame.public_flag(&source) {
                frame.owners.insert(identity, (target_owner, source_owner));
                frame.roots.insert(target.id(), source.clone());
            }
            frame.root_flags.push(identity);
        }
        let (owner, _) = target.grad_flag_handles();
        let identity = Arc::as_ptr(&owner) as usize;
        state.factory_flags.remove(&identity);
        state.transform_roots.insert(identity, owner);
        if let Some(tangent) = state.tangents.get(&source.id()).cloned() { state.tangents.insert(target.id(), tangent); }
        Ok(())
    })
}

pub(crate) fn record_actual_flags(output: &Tensor, inputs: &[Tensor]) {
    LEVEL.with(|level| {
        let mut state = level.borrow_mut();
        let Some(frame) = state.as_mut().and_then(|state| state.reverse_frame.as_mut()) else { return; };
        let flags = inputs.iter().map(|input| frame.public_flag(input)).collect::<Vec<_>>();
        let edges = inputs.iter().map(|input| frame.owner(&input.grad_flag_handles().0).load(Ordering::Relaxed)).collect::<Vec<_>>();
        let actual = output.has_gradient_edge() && frame.enabled && crate::autograd::is_grad_enabled() && flags.iter().any(|flag| *flag);
        frame.decisions.insert(output.id(), (edges, flags));
        let (owner, _) = output.grad_flag_handles();
        // An affected alias owns a fresh flag even if its bool agrees with the
        // logical wrapper. Otherwise enabling a rebased blocked view can mutate
        // the held wrapper's hidden own edge through a shared false flag Arc.
        if inputs.iter().any(|input| frame.affected(input)) || owner.load(Ordering::Relaxed) != actual {
            frame.owners.insert(Arc::as_ptr(&owner) as usize, (owner, Arc::new(AtomicBool::new(actual))));
        }
    });
}

#[doc(hidden)]
pub fn unwrap_values(values: &[&Tensor]) -> Result<Vec<Tensor>> {
    LEVEL.with(|level| {
        let mut state = level.borrow_mut();
        let state = state.as_mut().filter(|state| !state.suspended).ok_or_else(|| Error::Unsupported {
            op: "forward unwrap", msg: "unwrapping requires an active forward level".into() })?;
        if let Some(error) = &state.error { return Err(error.clone()); }
        let frame = state.reverse_frame.as_mut().ok_or_else(|| Error::Unsupported {
            op: "forward unwrap", msg: "unwrapping requires a reverse transform frame".into() })?;
        Ok(values.iter().map(|value| frame.unwrap(value)).collect())
    })
}

#[doc(hidden)]
pub fn current_tangent(source: &Tensor) -> Result<Option<Tensor>> {
    single_column(current_tangents(source)?)
}

fn single_column(values: Option<Vec<Tensor>>) -> Result<Option<Tensor>> {
    let count = LEVEL.with(|level| level.borrow().as_ref().map(|state| state.columns));
    if count != Some(1) { return Err(Error::Unsupported { op: "forward_ad tangent", msg: "single-direction lookup requires exactly one tangent column".into() }); }
    Ok(values.map(|mut values| values.remove(0)))
}

#[doc(hidden)]
pub fn current_tangents(source: &Tensor) -> Result<Option<Vec<Tensor>>> {
    LEVEL.with(|level| {
        let state = level.borrow();
        let state = state.as_ref().filter(|state| !state.suspended).ok_or_else(|| Error::Unsupported {
            op: "forward_ad tangent", msg: "tangent lookup requires an active forward level".into() })?;
        if let Some(error) = &state.error { return Err(error.clone()); }
        Ok(state.tangents.get(&source.id()).cloned())
    })
}

/// Reconstruct value/auxiliary aliases after a reverse transform invoked under
/// no_grad. Existing values keep identity; only genuine shared-storage views
/// of older real leaves keep public flags, with no reverse gradient edge.
#[doc(hidden)]
pub fn freeze_post(source: &Tensor, cutoff: usize, excluded: &[&Tensor]) -> Result<Tensor> {
    current_tangents(source)?;
    if source.id() < cutoff { return Ok(source.clone()); }
    if source.0.op.is_none() && !source.has_gradient_edge() {
        return Ok(if source.grad_flag_handles().1.is_some() { source.detach().inherit_no_grad_view(source) } else { source.clone() });
    }
    let mut current = source;
    loop {
        if excluded.iter().any(|excluded| excluded.id() == current.id()) { return Ok(source.detach()); }
        if current.id() < cutoff || current.is_inherited_grad_view() {
            return Ok(if current.requires_grad() {
                crate::autograd::with_grad_enabled(false, || source.detach().inherit_no_grad_view(current))
            } else { source.detach() });
        }
        let parent = current.0.op.as_ref().and_then(|op| op.inputs().iter().find(|parent| Arc::ptr_eq(&parent.0.storage, &current.0.storage)));
        let Some(parent) = parent else { return Ok(source.detach()); };
        current = parent;
    }
}

/// Private postprocessing may strip synthetic reverse provenance while keeping
/// an explicitly postprocessed forward tangent on the same primal data alias.
#[doc(hidden)]
pub fn post_alias(source: &Tensor, target: &Tensor, tangent: Option<&Tensor>) -> Result<()> {
    let count = LEVEL.with(|level| level.borrow().as_ref().map(|state| state.columns));
    if count != Some(1) { return Err(Error::Unsupported { op: "forward_ad alias", msg: "single-direction postprocessing requires exactly one tangent column".into() }); }
    match tangent {
        Some(tangent) => post_alias_columns(source, target, Some(&[tangent])),
        None => post_alias_columns(source, target, None),
    }
}

#[doc(hidden)]
pub fn post_alias_columns(source: &Tensor, target: &Tensor, tangents: Option<&[&Tensor]>) -> Result<()> {
    check_alias(source, target)?;
    if let Some(tangents) = tangents {
        for tangent in tangents {
            if tangent.shape() != target.shape() || !matches!(tangent.dtype(), DType::F32 | DType::F64) || tangent.device() != target.device() {
                return Err(Error::Unsupported { op: "forward_ad alias", msg: "postprocessed tangent metadata does not match its primal".into() });
            }
        }
    }
    LEVEL.with(|level| {
        let mut state = level.borrow_mut();
        let state = state.as_mut().filter(|state| !state.suspended).ok_or_else(|| Error::Unsupported {
            op: "forward_ad alias", msg: "transform postprocessing requires an active forward level".into() })?;
        if let Some(error) = &state.error { return Err(error.clone()); }
        if let Some(tangents) = tangents {
            if tangents.len() != state.columns { return Err(Error::Unsupported { op: "forward_ad alias", msg: "postprocessed tangent column count differs from the active level".into() }); }
            state.tangents.insert(target.id(), tangents.iter().map(|value| (*value).clone()).collect());
        } else { state.tangents.remove(&target.id()); }
        Ok(())
    })
}

pub(crate) fn records_grad(input: &Tensor) -> bool {
    let flag = input.requires_grad();
    LEVEL.with(|level| {
        let mut state = level.borrow_mut();
        flag && !state.as_mut().is_some_and(|state| ignored_factory(state, input))
    })
}

pub(crate) fn recording_inputs(inputs: Vec<Tensor>) -> Vec<Tensor> {
    let ignored = LEVEL.with(|level| {
        let mut state = level.borrow_mut();
        let state = state.as_mut()?;
        state.functional_cutoff?;
        Some(inputs.iter().map(|input| ignored_factory(state, input)).collect::<Vec<_>>())
    });
    let Some(ignored) = ignored else { return inputs; };
    // Store aliases with the transform's effective provenance, not merely a
    // masked traversal edge. Later graph-building VJPs run after this level
    // closes and must not reconnect an ignored factory through its saved value.
    // detach shares storage/layout/version, preserving mutation validation.
    inputs.into_iter().zip(ignored).map(|(input, ignored)| if ignored { input.detach() } else { input }).collect()
}

pub(crate) fn needs(inputs: &[Tensor]) -> bool {
    LEVEL.with(|level| {
        let mut state = level.borrow_mut();
        let Some(state) = state.as_mut() else { return false; };
        if state.suspended || state.error.is_some() { return false; }
        if crate::capture::is_recording() {
            state.error = Some(Error::Unsupported { op: "forward_ad", msg: "inference capture inside a forward AD level is unsupported".into() });
            return false;
        }
        inputs.iter().any(|input| state.tangents.contains_key(&input.id()))
    })
}

struct Resume(bool);
impl Drop for Resume { fn drop(&mut self) { LEVEL.with(|level| { if let Some(state) = level.borrow_mut().as_mut() { state.suspended = self.0; } }); } }

pub(crate) fn record(output: &Tensor, op: &Op) {
    let tangents = LEVEL.with(|level| {
        let mut state = level.borrow_mut();
        let state = state.as_mut()?;
        if state.suspended || state.error.is_some() { return None; }
        let values = op.inputs().iter().map(|input| state.tangents.get(&input.id()).cloned()).collect::<Vec<_>>();
        if values.iter().all(Option::is_none) { return None; }
        state.suspended = true;
        Some((state.columns, values))
    });
    let Some((columns, tangents)) = tangents else { return; };
    let _resume = Resume(false);
    let result = (|| {
        let mut results = Vec::new();
        results.try_reserve_exact(columns).map_err(|_| Error::Unsupported { op: "forward_ad", msg: "tangent column metadata exceeds available capacity".into() })?;
        for index in 0..columns {
            let column = tangents.iter().map(|values| values.as_ref().map(|values| values[index].clone())).collect::<Vec<_>>();
            let tangent = op.jvp(output, &column)?;
            if tangent.shape() != output.shape() { return Err(Error::ShapeMismatch { op: "forward_ad", lhs: output.shape().to_vec(), rhs: tangent.shape().to_vec() }); }
            if !matches!(tangent.dtype(), DType::F32 | DType::F64) || tangent.device() != Device::Cpu {
                return Err(Error::Unsupported { op: "forward_ad", msg: "local JVP must return a CPU floating-point tangent".into() });
            }
            results.push(tangent);
        }
        Ok(results)
    })();
    LEVEL.with(|level| {
        let mut state = level.borrow_mut();
        let state = state.as_mut().expect("active dual level");
        match result { Ok(tangent) => { state.tangents.insert(output.id(), tangent); }, Err(error) => state.error = Some(error) }
    });
}
