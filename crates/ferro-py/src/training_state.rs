use super::*;
use ferro_core::optim::{Adam, AdamW, Sgd, Adaptive, Lbfgs, Adafactor, Muon, OptimizerState};

#[derive(Clone)]
enum Optimizer { Sgd(Sgd), Adam(Adam), AdamW(AdamW), Adaptive(Adaptive), Lbfgs(Lbfgs), Adafactor(Adafactor), Muon(Muon) }
impl Optimizer {
    fn state(&self) -> &dyn OptimizerState {
        match self { Self::Sgd(o) => o, Self::Adam(o) => o, Self::AdamW(o) => o, Self::Adaptive(o) => o, Self::Lbfgs(o) => o, Self::Adafactor(o) => o, Self::Muon(o) => o }
    }
    fn state_mut(&mut self) -> &mut dyn OptimizerState {
        match self { Self::Sgd(o) => o, Self::Adam(o) => o, Self::AdamW(o) => o, Self::Adaptive(o) => o, Self::Lbfgs(o) => o, Self::Adafactor(o) => o, Self::Muon(o) => o }
    }
    fn empty(&self) -> Self {
        match self {
            Self::Sgd(o) => Self::Sgd(Sgd::new(o.state_parameters().unwrap(), 0.0)),
            Self::Adam(o) => Self::Adam(Adam::new(o.state_parameters().unwrap(), 0.0)),
            Self::AdamW(o) => Self::AdamW(AdamW::new(o.state_parameters().unwrap(), 0.0)),
            Self::Adafactor(o) => Self::Adafactor(Adafactor::new(vec![], &o.configuration()).expect("validated configuration")),
            Self::Muon(o) => Self::Muon(Muon::new(vec![], &o.configuration()).expect("validated configuration")),
            Self::Lbfgs(o) => Self::Lbfgs(Lbfgs::new(o.state_parameters().unwrap(), &o.configuration()).expect("validated LBFGS configuration")),
            Self::Adaptive(o) => Self::Adaptive(Adaptive::new(vec![], &o.configuration()).expect("validated configuration")),
        }
    }
}

type GradientHandles = Vec<(CoreTensor, Option<CoreTensor>)>;
fn gradient_handles(params: &[ferro_core::params::Param]) -> PyResult<GradientHandles> {
    if ferro_core::capture::is_recording() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err("Python optimizer steps require eager execution"));
    }
    let handles: GradientHandles = params.iter().map(|p| { let t=p.tensor(); let g=t.grad(); (t,g) }).collect();
    // Validate before updates, including gradients hidden by a frozen Param.
    for (tensor,gradient) in &handles { tensor.set_leaf_grad(gradient.clone()).map_err(map_err)?; }
    Ok(handles)
}
fn restore_gradient_handles(handles: GradientHandles) -> PyResult<()> {
    for (tensor,gradient) in handles { tensor.set_leaf_grad(gradient).map_err(map_err)?; }
    Ok(())
}

fn optimizer_parameters(params: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Param>> {
    let mut identities = std::collections::HashSet::new();
    params.into_iter().map(|value| {
        let parameter = if let Ok(parameter) = value.extract::<PyRef<'_, PyParameter>>() {
            parameter.inner.clone()
        } else {
            let tensor = value.extract::<PyRef<'_, PyTensor>>()?;
            Param::from_leaf(&tensor.inner).map_err(map_err)?
        };
        if !identities.insert(parameter.tensor().id()) {
            return Err(PyValueError::new_err("optimizer parameters must have distinct gradient slots"));
        }
        Ok(parameter)
    }).collect()
}

#[pyclass(name = "_Optimizer", unsendable)]
pub(crate) struct PyOptimizer { groups: Vec<Optimizer> }
impl PyOptimizer {
    fn state(&self) -> &dyn OptimizerState { self }
    fn state_mut(&mut self) -> &mut dyn OptimizerState { self }
    fn configurations_from(&self, config: &[f32]) -> ferro_core::Result<Vec<Vec<f32>>> {
        if self.groups.len() == 1 { return Ok(vec![config.to_vec()]); }
        let invalid = || ferro_core::Error::Format { op: "optim_restore", msg: "optimizer group configuration/partition mismatch".into() };
        if config.len() < 2 || config[0] != 4.0 || config[1] != self.groups.len() as f32 { return Err(invalid()); }
        let mut offset = 2;
        let mut result = Vec::new();
        for group in &self.groups {
            let count = group.state().state_parameters().unwrap().len();
            if offset + 2 > config.len() || config[offset] != count as f32 || !config[offset + 1].is_finite()
                || config[offset + 1] < 0.0 || config[offset + 1].fract() != 0.0 || config[offset + 1] as f64 > (config.len() - offset - 2) as f64 { return Err(invalid()); }
            let width = config[offset + 1] as usize;
            result.push(config[offset + 2..offset + 2 + width].to_vec());
            offset += 2 + width;
        }
        if offset != config.len() { return Err(invalid()); }
        Ok(result)
    }
    fn configure(&mut self, configurations: &[Vec<f32>]) -> ferro_core::Result<()> {
        for group in &self.groups { match group {
            Optimizer::Sgd(_) | Optimizer::Adaptive(_) => {},
            Optimizer::Adam(_) | Optimizer::AdamW(_) => ferro_core::optim::validate_adam_parameters(&group.state().state_parameters().unwrap())?,
            _ => ferro_core::optim::validate_f32_parameters(&group.state().state_parameters().unwrap())?,
        }}
        if configurations.len() != self.groups.len() {
            return Err(ferro_core::Error::Format { op: "optim_restore", msg: "optimizer group count mismatch".into() });
        }
        // Validate without copying live moments or allocating Adagrad accumulators.
        for (group, config) in self.groups.iter().zip(configurations) {
            match group {
                Optimizer::Adaptive(o) => o.validate_configuration(config)?,
                _ => group.empty().state_mut().restore_configuration(config)?,
            }
        }
        for (group, config) in self.groups.iter_mut().zip(configurations) {
            group.state_mut().restore_configuration(config).expect("validated optimizer configuration");
        }
        Ok(())
    }
}

impl OptimizerState for PyOptimizer {
    fn expected_restore_keys(&self, config: &[f32]) -> ferro_core::Result<Vec<String>> {
        let configurations = self.configurations_from(config)?;
        let mut keys = Vec::new();
        for (i, (group, config)) in self.groups.iter().zip(configurations).enumerate() {
            for name in group.state().expected_restore_keys(&config)? {
                keys.push(if self.groups.len() == 1 { name } else { format!("g{i}.{name}") });
            }
        }
        Ok(keys)
    }
    fn state_parameters(&self) -> Option<Vec<ferro_core::params::Param>> {
        Some(self.groups.iter().flat_map(|g| g.state().state_parameters().unwrap()).collect())
    }
    fn state_buffers(&self) -> Option<Vec<CoreTensor>> {
        Some(self.groups.iter().flat_map(|g| g.state().state_buffers().unwrap()).collect())
    }
    fn configuration(&self) -> Vec<f32> {
        if self.groups.len() == 1 { return self.groups[0].state().configuration(); }
        let mut out = vec![4.0, self.groups.len() as f32];
        for group in &self.groups {
            let config = group.state().configuration();
            out.extend([group.state().state_parameters().unwrap().len() as f32, config.len() as f32]);
            out.extend(config);
        }
        out
    }
    fn restore_configuration(&mut self, config: &[f32]) -> ferro_core::Result<()> {
        let configs = self.configurations_from(config)?;
        self.configure(&configs)
    }
    fn snapshot(&self) -> Vec<(String, CoreTensor)> {
        if self.groups.len() == 1 { return self.groups[0].state().snapshot(); }
        self.groups.iter().enumerate().flat_map(|(i,g)| {
            g.state().snapshot().into_iter().map(move |(key,value)| (format!("g{i}.{key}"),value))
        }).collect()
    }
    fn prepare_cpu_restore<'a>(&'a mut self, config: &[f32], state: &[(String, CoreTensor)]) -> ferro_core::Result<Box<dyn FnOnce() + 'a>> {
        let configs = self.configurations_from(config)?;
        if self.groups.len() == 1 { return self.groups[0].state_mut().prepare_cpu_restore(&configs[0], state); }
        let states: Vec<Vec<_>> = (0..self.groups.len()).map(|i| {
            let prefix = format!("g{i}.");
            state.iter().filter_map(|(key,value)| key.strip_prefix(&prefix).map(|name| (name.to_owned(),value.clone()))).collect()
        }).collect();
        if states.iter().map(Vec::len).sum::<usize>() != state.len() {
            return Err(ferro_core::Error::Format { op: "optim_restore", msg: "unexpected optimizer group state key".into() });
        }
        let mut commits = Vec::new();
        for ((group, config), state) in self.groups.iter_mut().zip(&configs).zip(&states) {
            commits.push(group.state_mut().prepare_cpu_restore(config, state)?);
        }
        Ok(Box::new(move || { for commit in commits { commit(); } }))
    }
    fn restore(&mut self, state: &[(String, CoreTensor)]) -> ferro_core::Result<()> {
        let config = self.configuration();
        self.prepare_cpu_restore(&config, state)?();
        Ok(())
    }
}
#[pymethods]
impl PyOptimizer {
    #[staticmethod]
    fn sgd(params: Vec<Bound<'_, PyAny>>, config: Vec<f32>) -> PyResult<Self> {
        let mut optimizer = Sgd::new(optimizer_parameters(params)?, 0.);
        optimizer.restore_configuration(&config).map_err(map_err)?;
        optimizer.validate_parameters().map_err(map_err)?;
        Ok(Self { groups: vec![Optimizer::Sgd(optimizer)] })
    }
    #[staticmethod]
    fn adam(params: Vec<Bound<'_, PyAny>>, config: Vec<f32>) -> PyResult<Self> {
        let slots = optimizer_parameters(params)?;
        ferro_core::optim::validate_adam_parameters(&slots).map_err(map_err)?;
        let mut optimizer = match config.first() {
            Some(2.) => Optimizer::Adam(Adam::new(slots, 0.)),
            Some(3.) => Optimizer::AdamW(AdamW::new(slots, 0.)),
            _ => return Err(PyValueError::new_err("invalid Adam family configuration")),
        };
        optimizer.state_mut().restore_configuration(&config).map_err(map_err)?;
        match &optimizer { Optimizer::Adam(o) => o.validate_parameters().map_err(map_err)?,
            Optimizer::AdamW(o) => o.validate_parameters().map_err(map_err)?, _ => unreachable!() }
        Ok(Self { groups: vec![optimizer] })
    }
    #[staticmethod]
    fn adaptive(params: Vec<Bound<'_, PyAny>>, config: Vec<f32>) -> PyResult<Self> {
        let slots = optimizer_parameters(params)?;
        Ok(Self { groups: vec![Optimizer::Adaptive(Adaptive::new(slots, &config).map_err(map_err)?)] })
    }
    #[staticmethod]
    fn adafactor(params: Vec<Bound<'_, PyAny>>, config: Vec<f32>) -> PyResult<Self> {
        let slots=optimizer_parameters(params)?;
        Ok(Self {groups:vec![Optimizer::Adafactor(Adafactor::new(slots,&config).map_err(map_err)?)]})
    }
    #[staticmethod]
    fn muon(params: Vec<Bound<'_, PyAny>>, config: Vec<f32>) -> PyResult<Self> {
        let slots=optimizer_parameters(params)?;
        Ok(Self {groups:vec![Optimizer::Muon(Muon::new(slots,&config).map_err(map_err)?)]})
    }
    #[staticmethod]
    fn lbfgs(params: Vec<Bound<'_, PyAny>>, config: Vec<f32>) -> PyResult<Self> {
        let slots = optimizer_parameters(params)?;
        Ok(Self { groups: vec![Optimizer::Lbfgs(Lbfgs::new(slots, &config).map_err(map_err)?)] })
    }
    #[staticmethod]
    #[pyo3(signature = (optimizer, closure, preserve_grads=false))]
    fn lbfgs_step(py: Python<'_>, optimizer: Py<PyOptimizer>, closure: Py<PyAny>, preserve_grads: bool) -> PyResult<Py<PyAny>> {
        // Release the PyO3 borrow while evaluating user code: a closure commonly
        // calls this same optimizer's zero_grad. The temporary optimizer shares
        // only its parameter slots. Public Python guards forbid other reentry.
        let slots = optimizer.borrow(py).state_parameters().unwrap();
        ferro_core::optim::validate_gradient_storage(&slots).map_err(map_err)?;
        if preserve_grads { gradient_handles(&slots)?; }
        let mut native = {
            let mut target = optimizer.borrow_mut(py);
            if target.groups.len()!=1 || !matches!(target.groups[0],Optimizer::Lbfgs(_)) {
                return Err(PyValueError::new_err("LBFGS needs one native LBFGS group"));
            }
            match std::mem::replace(&mut target.groups[0],Optimizer::Sgd(Sgd::new(slots.clone(),0.))) {
                Optimizer::Lbfgs(o)=>o, _=>unreachable!(),
            }
        };
        let mut original=None; let mut callback_error=None; let mut last_gradients=None;
        let outcome=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| native.step(|| {
            let result=(|| -> PyResult<f32> {
                let loss=closure.call0(py)?;
                ferro_core::optim::validate_gradient_storage(&slots).map_err(map_err)?;
                let value={
                    let tensor=loss.extract::<PyRef<'_,PyTensor>>(py)?;
                    if tensor.inner.dtype()!=ferro_core::DType::F32 || tensor.inner.device()!=ferro_core::Device::Cpu || !tensor.inner.shape().is_empty() {
                        return Err(PyValueError::new_err("LBFGS closure must return a CPU float32 scalar Tensor"));
                    }
                    tensor.inner.item()
                };
                if preserve_grads { last_gradients=Some(gradient_handles(&slots)?); }
                if original.is_none() {original=Some(loss);}
                Ok(value)
            })();
            result.map_err(|error| {
                callback_error=Some(error);
                ferro_core::Error::Format {op:"lbfgs",msg:"loss closure failed".into()}
            })
        })));
        optimizer.borrow_mut(py).groups[0]=Optimizer::Lbfgs(native);
        if let Some(error)=callback_error {return Err(error);}
        outcome.map_err(|_|PyValueError::new_err("LBFGS step failed; discard optimizer after an update error"))?.map_err(map_err)?;
        if let Some(handles)=last_gradients { restore_gradient_handles(handles)?; }
        Ok(original.expect("successful LBFGS evaluates its closure"))
    }
    #[new]
    fn new(kind: &str, params: Vec<Bound<'_, PyAny>>, lr: f32, beta1: f32, beta2: f32, eps: f32, decay: f32, momentum: f32, nesterov: bool) -> PyResult<Self> {
        if !lr.is_finite() || lr < 0.0 || !beta1.is_finite() || !(0.0..1.0).contains(&beta1)
            || !beta2.is_finite() || !(0.0..1.0).contains(&beta2) || !eps.is_finite() || eps <= 0.0
            || !decay.is_finite() || decay < 0.0 || !momentum.is_finite() || momentum < 0.0 || (nesterov && momentum == 0.0) {
            return Err(PyValueError::new_err("invalid optimizer configuration"));
        }
        let slots = optimizer_parameters(params)?;
        if kind != "SGD" { ferro_core::optim::validate_f32_parameters(&slots).map_err(map_err)?; }
        let inner = match kind {
            "SGD" => Optimizer::Sgd(Sgd::new(slots, lr).with_momentum(momentum).with_nesterov(nesterov)),
            "Adam" => Optimizer::Adam(Adam::new(slots, lr).with_betas(beta1, beta2).with_eps(eps)),
            "AdamW" => {
                let mut opt = AdamW::new(slots, lr);
                opt.restore_configuration(&[3.0, lr, beta1, beta2, eps, decay, -1.0]).map_err(map_err)?;
                Optimizer::AdamW(opt)
            },
            _ => return Err(PyValueError::new_err("unknown optimizer kind")),
        };
        Ok(Self { groups: vec![inner] })
    }
    #[staticmethod]
    fn grouped(groups: Vec<PyRef<'_, PyOptimizer>>) -> PyResult<Self> {
        if groups.is_empty() || groups.len() > (1 << 24) {
            return Err(PyValueError::new_err("invalid optimizer group count"));
        }
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        let kind = groups[0].groups[0].state().configuration()[0];
        for group in groups {
            if group.groups.len() != 1 || group.configuration()[0] != kind {
                return Err(PyValueError::new_err("groups must be single native optimizers of the same kind"));
            }
            let params = group.state_parameters().unwrap();
            if params.len() > (1 << 24) || params.iter().any(|p| !seen.insert(p.tensor().id())) {
                return Err(PyValueError::new_err("a parameter cannot occur in multiple optimizer groups"));
            }
            result.push(group.groups[0].clone());
        }
        Ok(Self { groups: result })
    }
    fn add_group(&mut self, group: PyRef<'_, PyOptimizer>) -> PyResult<()> {
        if group.groups.len() != 1 || group.configuration()[0] != self.groups[0].state().configuration()[0] {
            return Err(PyValueError::new_err("new group must have the same optimizer kind"));
        }
        let existing: std::collections::HashSet<_> = self.state_parameters().unwrap().iter().map(|p| p.tensor().id()).collect();
        let incoming = group.state_parameters().unwrap();
        if self.groups.len() >= (1 << 24) || incoming.len() > (1 << 24) {
            return Err(PyValueError::new_err("optimizer group size exceeds exact configuration representation"));
        }
        if incoming.iter().any(|p| existing.contains(&p.tensor().id())) {
            return Err(PyValueError::new_err("a parameter cannot occur in multiple optimizer groups"));
        }
        self.groups.push(group.groups[0].clone());
        Ok(())
    }
    fn configurations(&self) -> Vec<Vec<f32>> { self.groups.iter().map(|g| g.state().configuration()).collect() }
    fn set_configurations(&mut self, values: Vec<Vec<f32>>) -> PyResult<()> { self.configure(&values).map_err(map_err) }
    fn state_tensors(&self) -> Vec<(String, PyTensor)> { self.snapshot().into_iter().map(|(n,t)| (n,PyTensor::wrap(t))).collect() }
    fn restore_state(&mut self, config: Vec<f32>, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        ferro_core::checkpoint::validate_restore_context().map_err(map_err)?;
        let state: Vec<_> = state.into_iter().map(|(n,t)| (n,t.inner)).collect();
        self.prepare_cpu_restore(&config, &state).map_err(map_err)?();
        Ok(())
    }
    fn zero_grad(&self) {
        for group in &self.groups { match group { Optimizer::Sgd(o) => o.zero_grad(), Optimizer::Adam(o) => o.zero_grad(), Optimizer::AdamW(o) => o.zero_grad(), Optimizer::Adaptive(o) => o.zero_grad(), Optimizer::Lbfgs(o) => o.zero_grad(), Optimizer::Adafactor(o) => o.zero_grad(), Optimizer::Muon(o) => o.zero_grad() } }
    }
    #[pyo3(signature = (preserve_grads=false))]
    fn step(&mut self, preserve_grads: bool) -> PyResult<()> {
        self.validate_gradient_storage()?;
        let gradients=if preserve_grads {Some(gradient_handles(&self.state_parameters().unwrap())?)} else {None};
        let outcome=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for group in &mut self.groups { match group {
                Optimizer::Sgd(o) => o.step(), Optimizer::Adam(o) => o.step(), Optimizer::AdamW(o) => o.step(),
                Optimizer::Adaptive(o) => o.step().map_err(map_err)?,
                Optimizer::Adafactor(o) => o.step().map_err(map_err)?,
                Optimizer::Muon(o) => o.step().map_err(map_err)?,
                Optimizer::Lbfgs(_) => return Err(PyValueError::new_err("LBFGS requires a closure")),
            }}
            Ok(())
        })).map_err(|_| PyValueError::new_err("native optimizer step failed; discard optimizer after an update error"));
        if let Some(handles)=gradients { restore_gradient_handles(handles)?; }
        outcome?
    }
    fn validate_gradient_storage(&self) -> PyResult<()> {
        ferro_core::optim::validate_gradient_storage(&self.state_parameters().unwrap()).map_err(map_err)?;
        for group in &self.groups { match group {
            Optimizer::Sgd(o) => o.validate_parameters().map_err(map_err)?,
            Optimizer::Adam(o) => o.validate_parameters().map_err(map_err)?,
            Optimizer::AdamW(o) => o.validate_parameters().map_err(map_err)?,
            Optimizer::Adaptive(o) => o.validate_parameters().map_err(map_err)?,
            _ => ferro_core::optim::validate_f32_parameters(&group.state().state_parameters().unwrap()).map_err(map_err)?,
        }}
        Ok(())
    }
}


use ferro_core::checkpoint::Checkpoint;
use ferro_core::nn::Module;
use ferro_core::params::Param;
use std::cell::RefCell;

struct StateModule {
    params: Vec<(String, Param)>,
    buffers: Vec<(String, CoreTensor)>,
    scalars: RefCell<Vec<(String, u64)>>,
}
impl Module for StateModule {
    fn forward(&self, _: &CoreTensor) -> ferro_core::Result<CoreTensor> {
        Err(ferro_core::Error::Unsupported { op: "checkpoint", msg: "state-only adapter".into() })
    }
    fn named_parameters(&self) -> Vec<(String, Param)> { self.params.clone() }
    // Keep every name and the original StorageCell: core validates cross-model
    // storage aliases before any model, optimizer, scalar or RNG commit.
    fn named_buffers(&self) -> Vec<(String, CoreTensor)> { self.buffers.clone() }
    fn snapshot_scalars(&self) -> Vec<(String, u64)> { self.scalars.borrow().clone() }
    fn validate_scalars(&self, state: &[(String, u64)]) -> ferro_core::Result<()> {
        let expected = self.scalars.borrow();
        if state.len() != expected.len() || expected.iter().any(|(key, value)| {
            !state.iter().any(|(k, v)| k == key && if key.starts_with("mode.") { *v <= 1 }
                else if key.starts_with("optim_meta.") { true } else { v == value })
        }) {
            return Err(ferro_core::Error::Format { op: "checkpoint", msg: "Python module configuration/mode schema mismatch".into() });
        }
        Ok(())
    }
    fn commit_scalars(&self, state: &[(String, u64)]) { *self.scalars.borrow_mut() = state.to_vec(); }
}

fn adapter(params: Vec<(String, PyRef<'_, PyParameter>)>, buffers: Vec<(String, PyTensor)>,
    scalars: Vec<(String, u64)>) -> PyResult<StateModule> {
    let params: Vec<_> = params.into_iter().map(|(n,p)| (n,p.inner.clone())).collect();
    let buffers: Vec<_> = buffers.into_iter().map(|(n,t)| (n,t.inner)).collect();
    if params.iter().any(|(_,p)| !matches!(p.tensor().dtype(), ferro_core::dtype::DType::F32 | ferro_core::dtype::DType::F64)) {
        return Err(PyValueError::new_err("training checkpoint parameters must be float32 or float64"));
    }
    for t in params.iter().map(|(_,p)| p.tensor()).chain(buffers.iter().map(|(_,t)| t.clone())) {
        if t.device() != ferro_core::device::Device::Cpu || !matches!(t.dtype(), ferro_core::dtype::DType::F32 | ferro_core::dtype::DType::F64 | ferro_core::dtype::DType::I64 | ferro_core::dtype::DType::Bool) {
            return Err(PyValueError::new_err("training checkpoint requires CPU F32/F64 parameters and F32/F64/I64/Bool buffers"));
        }
        if t.grad().is_some() {
            return Err(PyValueError::new_err("pending gradients are unsupported; call zero_grad before checkpoint/restore"));
        }
    }
    Ok(StateModule { params, buffers, scalars: RefCell::new(scalars) })
}

#[pyclass(name = "_TrainingCheckpoint", unsendable)]
struct PyCheckpoint { inner: Checkpoint }
#[pymethods]
impl PyCheckpoint {
    #[staticmethod]
    fn snapshot(params: Vec<(String, PyRef<'_, PyParameter>)>, buffers: Vec<(String, PyTensor)>,
        scalars: Vec<(String, u64)>, optimizers: Vec<(String, PyRef<'_, PyOptimizer>)>, generator: &Generator, step: u64) -> PyResult<Self> {
        let module = adapter(params, buffers, scalars)?;
        let opts: Vec<_> = optimizers.iter().map(|(n,o)| (n.as_str(),o.state())).collect();
        let rng = generator.rng.lock().map_err(|_| PyValueError::new_err("generator lock poisoned"))?;
        let inner = Checkpoint::from_training_state(step, &module, &opts, &rng).map_err(map_err)?;
        Ok(Self { inner })
    }
    #[getter]
    fn step(&self) -> u64 { self.inner.step }
    fn save(&self, path: &str) -> PyResult<()> { self.inner.save_to_dir(path).map_err(map_err) }
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> { Ok(Self { inner: Checkpoint::load_from_dir(path).map_err(map_err)? }) }
    fn restore(&self, params: Vec<(String, PyRef<'_, PyParameter>)>, buffers: Vec<(String, PyTensor)>,
        scalars: Vec<(String, u64)>, mut optimizers: Vec<(String, PyRefMut<'_, PyOptimizer>)>, generator: &Generator) -> PyResult<Vec<(String, u64)>> {
        ferro_core::checkpoint::validate_restore_context().map_err(map_err)?;
        let module = adapter(params, buffers, scalars)?;
        for tensor in module.params.iter().map(|(_,parameter)| parameter.tensor()).chain(module.buffers.iter().map(|(_,tensor)| tensor.clone())) {
            ferro_core::checkpoint::validate_restore_target(&tensor).map_err(map_err)?;
        }
        let mut opts: Vec<_> = optimizers.iter_mut().map(|(n,o)| (n.as_str(),o.state_mut())).collect();
        let rng = generator.rng.lock().map_err(|_| PyValueError::new_err("generator lock poisoned"))?;
        self.inner.load_training_state_into(&module, &mut opts, &rng).map_err(map_err)?;
        Ok(module.snapshot_scalars())
    }
    // Owning inspection copies cannot mutate the checkpoint or live training state.
    fn tensors(&self) -> Vec<(String, PyTensor)> {
        self.inner.clone().tensors.into_iter().map(|(n,t)| (n,PyTensor::wrap(t))).collect()
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptimizer>()?;
    m.add_class::<PyCheckpoint>()?;
    Ok(())
}
