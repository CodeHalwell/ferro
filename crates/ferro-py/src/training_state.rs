use super::*;
use ferro_core::optim::{Adam, AdamW, Sgd, OptimizerState};

enum Optimizer { Sgd(Sgd), Adam(Adam), AdamW(AdamW) }

#[pyclass(name = "_Optimizer", unsendable)]
pub(crate) struct PyOptimizer { inner: Optimizer }
impl PyOptimizer {
    fn state(&self) -> &dyn OptimizerState {
        match &self.inner { Optimizer::Sgd(o) => o, Optimizer::Adam(o) => o, Optimizer::AdamW(o) => o }
    }
    fn state_mut(&mut self) -> &mut dyn OptimizerState {
        match &mut self.inner { Optimizer::Sgd(o) => o, Optimizer::Adam(o) => o, Optimizer::AdamW(o) => o }
    }
}
#[pymethods]
impl PyOptimizer {
    #[new]
    fn new(kind: &str, params: Vec<PyRef<'_, PyParameter>>, lr: f32, beta1: f32, beta2: f32, eps: f32, decay: f32, momentum: f32, nesterov: bool) -> PyResult<Self> {
        if !lr.is_finite() || lr < 0.0 || !beta1.is_finite() || !(0.0..1.0).contains(&beta1)
            || !beta2.is_finite() || !(0.0..1.0).contains(&beta2) || !eps.is_finite() || eps <= 0.0
            || !decay.is_finite() || decay < 0.0 || !momentum.is_finite() || momentum < 0.0 || (nesterov && momentum == 0.0) {
            return Err(PyValueError::new_err("invalid optimizer configuration"));
        }
        let slots = params.iter().map(|p| p.inner.clone()).collect();
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
        Ok(Self { inner })
    }
    fn zero_grad(&self) { match &self.inner { Optimizer::Sgd(o) => o.zero_grad(), Optimizer::Adam(o) => o.zero_grad(), Optimizer::AdamW(o) => o.zero_grad() } }
    fn step(&mut self) -> PyResult<()> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &mut self.inner {
            Optimizer::Sgd(o) => o.step(), Optimizer::Adam(o) => o.step(), Optimizer::AdamW(o) => o.step()
        })).map_err(|_| PyValueError::new_err("native optimizer step failed; discard optimizer after an update error"))
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
            !state.iter().any(|(k, v)| k == key && if key.starts_with("mode.") { *v <= 1 } else { v == value })
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
    for t in params.iter().map(|(_,p)| p.tensor()).chain(buffers.iter().map(|(_,t)| t.clone())) {
        if t.device() != ferro_core::device::Device::Cpu || t.dtype() != ferro_core::dtype::DType::F32 {
            return Err(PyValueError::new_err("training checkpoint requires contiguous CPU f32 state"));
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
        let module = adapter(params, buffers, scalars)?;
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
