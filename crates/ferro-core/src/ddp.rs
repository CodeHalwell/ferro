//! Single-process, multi-device data parallelism (DDP) for CPU+CUDA pairs.
//!
//! v1 design: each step replicates the batch and the named parameters to
//! every device, runs an independent forward/backward per replica, then
//! averages gradients by pulling every replica grad to the host, summing,
//! dividing by the replica count (exact mean, matching torch DDP), and
//! pushing the result back to the primary device's canonical parameters.
//!
//! This host round-trip is the deliberate v1 simplification: it needs no
//! collective communication primitives. A true allreduce slots in at exactly
//! one seam - `Ddp::average` - which would become a ring/tree allreduce over
//! device buffers instead of host gather + scatter. Nothing else in this file
//! changes when that lands.

use crate::device::Device;
use crate::error::{Error, Result};
use crate::params::Param;
use crate::tensor::Tensor;

/// Gradient-averaged data parallel wrapper over a set of named parameters.
/// The first device is the primary: averaged gradients land there and on the
/// original `Param` slots, so any optimizer over those params just works.
pub struct Ddp {
    params: Vec<(String, Param)>,
    devices: Vec<Device>,
}

impl Ddp {
    /// `devices` must be non-empty with no duplicates; `devices[0]` is primary.
    pub fn new(named: Vec<(String, Param)>, devices: Vec<Device>) -> Result<Ddp> {
        if named.is_empty() {
            return Err(Error::Unsupported {
                op: "ddp_new",
                msg: "at least one parameter is required".into(),
            });
        }
        if devices.is_empty() {
            return Err(Error::Unsupported {
                op: "ddp_new",
                msg: "at least one device is required".into(),
            });
        }
        let mut seen = std::collections::HashSet::new();
        for d in &devices {
            if !seen.insert(*d) {
                return Err(Error::Unsupported {
                    op: "ddp_new",
                    msg: format!("duplicate device {d}"),
                });
            }
        }
        Ok(Ddp {
            params: named,
            devices,
        })
    }

    pub fn primary(&self) -> Device {
        self.devices[0]
    }

    pub fn replicas(&self) -> &[Device] {
        &self.devices
    }

    pub fn named_parameters(&self) -> &[(String, Param)] {
        &self.params
    }

    fn zero_canonical_grads(&self) {
        for (_, p) in &self.params {
            p.zero_grad();
        }
    }

    fn replica_params(&self, dev: Device) -> Result<Vec<Tensor>> {
        self.params
            .iter()
            .map(|(_, p)| p.tensor().to_device(dev)?.requires_grad_(true))
            .collect()
    }

    /// Replicate `batch` to every device, run `f(x_replica, &replica_params)`
    /// per replica (must return a scalar loss), backward through each graph.
    /// Returns losses plus per-replica parameter gradients; grads[i][j] is
    /// param j's gradient on replica i's own device.
    pub fn backward_replicas(
        &self,
        batch: &Tensor,
        f: &mut dyn FnMut(&Tensor, &[Tensor]) -> Result<Tensor>,
    ) -> Result<(Vec<f32>, Vec<Vec<Tensor>>)> {
        self.zero_canonical_grads();
        let mut losses = Vec::with_capacity(self.devices.len());
        let mut grad_sets = Vec::with_capacity(self.devices.len());
        for &dev in &self.devices {
            let x = batch.to_device(dev)?;
            let reps = self.replica_params(dev)?;
            let loss = f(&x, &reps)?;
            if loss.numel() != 1 {
                return Err(Error::InvalidShape {
                    op: "ddp_step",
                    msg: format!(
                        "closure returned non-scalar loss of shape {:?}",
                        loss.shape()
                    ),
                });
            }
            loss.backward();
            losses.push(loss.item());
            let grads = reps
                .iter()
                .map(|t| {
                    t.grad().ok_or_else(|| Error::Unsupported {
                        op: "ddp_step",
                        msg: "replica param produced no gradient".into(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            grad_sets.push(grads);
        }
        Ok((losses, grad_sets))
    }

    /// Exact mean across replicas: pull each replica grad to the host, sum
    /// elementwise, divide by the replica count, upload to the primary
    /// device. This is the v1 allreduce stand-in (see module docs).
    pub fn average(&self, grad_sets: &[Vec<Tensor>]) -> Result<Vec<Tensor>> {
        if grad_sets.len() != self.devices.len() {
            return Err(Error::Unsupported {
                op: "ddp_average",
                msg: format!(
                    "expected {} replica gradient sets, got {}",
                    self.devices.len(),
                    grad_sets.len()
                ),
            });
        }
        let n = grad_sets.len() as f32;
        (0..self.params.len())
            .map(|j| {
                let mut sum = grad_sets[0][j].to_vec();
                for set in &grad_sets[1..] {
                    let g = &set[j];
                    let v = g.to_vec();
                    if v.len() != sum.len() {
                        return Err(Error::ShapeMismatch {
                            op: "ddp_average",
                            lhs: grad_sets[0][j].shape().to_vec(),
                            rhs: g.shape().to_vec(),
                        });
                    }
                    for (s, x) in sum.iter_mut().zip(v) {
                        *s += x;
                    }
                }
                Tensor::from_vec(sum.iter().map(|s| s / n).collect(), grad_sets[0][j].shape())?
                    .to_device(self.primary())
            })
            .collect()
    }

    /// One full DDP step: replicate, per-replica forward/backward, exact
    /// gradient mean onto the canonical parameters (primary device). Returns
    /// per-replica losses in replica order.
    pub fn step(
        &self,
        batch: &Tensor,
        f: &mut dyn FnMut(&Tensor, &[Tensor]) -> Result<Tensor>,
    ) -> Result<Vec<f32>> {
        let (losses, grad_sets) = self.backward_replicas(batch, f)?;
        let avg = self.average(&grad_sets)?;
        for ((_, p), g) in self.params.iter().zip(avg) {
            // accumulate_grad aligns g to the leaf's device; leaves are already
            // on the primary after Ddp::new validated replication, and their
            // grad slots were cleared at the top of this step.
            p.tensor().accumulate_grad(g);
        }
        Ok(losses)
    }
}
