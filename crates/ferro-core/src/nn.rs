//! Neural-network building blocks (Linear, activations, sequential containers).
//!
//! Built on the frozen `Tensor` API from `crate::tensor` / `crate::ops`:
//! `matmul`, `add` (broadcasts a bias row), `relu`, `sigmoid`, and the autograd
//! `backward()`.

use crate::params::Param;
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::Result;

pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<Param>;
}

/// Affine layer `y = x @ W + b` with He-initialized weights.
pub struct Linear {
    weight: Param,
    bias: Param,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, rng: &Rng) -> Linear {
        let scale = (2.0 / in_features as f32).sqrt();
        let w: Vec<f32> = (0..in_features * out_features).map(|_| rng.normal() * scale).collect();
        let weight = Param::new(Tensor::from_vec(w, &[in_features, out_features]).unwrap());
        let bias = Param::new(Tensor::zeros(&[out_features]));
        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight.tensor())?.add(&self.bias.tensor())
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

pub struct Relu;

impl Module for Relu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.relu())
    }

    fn parameters(&self) -> Vec<Param> {
        Vec::new()
    }
}

pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.sigmoid())
    }

    fn parameters(&self) -> Vec<Param> {
        Vec::new()
    }
}

/// Runs its layers in order, threading the output of each into the next.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    fn parameters(&self) -> Vec<Param> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
