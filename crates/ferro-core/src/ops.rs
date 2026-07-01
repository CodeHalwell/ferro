use crate::autograd::Op;
use crate::error::Result;
use crate::shape::numel;
use crate::tensor::{raw_binary, raw_matmul, raw_unary, Tensor};

/// Forward ops: compute the value with a detached raw kernel, then (only when a
/// gradient is needed) attach the autograd node that knows how to differentiate.
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("add", self, other, |a, b| a + b)?;
        let rg = self.requires_grad() || other.requires_grad();
        Ok(out.record(rg, || Op::Add(self.clone(), other.clone())))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("sub", self, other, |a, b| a - b)?;
        let rg = self.requires_grad() || other.requires_grad();
        Ok(out.record(rg, || Op::Sub(self.clone(), other.clone())))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("mul", self, other, |a, b| a * b)?;
        let rg = self.requires_grad() || other.requires_grad();
        Ok(out.record(rg, || Op::Mul(self.clone(), other.clone())))
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("div", self, other, |a, b| a / b)?;
        let rg = self.requires_grad() || other.requires_grad();
        Ok(out.record(rg, || Op::Div(self.clone(), other.clone())))
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_matmul(self, other)?;
        let rg = self.requires_grad() || other.requires_grad();
        Ok(out.record(rg, || Op::Matmul(self.clone(), other.clone())))
    }

    pub fn neg(&self) -> Tensor {
        let out = raw_unary(self, |x| -x);
        out.record(self.requires_grad(), || Op::Neg(self.clone()))
    }

    pub fn relu(&self) -> Tensor {
        let out = raw_unary(self, |x| x.max(0.0));
        out.record(self.requires_grad(), || Op::Relu(self.clone()))
    }

    pub fn exp(&self) -> Tensor {
        let out = raw_unary(self, |x| x.exp());
        let saved = out.detach_copy();
        out.record(self.requires_grad(), || Op::Exp(self.clone(), saved))
    }

    pub fn sigmoid(&self) -> Tensor {
        let out = raw_unary(self, |x| 1.0 / (1.0 + (-x).exp()));
        let saved = out.detach_copy();
        out.record(self.requires_grad(), || Op::Sigmoid(self.clone(), saved))
    }

    pub fn sum(&self) -> Tensor {
        let total: f32 = self.to_vec().iter().sum();
        let out = Tensor::scalar(total);
        out.record(self.requires_grad(), || Op::Sum(self.clone(), self.shape().to_vec()))
    }

    pub fn mean(&self) -> Tensor {
        let v = self.to_vec();
        let out = Tensor::scalar(v.iter().sum::<f32>() / numel(self.shape()).max(1) as f32);
        out.record(self.requires_grad(), || Op::Mean(self.clone(), self.shape().to_vec()))
    }
}
