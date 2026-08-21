use crate::dispatch::{BinaryKind, ReduceKind, UnaryKind};
use crate::error::Result;
use crate::reduce::pairwise_sum;
use crate::shape::numel;
use crate::tensor::{
    raw_binary_k, raw_matmul, raw_matmul_t, raw_reduce_dev, raw_unary_k, unbroadcast, Tensor,
};

/// Backends are looked up by device; Cpu is pre-registered and a device tensor
/// implies its backend was registered, so infallible forwards unwrap with this.
const REGISTERED: &str = "tensor's device backend is always registered";

/// Forward ops: compute the value with a detached raw kernel (named kind,
/// routed through the device's backend), then attach the backward closure via
/// `record_fn` (a no-op when no input requires grad). Backward closures also
/// use kind-routed kernels, so gradients stay on the input's device.
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary_k("add", self, other, BinaryKind::Add)?;
        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            vec![unbroadcast(g, &sa), unbroadcast(g, &sb)]
        }))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary_k("sub", self, other, BinaryKind::Sub)?;
        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let neg = raw_unary_k(g, UnaryKind::Neg).expect(REGISTERED);
            vec![unbroadcast(g, &sa), unbroadcast(&neg, &sb)]
        }))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary_k("mul", self, other, BinaryKind::Mul)?;
        let (a, b) = (self.clone(), other.clone());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let ga = raw_binary_k("mul_bw", g, &b, BinaryKind::Mul).unwrap();
            let gb = raw_binary_k("mul_bw", g, &a, BinaryKind::Mul).unwrap();
            vec![unbroadcast(&ga, a.shape()), unbroadcast(&gb, b.shape())]
        }))
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary_k("div", self, other, BinaryKind::Div)?;
        let (a, b) = (self.clone(), other.clone());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            // dL/da = g / b; dL/db = -(g * a) / b^2
            let ga = raw_binary_k("div_bw", g, &b, BinaryKind::Div).unwrap();
            let ga_num = raw_binary_k("div_bw", g, &a, BinaryKind::Mul).unwrap();
            let b2 = raw_binary_k("div_bw", &b, &b, BinaryKind::Mul).unwrap();
            let gb_pos = raw_binary_k("div_bw", &ga_num, &b2, BinaryKind::Div).unwrap();
            let gb = raw_unary_k(&gb_pos, UnaryKind::Neg).unwrap();
            vec![unbroadcast(&ga, a.shape()), unbroadcast(&gb, b.shape())]
        }))
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_matmul(self, other)?;
        let (a, b) = (self.clone(), other.clone());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            // C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC
            let da = raw_matmul_t(g, &b, false, true).unwrap();
            let db = raw_matmul_t(&a, g, true, false).unwrap();
            vec![da, db]
        }))
    }

    pub fn neg(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Neg).expect(REGISTERED);
        out.record_fn(vec![self.clone()], |g| {
            vec![raw_unary_k(g, UnaryKind::Neg).unwrap()]
        })
    }

    pub fn relu(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Relu).expect(REGISTERED);
        let a = self.clone();
        out.record_fn(vec![self.clone()], move |g| {
            let mask = raw_unary_k(&a, UnaryKind::Gtz).unwrap();
            vec![raw_binary_k("relu_bw", g, &mask, BinaryKind::Mul).unwrap()]
        })
    }

    pub fn exp(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Exp).expect(REGISTERED);
        if !self.requires_grad() {
            return out;
        }
        let y = out.detach_copy();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary_k("exp_bw", g, &y, BinaryKind::Mul).unwrap()]
        })
    }

    pub fn sigmoid(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Sigmoid).expect(REGISTERED);
        if !self.requires_grad() {
            return out;
        }
        let y = out.detach_copy();
        out.record_fn(vec![self.clone()], move |g| {
            // g * y * (1 - y) = g*y - (g*y)*y
            let gy = raw_binary_k("sigmoid_bw", g, &y, BinaryKind::Mul).unwrap();
            let gyy = raw_binary_k("sigmoid_bw", &gy, &y, BinaryKind::Mul).unwrap();
            vec![raw_binary_k("sigmoid_bw", &gy, &gyy, BinaryKind::Sub).unwrap()]
        })
    }

    pub fn sum(&self) -> Tensor {
        let out = raw_reduce_dev(self, ReduceKind::Sum)
            .unwrap_or_else(|| Tensor::scalar(pairwise_sum(&self.to_vec())));
        let in_shape = self.shape().to_vec();
        let device = self.device();
        out.record_fn(vec![self.clone()], move |g| {
            vec![Tensor::full_on(&in_shape, g.item(), device).unwrap()]
        })
    }

    pub fn mean(&self) -> Tensor {
        // No max(1) guard: torch's empty mean is NaN (0/0), not a silent 0.
        let n = numel(self.shape()) as f32;
        let out = raw_reduce_dev(self, ReduceKind::Mean)
            .unwrap_or_else(|| Tensor::scalar(pairwise_sum(&self.to_vec()) / n));
        let in_shape = self.shape().to_vec();
        let device = self.device();
        out.record_fn(vec![self.clone()], move |g| {
            vec![Tensor::full_on(&in_shape, g.item() / n, device).unwrap()]
        })
    }
}
