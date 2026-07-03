use crate::dispatch::{BinaryKind, UnaryKind};
use crate::error::Result;
use crate::shape::numel;
use crate::tensor::{raw_binary, raw_binary_k, raw_matmul, raw_unary, raw_unary_k, unbroadcast, Tensor};

/// Backends are looked up by device; Cpu is pre-registered and tensors are
/// Cpu-only today, so infallible forwards unwrap with this message.
const CPU_REGISTERED: &str = "cpu backend is always registered";

/// Forward ops: compute the value with a detached raw kernel (named kind,
/// routed through the device's backend), then attach the backward closure via
/// `record_fn` (a no-op when no input requires grad). Backward closures stay
/// on the host-closure kernels.
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
            let neg = raw_unary(g, |x| -x);
            vec![unbroadcast(g, &sa), unbroadcast(&neg, &sb)]
        }))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary_k("mul", self, other, BinaryKind::Mul)?;
        let (a, b) = (self.clone(), other.clone());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let ga = raw_binary("mul_bw", g, &b, |x, y| x * y).unwrap();
            let gb = raw_binary("mul_bw", g, &a, |x, y| x * y).unwrap();
            vec![unbroadcast(&ga, a.shape()), unbroadcast(&gb, b.shape())]
        }))
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary_k("div", self, other, BinaryKind::Div)?;
        let (a, b) = (self.clone(), other.clone());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let ga = raw_binary("div_bw", g, &b, |x, y| x / y).unwrap();
            let ga_num = raw_binary("div_bw", g, &a, |x, y| x * y).unwrap();
            let gb = raw_binary("div_bw", &ga_num, &b, |x, y| -x / (y * y)).unwrap();
            vec![unbroadcast(&ga, a.shape()), unbroadcast(&gb, b.shape())]
        }))
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_matmul(self, other)?;
        let (a, b) = (self.clone(), other.clone());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            // C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC
            let da = raw_matmul(g, &b.transpose_view(0, 1).unwrap()).unwrap();
            let db = raw_matmul(&a.transpose_view(0, 1).unwrap(), g).unwrap();
            vec![da, db]
        }))
    }

    pub fn neg(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Neg).expect(CPU_REGISTERED);
        out.record_fn(vec![self.clone()], |g| vec![raw_unary(g, |x| -x)])
    }

    pub fn relu(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Relu).expect(CPU_REGISTERED);
        let a = self.clone();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("relu_bw", g, &a, |gg, aa| if aa > 0.0 { gg } else { 0.0 }).unwrap()]
        })
    }

    pub fn exp(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Exp).expect(CPU_REGISTERED);
        if !self.requires_grad() {
            return out;
        }
        let y = out.detach_copy();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("exp_bw", g, &y, |x, yy| x * yy).unwrap()]
        })
    }

    pub fn sigmoid(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Sigmoid).expect(CPU_REGISTERED);
        if !self.requires_grad() {
            return out;
        }
        let y = out.detach_copy();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("sigmoid_bw", g, &y, |x, yy| x * yy * (1.0 - yy)).unwrap()]
        })
    }

    pub fn sum(&self) -> Tensor {
        let total: f32 = self.to_vec().iter().sum();
        let out = Tensor::scalar(total);
        let in_shape = self.shape().to_vec();
        out.record_fn(vec![self.clone()], move |g| vec![Tensor::full(&in_shape, g.item())])
    }

    pub fn mean(&self) -> Tensor {
        let v = self.to_vec();
        let n = numel(self.shape()).max(1) as f32;
        let out = Tensor::scalar(v.iter().sum::<f32>() / n);
        let in_shape = self.shape().to_vec();
        out.record_fn(vec![self.clone()], move |g| vec![Tensor::full(&in_shape, g.item() / n)])
    }
}
