use crate::dispatch::{BinaryKind, OpTag, ReduceKind, UnaryKind};
use crate::error::Result;
use crate::shape::numel;
use crate::tensor::{
    raw_binary_k, raw_matmul, raw_matmul_t, raw_reduce_dev, raw_unary_k, unbroadcast, Tensor,
};

/// Backends are looked up by device; Cpu is pre-registered and a device tensor
/// implies its backend was registered, so infallible forwards unwrap with this.
const REGISTERED: &str = "tensor's device backend is always registered";

/// Forward ops: compute the value with a detached raw kernel (named kind,
/// routed through the device's backend), then attach the backward closure via
/// `record_fn_tagged` (a no-op when no input requires grad). The tag names
/// which kernel produced the output so fusion planning can re-derive it.
/// Backward closures also use kind-routed kernels, so gradients stay on the
/// input's device.
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if let Some(out) = crate::typed_math::promoted_binary("add", self, other, BinaryKind::Add)? { return Ok(out); }
        let out = raw_binary_k("add", self, other, BinaryKind::Add)?;
        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn_tagged(
            vec![self.clone(), other.clone()],
            OpTag::Binary(BinaryKind::Add),
            move |g| vec![unbroadcast(g, &sa), unbroadcast(g, &sb)],
        ))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        if let Some(out) = crate::typed_math::promoted_binary("sub", self, other, BinaryKind::Sub)? { return Ok(out); }
        let out = raw_binary_k("sub", self, other, BinaryKind::Sub)?;
        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn_tagged(
            vec![self.clone(), other.clone()],
            OpTag::Binary(BinaryKind::Sub),
            move |g| {
                let neg = raw_unary_k(g, UnaryKind::Neg).expect(REGISTERED);
                vec![unbroadcast(g, &sa), unbroadcast(&neg, &sb)]
            },
        ))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if let Some(out) = crate::typed_math::promoted_binary("mul", self, other, BinaryKind::Mul)? { return Ok(out); }
        let out = raw_binary_k("mul", self, other, BinaryKind::Mul)?;
        let (a, b) = (self.clone(), other.clone());
        Ok(out.record_fn_tagged(
            vec![self.clone(), other.clone()],
            OpTag::Binary(BinaryKind::Mul),
            move |g| {
                let ga = raw_binary_k("mul_bw", g, &b, BinaryKind::Mul).unwrap();
                let gb = raw_binary_k("mul_bw", g, &a, BinaryKind::Mul).unwrap();
                vec![unbroadcast(&ga, a.shape()), unbroadcast(&gb, b.shape())]
            },
        ))
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        if let Some(out) = crate::typed_math::promoted_binary("div", self, other, BinaryKind::Div)? { return Ok(out); }
        let out = raw_binary_k("div", self, other, BinaryKind::Div)?;
        let (a, b) = (self.clone(), other.clone());
        Ok(out.record_fn_tagged(
            vec![self.clone(), other.clone()],
            OpTag::Binary(BinaryKind::Div),
            move |g| {
                // Preserve negation order for structural-zero cotangents.
                let ga = raw_binary_k("div_bw", g, &b, BinaryKind::Div).unwrap();
                let ratio = raw_binary_k("div_bw", &a, &b, BinaryKind::Div).unwrap();
                let slope = raw_binary_k("div_bw", &ratio, &b, BinaryKind::Div).unwrap();
                let negative = raw_unary_k(g, UnaryKind::Neg).unwrap();
                let gb = raw_binary_k("div_bw", &negative, &slope, BinaryKind::Mul).unwrap();
                vec![unbroadcast(&ga, a.shape()), unbroadcast(&gb, b.shape())]
            },
        ))
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 && (self.ndim() != 2 || other.ndim() != 2) {
            return crate::typed_math::matmul_general(self, other);
        }
        let out = raw_matmul(self, other)?;
        let (a, b) = (self.clone(), other.clone());
        Ok(out.record_fn_forward(vec![self.clone(), other.clone()], crate::autograd::ForwardOp::MatMul, move |g| {
            // C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC
            let da = raw_matmul_t(g, &b, false, true).unwrap();
            let db = raw_matmul_t(&a, g, true, false).unwrap();
            vec![da, db]
        }))
    }

    pub fn neg(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Neg).expect(REGISTERED);
        out.record_fn_tagged(vec![self.clone()], OpTag::Unary(UnaryKind::Neg), |g| {
            vec![raw_unary_k(g, UnaryKind::Neg).unwrap()]
        })
    }

    pub fn relu(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Relu).expect(REGISTERED);
        let a = self.clone();
        out.record_fn_tagged(
            vec![self.clone()],
            OpTag::Unary(UnaryKind::Relu),
            move |g| {
                vec![raw_binary_k("relu_bw", g, &a, BinaryKind::ReluBackward).unwrap()]
            },
        ).save_output_version()
    }

    pub fn exp(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Exp).expect(REGISTERED);
        if !self.requires_grad() && !crate::forward_ad::is_active() {
            return if crate::capture::is_recording() {
                out.record_fn_tagged(vec![self.clone()], crate::OpTag::Unary(UnaryKind::Exp), |_| unreachable!("inference node has no backward"))
            } else { out };
        }
        let y = out.detach_copy_layout().expect("validated exponential output snapshot");
        out.record_fn_tagged(
            vec![self.clone()],
            OpTag::Unary(UnaryKind::Exp),
            move |g| {
                vec![raw_binary_k("exp_bw", g, &y, BinaryKind::Mul).unwrap()]
            },
        ).save_output_version()
    }

    pub fn sigmoid(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Sigmoid).expect(REGISTERED);
        if !self.requires_grad() && !crate::forward_ad::is_active() {
            return if crate::capture::is_recording() {
                out.record_fn_tagged(vec![self.clone()], crate::OpTag::Unary(UnaryKind::Sigmoid), |_| unreachable!("inference node has no backward"))
            } else { out };
        }
        let y = out.detach_copy_layout().expect("validated sigmoid output snapshot");
        out.record_fn_tagged(
            vec![self.clone()],
            OpTag::Unary(UnaryKind::Sigmoid),
            move |g| {
                vec![crate::ops_ext::activation_backward::backward(crate::ops_ext::activation_backward::Kind::Sigmoid, g, &y).unwrap()]
            },
        ).save_output_version()
    }

    pub fn sum(&self) -> Tensor {
        if matches!(self.dtype(), crate::DType::F64 | crate::DType::I64) {
            let out = crate::typed_math::sum(self, None, false).expect("typed sum");
            let shape = self.shape().to_vec();
            return out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Sum, move |g| vec![g.broadcast_to(&shape).unwrap().detach_copy()]);
        }
        if self.dtype() == crate::DType::Bool { return self.bool_sum(None, false).expect("boolean sum requires eager CPU execution"); }
        let out = raw_reduce_dev(self, ReduceKind::Sum)
            .unwrap_or_else(|| Tensor::scalar(self.raw_host_sum()));
        let in_shape = self.shape().to_vec();
        out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Sum, move |g| {
            // d(sum)/dx = g: build a whole device buffer (fill+mul) rather
            // than broadcasting a stride-0 view - downstream backwards only
            // take the device fastpath on whole contiguous buffers, and a
            // view would push them onto the host path (per-step uploads).
            vec![g
                .mul(&Tensor::full_on(&in_shape, 1.0, g.device()).unwrap())
                .unwrap()]
        })
    }

    pub fn mean(&self) -> Tensor {
        if self.dtype() == crate::DType::F64 {
            return self.sum().div(&crate::typed_math::full_f64(&[], self.numel() as f64).unwrap()).unwrap();
        }
        crate::tensor::check_f32("mean", self).expect("mean requires floating-point storage");
        assert_ne!(self.dtype(), crate::DType::Bool, "mean requires floating-point storage");
        // No max(1) guard: torch's empty mean is NaN (0/0), not a silent 0.
        let n = numel(self.shape()) as f32;
        let out = raw_reduce_dev(self, ReduceKind::Mean)
            .unwrap_or_else(|| Tensor::scalar(self.raw_host_sum() / n));
        let in_shape = self.shape().to_vec();
        out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Mean, move |g| {
            // d(mean)/dx = g/n: same whole-buffer construction as sum's
            // backward; no g.item() host sync, no broadcast views.
            vec![g
                .mul(&Tensor::full_on(&in_shape, 1.0 / n, g.device()).unwrap())
                .unwrap()]
        })
    }
}

// The derivative is a recorded operation too: its input dependency matters for
// higher derivatives even though the derivative with respect to x is zero.
pub(crate) fn relu_backward(gradient: &Tensor, input: &Tensor) -> Result<Tensor> {
    let output = raw_binary_k("relu_backward", gradient, input, BinaryKind::ReluBackward)?;
    let saved = input.detach_copy();
    let shape = input.shape().to_vec();
    let dtype = input.dtype();
    let device = input.device();
    Ok(output.record_fn_tagged(vec![gradient.clone(), input.clone()], OpTag::Binary(BinaryKind::ReluBackward), move |g| {
        let zero = if dtype == crate::DType::F64 { crate::typed_math::full_f64(&shape, 0.0).unwrap() }
            else { Tensor::full_on(&shape, 0.0, device).unwrap() };
        vec![raw_binary_k("relu_backward_bw", g, &saved, BinaryKind::ReluBackward).unwrap(), zero]
    }))
}
