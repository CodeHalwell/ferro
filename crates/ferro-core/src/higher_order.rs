//! Graph-building VJPs on the existing record_fn Op graph, never a second tape.
use super::{ForwardOp, Op};
use crate::dispatch::{BinaryKind, OpTag, UnaryKind};
use crate::{Error, Result, Tensor};

fn reduce_to(g: &Tensor, shape: &[usize]) -> Result<Tensor> {
    let mut v = g.clone();
    while v.shape().len() > shape.len() { v = v.sum_dim(0, false)?; }
    for (d, &size) in shape.iter().enumerate() {
        if size == 1 && v.shape()[d] != 1 { v = v.sum_dim(d, true)?; }
    }
    v.reshape(shape)
}

impl Op {
    pub(super) fn graph_backward(&self, g: &Tensor) -> Result<Vec<Tensor>> {
        let a = &self.inputs[0];
        let one = || Tensor::full_on(a.shape(), 1.0, a.device());
        let grads = match self.tag {
            Some(OpTag::Binary(kind)) => {
                let b = &self.inputs[1];
                let (ga, gb) = match kind {
                    BinaryKind::Add => (g.clone(), g.clone()),
                    BinaryKind::Sub => (g.clone(), g.neg()),
                    BinaryKind::Mul => (g.mul(b)?, g.mul(a)?),
                    BinaryKind::Div => (g.div(b)?, g.mul(a)?.div(&b.mul(b)?)?.neg()),
                };
                vec![reduce_to(&ga, a.shape())?, reduce_to(&gb, b.shape())?]
            }
            Some(OpTag::Unary(UnaryKind::Neg)) => vec![g.neg()],
            Some(OpTag::Unary(UnaryKind::Exp)) => vec![g.mul(&a.exp())?],
            Some(OpTag::Unary(UnaryKind::Sqrt)) => {
                let half = Tensor::full_on(a.shape(), 0.5, a.device())?;
                vec![g.mul(&half)?.div(&a.sqrt())?]
            }
            Some(OpTag::Unary(UnaryKind::Tanh)) => {
                let y = a.tanh();
                vec![g.mul(&one()?.sub(&y.mul(&y)?)?)?]
            }
            Some(OpTag::Unary(UnaryKind::Sigmoid)) => {
                let y = a.sigmoid();
                vec![g.mul(&y)?.mul(&one()?.sub(&y)?)?]
            }
            _ => match &self.forward {
                Some(ForwardOp::Sum) => vec![g.mul(&one()?)?],
                Some(ForwardOp::Mean) => vec![g.mul(&Tensor::full_on(a.shape(), 1.0 / a.numel() as f32, a.device())?)?],
                Some(ForwardOp::MatMul) => vec![g.matmul(&self.inputs[1].transpose(0, 1)?)?, a.transpose(0, 1)?.matmul(g)?],
                Some(ForwardOp::Reshape(_)) => vec![g.reshape(a.shape())?],
                Some(ForwardOp::Transpose(d0, d1)) => vec![g.transpose(*d0, *d1)?],
                Some(ForwardOp::SumDim(dim, keep)) => {
                    let mut shape = a.shape().to_vec();
                    shape[*dim] = 1;
                    let expanded = if *keep { g.clone() } else { g.reshape(&shape)? };
                    vec![expanded.mul(&one()?)?]
                }
                _ => return Err(Error::Unsupported { op: "create_graph", msg: format!("no differentiable VJP registered for tag {:?}, forward {:?}", self.tag, self.forward) }),
            },
        };
        Ok(grads)
    }
}
