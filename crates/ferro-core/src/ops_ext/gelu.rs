//! GELU activation, both variants. `gelu` is the tanh approximation
//! (torch's `gelu(approximate="tanh")`):
//! y = 0.5 * x * (1 + tanh(c * (x + a*x^3))) with c = sqrt(2/pi), a = 0.044715.
//! Backward differentiates the same closed form: with u = c*(x + a*x^3) and
//! t = tanh(u), dy/dx = 0.5*(1 + t) + 0.5*x*(1 - t^2)*c*(1 + 3a*x^2).
//! `gelu_erf` is the exact form (torch's default): y = x * Phi(x) with Phi
//! the standard normal CDF, dy/dx = Phi(x) + x*phi(x) - the derivative is
//! its own named kernel (`GeluErfGrad`), mirroring relu's Gtz mask, so the
//! backward stays a single device launch plus a multiply.
//!
//! Forwards route through `UnaryKind::{Gelu, GeluErf}`, so device-resident
//! tensors take the backend's unary_dev kernel; backwards are composed from
//! kernels on g's device, keeping CUDA backward resident too.

use crate::dispatch::{BinaryKind, UnaryKind};
use crate::tensor::{raw_binary_k, raw_unary_k, Tensor};

const C: f32 = 0.797_884_6; // sqrt(2/pi)
const A: f32 = 0.044715;

impl Tensor {
    pub fn gelu(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Gelu)
            .expect("tensor's device backend is always registered");
        if !self.requires_grad() {
            return out;
        }
        let x = self.detach_copy();
        out.record_fn(vec![self.clone()], move |g| {
            let dev = g.device();
            let c = |v: f32| Tensor::scalar(v).to_device(dev).unwrap();
            let one = c(1.0);
            let x3 = x.powf(3.0);
            let u = x.add(&x3.mul(&c(A)).unwrap()).unwrap().mul(&c(C)).unwrap();
            let t = u.tanh();
            // p1 = 0.5*(1 + t); p2 = 0.5*x*(1 - t^2)*c*(1 + 3a*x^2)
            let p1 = t.add(&one).unwrap().mul(&c(0.5)).unwrap();
            let dt = one.sub(&t.mul(&t).unwrap()).unwrap();
            let p2 = x
                .mul(&x)
                .unwrap()
                .mul(&c(3.0 * A))
                .unwrap()
                .add(&one)
                .unwrap();
            let p2 = x
                .mul(&dt)
                .unwrap()
                .mul(&p2)
                .unwrap()
                .mul(&c(0.5 * C))
                .unwrap();
            vec![g.mul(&p1.add(&p2).unwrap()).unwrap()]
        })
    }

    pub fn gelu_erf(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::GeluErf)
            .expect("tensor's device backend is always registered");
        if !self.requires_grad() {
            return out;
        }
        let x = self.detach_copy();
        out.record_fn(vec![self.clone()], move |g| {
            let mask = raw_unary_k(&x, UnaryKind::GeluErfGrad).unwrap();
            vec![raw_binary_k("gelu_erf_bw", g, &mask, BinaryKind::Mul).unwrap()]
        })
    }
}
