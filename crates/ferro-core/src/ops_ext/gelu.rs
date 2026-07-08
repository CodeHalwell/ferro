//! GELU activation, tanh approximation (torch's `gelu(approximate="tanh")`):
//! y = 0.5 * x * (1 + tanh(c * (x + a*x^3))) with c = sqrt(2/pi), a = 0.044715.
//! Backward differentiates the same closed form: with u = c*(x + a*x^3) and
//! t = tanh(u), dy/dx = 0.5*(1 + t) + 0.5*x*(1 - t^2)*c*(1 + 3a*x^2).

use crate::tensor::Tensor;

const C: f32 = 0.797_884_6; // sqrt(2/pi)
const A: f32 = 0.044715;

impl Tensor {
    pub fn gelu(&self) -> Tensor {
        let xv = self.to_vec();
        let shape = self.shape().to_vec();
        let y: Vec<f32> = xv.iter().map(|&x| 0.5 * x * (1.0 + (C * (x + A * x * x * x)).tanh())).collect();
        let out = Tensor::from_vec(y, &shape).unwrap();
        if !self.requires_grad() {
            return out;
        }
        out.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let dx: Vec<f32> = xv
                .iter()
                .zip(&gd)
                .map(|(&x, &gg)| {
                    let t = (C * (x + A * x * x * x)).tanh();
                    gg * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * C * (1.0 + 3.0 * A * x * x))
                })
                .collect();
            vec![Tensor::from_vec(dx, &shape).unwrap()]
        })
    }
}
