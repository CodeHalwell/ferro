//! huber_loss: mean-reduced Huber loss. Per element, with d = input - target:
//! 0.5*d^2 if |d| <= delta, else delta*(|d| - 0.5*delta). Backward:
//! d/dd huber(d) = d if |d| <= delta, else delta*sign(d); the input gradient
//! is that times g/numel (the mean reduction), the target gradient is its
//! negation, each unbroadcast to its input's shape.

use crate::tensor::{raw_binary, unbroadcast, Tensor};
use crate::Result;

impl Tensor {
    pub fn huber_loss(&self, target: &Tensor, delta: f32) -> Result<Tensor> {
        let elementwise = raw_binary("huber_loss", self, target, move |a, b| {
            let d = a - b;
            let ad = d.abs();
            if ad <= delta {
                0.5 * d * d
            } else {
                delta * (ad - 0.5 * delta)
            }
        })?;
        let n = elementwise.numel() as f32;
        let out = Tensor::scalar(elementwise.to_vec().iter().sum::<f32>() / n);
        let (x, y) = (self.detach_copy(), target.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), target.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), target.clone()], move |g| {
            let dpart = raw_binary("huber_loss_dd", &x, &y, move |a, b| {
                let d = a - b;
                if d.abs() <= delta {
                    d
                } else {
                    delta * d.signum()
                }
            })
            .unwrap();
            let ga = raw_binary("huber_loss_bwa", g, &dpart, move |gg, p| gg * p / n).unwrap();
            let gb = raw_binary("huber_loss_bwb", g, &dpart, move |gg, p| -gg * p / n).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
