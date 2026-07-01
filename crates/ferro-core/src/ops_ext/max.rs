//! Global `max` reduction to a scalar. Forward returns the maximum over all
//! elements; backward routes the incoming scalar grad to the first argmax
//! position (ties break toward the lowest flat index) and 0 elsewhere.

use crate::tensor::Tensor;

impl Tensor {
    pub fn max(&self) -> Tensor {
        let xv = self.to_vec();
        let m = xv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out = Tensor::scalar(m);
        let shape = self.shape().to_vec();
        out.record_fn(vec![self.clone()], move |g| {
            let mut grad = vec![0.0; xv.len()];
            grad[xv.iter().position(|&e| e == m).unwrap()] = g.item();
            vec![Tensor::from_vec(grad, &shape).unwrap()]
        })
    }
}
