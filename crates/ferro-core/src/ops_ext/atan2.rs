//! atan2: y = atan2(a, b), matching torch.atan2(input, other). Backward:
//! dy/da = b / (a^2 + b^2), dy/db = -a / (a^2 + b^2).

use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn atan2(&self, other: &Tensor) -> crate::Result<Tensor> {
        let out = raw_binary("atan2", self, other, |a, b| a.atan2(b))?.to_device(self.device())?;
        let (x, y) = (self.detach_copy(), other.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let pa = raw_binary("atan2_pa", &x, &y, |a, b| b / (a * a + b * b)).unwrap();
            let pb = raw_binary("atan2_pb", &x, &y, |a, b| -a / (a * a + b * b)).unwrap();
            let ga = raw_binary("atan2_bwa", g, &pa, |gg, p| gg * p).unwrap();
            let gb = raw_binary("atan2_bwb", g, &pb, |gg, p| gg * p).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
