//! bce_loss: mean(-(t*ln(p) + (1-t)*ln(1-p))), p = self in (0,1), t = target.
//! Composed from existing ops, so autograd flows through the composition and
//! no backward closure is needed. The constant-1 tensors are built on each
//! operand's own device, not `Tensor::ones` (always cpu), so this stays
//! composable with device-resident inputs.
use crate::error::Result;
use crate::tensor::Tensor;

impl Tensor {
    pub fn bce_loss(&self, target: &Tensor) -> Result<Tensor> {
        let one_minus_p = Tensor::full_on(self.shape(), 1.0, self.device())?.sub(self)?;
        let one_minus_t = Tensor::full_on(target.shape(), 1.0, target.device())?.sub(target)?;
        let t_log_p = target.mul(&self.log())?;
        let nt_log_1mp = one_minus_t.mul(&one_minus_p.log())?;
        Ok(t_log_p.add(&nt_log_1mp)?.neg().mean())
    }
}
