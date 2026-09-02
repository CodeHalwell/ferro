//! `deg2rad`/`rad2deg`: exact inverses, y = x * pi/180 and y = x * 180/pi.
//! Both derivatives are the same constant scale factor, so dx = g * (that
//! constant). No device kernel exists for either op, so both forward and
//! backward go through the host `raw_binary` path.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

const DEG2RAD: f32 = std::f32::consts::PI / 180.0;
const RAD2DEG: f32 = 180.0 / std::f32::consts::PI;

impl Tensor {
    pub fn deg2rad(&self) -> Result<Tensor> {
        let out = raw_binary("deg2rad", self, self, |v, _| v * DEG2RAD)?;
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("deg2rad_bw", g, g, |gg, _| gg * DEG2RAD).unwrap()]
        }))
    }

    pub fn rad2deg(&self) -> Result<Tensor> {
        let out = raw_binary("rad2deg", self, self, |v, _| v * RAD2DEG)?;
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("rad2deg_bw", g, g, |gg, _| gg * RAD2DEG).unwrap()]
        }))
    }
}
