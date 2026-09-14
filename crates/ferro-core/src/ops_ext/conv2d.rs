//! Grouped NCHW convolution via reusable windows and CPU GEMM. No bias.
use super::window::{checked_len, invalid, require_f32, Window2d};
use crate::device::Device;
use crate::dispatch::backend_for;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

fn transpose(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

impl Tensor {
    /// Legacy isotropic convolution: dilation=1, groups=1.
    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Result<Tensor> {
        self.conv2d_with_options(weight, [stride; 2], [padding; 2], [1; 2], 1)
    }

    /// Weight is [Cout, Cin/groups, KH, KW]. Padding is symmetric per axis.
    /// Supports depthwise multipliers (groups=Cin, Cout a multiple of Cin).
    /// Host fallback: output is CPU, not a device-resident convolution kernel.
    pub fn conv2d_with_options(
        &self,
        weight: &Tensor,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Tensor> {
        require_f32(self, "conv2d")?;
        require_f32(weight, "conv2d")?;
        if self.device() != weight.device() {
            return Err(Error::DeviceMismatch {
                op: "conv2d",
                lhs: self.device(),
                rhs: weight.device(),
            });
        }
        if self.ndim() != 4 || weight.ndim() != 4 {
            return Err(invalid("conv2d", "expected NCHW input and OIHW weight"));
        }
        let xs = self.shape().to_vec();
        let ws = weight.shape().to_vec();
        let (n, cin, cout) = (xs[0], xs[1], ws[0]);
        if groups == 0 || cin == 0 || cout == 0 || cin % groups != 0 || cout % groups != 0 {
            return Err(invalid(
                "conv2d",
                "positive groups must divide positive input/output channels",
            ));
        }
        let (ic, oc) = (cin / groups, cout / groups);
        if ws[1] != ic {
            return Err(Error::ShapeMismatch {
                op: "conv2d",
                lhs: xs,
                rhs: ws,
            });
        }
        let win = Window2d::new([xs[2], xs[3]], [ws[2], ws[3]], stride, padding, dilation)?;
        let [oh, ow] = win.output_size();
        let pos = win.positions();
        let taps = checked_len("conv2d", &[ic, win.kernel_area()])?;
        let image = win.image_len(ic)?;
        let col_len = win.column_len(ic)?;
        let weight_group = checked_len("conv2d", &[oc, taps])?;
        let output_group = checked_len("conv2d", &[oc, pos])?;
        let out_len = checked_len("conv2d", &[n, groups, output_group])?;
        checked_len("conv2d", &[groups, output_group])?;
        checked_len("conv2d", &[n, groups, image])?;
        checked_len("conv2d", &[groups, weight_group])?;
        let (x, wt) = (self.to_vec(), weight.to_vec());
        let backend = backend_for(Device::Cpu)?;
        let mut out = vec![0.; out_len];
        let mut col = vec![0.; if n == 0 { 0 } else { col_len }];
        for ni in 0..n {
            for group in 0..groups {
                let io = (ni * groups + group) * image;
                let wo = group * weight_group;
                let yo = (ni * groups + group) * output_group;
                win.unfold_into(&x[io..io + image], ic, &mut col);
                let y = backend.matmul(&wt[wo..wo + weight_group], &col, oc, taps, pos);
                out[yo..yo + output_group].copy_from_slice(&y);
            }
        }
        Ok(Tensor::from_vec(out, &[n, cout, oh, ow])?.record_fn(
            vec![self.clone(), weight.clone()],
            move |g| {
                let gd = g.to_vec();
                let backend = backend_for(Device::Cpu).expect("cpu backend is always registered");
                let mut dx = vec![0.; x.len()];
                let mut dw = vec![0.; wt.len()];
                if n != 0 {
                    let mut col = vec![0.; col_len];
                    let mut ct = vec![0.; col_len];
                    let mut wtt = vec![0.; weight_group];
                    for group in 0..groups {
                        let wo = group * weight_group;
                        transpose(&wt[wo..wo + weight_group], &mut wtt, oc, taps);
                        for ni in 0..n {
                            let io = (ni * groups + group) * image;
                            let yo = (ni * groups + group) * output_group;
                            let gi = &gd[yo..yo + output_group];
                            win.unfold_into(&x[io..io + image], ic, &mut col);
                            transpose(&col, &mut ct, taps, pos);
                            let dwi = backend.matmul(gi, &ct, oc, pos, taps);
                            for (acc, v) in dw[wo..wo + weight_group].iter_mut().zip(dwi) {
                                *acc += v;
                            }
                            let dc = backend.matmul(&wtt, gi, taps, oc, pos);
                            win.fold_add(&dc, ic, &mut dx[io..io + image]);
                        }
                    }
                }
                vec![
                    Tensor::from_vec(dx, &xs).unwrap(),
                    Tensor::from_vec(dw, &ws).unwrap(),
                ]
            },
        ))
    }
}
