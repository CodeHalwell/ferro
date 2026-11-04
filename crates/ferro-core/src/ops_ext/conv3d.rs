//! Grouped NCDHW convolution via depth-stacked 2D windows and CPU GEMM. No bias.
//!
//! The column matrix of one image is [Cin/g*KD*KH*KW, OD*OH*OW]: for each
//! (channel, depth tap, output depth) the matching input depth plane is
//! unfolded by `Window2d` into the column block of that output depth, or
//! zeroed when the tap falls in depth padding. Backward folds through the
//! same mapping, so unfold and fold remain an adjoint pair.
use super::window::{checked_len, invalid, require_f32, Window2d};
use crate::device::Device;
use crate::dispatch::backend_for;
use crate::error::{Error, Result};
use crate::pool;
use crate::tensor::Tensor;

#[derive(Clone, Copy)]
struct Depth { input: usize, kernel: usize, stride: usize, padding: usize, dilation: usize, output: usize }

impl Depth {
    fn source(&self, kd: usize, od: usize) -> Option<usize> {
        (od * self.stride + kd * self.dilation).checked_sub(self.padding).filter(|&d| d < self.input)
    }
}

#[derive(Clone, Copy)]
struct Geometry { win: Window2d, depth: Depth, channels: usize }

impl Geometry {
    fn plane(&self) -> usize { self.win.image_len(1).expect("validated plane") }
    fn pos(&self) -> usize { self.win.positions() }

    /// `image` holds this group's channels; `col` is [taps, (od1-od0)*OH*OW]
    /// for output depths od0..od1.
    fn unfold(&self, image: &[f32], depths: std::ops::Range<usize>, col: &mut [f32]) {
        let (plane, pos, area) = (self.plane(), self.pos(), self.win.kernel_area());
        let ld = depths.len() * pos;
        for ci in 0..self.channels {
            for kd in 0..self.depth.kernel {
                let block = (ci * self.depth.kernel + kd) * area * ld;
                for (j, od) in depths.clone().enumerate() {
                    let at = &mut col[block + j * pos..];
                    match self.depth.source(kd, od) {
                        Some(d) => self.win.unfold_channel(&image[(ci * self.depth.input + d) * plane..][..plane], 0, at, ld),
                        None => (0..area).for_each(|t| at[t * ld..][..pos].fill(0.0)),
                    }
                }
            }
        }
    }

    fn fold_add(&self, col: &[f32], depths: std::ops::Range<usize>, image: &mut [f32]) {
        let (plane, pos, area) = (self.plane(), self.pos(), self.win.kernel_area());
        let ld = depths.len() * pos;
        for ci in 0..self.channels {
            for kd in 0..self.depth.kernel {
                let block = (ci * self.depth.kernel + kd) * area * ld;
                for (j, od) in depths.clone().enumerate() {
                    if let Some(d) = self.depth.source(kd, od) {
                        self.win.fold_channel_add(&col[block + j * pos..], ld, 0, &mut image[(ci * self.depth.input + d) * plane..][..plane]);
                    }
                }
            }
        }
    }
}

/// Column-matrix floats per output-depth chunk (16 MiB), as in conv2d.
const CHUNK: usize = 1 << 22;

fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dst = vec![0.; src.len()];
    for r in 0..rows { for c in 0..cols { dst[c * rows + r] = src[r * cols + c]; } }
    dst
}

impl Tensor {
    /// Weight is [Cout, Cin/groups, KD, KH, KW]; padding is symmetric per axis.
    /// Host fallback: output is CPU, not a device-resident convolution kernel.
    pub fn conv3d_with_options(&self, weight: &Tensor, stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], groups: usize) -> Result<Tensor> {
        require_f32(self, "conv3d")?;
        require_f32(weight, "conv3d")?;
        if self.device() != weight.device() { return Err(Error::DeviceMismatch { op: "conv3d", lhs: self.device(), rhs: weight.device() }); }
        if self.ndim() != 5 || weight.ndim() != 5 { return Err(invalid("conv3d", "expected NCDHW input and OIDHW weight")); }
        let (xs, ws) = (self.shape().to_vec(), weight.shape().to_vec());
        let (n, cin, cout) = (xs[0], xs[1], ws[0]);
        if groups == 0 || cin == 0 || cout == 0 || cin % groups != 0 || cout % groups != 0 {
            return Err(invalid("conv3d", "positive groups must divide positive input/output channels"));
        }
        let (ic, oc) = (cin / groups, cout / groups);
        if ws[1] != ic { return Err(Error::ShapeMismatch { op: "conv3d", lhs: xs, rhs: ws }); }
        if xs[2] == 0 || ws[2] == 0 || stride[0] == 0 || dilation[0] == 0 {
            return Err(invalid("conv3d", "input, kernel, stride and dilation must be positive"));
        }
        let padded = padding[0].checked_mul(2).and_then(|p| p.checked_add(xs[2])).ok_or_else(|| invalid("conv3d", "padding overflow"))?;
        let effective = (ws[2] - 1).checked_mul(dilation[0]).and_then(|k| k.checked_add(1)).ok_or_else(|| invalid("conv3d", "effective kernel overflow"))?;
        if effective > padded { return Err(invalid("conv3d", "effective kernel exceeds padded input")); }
        let depth = Depth { input: xs[2], kernel: ws[2], stride: stride[0], padding: padding[0], dilation: dilation[0], output: (padded - effective) / stride[0] + 1 };
        let win = Window2d::new([xs[3], xs[4]], [ws[3], ws[4]], [stride[1], stride[2]], [padding[1], padding[2]], [dilation[1], dilation[2]])?;
        let geo = Geometry { win, depth, channels: ic };
        let [oh, ow] = win.output_size();
        let (ld, taps) = (checked_len("conv3d", &[depth.output, win.positions()])?, checked_len("conv3d", &[ic, depth.kernel, win.kernel_area()])?);
        let image = checked_len("conv3d", &[ic, depth.input, xs[3], xs[4]])?;
        let (weight_group, output_group) = (checked_len("conv3d", &[oc, taps])?, checked_len("conv3d", &[oc, ld])?);
        let out_len = checked_len("conv3d", &[n, groups, output_group])?;
        checked_len("conv3d", &[n, groups, image])?;
        let (x, wt) = (self.to_vec(), weight.to_vec());
        let backend = backend_for(Device::Cpu)?;
        let mut out = vec![0.; out_len];
        let pos = win.positions();
        let step = (CHUNK / (taps * pos).max(1)).clamp(1, depth.output);
        let chunks: Vec<std::ops::Range<usize>> = (0..depth.output).step_by(step).map(|o| o..(o + step).min(depth.output)).collect();
        let mut col = vec![0.; taps * step * pos];
        for i in 0..n {
            for g in 0..groups {
                for depths in &chunks {
                    let len = depths.len() * pos;
                    geo.unfold(&x[(i * groups + g) * image..][..image], depths.clone(), &mut col[..taps * len]);
                    let y = backend.matmul(&wt[g * weight_group..][..weight_group], &col[..taps * len], oc, taps, len);
                    let base = (i * groups + g) * output_group + depths.start * pos;
                    for o in 0..oc { out[base + o * ld..][..len].copy_from_slice(&y[o * len..][..len]); }
                    pool::give(y);
                }
            }
        }
        Ok(Tensor::from_vec(out, &[n, cout, depth.output, oh, ow])?.record_fn(vec![self.clone(), weight.clone()], move |grad| {
            let gd = grad.to_vec();
            let backend = backend_for(Device::Cpu).expect("cpu backend is always registered");
            let (mut dx, mut dw) = (vec![0.; x.len()], vec![0.; wt.len()]);
            let mut col = vec![0.; taps * step * pos];
            let mut gy = vec![0.; oc * step * pos];
            let wtt: Vec<Vec<f32>> = (0..groups).map(|g| transpose(&wt[g * weight_group..][..weight_group], oc, taps)).collect();
            for i in 0..n {
                for g in 0..groups {
                    for depths in &chunks {
                        let len = depths.len() * pos;
                        let base = (i * groups + g) * output_group + depths.start * pos;
                        for o in 0..oc { gy[o * len..][..len].copy_from_slice(&gd[base + o * ld..][..len]); }
                        let gy = &gy[..oc * len];
                        geo.unfold(&x[(i * groups + g) * image..][..image], depths.clone(), &mut col[..taps * len]);
                        // dW^T = col @ gy^T, accumulated over images and chunks.
                        let dwt = backend.matmul(&col[..taps * len], &transpose(gy, oc, len), taps, len, oc);
                        let dwg = &mut dw[g * weight_group..][..weight_group];
                        for o in 0..oc { for t in 0..taps { dwg[o * taps + t] += dwt[t * oc + o]; } }
                        pool::give(dwt);
                        let dc = backend.matmul(&wtt[g], gy, taps, oc, len);
                        geo.fold_add(&dc, depths.clone(), &mut dx[(i * groups + g) * image..][..image]);
                        pool::give(dc);
                    }
                }
            }
            vec![Tensor::from_vec(dx, &xs).unwrap(), Tensor::from_vec(dw, &ws).unwrap()]
        }))
    }
}
