//! Grouped NCHW convolution via reusable windows and CPU GEMM. No bias.
//!
//! Images are processed in chunks whose column matrices sit side by side as
//! one [taps, chunk*positions] matrix, so each (group, chunk) is a single
//! large GEMM the backend can block and thread. Unfold runs across channels
//! and fold across images on scoped threads when the chunk is large. Copies
//! and per-pixel fold order do not depend on the thread count.
use super::window::{checked_len, invalid, require_f32, resident, whole_on, Window2d};
use crate::device::Device;
use crate::dispatch::{backend_for, Window2dDev};
use crate::error::{Error, Result};
use crate::pool;
use crate::tensor::{device_leaf, raw_matmul_t, Tensor};

/// Column-matrix floats per chunk (16 MiB).
const CHUNK: usize = 1 << 19;
/// Column-matrix floats below which unfold/fold stay on the calling thread.
const PAR_MIN: usize = 1 << 20;

fn threads(len: usize, units: usize) -> usize {
    if len < PAR_MIN {
        return 1;
    }
    std::thread::available_parallelism().map_or(1, |p| p.get()).min(units).max(1)
}

/// Run `f(first_index, items)` over `items` split into contiguous groups,
/// one per thread, the first on the calling thread.
fn par<T: Send>(items: &mut [T], threads: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let per = items.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        let mut groups = items.chunks_mut(per).enumerate();
        let first = groups.next();
        for (g, part) in groups {
            let f = &f;
            s.spawn(move || f(g * per, part));
        }
        if let Some((_, part)) = first {
            f(0, part);
        }
    });
}

/// Unfold `images` (each one group's channels) into `col`, image j's columns
/// at offset j*positions of every row, rows `ld` apart.
fn unfold_chunk(win: &Window2d, images: &[&[f32]], channels: usize, col: &mut [f32], ld: usize) {
    let rows = win.kernel_area() * ld;
    let pos = win.positions();
    let mut planes: Vec<&mut [f32]> = col.chunks_mut(rows).take(channels).collect();
    par(&mut planes, threads(channels * rows, channels), |c0, part| {
        for (k, plane) in part.iter_mut().enumerate() {
            for (j, img) in images.iter().enumerate() {
                win.unfold_channel(img, c0 + k, &mut plane[j * pos..], ld);
            }
        }
    });
}

impl Tensor {
    /// Legacy isotropic convolution: dilation=1, groups=1.
    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Result<Tensor> {
        self.conv2d_with_options(weight, [stride; 2], [padding; 2], [1; 2], 1)
    }

    /// Weight is [Cout, Cin/groups, KH, KW]. Padding is symmetric per axis.
    /// Supports depthwise multipliers (groups=Cin, Cout a multiple of Cin).
    /// Device tensors stay resident (groups = 1) on backends with the spatial
    /// seams; otherwise this is a host fallback returning a CPU tensor.
    pub fn conv2d_with_options(
        &self,
        weight: &Tensor,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Tensor> {
        if self.dtype() == crate::DType::F64 {
            return super::conv2d_f64::convolution(self, weight, stride, padding, dilation, groups);
        }
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
        if groups == 1 && n > 0 && self.device_resident_whole() && weight.device_resident_whole() {
            if let Some(out) = conv_resident(self, weight, win.device(n, cin), cout) {
                return out;
            }
        }
        let (x, wt) = (self.to_vec(), weight.to_vec());
        let backend = backend_for(Device::Cpu)?;
        let nb = if n == 0 { 0 } else { n.div_ceil(n.div_ceil((CHUNK / col_len).max(1))) };
        let mut out = vec![0.; out_len];
        let mut col = vec![0.; nb * col_len];
        for group in 0..groups {
            let wg = &wt[group * weight_group..][..weight_group];
            for i0 in (0..n).step_by(nb.max(1)) {
                let cnt = nb.min(n - i0);
                let ld = cnt * pos;
                let imgs: Vec<&[f32]> = (i0..i0 + cnt).map(|i| &x[(i * groups + group) * image..][..image]).collect();
                unfold_chunk(&win, &imgs, ic, &mut col[..taps * ld], ld);
                let y = backend.matmul(wg, &col[..taps * ld], oc, taps, ld);
                for (j, i) in (i0..i0 + cnt).enumerate() {
                    let yo = (i * groups + group) * output_group;
                    for o in 0..oc {
                        out[yo + o * pos..][..pos].copy_from_slice(&y[o * ld + j * pos..][..pos]);
                    }
                }
                pool::give(y);
            }
        }
        Ok(Tensor::from_vec(out, &[n, cout, oh, ow])?.record_fn(
            vec![self.clone(), weight.clone()],
            move |g| {
                let gd = g.to_vec();
                let backend = backend_for(Device::Cpu).expect("cpu backend is always registered");
                let mut dx = vec![0.; x.len()];
                let mut dw = vec![0.; wt.len()];
                let mut col = vec![0.; nb * col_len];
                let mut gyc = vec![0.; nb * output_group];
                let rows = win.kernel_area();
                for group in 0..groups {
                    let wo = group * weight_group;
                    for i0 in (0..n).step_by(nb.max(1)) {
                        let cnt = nb.min(n - i0);
                        let ld = cnt * pos;
                        let imgs: Vec<&[f32]> = (i0..i0 + cnt).map(|i| &x[(i * groups + group) * image..][..image]).collect();
                        unfold_chunk(&win, &imgs, ic, &mut col[..taps * ld], ld);
                        for (j, i) in (i0..i0 + cnt).enumerate() {
                            let yo = (i * groups + group) * output_group;
                            for o in 0..oc {
                                gyc[o * ld + j * pos..][..pos].copy_from_slice(&gd[yo + o * pos..][..pos]);
                            }
                        }
                        // dW = gy @ col^T: K runs over every position of the chunk.
                        let dwc = backend.matmul_t(&gyc[..oc * ld], &col[..taps * ld], oc, ld, taps, false, true);
                        for (d, v) in dw[wo..wo + weight_group].iter_mut().zip(&dwc) {
                            *d += v;
                        }
                        pool::give(dwc);
                        let dc = backend.matmul_t(&wt[wo..wo + weight_group], &gyc[..oc * ld], taps, oc, ld, true, false);
                        let mut dst: Vec<&mut [f32]> = dx.chunks_mut(image).skip(i0 * groups + group).step_by(groups).take(cnt).collect();
                        par(&mut dst, threads(taps * ld, cnt), |j0, part| {
                            for (k, img) in part.iter_mut().enumerate() {
                                for ci in 0..ic {
                                    win.fold_channel_add(&dc[ci * rows * ld + (j0 + k) * pos..], ld, ci, img);
                                }
                            }
                        });
                        pool::give(dc);
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

fn unfold_resident(x: &Tensor, g: &Window2dDev) -> Option<Result<Tensor>> {
    resident(x, |b, buf| b.unfold2d_dev(buf, g)).map(|r| r.map(|col| device_leaf(col, &[g.taps(), g.positions()], x.device())))
}

/// Resident im2col convolution: one unfold kernel, one GEMM over every
/// image, and a [Cout, N, P] -> [N, Cout, P] layout copy. Backward
/// recomputes the columns instead of holding them across the step.
fn conv_resident(x: &Tensor, weight: &Tensor, g: Window2dDev, cout: usize) -> Option<Result<Tensor>> {
    let col = match unfold_resident(x, &g)? { Ok(c) => c, Err(e) => return Some(Err(e)) };
    let (dev, taps, pos) = (x.device(), g.taps(), g.oh * g.ow);
    let forward = || -> Result<Tensor> {
        let wm = weight.detach().reshape(&[cout, taps])?;
        let y = raw_matmul_t(&wm, &col, false, false)?.reshape(&[cout, g.n, pos])?;
        y.permute(&[1, 0, 2])?.detach_copy().reshape(&[g.n, cout, g.oh, g.ow])
    };
    let y = match forward() { Ok(y) => y, Err(e) => return Some(Err(e)) };
    let (xs, ws) = (x.detach(), weight.detach());
    let wshape = weight.shape().to_vec();
    Some(Ok(y.record_fn(vec![x.clone(), weight.clone()], move |gy| {
        let gy = whole_on(gy, dev).reshape(&[g.n, cout, pos]).unwrap();
        let gcat = gy.permute(&[1, 0, 2]).unwrap().detach_copy().reshape(&[cout, g.positions()]).unwrap();
        let col = unfold_resident(&xs, &g).expect("unfold2d_dev seam vanished").expect("unfold2d_dev");
        let wm = ws.reshape(&[cout, taps]).unwrap();
        let dw = raw_matmul_t(&gcat, &col, false, true).unwrap().reshape(&wshape).unwrap();
        let dcol = raw_matmul_t(&wm, &gcat, true, false).unwrap();
        let dx = resident(&dcol, |b, buf| b.fold2d_dev(buf, &g)).expect("fold2d_dev seam missing").expect("fold2d_dev");
        vec![device_leaf(dx, &[g.n, g.c, g.h, g.w], dev), dw]
    })))
}
