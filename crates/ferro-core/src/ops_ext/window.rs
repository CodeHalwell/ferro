//! Validated rectangular sliding windows. CPU unfold/fold are an adjoint pair.
use crate::{
    dtype::DType,
    error::{Error, Result},
    Tensor,
};

pub(crate) fn invalid(op: &'static str, msg: &str) -> Error {
    Error::InvalidShape {
        op,
        msg: msg.into(),
    }
}

pub(crate) fn checked_len(op: &'static str, dims: &[usize]) -> Result<usize> {
    let len = dims
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| invalid(op, "shape product overflow"))?;
    if len > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(invalid(op, "buffer byte length overflow"));
    }
    Ok(len)
}

pub(crate) fn require_f32(t: &Tensor, op: &'static str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::DtypeMismatch {
            op,
            expected: DType::F32,
            got: t.dtype(),
        });
    }
    Ok(())
}

/// Symmetric padding per axis, with independent height/width stride and dilation.
/// Zero batch is supported by tensor operations; spatial/channel sizes must be positive.
#[derive(Clone, Copy, Debug)]
pub struct Window2d {
    input: [usize; 2],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    output: [usize; 2],
}

impl Window2d {
    pub fn new(
        input: [usize; 2],
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self> {
        let mut output = [0; 2];
        for axis in 0..2 {
            if input[axis] == 0 || kernel[axis] == 0 || stride[axis] == 0 || dilation[axis] == 0 {
                return Err(invalid(
                    "window2d",
                    "input, kernel, stride and dilation must be positive",
                ));
            }
            let padded = padding[axis]
                .checked_mul(2)
                .and_then(|p| input[axis].checked_add(p))
                .ok_or_else(|| invalid("window2d", "padding overflow"))?;
            let effective = (kernel[axis] - 1)
                .checked_mul(dilation[axis])
                .and_then(|k| k.checked_add(1))
                .ok_or_else(|| invalid("window2d", "effective kernel overflow"))?;
            if effective > padded {
                return Err(invalid("window2d", "effective kernel exceeds padded input"));
            }
            output[axis] = (padded - effective) / stride[axis] + 1;
        }
        checked_len("window2d", &input)?;
        checked_len("window2d", &kernel)?;
        checked_len("window2d", &output)?;
        checked_len("window2d", &[kernel[0], kernel[1], output[0], output[1]])?;
        Ok(Self {
            input,
            kernel,
            stride,
            padding,
            dilation,
            output,
        })
    }

    pub fn output_size(&self) -> [usize; 2] {
        self.output
    }
    pub(crate) fn positions(&self) -> usize {
        self.output[0] * self.output[1]
    }
    pub(crate) fn kernel_area(&self) -> usize {
        self.kernel[0] * self.kernel[1]
    }
    pub(crate) fn image_len(&self, channels: usize) -> Result<usize> {
        checked_len("window2d", &[channels, self.input[0], self.input[1]])
    }
    pub(crate) fn column_len(&self, channels: usize) -> Result<usize> {
        checked_len(
            "window2d",
            &[channels, self.kernel_area(), self.positions()],
        )
    }

    /// In-bounds output columns [x0, x1) for kernel column `c`, and the
    /// input column of x = 0 plus padding (ix = x*sw + base - pw).
    fn columns(&self, c: usize) -> (usize, usize, usize) {
        let ([w, ow], sw, pw) = ([self.input[1], self.output[1]], self.stride[1], self.padding[1]);
        let off = c * self.dilation[1];
        let x0 = pw.saturating_sub(off).div_ceil(sw).min(ow);
        let x1 = if w + pw > off { ((w + pw - off - 1) / sw + 1).min(ow) } else { 0 };
        (x0, x1.max(x0), off)
    }

    /// Input row for kernel row `r` at output row `y`, if in bounds.
    fn input_row(&self, r: usize, y: usize) -> Option<usize> {
        (y * self.stride[0] + r * self.dilation[0]).checked_sub(self.padding[0]).filter(|&iy| iy < self.input[0])
    }

    /// Both directions share this mapping, including the padding mask. For
    /// tap t = r*KW + c and output row y with an in-bounds input row,
    /// `f(t, y*OW + x0, src, len)` receives the in-bounds column range, with
    /// `src` the input offset of x0 inside the channel plane (consecutive x
    /// step by the width stride).
    fn rows(&self, mut f: impl FnMut(usize, usize, usize, usize)) {
        let (w, [kh, kw], [oh, ow]) = (self.input[1], self.kernel, self.output);
        let (sw, pw) = (self.stride[1], self.padding[1]);
        for r in 0..kh {
            for c in 0..kw {
                let (x0, x1, off) = self.columns(c);
                if x0 == x1 {
                    continue;
                }
                for y in 0..oh {
                    if let Some(iy) = self.input_row(r, y) {
                        f(r * kw + c, y * ow + x0, iy * w + x0 * sw + off - pw, x1 - x0);
                    }
                }
            }
        }
    }

    /// Unfold channel `ci` of `image` into KH*KW column rows of `col`
    /// spaced `ld` apart (row t starts at `col[t*ld]`, holding OH*OW values).
    /// Every slot is written exactly once (padding as zero).
    pub(crate) fn unfold_channel(&self, image: &[f32], ci: usize, col: &mut [f32], ld: usize) {
        let ([h, w], [kh, kw], [oh, ow]) = (self.input, self.kernel, self.output);
        let (sw, pw) = (self.stride[1], self.padding[1]);
        let src = &image[ci * h * w..(ci + 1) * h * w];
        for r in 0..kh {
            for c in 0..kw {
                let (x0, x1, off) = self.columns(c);
                let row = &mut col[(r * kw + c) * ld..][..oh * ow];
                for (y, seg) in row.chunks_exact_mut(ow).enumerate() {
                    let Some(iy) = self.input_row(r, y) else {
                        seg.fill(0.0);
                        continue;
                    };
                    seg[..x0].fill(0.0);
                    seg[x1..].fill(0.0);
                    if x0 < x1 {
                        let s = &src[iy * w + x0 * sw + off - pw..];
                        if sw == 1 {
                            seg[x0..x1].iter_mut().zip(s).for_each(|(o, &v)| *o = v);
                        } else {
                            seg[x0..x1].iter_mut().zip(s.iter().step_by(sw)).for_each(|(o, &v)| *o = v);
                        }
                    }
                }
            }
        }
    }

    /// Adjoint of `unfold_channel`: accumulate the column rows into channel `ci`.
    pub(crate) fn fold_channel_add(&self, col: &[f32], ld: usize, ci: usize, image: &mut [f32]) {
        let (plane, sw) = (self.input[0] * self.input[1], self.stride[1]);
        let dst = &mut image[ci * plane..(ci + 1) * plane];
        self.rows(|t, src, d, len| {
            let s = &col[t * ld + src..][..len];
            dst[d..].iter_mut().step_by(sw).zip(s).for_each(|(o, &v)| *o += v);
        });
    }

    pub(crate) fn unfold_into(&self, image: &[f32], channels: usize, col: &mut [f32]) {
        let rows = self.kernel_area() * self.positions();
        for ci in 0..channels {
            self.unfold_channel(image, ci, &mut col[ci * rows..(ci + 1) * rows], self.positions());
        }
    }
    pub(crate) fn fold_add(&self, col: &[f32], channels: usize, image: &mut [f32]) {
        let rows = self.kernel_area() * self.positions();
        for ci in 0..channels {
            self.fold_channel_add(&col[ci * rows..(ci + 1) * rows], self.positions(), ci, image);
        }
    }
}

impl Tensor {
    /// NCHW -> [N, C*KH*KW, OH*OW], with zero-filled padding. CPU fallback.
    pub fn unfold2d(
        &self,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Tensor> {
        require_f32(self, "unfold2d")?;
        if self.ndim() != 4 || self.shape()[1] == 0 {
            return Err(invalid("unfold2d", "expected NCHW with positive channels"));
        }
        let shape = self.shape().to_vec();
        let (n, c) = (shape[0], shape[1]);
        let win = Window2d::new([shape[2], shape[3]], kernel, stride, padding, dilation)?;
        let image_len = win.image_len(c)?;
        let col_len = win.column_len(c)?;
        let data = self.to_vec();
        let mut out = vec![0.; checked_len("unfold2d", &[n, col_len])?];
        for i in 0..n {
            win.unfold_into(
                &data[i * image_len..(i + 1) * image_len],
                c,
                &mut out[i * col_len..(i + 1) * col_len],
            );
        }
        Ok(
            Tensor::from_vec(out, &[n, c * win.kernel_area(), win.positions()])?.record_fn(
                vec![self.clone()],
                move |g| {
                    let mut dx = vec![0.; n * image_len];
                    let gv = g.to_vec();
                    for i in 0..n {
                        win.fold_add(
                            &gv[i * col_len..(i + 1) * col_len],
                            c,
                            &mut dx[i * image_len..(i + 1) * image_len],
                        );
                    }
                    vec![Tensor::from_vec(dx, &shape).unwrap()]
                },
            ),
        )
    }

    /// [N, C*KH*KW, OH*OW] -> NCHW; overlapping windows are summed, not averaged.
    pub fn fold2d(
        &self,
        output_size: [usize; 2],
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Tensor> {
        require_f32(self, "fold2d")?;
        let win = Window2d::new(output_size, kernel, stride, padding, dilation)?;
        if self.ndim() != 3
            || self.shape()[1] == 0
            || self.shape()[1] % win.kernel_area() != 0
            || self.shape()[2] != win.positions()
        {
            return Err(invalid(
                "fold2d",
                "expected [N, C*KH*KW, OH*OW] matching geometry",
            ));
        }
        let shape = self.shape().to_vec();
        let (n, c) = (shape[0], shape[1] / win.kernel_area());
        let image_len = win.image_len(c)?;
        let col_len = win.column_len(c)?;
        let data = self.to_vec();
        let mut out = vec![0.; checked_len("fold2d", &[n, image_len])?];
        for i in 0..n {
            win.fold_add(
                &data[i * col_len..(i + 1) * col_len],
                c,
                &mut out[i * image_len..(i + 1) * image_len],
            );
        }
        Ok(
            Tensor::from_vec(out, &[n, c, output_size[0], output_size[1]])?.record_fn(
                vec![self.clone()],
                move |g| {
                    let mut dx = vec![0.; n * col_len];
                    let gv = g.to_vec();
                    for i in 0..n {
                        win.unfold_into(
                            &gv[i * image_len..(i + 1) * image_len],
                            c,
                            &mut dx[i * col_len..(i + 1) * col_len],
                        );
                    }
                    vec![Tensor::from_vec(dx, &shape).unwrap()]
                },
            ),
        )
    }
}
