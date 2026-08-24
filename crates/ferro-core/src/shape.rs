use crate::error::{Error, Result};

/// Row-major (C-contiguous) strides for a shape.
pub fn default_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        stride[i] = acc;
        acc *= shape[i];
    }
    stride
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// NumPy/PyTorch broadcasting rules: align shapes on the right, each dim must be
/// equal or one of them must be 1; the result takes the max of the two.
pub fn broadcast_shapes(op: &'static str, a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        let ad = if i < n - a.len() {
            1
        } else {
            a[i - (n - a.len())]
        };
        let bd = if i < n - b.len() {
            1
        } else {
            b[i - (n - b.len())]
        };
        out[i] = if ad == bd {
            ad
        } else if ad == 1 {
            bd
        } else if bd == 1 {
            ad
        } else {
            return Err(Error::ShapeMismatch {
                op,
                lhs: a.to_vec(),
                rhs: b.to_vec(),
            });
        };
    }
    Ok(out)
}
