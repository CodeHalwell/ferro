//! Immutable sparse topology, with integer coordinates and explicit duplicate edges.
//! Tensor operations are CPU f32, first-order autograd only; no graph replay support.
use crate::{Error, Result, Tensor};
use crate::segment;

fn invalid(op: &'static str, msg: &str) -> Error { Error::InvalidShape { op, msg: msg.into() } }

/// Coordinates are (destination row, source column). Input edge order is retained.
/// Duplicates remain independent edges unless explicitly coalesced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coo {
    shape: [usize; 2],
    rows: Vec<usize>,
    cols: Vec<usize>,
}

impl Coo {
    pub fn new(nrows: usize, ncols: usize, rows: Vec<usize>, cols: Vec<usize>) -> Result<Self> {
        if rows.len() != cols.len() || rows.iter().any(|&r| r >= nrows) || cols.iter().any(|&c| c >= ncols) {
            return Err(invalid("coo", "coordinate lengths differ or coordinate out of bounds"));
        }
        Ok(Self { shape: [nrows, ncols], rows, cols })
    }

    /// Checked signed-integer conversion, never through floating point.
    pub fn from_i64(nrows: usize, ncols: usize, rows: &[i64], cols: &[i64]) -> Result<Self> {
        let convert = |v: &[i64]| v.iter().map(|&i| usize::try_from(i)
            .map_err(|_| invalid("coo", "negative or unrepresentable coordinate"))).collect::<Result<Vec<_>>>();
        Self::new(nrows, ncols, convert(rows)?, convert(cols)?)
    }
    pub fn shape(&self) -> [usize; 2] { self.shape }
    pub fn nnz(&self) -> usize { self.rows.len() }
    pub fn rows(&self) -> &[usize] { &self.rows }
    pub fn cols(&self) -> &[usize] { &self.cols }

    /// Stable lexicographic conversion; duplicates are NOT merged.
    /// Returned order maps each CSR entry to its original COO edge index.
    pub fn to_csr(&self) -> Result<(Csr, Vec<usize>)> {
        let len = self.shape[0].checked_add(1).ok_or_else(|| invalid("coo_to_csr", "row count overflow"))?;
        let mut order: Vec<_> = (0..self.nnz()).collect();
        order.sort_by_key(|&e| (self.rows[e], self.cols[e]));
        let mut offsets = vec![0usize; len];
        for &r in &self.rows { offsets[r + 1] += 1; }
        for r in 0..self.shape[0] { offsets[r + 1] += offsets[r]; }
        let cols = order.iter().map(|&e| self.cols[e]).collect();
        Ok((Csr::new(self.shape[0], self.shape[1], offsets, cols)?, order))
    }
}

/// Validated CSR; columns may be unsorted and duplicated within each row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Csr {
    shape: [usize; 2],
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
}
impl Csr {
    pub fn new(nrows: usize, ncols: usize, row_offsets: Vec<usize>, col_indices: Vec<usize>) -> Result<Self> {
        if nrows.checked_add(1) != Some(row_offsets.len()) || row_offsets.first() != Some(&0)
            || row_offsets.last() != Some(&col_indices.len()) || row_offsets.windows(2).any(|w| w[0] > w[1])
            || col_indices.iter().any(|&c| c >= ncols) {
            return Err(invalid("csr", "invalid offsets or column out of bounds"));
        }
        Ok(Self { shape: [nrows,ncols], row_offsets, col_indices })
    }
    pub fn shape(&self) -> [usize; 2] { self.shape }
    pub fn nnz(&self) -> usize { self.col_indices.len() }
    pub fn row_offsets(&self) -> &[usize] { &self.row_offsets }
    pub fn col_indices(&self) -> &[usize] { &self.col_indices }
    pub fn to_coo(&self) -> Coo {
        let mut rows = Vec::with_capacity(self.nnz());
        for r in 0..self.shape[0] {
            rows.extend(std::iter::repeat(r).take(self.row_offsets[r+1] - self.row_offsets[r]));
        }
        Coo { shape: self.shape, rows, cols: self.col_indices.clone() }
    }
}


impl Coo {
    /// Merge equal coordinates by summing values [nnz, ...], sorted row then column.
    /// Every original duplicate receives the full coalesced-value adjoint.
    pub fn coalesce(&self, values: &Tensor) -> Result<(Self, Tensor)> {
        segment::cpu_f32(values, "coalesce")?;
        if values.shape().first() != Some(&self.nnz()) {
            return Err(invalid("coalesce", "expected one value row per edge"));
        }
        let mut order: Vec<_> = (0..self.nnz()).collect();
        order.sort_by_key(|&e| (self.rows[e], self.cols[e]));
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut ids = vec![0; self.nnz()];
        for e in order {
            if rows.last() != Some(&self.rows[e]) || cols.last() != Some(&self.cols[e]) {
                rows.push(self.rows[e]); cols.push(self.cols[e]);
            }
            ids[e] = rows.len() - 1;
        }
        let out = segment::sum(values, &ids, rows.len())?;
        Ok((Self { shape: self.shape, rows, cols }, out))
    }

    /// A @ dense, with scalar values [nnz] and dense [ncols, features].
    /// Duplicate entries add, isolated destination rows are zero.
    pub fn spmm(&self, values: &Tensor, dense: &Tensor) -> Result<Tensor> {
        segment::cpu_f32(values, "spmm")?;
        segment::cpu_f32(dense, "spmm")?;
        if values.shape() != [self.nnz()] || dense.shape().len() != 2 || dense.shape()[0] != self.shape[1] {
            return Err(invalid("spmm", "expected values [nnz] and dense [ncols, features]"));
        }
        let width = dense.shape()[1];
        let size = self.shape[0].checked_mul(width).ok_or_else(|| invalid("spmm", "output size overflow"))?;
        let w = values.to_vec();
        let x = dense.to_vec();
        let mut out = vec![0.; size];
        for e in 0..self.nnz() {
            for f in 0..width { out[self.rows[e] * width + f] += w[e] * x[self.cols[e] * width + f]; }
        }
        let a = self.clone();
        let xshape = dense.shape().to_vec();
        Ok(Tensor::from_vec(out, &[self.shape[0], width])?.record_fn(vec![values.clone(), dense.clone()], move |g| {
            let g = g.to_vec();
            let mut dw = vec![0.; a.nnz()];
            let mut dx = vec![0.; x.len()];
            for e in 0..a.nnz() {
                for f in 0..width {
                    let gg = g[a.rows[e] * width + f];
                    dw[e] += gg * x[a.cols[e] * width + f];
                    dx[a.cols[e] * width + f] += gg * w[e];
                }
            }
            vec![Tensor::from_vec(dw, &[a.nnz()]).unwrap(), Tensor::from_vec(dx, &xshape).unwrap()]
        }))
    }
}
impl Csr {
    /// Values follow CSR entry order (apply the permutation from COO conversion).
    pub fn spmm(&self, values: &Tensor, dense: &Tensor) -> Result<Tensor> {
        self.to_coo().spmm(values, dense)
    }
}


impl Coo {
    /// Sample (left @ right.T) at each edge, returning [nnz] in original order.
    /// Left is [nrows, features], right is [ncols, features]. Duplicates are
    /// separate outputs; their adjoints accumulate into both node operands.
    pub fn sddmm(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        segment::cpu_f32(left, "sddmm")?;
        segment::cpu_f32(right, "sddmm")?;
        if left.shape().len() != 2 || right.shape().len() != 2
            || left.shape()[0] != self.shape[0] || right.shape()[0] != self.shape[1]
            || left.shape()[1] != right.shape()[1] {
            return Err(invalid("sddmm", "expected [nrows, features] and [ncols, features]"));
        }
        let width = left.shape()[1];
        let l = left.to_vec();
        let r = right.to_vec();
        let mut out = vec![0.; self.nnz()];
        for e in 0..self.nnz() {
            for f in 0..width { out[e] += l[self.rows[e] * width + f] * r[self.cols[e] * width + f]; }
        }
        let a = self.clone();
        Ok(Tensor::from_vec(out, &[self.nnz()])?.record_fn(vec![left.clone(), right.clone()], move |g| {
            let g = g.to_vec();
            let mut dl = vec![0.; l.len()];
            let mut dr = vec![0.; r.len()];
            for e in 0..a.nnz() {
                for f in 0..width {
                    dl[a.rows[e] * width + f] += g[e] * r[a.cols[e] * width + f];
                    dr[a.cols[e] * width + f] += g[e] * l[a.rows[e] * width + f];
                }
            }
            vec![Tensor::from_vec(dl, &[a.shape[0],width]).unwrap(), Tensor::from_vec(dr, &[a.shape[1],width]).unwrap()]
        }))
    }
}
