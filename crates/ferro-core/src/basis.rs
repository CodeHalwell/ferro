//! Fixed-knot CPU f32 splines. Knots are immutable, not trainable.
//! The basis VJP is recorded in the shared engine; create_graph through the
//! basis returns Unsupported (piecewise derivatives, no higher-order promise).
use crate::{DType, Device, Error, Result, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutsideDomain {
    /// Values and derivatives are zero strictly outside the closed domain.
    Zero,
    /// Reject any input outside the closed domain.
    Error,
}

#[derive(Clone, Debug)]
pub struct BSplineBasis {
    knots: Vec<f32>,
    degree: usize,
    outside: OutsideDomain,
}

impl BSplineBasis {
    /// Nondecreasing finite knots, degree < number of basis functions, and a
    /// nonempty domain [knots[degree], knots[n_basis]] are required.
    pub fn new(knots: Vec<f32>, degree: usize, outside: OutsideDomain) -> Result<Self> {
        let valid_len = degree.checked_add(1).and_then(|d| d.checked_mul(2));
        if valid_len.is_none_or(|n| knots.len() < n) || knots.iter().any(|v| !v.is_finite())
            || knots.windows(2).any(|p| p[0] > p[1]) {
            return Err(Error::InvalidShape { op: "bspline", msg: "finite nondecreasing knots and at least 2*(degree+1) knots required".into() });
        }
        let n = knots.len() - degree - 1;
        if knots[degree] >= knots[n] {
            return Err(Error::InvalidShape { op: "bspline", msg: "empty base domain".into() });
        }
        Ok(Self { knots, degree, outside })
    }

    pub fn knots(&self) -> &[f32] { &self.knots }
    pub fn degree(&self) -> usize { self.degree }
    pub fn num_basis(&self) -> usize { self.knots.len() - self.degree - 1 }
    pub fn domain(&self) -> (f32, f32) { (self.knots[self.degree], self.knots[self.num_basis()]) }

    /// Append a basis axis to any input shape. Interior knots use right-hand
    /// values/derivatives; the domain's right endpoint uses the left limit.
    /// Repeated-knot zero denominators contribute zero, including degree zero.
    pub fn evaluate(&self, x: &Tensor) -> Result<Tensor> {
        cpu_f32(x, "bspline")?;
        let xs = x.to_vec();
        let (lo, hi) = self.domain();
        if xs.iter().any(|v| !v.is_finite() || (self.outside == OutsideDomain::Error && (*v < lo || *v > hi))) {
            return Err(Error::InvalidShape { op: "bspline", msg: "nonfinite input or input outside base domain".into() });
        }
        let n = self.num_basis();
        let size = xs.len().checked_mul(n).ok_or_else(|| Error::InvalidShape { op: "bspline", msg: "output size overflow".into() })?;
        let mut values = vec![0.; size];
        let mut deriv = vec![0.; size];
        for (row, &v) in xs.iter().enumerate() {
            if v < lo || v > hi { continue; }
            let t = &self.knots;
            let mut level = vec![0f64; t.len() - 1];
            for i in 0..level.len() {
                let active = if v == hi { t[i] < v && v <= t[i+1] } else { t[i] <= v && v < t[i+1] };
                level[i] = if active { 1. } else { 0. };
            }
            for d in 1..=self.degree {
                let mut next = vec![0.; level.len()-1];
                for i in 0..next.len() {
                    let left = t[i+d] as f64 - t[i] as f64;
                    let right = t[i+d+1] as f64 - t[i+1] as f64;
                    let a = if left > 0. { level[i] / left } else { 0. };
                    let b = if right > 0. { level[i+1] / right } else { 0. };
                    next[i] = (v as f64 - t[i] as f64)*a + (t[i+d+1] as f64 - v as f64)*b;
                    if d == self.degree { deriv[row*n+i] = (d as f64*(a-b)) as f32; }
                }
                level = next;
            }
            for i in 0..n { values[row*n+i] = level[i] as f32; }
        }
        let mut shape = x.shape().to_vec();
        shape.push(n);
        let input_shape = x.shape().to_vec();
        Ok(Tensor::from_vec(values, &shape)?.record_fn(vec![x.clone()], move |g| {
            let gv = g.to_vec();
            let dx = gv.chunks(n).zip(deriv.chunks(n)).map(|(g,d)| {
                g.iter().zip(d).map(|(&a,&b)| a as f64*b as f64).sum::<f64>() as f32
            }).collect();
            vec![Tensor::from_vec(dx, &input_shape).expect("validated basis gradient shape")]
        }))
    }
}

/// True edge-wise spline contraction: x [batch,input], coefficients
/// [output,input,basis], y [batch,output], y[r,j] = sum_i,k c[j,i,k] B_k(x[r,i]).
/// No node activation, residual MLP, bias, or implicit device fallback.
/// Tensor reshape/transpose/matmul preserve the coefficient graph. Higher-order
/// coefficient-only derivatives are supported; paths through x's basis are not.
pub fn kan(x: &Tensor, coefficients: &Tensor, basis: &BSplineBasis) -> Result<Tensor> {
    cpu_f32(x, "kan")?;
    cpu_f32(coefficients, "kan")?;
    let c = coefficients.shape();
    let s = x.shape();
    if s.len() != 2 || c.len() != 3 || s[1] == 0 || c[0] == 0
        || c[1] != s[1] || c[2] != basis.num_basis() {
        return Err(Error::InvalidShape { op: "kan", msg: "expected x [batch,input>0], coefficients [output>0,input,num_basis]".into() });
    }
    let width = s[1].checked_mul(c[2]).ok_or_else(|| Error::InvalidShape { op: "kan", msg: "feature size overflow".into() })?;
    let features = basis.evaluate(x)?.reshape(&[s[0], width])?;
    features.matmul(&coefficients.reshape(&[c[0], width])?.transpose(0,1)?)
}

/// A trainable bank of independent univariate splines on input-output edges.
/// Parameters use the existing Param slot and any shared-core optimizer.
#[derive(Clone)]
pub struct KanLayer {
    basis: BSplineBasis,
    coefficients: crate::Param,
}

impl KanLayer {
    /// Copy caller-provided leaf coefficients [output,input,num_basis].
    /// Initialization is explicit: no random defaults or hidden base activation.
    pub fn from_coefficients(basis: BSplineBasis, coefficients: Tensor) -> Result<Self> {
        cpu_f32(&coefficients, "kan_layer")?;
        let c = coefficients.shape();
        if c.len() != 3 || c[0] == 0 || c[1] == 0 || c[2] != basis.num_basis() {
            return Err(Error::InvalidShape { op: "kan_layer", msg: "expected coefficients [output>0,input>0,num_basis]".into() });
        }
        let leaf = coefficients.requires_grad_(true)?;
        Ok(Self { basis, coefficients: crate::Param::new(leaf) })
    }
    pub fn coefficients(&self) -> crate::Param { self.coefficients.clone() }
    pub fn basis(&self) -> &BSplineBasis { &self.basis }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        kan(x, &self.coefficients.tensor(), &self.basis)
    }
}

fn cpu_f32(t: &Tensor, op: &'static str) -> Result<()> {
    // Refuse devices before any to_vec/download or backend dispatch.
    if t.device() != Device::Cpu { return Err(Error::Unsupported { op, msg: "CPU tensors only; no implicit device download".into() }); }
    if t.dtype() != DType::F32 { return Err(Error::DtypeMismatch { op, expected: DType::F32, got: t.dtype() }); }
    Ok(())
}
