//! Whole-sequence RNN/GRU/LSTM recorded as ONE first-order node. The composed
//! unroll records ~15 ops per timestep and slices the hoisted projection once
//! per step, so backward pays a full-size slice adjoint per step (quadratic in
//! T). Here the forward keeps activations in flat buffers and the backward is
//! analytic BPTT: one [B,G]x[G,H] GEMM per step plus three batched GEMMs for
//! the weight and input gradients.
//!
//! Callers guarantee: CPU float32, validated shapes, no resets, truncation only
//! with full-length rows, padded input positions already zeroed. Rows past
//! their length keep their state and emit zeros, exactly like `unroll`; their
//! gate cotangents are exactly zero (never 0 * local derivative).

use super::{CellKind, RecurrentState, UnrollOutput};
use crate::autograd::with_grad_enabled;
use crate::{Result, Tensor};
use std::sync::Arc;

/// a [n,p] (row-major) times b; b is any [p,q] tensor.
fn mm(a: Vec<f32>, n: usize, p: usize, b: &Tensor) -> Result<Vec<f32>> { Ok(Tensor::from_vec(a, &[n, p])?.matmul(b)?.to_vec()) }

/// a^T b for row-major a [n,p], b [n,q]. The transpose is materialized in
/// cache tiles: a strided view would be gathered element by element.
fn mm_tn(a: &[f32], b: &[f32], n: usize, p: usize, q: usize) -> Result<Vec<f32>> {
    const TILE: usize = 32;
    let mut at = vec![0f32; n * p];
    for r0 in (0..n).step_by(TILE) {
        for c0 in (0..p).step_by(TILE) {
            for r in r0..(r0 + TILE).min(n) { for c in c0..(c0 + TILE).min(p) { at[c * n + r] = a[r * p + c]; } }
        }
    }
    Ok(Tensor::from_vec(at, &[p, n])?.matmul(&Tensor::from_vec(b.to_vec(), &[n, q])?)?.to_vec())
}

fn column_sums(a: &[f32], cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; cols];
    for row in a.chunks_exact(cols) { for (o, v) in out.iter_mut().zip(row) { *o += v; } }
    out
}

struct Saved {
    kind: CellKind,
    dims: [usize; 5],
    lengths: Vec<usize>,
    truncate: Option<usize>,
    /// State entering each step, [T,B,H] (h and, for LSTM, c).
    h_in: Vec<f32>,
    c_in: Vec<f32>,
    /// Activated gates [T,B,G]; for LSTM also tanh(c_t) [T,B,H]; for GRU the recurrent new-gate affine [T,B,H].
    acts: Vec<f32>,
    extra: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sequence(kind: CellKind, x: &Tensor, initial: &RecurrentState, wi: &Tensor, wh: &Tensor, bi: Option<&Tensor>, bh: Option<&Tensor>, lengths: &[usize], truncate: Option<usize>) -> Result<UnrollOutput> {
    let (time, batch, features) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let width = initial.h.shape()[1];
    let gates = wh.shape()[0];
    let (projected, recurrent) = with_grad_enabled(false, || -> Result<_> {
        Ok((x.reshape(&[time * batch, features])?.matmul(&wi.transpose(0, 1)?)?.to_vec(), wh.transpose(0, 1)?.contiguous()?))
    })?;
    let zeros = || vec![0f32; gates];
    let (bias_i, bias_h) = (bi.map(Tensor::to_vec).unwrap_or_else(zeros), bh.map(Tensor::to_vec).unwrap_or_else(zeros));
    let (bh_len, hw) = (batch * width, batch * gates);
    let mut h = initial.h.to_vec();
    let mut c = initial.c.as_ref().map(Tensor::to_vec).unwrap_or_default();
    let mut out = vec![0f32; time * bh_len];
    let mut saved = Saved { kind, dims: [time, batch, features, width, gates], lengths: lengths.to_vec(), truncate, h_in: Vec::with_capacity(time * bh_len), c_in: Vec::new(), acts: vec![0.; time * hw], extra: vec![0.; time * bh_len] };
    if kind == CellKind::Lstm { saved.c_in.reserve(time * bh_len); }
    let mut pre = vec![0f32; hw];
    let mut second = vec![0f32; bh_len];
    for t in 0..time {
        saved.h_in.extend_from_slice(&h);
        if kind == CellKind::Lstm { saved.c_in.extend_from_slice(&c); }
        let rec = with_grad_enabled(false, || mm(h.clone(), batch, width, &recurrent))?;
        // GRU keeps the recurrent new-gate affine out of `pre`: it is scaled by the reset gate first.
        let split = if kind == CellKind::Gru { 2 * width } else { gates };
        for b in 0..batch {
            let (a, r, p) = (&projected[(t * batch + b) * gates..][..gates], &rec[b * gates..][..gates], &mut pre[b * gates..][..gates]);
            for k in 0..gates { p[k] = a[k] + bias_i[k] + if k < split { r[k] + bias_h[k] } else { 0. }; }
            if kind == CellKind::Gru { for j in 0..width { saved.extra[t * bh_len + b * width + j] = r[split + j] + bias_h[split + j]; } }
        }
        let act = |v: &[f32], f: fn(&Tensor) -> Tensor| with_grad_enabled(false, || Ok::<_, crate::Error>(f(&Tensor::from_vec(v.to_vec(), &[v.len()])?).to_vec()));
        let acts = &mut saved.acts[t * hw..][..hw];
        let extra = &mut saved.extra[t * bh_len..][..bh_len];
        match kind {
            CellKind::Rnn => acts.copy_from_slice(&act(&pre, Tensor::tanh)?),
            CellKind::Gru => {
                let sg = act(&pre, Tensor::sigmoid)?;
                for b in 0..batch { for j in 0..width { second[b * width + j] = pre[b * gates + 2 * width + j] + sg[b * gates + j] * extra[b * width + j]; } }
                let n = act(&second, Tensor::tanh)?;
                for b in 0..batch {
                    let row = &mut acts[b * gates..][..gates];
                    row[..2 * width].copy_from_slice(&sg[b * gates..][..2 * width]);
                    row[2 * width..].copy_from_slice(&n[b * width..][..width]);
                }
            }
            CellKind::Lstm => {
                let (sg, th) = (act(&pre, Tensor::sigmoid)?, act(&pre, Tensor::tanh)?);
                for b in 0..batch {
                    let row = &mut acts[b * gates..][..gates];
                    row.copy_from_slice(&sg[b * gates..][..gates]);
                    row[2 * width..3 * width].copy_from_slice(&th[b * gates + 2 * width..][..width]);
                    for j in 0..width { second[b * width + j] = row[width + j] * c[b * width + j] + row[j] * row[2 * width + j]; }
                }
                extra.copy_from_slice(&act(&second, Tensor::tanh)?);
            }
        }
        for b in (0..batch).filter(|&b| t < lengths[b]) {
            let (row, o) = (b * width, &acts[b * gates..][..gates]);
            for j in 0..width {
                let v = match kind {
                    CellKind::Rnn => o[j],
                    CellKind::Gru => (1. - o[width + j]) * o[2 * width + j] + o[width + j] * h[row + j],
                    CellKind::Lstm => { c[row + j] = second[row + j]; o[3 * width + j] * extra[row + j] }
                };
                h[row + j] = v;
                out[t * bh_len + row + j] = v;
            }
        }
    }
    let lstm = kind == CellKind::Lstm;
    let mut outputs = vec![Tensor::from_vec(out, &[time, batch, width])?, Tensor::from_vec(h, &[batch, width])?];
    if lstm { outputs.push(Tensor::from_vec(c, &[batch, width])?); }
    let mut inputs = vec![x.clone(), initial.h.clone()];
    inputs.extend(initial.c.clone());
    inputs.extend([wi.clone(), wh.clone()]);
    inputs.extend(bi.cloned());
    inputs.extend(bh.cloned());
    let (has_bi, has_bh) = (bi.is_some(), bh.is_some());
    let (x_saved, wi_saved, wh_saved) = (x.detach_copy(), wi.detach_copy(), wh.detach_copy());
    let saved = Arc::new(saved);
    let differentiable = vec![true; outputs.len()];
    let mut outputs = Tensor::record_many_optional(outputs, inputs, &differentiable, move |seeds| {
        let grads = backward(&saved, seeds, &x_saved, &wi_saved, &wh_saved).expect("fused recurrent backward on validated shapes");
        let [dx, dh0, dc0, dwi, dwh, dbi, dbh] = grads;
        let mut result = vec![Some(dx), Some(dh0)];
        if lstm { result.push(Some(dc0)); }
        result.extend([Some(dwi), Some(dwh)]);
        if has_bi { result.push(Some(dbi)); }
        if has_bh { result.push(Some(dbh)); }
        result
    })?.into_iter();
    let outputs_seq = outputs.next().expect("sequence output");
    let h = outputs.next().expect("final hidden");
    Ok(UnrollOutput { outputs: outputs_seq, state: RecurrentState { h, c: outputs.next() } })
}

fn backward(s: &Saved, seeds: &[Option<Tensor>], x: &Tensor, wi: &Tensor, wh: &Tensor) -> Result<[Tensor; 7]> {
    let [time, batch, features, width, gates] = s.dims;
    let (bh_len, hw) = (batch * width, batch * gates);
    let seed = |k: usize| seeds.get(k).and_then(Option::as_ref).map(Tensor::to_vec);
    let gout = seed(0);
    let mut dh = seed(1).unwrap_or_else(|| vec![0.; bh_len]);
    let mut dc = if s.kind == CellKind::Lstm { seed(2).unwrap_or_else(|| vec![0.; bh_len]) } else { Vec::new() };
    // Cotangents of the input-side affine (a) and the recurrent-side affine (b); they differ only in GRU's new gate.
    let mut da = vec![0f32; time * hw];
    let mut db = if s.kind == CellKind::Gru { vec![0f32; time * hw] } else { Vec::new() };
    for t in (0..time).rev() {
        let mut dh_prev = dh.clone();
        let mut dc_prev = dc.clone();
        for b in (0..batch).filter(|&b| t < s.lengths[b]) {
            let row = b * width;
            let dhb = &mut dh[row..][..width];
            if let Some(g) = &gout { for (d, v) in dhb.iter_mut().zip(&g[t * bh_len + row..][..width]) { *d += v; } }
            let act = &s.acts[t * hw + b * gates..][..gates];
            let extra = &s.extra[t * bh_len + row..][..width];
            let hp = &s.h_in[t * bh_len + row..][..width];
            let dab = &mut da[t * hw + b * gates..][..gates];
            for v in &mut dh_prev[row..][..width] { *v = 0.; }
            match s.kind {
                CellKind::Rnn => for j in 0..width { dab[j] = dhb[j] * (1. - act[j] * act[j]); },
                CellKind::Gru => {
                    let dbb = &mut db[t * hw + b * gates..][..gates];
                    for j in 0..width {
                        let (r, z, n, bn) = (act[j], act[width + j], act[2 * width + j], extra[j]);
                        let dn = dhb[j] * (1. - z) * (1. - n * n);
                        let dz = dhb[j] * (hp[j] - n) * z * (1. - z);
                        let dr = dn * bn * r * (1. - r);
                        (dab[j], dab[width + j], dab[2 * width + j]) = (dr, dz, dn);
                        (dbb[j], dbb[width + j], dbb[2 * width + j]) = (dr, dz, dn * r);
                        dh_prev[row + j] = dhb[j] * z;
                    }
                }
                CellKind::Lstm => {
                    let cp = &s.c_in[t * bh_len + row..][..width];
                    for j in 0..width {
                        let (i, f, g, o, tc) = (act[j], act[width + j], act[2 * width + j], act[3 * width + j], extra[j]);
                        let dct = dc[row + j] + dhb[j] * o * (1. - tc * tc);
                        dab[j] = dct * g * i * (1. - i);
                        dab[width + j] = dct * cp[j] * f * (1. - f);
                        dab[2 * width + j] = dct * i * (1. - g * g);
                        dab[3 * width + j] = dhb[j] * tc * o * (1. - o);
                        dc_prev[row + j] = dct * f;
                    }
                }
            }
        }
        let side = if s.kind == CellKind::Gru { &db } else { &da };
        let through = with_grad_enabled(false, || mm(side[t * hw..][..hw].to_vec(), batch, gates, wh))?;
        for b in (0..batch).filter(|&b| t < s.lengths[b]) {
            for j in b * width..(b + 1) * width { dh_prev[j] += through[j]; }
        }
        if s.truncate.is_some_and(|k| t > 0 && t % k == 0) {
            dh_prev.iter_mut().for_each(|v| *v = 0.);
            dc_prev.iter_mut().for_each(|v| *v = 0.);
        }
        dh = dh_prev;
        dc = dc_prev;
    }
    let rows = time * batch;
    let side = if s.kind == CellKind::Gru { &db } else { &da };
    with_grad_enabled(false, || {
        let dwh = Tensor::from_vec(mm_tn(side, &s.h_in, rows, gates, width)?, &[gates, width])?;
        let dbh = Tensor::from_vec(column_sums(side, gates), &[gates])?;
        let dwi = Tensor::from_vec(mm_tn(&da, &x.to_vec(), rows, gates, features)?, &[gates, features])?;
        let dbi = Tensor::from_vec(column_sums(&da, gates), &[gates])?;
        let dx = Tensor::from_vec(mm(da.clone(), rows, gates, wi)?, &[time, batch, features])?;
        let dc0 = if s.kind == CellKind::Lstm { Tensor::from_vec(dc, &[batch, width])? } else { Tensor::zeros(&[batch, width]) };
        Ok([dx, Tensor::from_vec(dh, &[batch, width])?, dc0, dwi, dwh, dbi, dbh])
    })
}
