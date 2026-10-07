//! AVX2+FMA kernels behind elementwise.rs (x86_64 only; every entry point
//! assumes the caller detected avx2 and fma at runtime).
//!
//! Transcendentals are Cephes-style polynomials with range reduction. Their
//! accuracy contract, enforced by the tolerance tests in elementwise.rs: no
//! lane is more than 4 ULP further from the same formula evaluated in f64
//! than CpuBackend's scalar libm result (measured: <= 2.8). Special values
//! (NaN, +-inf, +-0, denormals, overflow/underflow) follow CpuBackend.
//!
//! Sums replay ferro-core reduce.rs's fixed-shape pairwise tree exactly, so
//! they are bitwise identical to CpuBackend at any thread count.

use std::arch::x86_64::*;

use ferro_core::dispatch::UnaryKind;

use crate::elementwise::par_split;
use crate::workers;

/// Mirrors ferro-core reduce.rs's BASE; the bitwise sum tests pin the match.
const BASE: usize = 128;
/// Pool split points, measured on the 4-core dev VM against the serial path:
/// sums stream at ~one add per element, so like the streaming elementwise
/// kinds they only win once the input leaves L3 (128K-1M was a wash or a
/// loss); softmax does an exp per element and already wins at 128K.
const PAR_SUM: usize = 1 << 21;
const PAR_SOFTMAX: usize = 1 << 17;

#[target_feature(enable = "avx2,fma")]
fn s(v: f32) -> __m256 {
    _mm256_set1_ps(v)
}

#[target_feature(enable = "avx2,fma")]
fn poly(x: __m256, c: &[f32]) -> __m256 {
    let mut y = s(c[0]);
    for &k in &c[1..] {
        y = _mm256_fmadd_ps(y, x, s(k));
    }
    y
}

/// e^x = 2^n * p(r), r = x - n*ln2 split hi/lo (Cephes expf). When every
/// lane has |x| < 87 (n in [-126, 126]) the scale is one exponent-field
/// build; otherwise `exp_wide` takes the whole vector.
#[target_feature(enable = "avx2,fma")]
fn exp(x: __m256) -> __m256 {
    // NLT_UQ is also true for NaN, which the wide path propagates.
    let wide = _mm256_cmp_ps::<_CMP_NLT_UQ>(_mm256_andnot_ps(s(-0.0), x), s(87.0));
    if _mm256_movemask_ps(wide) != 0 {
        return exp_wide(x);
    }
    // Adding 1.5*2^23 rounds x*log2(e) to an integer held in the low
    // mantissa bits: n as a float is t - magic, as an int it is the bit delta.
    let magic = s(12_582_912.0);
    let t = _mm256_fmadd_ps(x, s(std::f32::consts::LOG2_E), magic);
    let n = _mm256_sub_ps(t, magic);
    let ni = _mm256_sub_epi32(_mm256_castps_si256(t), _mm256_castps_si256(magic));
    let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(ni, _mm256_set1_epi32(127))));
    _mm256_mul_ps(exp_poly(x, n), scale)
}

#[target_feature(enable = "avx2,fma")]
fn exp_poly(x: __m256, n: __m256) -> __m256 {
    let r = _mm256_fnmadd_ps(n, s(0.693_359_4), x);
    let r = _mm256_fnmadd_ps(n, s(-2.121_944_4e-4), r);
    let p = poly(r, &[1.987_569_1e-4, 1.398_2e-3, 8.333_452e-3, 4.166_579_6e-2, 0.166_666_65, 0.5]);
    _mm256_add_ps(_mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r), s(1.0))
}

/// Full-range e^x: x clamped to [-104, 89] (beyond which the result is 0 or
/// inf anyway) and 2^n applied as two halves, so overflow lands on inf,
/// denormal results round once instead of flushing, and NaN survives.
#[cold]
#[target_feature(enable = "avx2,fma")]
fn exp_wide(x: __m256) -> __m256 {
    // max/min return their second operand when either is NaN.
    let x = _mm256_min_ps(s(89.0), _mm256_max_ps(s(-104.0), x));
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(x, s(std::f32::consts::LOG2_E)));
    let ni = _mm256_cvtps_epi32(n);
    let h = _mm256_srai_epi32::<1>(ni);
    let bias = _mm256_set1_epi32(127);
    let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(h, bias)));
    let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_sub_epi32(ni, h), bias)));
    _mm256_mul_ps(_mm256_mul_ps(exp_poly(x, n), s1), s2)
}

/// ln(x): frexp to m in [sqrt(1/2), sqrt(2)), degree-8 polynomial in m-1.
#[target_feature(enable = "avx2,fma")]
fn ln(x: __m256) -> __m256 {
    let tiny = _mm256_cmp_ps::<_CMP_LT_OQ>(x, s(f32::MIN_POSITIVE));
    let xs = _mm256_blendv_ps(x, _mm256_mul_ps(x, s(8_388_608.0)), tiny);
    let xi = _mm256_castps_si256(xs);
    let e = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32::<23>(xi), _mm256_set1_epi32(126)));
    let e = _mm256_sub_ps(e, _mm256_and_ps(tiny, s(23.0)));
    let m = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(xi, _mm256_set1_epi32(0x007f_ffff)), _mm256_set1_epi32(0x3f00_0000)));
    let small = _mm256_cmp_ps::<_CMP_LT_OQ>(m, s(std::f32::consts::FRAC_1_SQRT_2));
    let e = _mm256_sub_ps(e, _mm256_and_ps(small, s(1.0)));
    let m = _mm256_add_ps(_mm256_sub_ps(m, s(1.0)), _mm256_and_ps(small, m));
    let z = _mm256_mul_ps(m, m);
    let c = [7.037_683_6e-2, -0.115_146_1, 0.116_769_984, -0.124_201_41, 0.142_493_23, -0.166_680_57, 0.200_007_15, -0.249_999_94, 0.333_333_3];
    let y = _mm256_mul_ps(_mm256_mul_ps(poly(m, &c), m), z);
    let y = _mm256_fmadd_ps(e, s(-2.121_944_4e-4), y);
    let y = _mm256_fnmadd_ps(s(0.5), z, y);
    let r = _mm256_fmadd_ps(e, s(0.693_359_4), _mm256_add_ps(m, y));
    let r = _mm256_blendv_ps(r, s(f32::NEG_INFINITY), _mm256_cmp_ps::<_CMP_EQ_OQ>(x, s(0.0)));
    let r = _mm256_blendv_ps(r, s(f32::NAN), _mm256_cmp_ps::<_CMP_LT_OQ>(x, s(0.0)));
    let r = _mm256_blendv_ps(r, x, _mm256_cmp_ps::<_CMP_EQ_OQ>(x, s(f32::INFINITY)));
    _mm256_blendv_ps(r, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// tanh: odd polynomial below |x| = 0.625 (where 1 - 2/(e^2x + 1) would
/// cancel), the exp form above it.
#[target_feature(enable = "avx2,fma")]
fn tanh(x: __m256) -> __m256 {
    let sign = _mm256_and_ps(x, s(-0.0));
    let ax = _mm256_andnot_ps(s(-0.0), x);
    let z = _mm256_mul_ps(x, x);
    let p = poly(z, &[-5.704_988_7e-3, 2.063_908_9e-2, -5.373_971_6e-2, 0.133_314_42, -0.333_332_82]);
    let small = _mm256_fmadd_ps(_mm256_mul_ps(p, z), x, x);
    let e = exp(_mm256_add_ps(ax, ax));
    let large = _mm256_or_ps(_mm256_sub_ps(s(1.0), _mm256_div_ps(s(2.0), _mm256_add_ps(e, s(1.0)))), sign);
    _mm256_blendv_ps(large, small, _mm256_cmp_ps::<_CMP_LT_OQ>(ax, s(0.625)))
}

/// num / (1 + e^-arg): sigmoid (num = 1) and silu (num = arg = x).
#[target_feature(enable = "avx2,fma")]
fn logistic(num: __m256, arg: __m256) -> __m256 {
    _mm256_div_ps(num, _mm256_add_ps(s(1.0), exp(_mm256_sub_ps(s(0.0), arg))))
}

/// tanh-GELU as v / (1 + e^-2u), since 0.5*(1 + tanh(u)) = sigmoid(2u);
/// unlike 1 + tanh(u) this does not cancel for negative v. A rounding in 2u = 2c*(v + a*v^3) is
/// amplified by |2u| in the output, so 2u is carried as hi + lo (TwoSum for
/// the add, FMA residual for the multiply) and e^-(hi+lo) = e^-hi * (1 - lo).
#[target_feature(enable = "avx2,fma")]
fn gelu(v: __m256) -> __m256 {
    let w = _mm256_mul_ps(_mm256_mul_ps(s(0.044715), _mm256_mul_ps(v, v)), v);
    let inner = _mm256_add_ps(v, w);
    let bb = _mm256_sub_ps(inner, v);
    let err = _mm256_add_ps(_mm256_sub_ps(v, _mm256_sub_ps(inner, bb)), _mm256_sub_ps(w, bb));
    let k = s(2.0 * 0.797_884_6);
    let hi = _mm256_mul_ps(k, inner);
    let lo = _mm256_fmadd_ps(k, err, _mm256_fmsub_ps(k, inner, hi));
    let lo = _mm256_and_ps(lo, _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_andnot_ps(s(-0.0), lo), s(f32::INFINITY)));
    let e = exp(_mm256_sub_ps(s(0.0), hi));
    let e = _mm256_blendv_ps(_mm256_fnmadd_ps(e, lo, e), e, _mm256_cmp_ps::<_CMP_EQ_OQ>(e, s(f32::INFINITY)));
    _mm256_div_ps(v, _mm256_add_ps(s(1.0), e))
}

/// For 4 lanes, in f64 like `ferro_core::dispatch::erf_f32` (the A-S 7.1.26
/// polynomial's alternating terms cancel ~4x, which f32 Horner would amplify
/// into several ULP): h = p/2 with q = 1 - erf(|v|/sqrt 2) = p*e, and
/// g = v*c +- h, so that Phi(v) + v*phi(v) = e*g (v < 0) or 1 + e*g, with
/// the cancellation between the two terms happening here in f64.
#[target_feature(enable = "avx2,fma")]
fn erf_terms(v: __m128) -> (__m128, __m128) {
    let d = |k: f64| _mm256_set1_pd(k);
    let vd = _mm256_cvtps_pd(v);
    let ax = _mm256_mul_pd(_mm256_andnot_pd(d(-0.0), vd), d(std::f64::consts::FRAC_1_SQRT_2));
    let t = _mm256_div_pd(d(1.0), _mm256_fmadd_pd(d(0.3275911), ax, d(1.0)));
    let mut p = d(1.061405429);
    for k in [-1.453152027, 1.421413741, -0.284496736, 0.254829592] {
        p = _mm256_fmadd_pd(p, t, d(k));
    }
    let h = _mm256_mul_pd(_mm256_mul_pd(p, t), d(0.5));
    let neg = _mm256_cmp_pd::<_CMP_LT_OQ>(vd, d(0.0));
    let g = _mm256_add_pd(_mm256_mul_pd(vd, d(0.398_942_28f32 as f64)), _mm256_blendv_pd(_mm256_sub_pd(d(0.0), h), h, neg));
    (_mm256_cvtpd_ps(h), _mm256_cvtpd_ps(g))
}

/// (h, g, e) for all 8 lanes, e = e^(-v^2/2) with v^2's FMA residual folded
/// in so the exp argument is exact to f32 precision.
#[target_feature(enable = "avx2,fma")]
fn erf_parts(v: __m256) -> (__m256, __m256, __m256) {
    let (hl, gl) = erf_terms(_mm256_castps256_ps128(v));
    let (hh, gh) = erf_terms(_mm256_extractf128_ps::<1>(v));
    let vv = _mm256_mul_ps(v, v);
    // The residual is NaN once v*v overflows; e is 0 there anyway.
    let lo = _mm256_and_ps(_mm256_fmsub_ps(v, v, vv), _mm256_cmp_ps::<_CMP_LT_OQ>(vv, s(f32::INFINITY)));
    let e = exp(_mm256_mul_ps(s(-0.5), vv));
    let e = _mm256_fnmadd_ps(_mm256_mul_ps(s(0.5), lo), e, e);
    (_mm256_set_m128(hh, hl), _mm256_set_m128(gh, gl), e)
}

/// v * Phi(v), Phi formed as h*e or 1 - h*e so the left tail never cancels.
#[target_feature(enable = "avx2,fma")]
fn gelu_erf(v: __m256) -> __m256 {
    let (h, _, e) = erf_parts(v);
    let he = _mm256_mul_ps(h, e);
    let neg = _mm256_cmp_ps::<_CMP_LT_OQ>(v, s(0.0));
    _mm256_mul_ps(v, _mm256_blendv_ps(_mm256_sub_ps(s(1.0), he), he, neg))
}

#[target_feature(enable = "avx2,fma")]
fn gelu_erf_grad(v: __m256) -> __m256 {
    let (_, g, e) = erf_parts(v);
    let eg = _mm256_mul_ps(e, g);
    let r = _mm256_blendv_ps(_mm256_add_ps(s(1.0), eg), eg, _mm256_cmp_ps::<_CMP_LT_OQ>(v, s(0.0)));
    // The reference's v*phi(v) is inf*0 = NaN at v = +-inf; e*g loses that.
    _mm256_blendv_ps(r, s(f32::NAN), _mm256_cmp_ps::<_CMP_EQ_OQ>(_mm256_andnot_ps(s(-0.0), v), s(f32::INFINITY)))
}

/// Vectorized unary kinds; returns false (writing nothing) for kinds that
/// stay on the reference scalar formulas.
#[target_feature(enable = "avx2,fma")]
pub fn unary(kind: UnaryKind, x: &[f32], out: &mut [f32]) -> bool {
    match kind {
        UnaryKind::Exp => map8(x, out, |v| exp(v)),
        UnaryKind::Log => map8(x, out, |v| ln(v)),
        UnaryKind::Tanh => map8(x, out, |v| tanh(v)),
        UnaryKind::Sigmoid => map8(x, out, |v| logistic(s(1.0), v)),
        UnaryKind::Silu => map8(x, out, |v| logistic(v, v)),
        UnaryKind::Gelu => map8(x, out, |v| gelu(v)),
        UnaryKind::GeluErf => map8(x, out, |v| gelu_erf(v)),
        UnaryKind::GeluErfGrad => map8(x, out, |v| gelu_erf_grad(v)),
        _ => return false,
    }
    true
}

/// 8 lanes at a time; the tail goes through the same vector formula on a
/// zero-padded copy, so a value's result never depends on its position.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn map8(x: &[f32], out: &mut [f32], f: impl Fn(__m256) -> __m256) {
    assert_eq!(x.len(), out.len(), "unary output length must match input");
    let main = x.len() / 8 * 8;
    for (xc, oc) in x[..main].chunks_exact(8).zip(out[..main].chunks_exact_mut(8)) {
        // SAFETY: both chunks hold exactly 8 f32s.
        unsafe { _mm256_storeu_ps(oc.as_mut_ptr(), f(_mm256_loadu_ps(xc.as_ptr()))) };
    }
    if main < x.len() {
        let mut buf = [0f32; 8];
        let r = x.len() - main;
        buf[..r].copy_from_slice(&x[main..]);
        // SAFETY: buf holds 8 f32s.
        unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), f(_mm256_loadu_ps(buf.as_ptr()))) };
        out[main..].copy_from_slice(&buf[..r]);
    }
}

/// Full sum. Large inputs cut the tree at a fixed depth into independent
/// subtrees, sum those on the pool, then rebuild the top of the same tree
/// from their sums - the result does not depend on the thread count.
pub fn sum(x: &[f32]) -> f32 {
    if x.len() < PAR_SUM || workers::threads() == 1 {
        // SAFETY: callers detected avx2 and fma.
        return unsafe { psum(x) };
    }
    let depth = (4 * workers::threads()).next_power_of_two().trailing_zeros() as usize;
    let mut nodes = Vec::new();
    split(0, x.len(), depth, &mut nodes);
    let mut sums = vec![0f32; nodes.len()];
    par_split(&mut sums, 1, true, |i, o| {
        for (s, &(off, n)) in o.iter_mut().zip(&nodes[i..]) {
            // SAFETY: callers detected avx2 and fma.
            *s = unsafe { psum(&x[off..off + n]) };
        }
    });
    combine(x.len(), depth, &mut sums.iter())
}

/// Keepdim sum over `dim` of a contiguous buffer of `shape`.
pub fn sum_dim(x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let n = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    let mut out = ferro_core::pool::take_uninit(outer * inner);
    let par = x.len() >= PAR_SUM;
    if inner == 1 {
        par_split(&mut out, 1, par, |r0, o| {
            for (r, s) in (r0..).zip(o.iter_mut()) {
                // SAFETY: callers detected avx2 and fma.
                *s = unsafe { psum(&x[r * n..(r + 1) * n]) };
            }
        });
    } else {
        // SAFETY: callers detected avx2 and fma.
        par_split(&mut out, 8, par, |j0, o| unsafe { cols_sum(x, n, inner, j0, o) });
    }
    out
}

/// Appends the tree's nodes at `depth` (or leaves above it), left to right.
fn split(off: usize, n: usize, depth: usize, nodes: &mut Vec<(usize, usize)>) {
    if depth == 0 || n <= BASE {
        nodes.push((off, n));
        return;
    }
    split(off, n / 2, depth - 1, nodes);
    split(off + n / 2, n - n / 2, depth - 1, nodes);
}

/// Rebuilds the tree above the nodes `split` produced, consuming their sums.
fn combine(n: usize, depth: usize, sums: &mut std::slice::Iter<f32>) -> f32 {
    if depth == 0 || n <= BASE {
        return *sums.next().expect("one sum per split node");
    }
    let left = combine(n / 2, depth - 1, sums);
    left + combine(n - n / 2, depth - 1, sums)
}

/// Core's pairwise tree. Nodes of up to 4*BASE are evaluated as their (up to
/// four) leaves at once, so the leaves' dependent add chains interleave and
/// their lane combines share three hadds.
#[target_feature(enable = "avx2,fma")]
fn psum(x: &[f32]) -> f32 {
    let n = x.len();
    if n > 4 * BASE {
        return psum(&x[..n / 2]) + psum(&x[n / 2..]);
    }
    if n <= BASE {
        return leaves4([x, &[], &[], &[]])[0];
    }
    let (l, r) = x.split_at(n / 2);
    if n <= 2 * BASE {
        let s = leaves4([l, r, &[], &[]]);
        return s[0] + s[1];
    }
    fn halve(h: &[f32]) -> (&[f32], &[f32]) {
        if h.len() > BASE { h.split_at(h.len() / 2) } else { (h, &[]) }
    }
    let ((ll, lr), (rl, rr)) = (halve(l), halve(r));
    let s = leaves4([ll, lr, rl, rr]);
    let left = if lr.is_empty() { s[0] } else { s[0] + s[1] };
    left + if rr.is_empty() { s[2] } else { s[2] + s[3] }
}

/// reduce.rs's base case for up to four leaves of <= BASE elements (empty =
/// absent): lane l accumulates elements 8c+l in c order, lanes combine as
/// ((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7)), then the tail adds left to right.
#[target_feature(enable = "avx2,fma")]
fn leaves4(leaves: [&[f32]; 4]) -> [f32; 4] {
    static ZEROS: [f32; BASE] = [0.0; BASE];
    let src = leaves.map(|l| if l.is_empty() { &ZEROS[..] } else { l });
    let chunks = leaves.map(|l| l.len() / 8);
    let common = leaves.iter().filter(|l| !l.is_empty()).map(|l| l.len() / 8).min().unwrap_or(0);
    // SAFETY: c < common <= every non-empty leaf's chunk count, and absent
    // leaves read ZEROS, which holds BASE >= 8 * common elements.
    let ld = |j: usize, c: usize| unsafe { _mm256_loadu_ps(src[j].as_ptr().add(8 * c)) };
    let mut a = [_mm256_setzero_ps(); 4];
    for c in 0..common {
        a[0] = _mm256_add_ps(a[0], ld(0, c));
        a[1] = _mm256_add_ps(a[1], ld(1, c));
        a[2] = _mm256_add_ps(a[2], ld(2, c));
        a[3] = _mm256_add_ps(a[3], ld(3, c));
    }
    for j in 0..4 {
        for c in common..chunks[j] {
            a[j] = _mm256_add_ps(a[j], ld(j, c));
        }
    }
    // hadd pairs adjacent lanes, so two rounds then a 128-bit fold give
    // exactly the reference's per-leaf combine order, four leaves at a time.
    let h = _mm256_hadd_ps(_mm256_hadd_ps(a[0], a[1]), _mm256_hadd_ps(a[2], a[3]));
    let h = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps::<1>(h));
    let mut s = [0f32; 4];
    // SAFETY: s holds 4 f32s.
    unsafe { _mm_storeu_ps(s.as_mut_ptr(), h) };
    for (sum, l) in s.iter_mut().zip(leaves) {
        for &v in &l[l.len() / 8 * 8..] {
            *sum += v;
        }
    }
    s
}

/// Keepdim sums for flat output slots j0..j0+out.len() of an (outer, n,
/// inner) buffer, eight adjacent columns per vector; masked lanes cover a
/// row's last partial group.
#[target_feature(enable = "avx2,fma")]
fn cols_sum(x: &[f32], n: usize, inner: usize, j0: usize, out: &mut [f32]) {
    let mut j = 0;
    while j < out.len() {
        let (o, i) = ((j0 + j) / inner, (j0 + j) % inner);
        let lanes = 8.min(inner - i).min(out.len() - j);
        let mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(lanes as i32), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        let mut v = [0f32; 8];
        // SAFETY: v holds 8 f32s.
        unsafe { _mm256_storeu_ps(v.as_mut_ptr(), psum_cols(x, o * n * inner + i, n, inner, lanes, mask)) };
        out[j..j + lanes].copy_from_slice(&v[..lanes]);
        j += lanes;
    }
}

/// reduce.rs's strided pairwise tree for `lanes` adjacent columns at once.
#[target_feature(enable = "avx2,fma")]
fn psum_cols(x: &[f32], off: usize, n: usize, stride: usize, lanes: usize, mask: __m256i) -> __m256 {
    let ld = |i: usize| {
        assert!(i + lanes <= x.len());
        // SAFETY: lanes >= `lanes` are masked off and never touched, and the
        // pointer is formed with wrapping_add so it need not stay in bounds.
        unsafe { _mm256_maskload_ps(x.as_ptr().wrapping_add(i), mask) }
    };
    if n > BASE {
        let left = psum_cols(x, off, n / 2, stride, lanes, mask);
        return _mm256_add_ps(left, psum_cols(x, off + n / 2 * stride, n - n / 2, stride, lanes, mask));
    }
    let mut a = [_mm256_setzero_ps(); 8];
    for c in 0..n / 8 {
        for (l, acc) in a.iter_mut().enumerate() {
            *acc = _mm256_add_ps(*acc, ld(off + (8 * c + l) * stride));
        }
    }
    let lo = _mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3]));
    let hi = _mm256_add_ps(_mm256_add_ps(a[4], a[5]), _mm256_add_ps(a[6], a[7]));
    let mut acc = _mm256_add_ps(lo, hi);
    for k in n / 8 * 8..n {
        acc = _mm256_add_ps(acc, ld(off + k * stride));
    }
    acc
}

/// Row softmax/log_softmax over a contiguous rows x cols buffer.
pub fn softmax(x: &[f32], rows: usize, cols: usize, log: bool) -> Vec<f32> {
    let mut out = ferro_core::pool::take_uninit(rows * cols);
    if cols > 0 {
        // SAFETY: callers detected avx2 and fma.
        par_split(&mut out, cols, x.len() >= PAR_SOFTMAX, |i, o| unsafe { softmax_chunk(&x[i..i + o.len()], o, cols, log) });
    }
    out
}

/// CpuBackend's row algorithm (max, exp(x - max), the same pairwise sum, then
/// divide or subtract the log-sum-exp) with the vector exp. The divide is a
/// reciprocal multiply plus one FMA residual step, which rounds like y / sum
/// at a fraction of vdivps's cost.
#[target_feature(enable = "avx2,fma")]
fn softmax_chunk(x: &[f32], out: &mut [f32], cols: usize, log: bool) {
    let main = cols / 8 * 8;
    for (xr, yr) in x.chunks_exact(cols).zip(out.chunks_exact_mut(cols)) {
        let mut mv = s(f32::NEG_INFINITY);
        for c in xr[..main].chunks_exact(8) {
            // SAFETY: c holds 8 f32s.
            mv = _mm256_max_ps(mv, unsafe { _mm256_loadu_ps(c.as_ptr()) });
        }
        let mut l = [0f32; 8];
        // SAFETY: l holds 8 f32s.
        unsafe { _mm256_storeu_ps(l.as_mut_ptr(), mv) };
        let m = l.iter().chain(&xr[main..]).fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let mv = s(m);
        map8(xr, yr, |v| exp(_mm256_sub_ps(v, mv)));
        let sum = psum(yr);
        if log {
            let lse = m + sum.ln();
            for (y, &v) in yr.iter_mut().zip(xr) {
                *y = v - lse;
            }
            continue;
        }
        let (sv, inv) = (s(sum), s(1.0 / sum));
        let div = |y: __m256| {
            let q = _mm256_mul_ps(y, inv);
            _mm256_fmadd_ps(_mm256_fnmadd_ps(q, sv, y), inv, q)
        };
        for c in yr[..main].chunks_exact_mut(8) {
            // SAFETY: c holds 8 f32s.
            unsafe { _mm256_storeu_ps(c.as_mut_ptr(), div(_mm256_loadu_ps(c.as_ptr()))) };
        }
        for y in &mut yr[main..] {
            *y /= sum;
        }
    }
}
