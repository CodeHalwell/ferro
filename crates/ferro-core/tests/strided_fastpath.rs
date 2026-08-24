//! Bitwise-parity coverage for the strided/broadcast elementwise fast paths
//! in raw_unary_k/raw_binary_k (CAPABILITY.md 5.3, gate G8): a fast path must
//! never change *what* is computed, only how many bytes it moves. Every case
//! is checked bit-for-bit (`to_bits`) against a manually reproduced "old"
//! materialize-then-compute reference built only from the public API
//! (`broadcast` + `to_vec` + `CpuBackend`), so the comparison never
//! accidentally exercises the same new code path twice.

use ferro_core::testkit::grad_check;
use ferro_core::{Backend, BinaryKind, CpuBackend, Tensor, UnaryKind};

fn lcg_fill(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// Like `lcg_fill` but shifted away from zero, for safe division denominators.
fn lcg_fill_nonzero(seed: u64, n: usize) -> Vec<f32> {
    lcg_fill(seed, n)
        .into_iter()
        .map(|v| if v >= 0.0 { v + 0.2 } else { v - 0.2 })
        .collect()
}

fn numpy_broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
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
            // Mirrors shape.rs's broadcast_shapes: equal wins outright, else
            // whichever side isn't the 1 wins (a 0-size dim is never a 1, so
            // this stays correct for empty-tensor broadcasts too - unlike a
            // naive max(), which would wrongly turn a 0 vs 1 pairing into 1).
            if ad == bd {
                ad
            } else if ad == 1 {
                bd
            } else {
                ad
            }
        })
        .collect()
}

/// Reference broadcast materialization, reimplemented locally against plain
/// slices (mirrors what `Tensor::broadcast_to` + `to_vec` used to do before
/// this fast path existed).
fn broadcast_materialize(v: &[f32], shape: &[usize], out_shape: &[usize]) -> Vec<f32> {
    let pad = out_shape.len() - shape.len();
    let mut stride = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        stride[i] = acc;
        acc *= shape[i];
    }
    let bstride: Vec<usize> = (0..out_shape.len())
        .map(|i| {
            if i < pad || shape[i - pad] != out_shape[i] {
                0
            } else {
                stride[i - pad]
            }
        })
        .collect();
    let ndim = out_shape.len();
    let n: usize = out_shape.iter().product();
    let mut out = vec![0f32; n];
    let mut idx = vec![0usize; ndim];
    for o in out.iter_mut() {
        let off: usize = (0..ndim).map(|d| idx[d] * bstride[d]).sum();
        *o = v[off];
        for d in (0..ndim).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

fn reference_binary(
    kind: BinaryKind,
    a: &[f32],
    sa: &[usize],
    b: &[f32],
    sb: &[usize],
) -> Vec<f32> {
    let out_shape = numpy_broadcast_shape(sa, sb);
    let ba = broadcast_materialize(a, sa, &out_shape);
    let bb = broadcast_materialize(b, sb, &out_shape);
    CpuBackend.binary(kind, &ba, &bb)
}

fn run_binary(kind: BinaryKind, a: &Tensor, b: &Tensor) -> Tensor {
    match kind {
        BinaryKind::Add => a.add(b).unwrap(),
        BinaryKind::Sub => a.sub(b).unwrap(),
        BinaryKind::Mul => a.mul(b).unwrap(),
        BinaryKind::Div => a.div(b).unwrap(),
    }
}

fn assert_bitwise(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&x, &y)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "mismatch at {i}: {x} vs {y}");
    }
}

/// Runs add/mul/div on tensors built from `va`/`vb` and checks every result
/// bit-for-bit against `reference_binary`'s manual materialize-then-compute.
fn check_binary(a: &Tensor, va: &[f32], b: &Tensor, vb: &[f32]) {
    for kind in [BinaryKind::Add, BinaryKind::Mul, BinaryKind::Div] {
        let want = reference_binary(kind, va, a.shape(), vb, b.shape());
        let got = run_binary(kind, a, b).to_vec();
        assert_bitwise(&got, &want);
    }
}

fn check_unary(t: &Tensor) {
    let host = t.to_vec();
    assert_bitwise(
        &t.relu().to_vec(),
        &CpuBackend.unary(UnaryKind::Relu, &host),
    );
    assert_bitwise(&t.exp().to_vec(), &CpuBackend.unary(UnaryKind::Exp, &host));
}

#[test]
fn equal_shape_contiguous() {
    let (sa, sb) = (&[7usize, 5][..], &[7usize, 5][..]);
    let va = lcg_fill(1, 35);
    let vb = lcg_fill_nonzero(2, 35);
    let a = Tensor::from_vec(va.clone(), sa).unwrap();
    let b = Tensor::from_vec(vb.clone(), sb).unwrap();
    check_binary(&a, &va, &b, &vb);
    check_unary(&a);
}

#[test]
fn broadcast_bias_row() {
    // [n, m] + [m]
    let va = lcg_fill(3, 30);
    let vb = lcg_fill_nonzero(4, 5);
    let a = Tensor::from_vec(va.clone(), &[6, 5]).unwrap();
    let b = Tensor::from_vec(vb.clone(), &[5]).unwrap();
    check_binary(&a, &va, &b, &vb);
}

#[test]
fn broadcast_bias_row_keepdim() {
    // [n, m] + [1, m]
    let va = lcg_fill(5, 30);
    let vb = lcg_fill_nonzero(6, 5);
    let a = Tensor::from_vec(va.clone(), &[6, 5]).unwrap();
    let b = Tensor::from_vec(vb.clone(), &[1, 5]).unwrap();
    check_binary(&a, &va, &b, &vb);
}

#[test]
fn broadcast_column_times_row() {
    // [n, 1] + [1, m]: both operands broadcast, on different axes.
    let va = lcg_fill(7, 6);
    let vb = lcg_fill_nonzero(8, 5);
    let a = Tensor::from_vec(va.clone(), &[6, 1]).unwrap();
    let b = Tensor::from_vec(vb.clone(), &[1, 5]).unwrap();
    check_binary(&a, &va, &b, &vb);
}

#[test]
fn broadcast_scalar() {
    let va = lcg_fill(9, 30);
    let vb = lcg_fill_nonzero(10, 1);
    let a = Tensor::from_vec(va.clone(), &[6, 5]).unwrap();
    let b = Tensor::scalar(vb[0]);
    check_binary(&a, &va, &b, &vb);
}

#[test]
fn empty_tensors() {
    // Equal-shape empty.
    let a = Tensor::from_vec(vec![], &[3, 0]).unwrap();
    let b = Tensor::from_vec(vec![], &[3, 0]).unwrap();
    check_binary(&a, &[], &b, &[]);
    check_unary(&a);

    // Broadcast against an empty leading dim.
    let vb = lcg_fill_nonzero(11, 5);
    let a = Tensor::from_vec(vec![], &[0, 5]).unwrap();
    let b = Tensor::from_vec(vb.clone(), &[5]).unwrap();
    check_binary(&a, &[], &b, &vb);
}

#[test]
fn transpose_view_falls_back_and_stays_correct() {
    // A non-square transpose is never contiguous, so this must take the
    // materializing fallback path - checked for correctness, not speed.
    let va = lcg_fill(13, 12);
    let a = Tensor::from_vec(va, &[4, 3])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert!(a.shape() == [3, 4]);
    let vb = lcg_fill_nonzero(14, 12);
    let b = Tensor::from_vec(vb.clone(), &[3, 4]).unwrap();
    check_binary(&a, &a.to_vec(), &b, &vb);
    check_unary(&a);
}

#[test]
fn transpose_view_that_stays_contiguous() {
    // A transpose across a size-1 dim ([1,n] -> [n,1]) is still row-major
    // contiguous, so it shares storage with the original and takes the fast
    // path directly - the closest thing to an "offset view" reachable
    // through the public API today (ferro-core has no narrow/as_strided yet;
    // see docs/CAPABILITY.md 2.2). raw_unary_k/raw_binary_k's offset
    // arithmetic itself is exercised directly in tensor.rs's own
    // #[cfg(test)] unit tests, which can reach pub(crate) Tensor::from_parts.
    let va = lcg_fill(15, 6);
    let a = Tensor::from_vec(va, &[1, 6])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert!(a.shape() == [6, 1]);
    assert!(a.numel() == 6);

    let vb = lcg_fill_nonzero(16, 6);
    let b = Tensor::from_vec(vb.clone(), &[6, 1]).unwrap();
    check_binary(&a, &a.to_vec(), &b, &vb);
    check_unary(&a);

    // Reused in a broadcast pattern: [6,1] + [1,4].
    let vc = lcg_fill_nonzero(17, 4);
    let c = Tensor::from_vec(vc.clone(), &[1, 4]).unwrap();
    check_binary(&a, &a.to_vec(), &c, &vc);
}

#[test]
fn reshape_shares_storage() {
    let va = lcg_fill(19, 24);
    let a = Tensor::from_vec(va, &[2, 3, 4])
        .unwrap()
        .reshape(&[4, 6])
        .unwrap();
    let vb = lcg_fill_nonzero(20, 24);
    let b = Tensor::from_vec(vb.clone(), &[4, 6]).unwrap();
    check_binary(&a, &a.to_vec(), &b, &vb);
    check_unary(&a);

    // A reshaped tensor as the smaller side of a broadcast op.
    let vbias = lcg_fill_nonzero(21, 6);
    let bias = Tensor::from_vec(vbias.clone(), &[2, 3])
        .unwrap()
        .reshape(&[6])
        .unwrap();
    let vc = lcg_fill(22, 24);
    let c = Tensor::from_vec(vc.clone(), &[4, 6]).unwrap();
    check_binary(&c, &vc, &bias, &vbias);
}

#[test]
fn grad_check_bias_add_composition() {
    // x @ w + bias, then a smooth nonlinearity and a sum: exercises matmul's
    // own backward plus the broadcast-add and unary fast paths in the same
    // graph. exp (not relu) keeps the whole thing away from a
    // finite-difference kink, per CLAUDE.md's grad_check conventions.
    let x = Tensor::from_vec(lcg_fill(23, 3 * 4), &[3, 4]);
    let w = Tensor::from_vec(lcg_fill(29, 4 * 5), &[4, 5]);
    let bias = Tensor::from_vec(lcg_fill(31, 5), &[5]);
    grad_check(&[x.unwrap(), w.unwrap(), bias.unwrap()], |leaves| {
        leaves[0]
            .matmul(&leaves[1])
            .unwrap()
            .add(&leaves[2])
            .unwrap()
            .exp()
            .sum()
    });
}
