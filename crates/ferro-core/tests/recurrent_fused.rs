use ferro_core::recurrent::*;
use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn t(v: &[f32], shape: &[usize]) -> Tensor { Tensor::from_vec(v.to_vec(), shape).unwrap() }
fn values(n: usize, seed: usize) -> Vec<f32> { (0..n).map(|i| (((i * 7 + seed * 11) % 23) as f32 - 11.0) * 0.06).collect() }
fn leaf(n: usize, seed: usize, shape: &[usize]) -> Tensor { t(&values(n, seed), shape).requires_grad_(true).unwrap() }

fn composed(kind: CellKind, x: &Tensor, init: &RecurrentState, w: &[Tensor; 2], b: [Option<&Tensor>; 2], lengths: &[usize], trunc: Option<usize>) -> UnrollOutput {
    unroll(x, init, lengths, None, trunc, |x, s| Ok(match kind {
        CellKind::Rnn => RecurrentState { h: rnn_cell(x, &s.h, &w[0], &w[1], b[0], b[1])?, c: None },
        CellKind::Gru => RecurrentState { h: gru_cell(x, &s.h, &w[0], &w[1], b[0], b[1])?, c: None },
        CellKind::Lstm => { let (h, c) = lstm_cell(x, &s.h, s.c.as_ref().unwrap(), &w[0], &w[1], b[0], b[1])?; RecurrentState { h, c: Some(c) } }
    })).unwrap()
}

#[test]
fn fused_sequence_matches_composed_values_and_every_gradient() {
    let (time, batch, features, width) = (6, 3, 2, 3);
    for (kind, gates) in [(CellKind::Rnn, 1), (CellKind::Gru, 3), (CellKind::Lstm, 4)] {
        for (lengths, trunc, biased) in [([6, 6, 6], Some(2), true), ([6, 6, 6], None, false), ([6, 2, 4], None, true), ([6, 6, 6], Some(9), true)] {
            let x = leaf(time * batch * features, 1, &[time, batch, features]);
            let h0 = leaf(batch * width, 2, &[batch, width]);
            let c0 = (kind == CellKind::Lstm).then(|| leaf(batch * width, 3, &[batch, width]));
            let w = [leaf(gates * width * features, 4, &[gates * width, features]), leaf(gates * width * width, 5, &[gates * width, width])];
            let bias = [leaf(gates * width, 6, &[gates * width]), leaf(gates * width, 7, &[gates * width])];
            let b = if biased { [Some(&bias[0]), Some(&bias[1])] } else { [None, None] };
            let init = RecurrentState { h: h0.clone(), c: c0.clone() };
            let probes = [t(&values(time * batch * width, 8), &[time, batch, width]), t(&values(batch * width, 9), &[batch, width]), t(&values(batch * width, 10), &[batch, width])];
            let leaves = [x.clone(), h0.clone(), w[0].clone(), w[1].clone(), bias[0].clone(), bias[1].clone()].into_iter().chain(c0.clone()).collect::<Vec<_>>();
            let run = |fused: bool| {
                let out = if fused { unroll_cell(kind, &x, &init, &w[0], &w[1], b[0], b[1], &lengths, None, trunc).unwrap() } else { composed(kind, &x, &init, &w, b, &lengths, trunc) };
                let mut loss = out.outputs.mul(&probes[0]).unwrap().sum().add(&out.state.h.mul(&probes[1]).unwrap().sum()).unwrap();
                if let Some(c) = &out.state.c { loss = loss.add(&c.mul(&probes[2]).unwrap().sum()).unwrap(); }
                loss.backward();
                let grads = leaves.iter().map(|p| { let g = p.grad().map(|g| g.to_vec()); p.zero_grad(); g }).collect::<Vec<_>>();
                (out.outputs.to_vec(), out.state.h.to_vec(), out.state.c.map(|c| c.to_vec()), grads)
            };
            let (fused, reference) = (run(true), run(false));
            let label = format!("{kind:?} lengths={lengths:?} trunc={trunc:?} biased={biased}");
            let near = |a: &[f32], b: &[f32], what: &str| {
                assert_eq!(a.len(), b.len(), "{label} {what}");
                for (p, q) in a.iter().zip(b) { assert!((p - q).abs() < 2e-5, "{label} {what}: {p} != {q}"); }
            };
            near(&fused.0, &reference.0, "outputs"); near(&fused.1, &reference.1, "h");
            if let (Some(a), Some(b)) = (&fused.2, &reference.2) { near(a, b, "c"); }
            for (i, (a, b)) in fused.3.iter().zip(&reference.3).enumerate() {
                match (a, b) {
                    (Some(a), Some(b)) => near(a, b, &format!("grad {i}")),
                    (None, None) => {}
                    (a, b) => panic!("{label} grad {i} presence differs: fused {} composed {}", a.is_some(), b.is_some()),
                }
            }
        }
    }
}

#[test]
fn fused_lstm_gradients_match_finite_differences() {
    let (time, batch, features, width) = (3, 2, 2, 2);
    let inputs = [values(time * batch * features, 1), values(batch * width, 2), values(batch * width, 3), values(8 * features, 4), values(8 * width, 5), values(8, 6), values(8, 7)];
    let shapes: [&[usize]; 7] = [&[time, batch, features], &[batch, width], &[batch, width], &[8, features], &[8, width], &[8], &[8]];
    let leaves = inputs.iter().zip(shapes).map(|(v, s)| leaf(v.len(), 0, s).detach_copy().add(&t(v, s)).unwrap().requires_grad_(true).unwrap()).collect::<Vec<_>>();
    let probe = t(&values(time * batch * width, 8), &[time, batch, width]);
    grad_check(&leaves, |p| {
        let out = unroll_cell(CellKind::Lstm, &p[0], &RecurrentState { h: p[1].clone(), c: Some(p[2].clone()) }, &p[3], &p[4], Some(&p[5]), Some(&p[6]), &[3, 2], None, None).unwrap();
        out.outputs.mul(&probe).unwrap().sum().add(&out.state.c.unwrap().sum()).unwrap()
    });
}

#[test]
fn fused_sequence_is_first_order_and_records_without_grad_mode() {
    let x = leaf(2 * 1 * 2, 1, &[2, 1, 2]);
    let w = [leaf(4, 2, &[2, 2]), leaf(4, 3, &[2, 2])];
    let init = RecurrentState { h: t(&[0.1, -0.2], &[1, 2]), c: None };
    let out = unroll_cell(CellKind::Rnn, &x, &init, &w[0], &w[1], None, None, &[2], None, None).unwrap();
    assert!(out.outputs.sum().grad_wrt(&[&w[0]], true).is_err());
    let first = out.outputs.sum().grad_wrt(&[&w[0], &x], false).unwrap();
    assert!(first.iter().all(|g| g.to_vec().iter().all(|v| v.is_finite() && *v != 0.)));
    let frozen = ferro_core::autograd::with_grad_enabled(false, || unroll_cell(CellKind::Rnn, &x, &init, &w[0], &w[1], None, None, &[2], None, None).unwrap());
    assert!(!frozen.outputs.requires_grad());
    assert_eq!(frozen.outputs.to_vec(), out.outputs.to_vec());
}
