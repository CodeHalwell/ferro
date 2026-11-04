use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::DualLevel;

fn activation(x: &Tensor, kind: usize) -> Tensor { match kind { 0 => x.exp(), 1 => x.sigmoid(), _ => x.tanh() } }
fn leaf(values: &[f64], dtype: DType) -> Tensor {
    Tensor::from_vec_f64(values.to_vec(), &[values.len()]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap()
}
fn exact(a: f64, b: f64) { assert!(a.is_nan() && b.is_nan() || a.to_bits() == b.to_bits(), "{a:?} != {b:?}"); }
fn stage(kind: usize, g: f64, y: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 { let (g, y) = (g as f32, y as f32); return match kind { 0 => g*y, 1 => (g*(1.-y))*y, _ => g*(-y).mul_add(y,1.) } as f64; }
    match kind { 0 => g*y, 1 => (g*(1.-y))*y, _ => g*(-y).mul_add(y,1.) }
}

#[test]
fn activation_adjoint_stages_match_saved_output_at_nonfinite_and_subnormal_values() {
    for dtype in [DType::F32, DType::F64] { for kind in 0..3 {
        let tiny = if dtype == DType::F32 { f32::from_bits(1) as f64 } else { f64::from_bits(1) };
        let x = leaf(&[-f64::INFINITY, -30., -0.7, -0., 0.2, 2., 30., f64::INFINITY, f64::NAN], dtype);
        let output = activation(&x, kind);
        for seed in [0., -0., tiny, -tiny, 0.7, f64::INFINITY, -f64::INFINITY, f64::NAN] {
            let g = Tensor::from_vec_f64(vec![seed; 9], &[9]).unwrap().to_dtype(dtype);
            for graph in [false, true].into_iter().filter(|graph| !graph || dtype == DType::F32) {
                let actual = output.vjp_wrt(&[&x], &g, graph).unwrap().remove(0);
                for ((actual, y), g) in actual.to_vec_f64().into_iter().zip(output.to_vec_f64()).zip(g.to_vec_f64()) {
                    exact(actual, stage(kind, g, y, dtype));
                }
            }
        }
        assert!(x.grad().is_none());
    } }
}

#[test]
fn activation_input_changes_leave_old_output_derivatives_live_and_output_changes_reject() {
    for dtype in [DType::F32, DType::F64] { for kind in 0..3 {
        let x = leaf(&[0.2, -0.7], dtype);
        let output = activation(&x, kind);
        let before = output.sum().grad_wrt(&[&x], false).unwrap().remove(0);
        with_grad_enabled(false, || x.copy_leaf_from(&Tensor::full(&[2], 8.).to_dtype(dtype))).unwrap();
        let after = output.sum().grad_wrt(&[&x], false).unwrap().remove(0);
        assert_eq!(before.to_vec_f64(), after.to_vec_f64());
        if dtype == DType::F32 {
            let first = output.sum().grad_wrt(&[&x], true).unwrap().remove(0);
            assert_eq!(before.to_vec(), first.to_vec());
            let second = first.sum().grad_wrt(&[&x], true).unwrap().remove(0);
            let third = second.sum().grad_wrt(&[&x], false).unwrap().remove(0);
            let y = output.to_vec();
            for (i, (&second, third)) in second.to_vec().iter().zip(third.to_vec()).enumerate() {
                let (d2, d3) = match kind {
                    0 => (y[i], y[i]),
                    1 => { let d = y[i]*(1.-y[i]); (d*(1.-2.*y[i]), d*(1.-6.*y[i]+6.*y[i]*y[i])) },
                    _ => { let d = 1.-y[i]*y[i]; (-2.*y[i]*d, (6.*y[i]*y[i]-2.)*d) },
                };
                assert!((second-d2).abs()<3e-7, "second {kind} {second} {d2}");
                assert!((third-d3).abs()<5e-7, "third {kind} {third} {d3}");
            }
        }
        output.sum().backward();
        assert_eq!(x.grad().unwrap().to_vec_f64(), before.to_vec_f64());
        output.detach().copy_from(&Tensor::zeros(&[2]).to_dtype(dtype)).unwrap();
        assert!(output.sum().grad_wrt(&[&x], false).is_err());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| output.sum().backward())).is_err());
        assert_eq!(x.grad().unwrap().to_vec_f64(), before.to_vec_f64());
    } }
}

#[test]
fn direct_activation_jvp_promotes_saved_output_before_kernel_and_preserves_reverse_history() {
    for kind in 0..3 { for direction_dtype in [DType::F32, DType::F64] { for enabled in [false, true] {
        let x = leaf(&[0.2, -0.7, 30.], DType::F32);
        let v = leaf(&[0.7, -0.3, 1e-30], direction_dtype);
        let (out, tangent) = with_grad_enabled(enabled, || {
            let level = DualLevel::new(&[&x], &[&v]).unwrap();
            let out = activation(&x, kind);
            let tangent = level.tangent(&out).unwrap().unwrap();
            (out, tangent)
        });
        assert_eq!(tangent.dtype(), direction_dtype);
        assert_eq!(out.requires_grad(), enabled);
        assert_eq!(tangent.requires_grad(), enabled);
        for ((actual, y), g) in tangent.to_vec_f64().into_iter().zip(out.to_vec_f64()).zip(v.to_vec_f64()) { exact(actual, stage(kind, g, y, direction_dtype)); }
        if enabled {
            let derivatives = tangent.sum().grad_wrt(&[&x, &v], false).unwrap();
            assert_eq!(derivatives[0].dtype(), DType::F32);
            assert_eq!(derivatives[1].dtype(), direction_dtype);
            out.detach().copy_from(&Tensor::zeros(&[3])).unwrap();
            assert!(tangent.sum().grad_wrt(&[&x, &v], false).is_err());
        }
        assert!(x.grad().is_none() && v.grad().is_none());
    } } }
}

#[test]
fn activations_cover_empty_offset_strides_inference_replay_and_gradcheck() {
    for dtype in [DType::F32, DType::F64] { for kind in 0..3 {
        let x = leaf(&[0.2, -0.4, 0.3, 0.7, -0.1, 0.6], dtype).reshape(&[2,3]).unwrap();
        let strided = x.transpose(0,1).unwrap();
        let out = activation(&strided, kind);
        out.sum().backward();
        assert_eq!(x.grad().unwrap().shape(), &[2,3]);
        let empty = Tensor::zeros(&[0,8]).to_dtype(dtype).requires_grad_(true).unwrap();
        activation(&empty.narrow(1,4,4).unwrap(), kind).sum().backward();
        assert_eq!(empty.grad().unwrap().shape(), &[0,8]);
        if dtype == DType::F32 {
            ferro_core::testkit::grad_check(&[strided.detach_copy()], |inputs| activation(&inputs[0], kind).sum());
            let source = Tensor::from_vec(vec![0.2,-0.4], &[2]).unwrap();
            let captured = ferro_core::capture(|| activation(&source, kind));
            let replay = ferro_core::graph::CompiledChain::compile(&captured).unwrap();
            source.fill_(0.7).unwrap();
            assert_eq!(replay.replay().unwrap().to_vec(), activation(&source, kind).to_vec());
        }
    } }
}
