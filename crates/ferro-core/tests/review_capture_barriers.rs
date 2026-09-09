use ferro_core::{capture, Tensor};
use ferro_core::graph::{CompiledChain, Graph};

#[test]
fn triangular_masks_remain_capture_barriers() {
    for upper in [true, false] {
        let x = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
        let y = capture(|| if upper { x.triu(0) } else { x.tril(0) }.unwrap().neg());
        x.fill_(9.).unwrap();
        assert!(Graph::from_root(&y).nodes.contains_key(&x.id()), "triangular input disappeared");
        assert!(CompiledChain::compile(&y).is_err(), "must not freeze a triangular intermediate");
    }
}

#[test]
fn residual_layernorm_remains_capture_barrier() {
    let x = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
    let r = Tensor::zeros(&[2, 2]);
    let y = capture(|| x.residual_layernorm(&r, None, None, 1e-5).unwrap().neg());
    x.fill_(9.).unwrap();
    assert!(Graph::from_root(&y).nodes.contains_key(&x.id()), "residual layernorm input disappeared");
    assert!(CompiledChain::compile(&y).is_err());
}
