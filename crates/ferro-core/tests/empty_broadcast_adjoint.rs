use ferro_core::{DType, Tensor};
use ferro_core::forward_ad::DualLevel;
use ferro_core::ops_ext::add_alpha::Alpha;

fn empty(dtype: DType) -> Tensor {
    Tensor::from_vec_f64(Vec::new(), &[0, 1usize << 60]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap()
}

fn scalar_matrix(dtype: DType) -> Tensor {
    Tensor::from_vec_f64(vec![2.], &[1, 1]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap()
}

fn operation(a: &Tensor, b: &Tensor, kind: usize) -> Tensor {
    match kind {
        0 => a.mul(b).unwrap(),
        1 => a.add(b).unwrap(),
        2 => a.add_alpha(b, Alpha::F64(2.5)).unwrap(),
        _ => a.sub_alpha(b, Alpha::F64(2.5)).unwrap(),
    }
}

#[test]
fn huge_empty_ordinary_and_functional_reductions_do_not_allocate_intermediate_axes() {
    for dtype in [DType::F32, DType::F64] { for kind in 0..4 { for functional in [false, true] {
        let a = empty(dtype);
        let b = scalar_matrix(dtype);
        let output = operation(&a, &b, kind);
        let gradients = if functional { output.sum().grad_wrt(&[&a, &b], false).unwrap() }
            else { output.sum().backward(); vec![a.grad().unwrap(), b.grad().unwrap()] };
        assert_eq!(gradients[0].shape(), a.shape());
        assert_eq!(gradients[1].shape(), b.shape());
        assert_eq!(gradients[0].dtype(), dtype);
        assert_eq!(gradients[1].to_dtype(DType::F64).to_vec_f64(), vec![0.]);
        if functional { assert!(a.grad().is_none() && b.grad().is_none()); }
    } } }
}

#[test]
fn empty_graph_reductions_keep_live_seed_and_exact_shape_identity() {
    for kind in 0..4 {
        let a = empty(DType::F32);
        let b = scalar_matrix(DType::F32);
        let seed = empty(DType::F32);
        let output = operation(&a, &b, kind);
        let first = output.vjp_wrt(&[&a, &b], &seed, true).unwrap();
        assert_eq!(first[0].shape(), a.shape());
        assert!(first[1].requires_grad());
        assert_eq!(first[1].to_vec(), vec![0.]);
        if kind != 0 { assert_eq!(first[0].id(), seed.id()); }
        let second = first[1].sum().grad_wrt(&[&seed, &a], true).unwrap();
        assert_eq!(second[0].shape(), seed.shape());
        assert_eq!(second[1].shape(), a.shape());
        assert!(a.grad().is_none() && b.grad().is_none() && seed.grad().is_none());
    }
    for graph in [false, true] {
        let a = Tensor::zeros(&[0, 2, 7]).requires_grad_(true).unwrap();
        let b = Tensor::zeros(&[0, 1, 7]).requires_grad_(true).unwrap();
        let seed = Tensor::zeros(a.shape()).requires_grad_(true).unwrap();
        let output = a.add(&b).unwrap();
        let gradients = if graph { output.vjp_wrt(&[&a, &b], &seed, true).unwrap() }
            else { output.vjp_wrt_preserve(&[&a, &b], &seed).unwrap() };
        assert_eq!(gradients[0].id(), seed.id());
        assert_eq!(gradients[1].shape(), b.shape());
        assert_eq!(gradients[1].requires_grad(), graph);
        assert_eq!(gradients[1].is_leaf(), !graph);
    }
}

#[test]
fn empty_forward_duals_and_reverse_over_dual_use_bounded_reductions() {
    for kind in 0..4 {
        let a = empty(DType::F32);
        let b = scalar_matrix(DType::F32);
        let va = empty(DType::F32);
        let vb = scalar_matrix(DType::F32);
        let level = DualLevel::new(&[&a, &b], &[&va, &vb]).unwrap();
        let output = operation(&a, &b, kind);
        let tangent = level.tangent(&output).unwrap().unwrap();
        drop(level);
        assert_eq!(tangent.shape(), a.shape());
        let gradients = tangent.sum().grad_wrt(&[&b, &vb], false).unwrap();
        for value in gradients { assert_eq!(value.shape(), &[1, 1]); assert_eq!(value.to_vec(), vec![0.]); }
        assert!(a.grad().is_none() && b.grad().is_none() && va.grad().is_none() && vb.grad().is_none());
    }
}
