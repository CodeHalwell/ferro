use ferro_core::{Result, Tensor};

#[test]
fn leaf_accepts_requires_grad() {
    let x = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let x = x.requires_grad_(true).unwrap();
    assert!(x.requires_grad());
    // Same storage: values and device are preserved.
    assert_eq!(x.to_vec(), vec![1.0, 2.0]);
    assert_eq!(x.device(), ferro_core::Device::Cpu);
    assert_eq!(x.shape(), &[2]);
}

#[test]
fn interior_node_preserves_unchanged_flags_and_rejects_detaching_changes() {
    let a = Tensor::from_vec(vec![3.0, -2.0], &[2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let b = a.mul(&a).unwrap();
    assert!(
        b.requires_grad(),
        "op output of a requires-grad input is interior"
    );
    let same = b.requires_grad_(true).unwrap();
    assert_eq!(same.id(), b.id());
    let err = match b.requires_grad_(false) {
        Err(e) => e,
        Ok(_) => panic!("interior node accepted by requires_grad_(false)"),
    };
    let msg = format!("{err}");
    assert!(msg.contains("requires_grad_"), "unexpected error: {msg}");
    same.sum().backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![6.0, -4.0]);
    assert!(!b.detach_copy().requires_grad());
}

#[test]
fn unchanged_leaf_flags_preserve_existing_graph_identity_and_slots() {
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let leaf = Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap().to_dtype(dtype);
        assert_eq!(leaf.requires_grad_(false).unwrap().id(), leaf.id());
        let leaf = leaf.requires_grad_(true).unwrap();
        let loss = leaf.mul(&leaf).unwrap().sum();
        let same = leaf.requires_grad_(true).unwrap();
        assert_eq!(same.id(), leaf.id());
        loss.backward();
        let gradient_id = same.grad().unwrap().id();
        let again = same.requires_grad_(true).unwrap();
        assert_eq!(again.id(), leaf.id());
        assert_eq!(again.grad().unwrap().id(), gradient_id);
        loss.backward();
        assert_eq!(again.grad().unwrap().to_vec_f64(), vec![8.0, 12.0]);
        assert_eq!(leaf.grad().unwrap().to_vec_f64(), vec![8.0, 12.0]);
    }
}

#[test]
fn gradient_path_unaffected_for_leaves() {
    let w = Tensor::from_vec(vec![2.0, -1.0], &[2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let x = Tensor::from_vec(vec![1.0, 3.0], &[2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let loss = w.mul(&x).unwrap().sum();
    loss.backward();
    let (gw, gx) = (w.grad().expect("w grad"), x.grad().expect("x grad"));
    assert_eq!(gw.to_vec(), vec![1.0, 3.0]);
    assert_eq!(gx.to_vec(), vec![2.0, -1.0]);
    // Leaf grads accumulate across backward calls; marking again keeps that.
    loss.backward();
    assert_eq!(w.grad().unwrap().to_vec(), vec![2.0, 6.0]);
}

#[test]
fn integer_leaf_rejected_with_result_error() -> Result<()> {
    let t = Tensor::from_vec_i64(vec![1], &[1]).unwrap();
    let err = match t.requires_grad_(true) {
        Err(e) => e,
        Ok(_) => panic!("integer leaf accepted by requires_grad_(true)"),
    };
    assert!(format!("{err}").contains("f32"));
    Ok(())
}

#[test]
fn parameter_leaf_guards_do_not_depend_on_the_flag_setter() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let source = Tensor::ones(&[1, 1, 2]).requires_grad_(true).unwrap();
    let nonleaf = source.mul(&source).unwrap();
    assert_eq!(nonleaf.requires_grad_(true).unwrap().id(), nonleaf.id());
    assert!(catch_unwind(AssertUnwindSafe(|| ferro_core::Param::new(nonleaf.clone()))).is_err());
    let parameter = ferro_core::Param::new(source.clone());
    let before = parameter.tensor();
    assert!(catch_unwind(AssertUnwindSafe(|| parameter.set(nonleaf.clone()))).is_err());
    assert_eq!(parameter.tensor().id(), before.id());
    assert_eq!(parameter.tensor().to_vec(), before.to_vec());
    let basis = ferro_core::basis::BSplineBasis::new(vec![0., 0., 1., 1.], 1, ferro_core::basis::OutsideDomain::Zero).unwrap();
    assert!(ferro_core::basis::KanLayer::from_coefficients(basis, nonleaf).is_err());
    assert!(source.grad().is_none());
}

#[test]
fn gradient_checker_uses_fresh_leaf_slots_without_resetting_callers() {
    let source = Tensor::from_vec(vec![0.3, -0.4], &[2]).unwrap().requires_grad_(true).unwrap();
    source.mul(&source).unwrap().sum().backward();
    let before = source.grad().unwrap();
    ferro_core::testkit::grad_check(&[source.clone()], |inputs| inputs[0].mul(&inputs[0]).unwrap().sum());
    assert_eq!(source.grad().unwrap().id(), before.id());
    assert_eq!(source.grad().unwrap().to_vec(), before.to_vec());
}
