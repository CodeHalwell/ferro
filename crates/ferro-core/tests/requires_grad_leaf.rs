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
fn interior_node_is_rejected_not_silently_detached() {
    let a = Tensor::from_vec(vec![3.0, -2.0], &[2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let b = a.mul(&a).unwrap();
    assert!(
        b.requires_grad(),
        "op output of a requires-grad input is interior"
    );
    let err = match b.requires_grad_(true) {
        Err(e) => e,
        Ok(_) => panic!("interior node accepted by requires_grad_(true)"),
    };
    let msg = format!("{err}");
    assert!(msg.contains("requires_grad_"), "unexpected error: {msg}");
    // requires_grad_(false) stays legal on an interior node (dropping the flag
    // is not a graph cut the way re-marking would be).
    assert!(!b.detach_copy().requires_grad());
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
fn non_f32_leaf_rejected_with_result_error() -> Result<()> {
    let t = Tensor::from_vec_f64(vec![1.0], &[1]).unwrap();
    let err = match t.requires_grad_(true) {
        Err(e) => e,
        Ok(_) => panic!("non-f32 leaf accepted by requires_grad_(true)"),
    };
    assert!(format!("{err}").contains("f32"));
    Ok(())
}
