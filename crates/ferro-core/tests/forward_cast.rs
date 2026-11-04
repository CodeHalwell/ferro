use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::DualLevel;
use ferro_core::{DType, Tensor};

#[test]
fn allocating_primal_cast_also_copies_tangents_and_obeys_reverse_mode() {
    for enabled in [false, true] {
        for live in [false, true] {
            for dtype in [DType::F32, DType::F64] {
                for output_dtype in [DType::F32, DType::F64] {
                    let x = Tensor::from_vec(vec![0.2, -0.4], &[2]).unwrap().requires_grad_(true).unwrap();
                    let v = Tensor::from_vec(vec![0.3, 0.7], &[2]).unwrap().to_dtype_autograd(dtype, false).unwrap().requires_grad_(live).unwrap();
                    let level = DualLevel::functional(&[&x], &[&v]).unwrap();
                    let output = with_grad_enabled(enabled, || x.to_dtype_autograd(output_dtype, true).unwrap());
                    let tangent = level.tangent(&output).unwrap().unwrap();
                    drop(level);
                    assert_eq!(tangent.dtype(), output_dtype);
                    assert_eq!(tangent.requires_grad(), enabled && live);
                    assert_ne!(tangent.id(), v.id());
                    assert_ne!(tangent._storage_ptr(), v._storage_ptr());
                    assert_eq!(tangent.to_vec_f64(), v.to_vec_f64());
                    if tangent.requires_grad() {
                        let gradient = tangent.sum().grad_wrt(&[&v], false).unwrap().remove(0);
                        assert_eq!(gradient.dtype(), dtype);
                        assert_eq!(gradient.to_vec_f64(), vec![1., 1.]);
                    }
                    assert!(x.grad().is_none() && v.grad().is_none());
                }
            }
        }
    }
}

#[test]
fn copied_cast_tangents_materialize_symbolic_zero_including_empty_columns() {
    for shape in [vec![2], vec![0, 2]] {
        let x = Tensor::zeros(&shape).requires_grad_(true).unwrap();
        let seed = Tensor::ones(&shape);
        let level = DualLevel::functional(&[&x], &[&seed]).unwrap();
        let product = x.mul(&Tensor::scalar(2.)).unwrap();
        let direction = level.tangent(&product).unwrap().unwrap();
        drop(level);
        let v = direction.sum().grad_wrt(&[&x], false).unwrap().remove(0);
        assert!(v.is_structural_zero());
        let ordinary = Tensor::ones(&shape).to_dtype_autograd(DType::F64, false).unwrap();
        for dtype in [DType::F32, DType::F64] {
            let level = DualLevel::functional_columns(&[&x], &[&[&v], &[&ordinary]]).unwrap();
            let output = x.to_dtype_autograd(dtype, true).unwrap();
            let tangents = level.tangents(&output).unwrap().unwrap();
            drop(level);
            for (tangent, source) in tangents.iter().zip([&v, &ordinary]) {
                assert_eq!(tangent.dtype(), dtype);
                assert_eq!(tangent.shape(), shape);
                assert!(!tangent.is_structural_zero());
                assert_eq!(tangent.to_vec_f64(), source.to_vec_f64());
                assert_ne!(tangent.id(), source.id());
            }
        }
    }
}
