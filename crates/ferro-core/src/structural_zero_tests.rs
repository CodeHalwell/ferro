use super::*;

fn zero(shape: &[usize], dtype: DType) -> Tensor { Tensor::structural_zero(shape, dtype, Device::Cpu).unwrap() }

#[test]
fn transposed_symbolic_zero_unary_and_explicit_casts_materialize_contiguous_storage() {
    for dtype in [DType::F32,DType::F64] {
        let original=zero(&[2,3],dtype);let view=original.transpose(0,1).unwrap();
        assert_eq!(view.strides(),&[1,3]);assert!(view.is_structural_zero());
        for output in [view.neg(),view.exp(),view.sigmoid(),view.tanh(),view.sqrt(),view.relu(),view.sign().unwrap(),view.to_dtype_autograd(dtype,true).unwrap(),view.to_dtype_autograd(if dtype==DType::F32{DType::F64}else{DType::F32},false).unwrap()] {
            assert_eq!(output.strides(),&[2,1]);assert!(!output.is_structural_zero());
            assert_ne!(output._storage_ptr(),original._storage_ptr());
        }
        assert!(view.neg().to_vec_f64().into_iter().all(|v|v.to_bits()==(-0.0f64).to_bits()));
        let copied=view.clone_tensor(true).unwrap();assert_eq!(copied.strides(),&[1,3]);assert!(!copied.is_structural_zero());
        assert_eq!(view.to_dtype_autograd(dtype,false).unwrap().id(),view.id());
        assert!(view.is_structural_zero()&&view.to_vec_f64().iter().all(|&v|v==0.));
    }
}
fn ordinary(values: &[f64], shape: &[usize], dtype: DType) -> Tensor {
    Tensor::from_vec_f64(values.to_vec(), shape).unwrap().to_dtype(dtype)
}

#[test]
fn special_binary_values_match_efficient_zero_rules_without_numeric_zero_detection() {
    for dtype in [DType::F32, DType::F64] {
        let z = zero(&[6], dtype);
        let values = [-2., -0., 0., f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let x = ordinary(&values, &[6], dtype);
        for out in [z.mul(&x).unwrap(), x.mul(&z).unwrap(), z.div(&x).unwrap(), z.add(&z).unwrap(), z.sub(&z).unwrap()] {
            assert!(out.is_structural_zero());
            assert_eq!(out.to_vec_f64(), vec![0.; 6]);
        }
        let both = z.div(&z).unwrap();
        assert!(!both.is_structural_zero());
        assert!(both.to_vec_f64().iter().all(|v| v.is_nan()));
        let denominator = x.div(&z).unwrap();
        assert!(!denominator.is_structural_zero());
        assert_eq!(denominator.to_vec_f64(), vec![f64::INFINITY; 6]);
        for out in [z.add(&x).unwrap(), x.add(&z).unwrap(), x.sub(&z).unwrap()] {
            assert!(!out.is_structural_zero());
            for (actual, expected) in out.to_vec_f64().iter().zip(values) {
                assert!(actual.is_nan() && expected.is_nan() || actual.to_bits() == expected.to_bits());
            }
        }
        let negative = z.sub(&x).unwrap().to_vec_f64();
        assert!(negative[1].is_sign_positive() && negative[2].is_sign_negative());
        assert_eq!(negative[0], 2.);
        let normal_zero = ordinary(&[0.; 6], &[6], dtype);
        assert!(!normal_zero.mul(&x).unwrap().is_structural_zero());
        assert!(normal_zero.mul(&x).unwrap().to_vec_f64()[3].is_nan());
    }
}

#[test]
fn views_snapshots_and_all_mutation_seams_keep_symbolic_storage_immutable() {
    for dtype in [DType::F32, DType::F64] {
        let z = zero(&[2, 2], dtype);
        let aliases = [z.detach(), z.transpose(0, 1).unwrap(), z.reshape(&[4]).unwrap(), z.narrow(0, 0, 1).unwrap()];
        for alias in aliases {
            assert!(alias.is_structural_zero());
            assert_eq!(alias._storage_ptr(), z._storage_ptr());
            assert!(alias.copy_from(&Tensor::ones(alias.shape()).to_dtype(dtype)).is_err());
        }
        assert!(z.copy_from(&z).is_err());
        assert!(z.zero_().is_err());
        assert!(z.fill_typed_(if dtype == DType::F32 { crate::inplace::FillValue::F32(0.) } else { crate::inplace::FillValue::F64(0.) }).is_err());
        assert!(crate::inplace::raw_copy_("zero_test", &z, &z).is_err());
        if dtype == DType::F32 {
            assert!(crate::inplace::raw_fill_("zero_test", &z, 0.).is_err());
            assert!(crate::inplace::raw_affine_("zero_test", &z, 1., 0.).is_err());
            assert!(crate::inplace::raw_binary_("zero_test", BinaryKind::Add, &z, &z).is_err());
        }
        let snapshot = z.detach_copy();
        assert!(snapshot.is_structural_zero());
        assert_ne!(snapshot._storage_ptr(), z._storage_ptr());
        let cloned = z.clone_tensor(true).unwrap();
        assert!(!cloned.is_structural_zero());
        cloned.copy_from(&Tensor::ones(z.shape()).to_dtype(dtype)).unwrap();
        assert_eq!(z.to_vec_f64(), vec![0.; 4]);
        assert_eq!(cloned.to_vec_f64(), vec![1.; 4]);
        assert_eq!(z._version(), 0);
    }
}

#[test]
fn unary_reductions_explicit_casts_and_matmul_materialize_like_torch() {
    for dtype in [DType::F32, DType::F64] {
        let z = zero(&[2], dtype);
        let negative = z.neg();
        assert!(!negative.is_structural_zero());
        assert!(negative.to_vec_f64().iter().all(|v| *v == 0. && v.is_sign_negative()));
        for out in [z.sum(), z.mean(), z.exp(), z.relu(), z.square().unwrap()] { assert!(!out.is_structural_zero()); }
        let empty = z.narrow(0, 0, 0).unwrap().mean();
        assert!(!empty.is_structural_zero() && empty.to_vec_f64()[0].is_nan());
        assert!(z.to_dtype_autograd(dtype, false).unwrap().is_structural_zero());
        assert!(!z.to_dtype_autograd(dtype, true).unwrap().is_structural_zero());
        assert!(!z.to_dtype_autograd(if dtype == DType::F32 { DType::F64 } else { DType::F32 }, false).unwrap().is_structural_zero());
        let matrix = z.reshape(&[1, 2]).unwrap().matmul(&ordinary(&[f64::INFINITY, f64::NAN], &[2, 1], dtype)).unwrap();
        assert!(!matrix.is_structural_zero() && matrix.to_vec_f64()[0].is_nan());
    }
}

#[test]
fn implicit_promotion_preserves_zero_dispatch_and_records_real_cast_adjoints() {
    for (zd, xd) in [(DType::F32, DType::F64), (DType::F64, DType::F32)] {
        for scalar in [false, true] {
            let z = zero(if scalar { &[] } else { &[2] }, zd);
            let x = ordinary(&[2., 3.], &[2], xd);
            z.set_requires_grad(true).unwrap();
            x.set_requires_grad(true).unwrap();
            let result = z.mul(&x).unwrap();
            let dtype = if scalar { xd } else { DType::F64 };
            assert_eq!(result.dtype(), dtype);
            assert!(result.is_structural_zero() && result.requires_grad());
            result.sum().backward();
            assert_eq!(z.grad().unwrap().dtype(), zd);
            assert_eq!(z.grad().unwrap().to_vec_f64(), if scalar { vec![5.] } else { vec![2., 3.] });
            assert_eq!(x.grad().unwrap().dtype(), xd);
            assert_eq!(x.grad().unwrap().to_vec_f64(), vec![0.; 2]);
        }
    }
}

#[test]
fn reverse_callbacks_keep_structural_zero_gradients_at_nonfinite_divisors() {
    let x = ordinary(&[0.2, -0.4], &[2], DType::F32).requires_grad_(true).unwrap();
    let v = ordinary(&[0.3, -0.7], &[2], DType::F32).requires_grad_(true).unwrap();
    let c = Tensor::scalar(0.).requires_grad_(true).unwrap();
    let absent = zero(&[], DType::F32);
    let tangent = v.sub(&absent.mul(&x.div(&c).unwrap()).unwrap()).unwrap().div(&c).unwrap();
    let gradient = tangent.sum().grad_wrt(&[&x, &v, &c], false).unwrap();
    assert_eq!(gradient[0].to_vec(), vec![0.; 2]);
    assert!(gradient[0].is_structural_zero());
    assert_eq!(gradient[1].to_vec(), vec![f32::INFINITY; 2]);
    assert!(gradient[2].to_vec()[0].is_nan());
    assert!(x.grad().is_none() && v.grad().is_none() && c.grad().is_none());
}

#[test]
fn invalid_shapes_devices_and_capture_reject_before_materialization() {
    assert!(Tensor::structural_zero(&[2], DType::Bool, Device::Cpu).is_err());
    assert!(Tensor::structural_zero(&[2], DType::F32, Device::Cuda(127)).is_err());
    let huge = zero(&[], DType::F64).expand(&[isize::MAX as usize / 8 + 1]).unwrap();
    let one = ordinary(&[1.], &[], DType::F64);
    assert!(huge.add(&one).is_err());
    assert!(huge.mul(&one).is_err());
    assert!(huge.clone_tensor(true).is_err());
    let z = zero(&[2], DType::F32);
    assert!(z.mul(&Tensor::ones(&[3])).is_err());
    assert!(crate::capture(|| z.mul(&Tensor::ones(&[2]))).is_err());
    assert!(!crate::capture::is_recording());
    assert!(crate::capture(|| Tensor::ones(&[2]).mul(&Tensor::ones(&[2]))).is_ok());
}

#[test]
fn parameter_adoption_rejects_immutable_zeros_and_accepts_explicit_owning_clones() {
    use crate::{Param, optim::Sgd};
    for requires_grad in [false, true] {
        let z = zero(&[2], DType::F32).requires_grad_(requires_grad).unwrap();
        let gradient = Tensor::full(&[2], 2.);
        z.set_leaf_grad(Some(gradient.clone())).unwrap();
        for shared in [false, true] {
            let rejected = if shared { Param::from_shared(&z, !requires_grad) } else { Param::from_leaf(&z) };
            assert!(rejected.is_err());
            assert!(rejected.err().unwrap().to_string().contains("clone before"));
            assert_eq!(z.requires_grad(), requires_grad);
            assert_eq!(z.grad().unwrap().id(), gradient.id());
            assert_eq!(z._version(), 0);
            let cloned = z.clone_tensor(true).unwrap().detach().requires_grad_(requires_grad).unwrap();
            assert!(!cloned.is_structural_zero());
            let parameter = if shared { Param::from_shared(&cloned, requires_grad).unwrap() } else { Param::from_leaf(&cloned).unwrap() };
            assert_eq!(parameter.tensor()._storage_ptr(), cloned._storage_ptr());
            assert_eq!(parameter.tensor().id() == cloned.id(), !shared);
            parameter.tensor().set_leaf_grad(Some(gradient.clone())).unwrap();
            Sgd::new(vec![parameter], 0.1).step();
            assert_eq!(cloned.to_vec(), vec![-0.2; 2]);
            assert_eq!(z.to_vec(), vec![0.; 2]);
            assert_eq!(z._version(), 0);
        }
    }
}

#[test]
fn late_immutable_restore_target_preserves_named_and_complete_training_state() {
    use crate::{checkpoint::Checkpoint, nn::Module, optim::Adam, Param, Rng};
    use std::cell::Cell;
    struct Model { weight: Param, buffer: Tensor, mode: Cell<u64> }
    impl Module for Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> { Ok(input.clone()) }
        fn named_parameters(&self) -> Vec<(String, Param)> { vec![("weight".into(), self.weight.clone())] }
        fn named_buffers(&self) -> Vec<(String, Tensor)> { vec![("late".into(), self.buffer.clone())] }
        fn snapshot_scalars(&self) -> Vec<(String, u64)> { vec![("mode".into(), self.mode.get())] }
        fn validate_scalars(&self, values: &[(String, u64)]) -> Result<()> {
            if values.len() != 1 || values[0].0 != "mode" { return Err(Error::Format { op: "zero_restore", msg: "mode schema mismatch".into() }); }
            Ok(())
        }
        fn commit_scalars(&self, values: &[(String, u64)]) { self.mode.set(values[0].1); }
    }
    let source = Model { weight: Param::new(Tensor::ones(&[2])), buffer: Tensor::ones(&[2]), mode: Cell::new(1) };
    let target = Model { weight: Param::new(Tensor::full(&[2], 9.)), buffer: zero(&[2], DType::F32), mode: Cell::new(7) };
    let named = Checkpoint::new(0).with_named_state(&[("first".into(), source.weight.tensor()), ("late".into(), source.buffer.clone())]).unwrap();
    let targets = [("first".into(), target.weight.tensor()), ("late".into(), target.buffer.clone())];
    assert!(named.load_named_state_into(&targets).is_err());
    assert_eq!(target.weight.tensor().to_vec(), vec![9.; 2]);
    assert_eq!(target.weight.tensor()._version(), 0); assert_eq!(target.buffer._version(), 0);
    let source_rng = Rng::new(7); let target_rng = Rng::new(11);
    let source_opt = Adam::new(source.parameters(), 0.01);
    let mut target_opt = Adam::new(target.parameters(), 0.03);
    target.weight.tensor().square().unwrap().sum().backward(); target_opt.step();
    target.weight.set_trainable(false); target.weight.tensor().set_requires_grad(false).unwrap();
    let snapshot = |model: &Model, optimizer: &Adam, rng: &Rng| {
        let checkpoint = Checkpoint::from_training_state(0, model, &[("main", optimizer)], rng).unwrap();
        crate::safetensors::to_safetensors_bytes(&checkpoint.tensors.iter().map(|(name, value)| (name.as_str(), value)).collect::<Vec<_>>()).unwrap()
    };
    let before = snapshot(&target, &target_opt, &target_rng);
    let handles = (target.weight.tensor().id(), target.weight.tensor()._version(), target.buffer.id(), target.buffer._version());
    let saved = Checkpoint::from_training_state(0, &source, &[("main", &source_opt)], &source_rng).unwrap();
    assert!(saved.load_training_state_into(&target, &mut [("main", &mut target_opt)], &target_rng).is_err());
    assert_eq!(snapshot(&target, &target_opt, &target_rng), before);
    assert_eq!((target.weight.tensor().id(), target.weight.tensor()._version(), target.buffer.id(), target.buffer._version()), handles);
    assert_eq!(target.mode.get(), 7);
    let control = Rng::new(0); control.restore_state(target_rng.state()).unwrap();
    assert_eq!(target_rng.normal(), control.normal());
}

#[test]
fn unary_adjoint_expressions_distinguish_efficient_zero_from_numeric_zero() {
    for dtype in [DType::F32,DType::F64] {
        let x=ordinary(&[-1.,-0.,0.,1.,f64::INFINITY,f64::NAN],&[2,3],dtype).transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
        let seed=zero(&[2,3],dtype).transpose(0,1).unwrap();
        for kind in 0..5 {
            let out=match kind {0=>x.sqrt(),1=>x.expm1().unwrap(),2=>x.exp2().unwrap(),3=>x.tan().unwrap(),_=>x.reciprocal().unwrap()};
            let grad=Tensor::vjp_many_wrt_optional(&[&out],&[&x],&[&seed],false).unwrap().remove(0).unwrap();
            assert_eq!(grad.is_structural_zero(),kind!=4);
            if kind!=4 {assert_eq!(grad.to_vec_f64(),vec![0.;6]);}
            else {assert!(grad.to_vec_f64().into_iter().any(f64::is_nan));assert_eq!(grad.strides(),&[2,1]);}
        }
    }
    for empty in [false,true] {
        let shape=if empty{vec![2,0]}else{vec![2,3]};
        let x=if empty{Tensor::from_vec_f64(vec![],&shape).unwrap()}else{ordinary(&[-1.,-0.,0.,1.,f64::INFINITY,f64::NAN],&shape,DType::F64)}.transpose(0,1).unwrap().requires_grad_(true).unwrap();
        let seed=zero(&shape,DType::F64).transpose(0,1).unwrap();
        let outputs=[x.clamp_f64(-0.5,0.5).unwrap(),x.gelu(),x.silu(),x.powf64(0.).unwrap()];
        for out in outputs {
            let grad=Tensor::vjp_many_wrt_optional(&[&out],&[&x],&[&seed],false).unwrap().remove(0).unwrap();
            assert!(!grad.is_structural_zero());
            if empty {assert_eq!(grad.strides(),&[2,1]);assert_eq!(grad.numel(),0);}
        }
    }
}
