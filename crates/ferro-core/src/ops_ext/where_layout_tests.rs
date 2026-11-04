use crate::{DType, Device, Tensor};
use crate::forward_ad::DualLevel;

struct Case {
    shapes: [&'static [usize]; 3], strides: [&'static [usize]; 3],
    output: &'static [usize], output_stride: &'static [usize],
    values: &'static [f64], booleans: &'static [bool],
}
struct MixedCase { base: usize, left: DType, right: DType, dtype: DType, stride: &'static [usize], values: &'static [f64] }
include!("where_layout_goldens.rs");

fn operand(shape: &[usize], strides: &[usize], dtype: DType, role: usize) -> Tensor {
    let count = if shape.contains(&0) { 3 } else { 4 + shape.iter().zip(strides).map(|(&d, &s)| (d - 1) * s).sum::<usize>() };
    let values = (0..count).map(|i| match role { 0 => if i % 3 == 0 { 1. } else { 0. }, 1 => (i % 7) as f64 - 3., _ => 8. - (i % 5) as f64 }).collect();
    let raw = Tensor::from_vec_f64(values, &[count]).unwrap().to_dtype(dtype);
    Tensor::from_parts(raw.0.storage.clone(), shape.to_vec(), strides.to_vec(), 2, Device::Cpu, false, None)
}

#[test]
fn actual_torch_typed_ternary_values_and_physical_layout_goldens() {
    for (index, case) in CASES.iter().enumerate() { for dtype in [DType::Bool, DType::I64, DType::F32, DType::F64] {
        let c = operand(case.shapes[0], case.strides[0], DType::Bool, 0);
        let a = operand(case.shapes[1], case.strides[1], dtype, 1);
        let b = operand(case.shapes[2], case.strides[2], dtype, 2);
        let output = Tensor::where_cond(&c, &a, &b).unwrap();
        assert_eq!(output.shape(), case.output, "case {index} {dtype:?}");
        assert_eq!(output.strides(), case.output_stride, "case {index} {dtype:?}");
        if dtype == DType::Bool { assert_eq!(output.to_vec_bool(), case.booleans, "case {index}"); }
        else { assert_eq!(output.to_vec_f64(), case.values, "case {index} {dtype:?}"); }
        assert_eq!(output.storage_offset(), 0);
        for input in [&c, &a, &b] { assert_ne!(output._storage_ptr(), input._storage_ptr()); }
    } }
}

#[test]
fn actual_torch_mixed_branch_empty_and_scalar_promotion_goldens() {
    for (index, mixed) in MIXED_CASES.iter().enumerate() {
        let case = &CASES[mixed.base];
        let c = operand(case.shapes[0], case.strides[0], DType::Bool, 0);
        let a = operand(case.shapes[1], case.strides[1], mixed.left, 1);
        let b = operand(case.shapes[2], case.strides[2], mixed.right, 2);
        let output = Tensor::where_cond(&c, &a, &b).unwrap();
        assert_eq!(output.dtype(), mixed.dtype, "mixed {index}");
        assert_eq!(output.strides(), mixed.stride, "mixed {index}");
        assert_eq!(output.to_vec_f64(), mixed.values, "mixed {index}");
    }
}

fn transpose(dtype: DType) -> Tensor {
    Tensor::from_vec_f64(vec![-2., -0.5, 0., 0.5, 1., 2.], &[2, 3]).unwrap().to_dtype(dtype).transpose(0, 1).unwrap()
}

#[test]
fn comparison_and_logical_masks_keep_real_iteration_layout() {
    for dtype in [DType::Bool, DType::I64, DType::F32, DType::F64] {
        let x = transpose(dtype);
        let zero = Tensor::scalar(0.).to_dtype(dtype);
        let y = x.ge_tensor(&zero).unwrap();
        let z = x.le_tensor(&Tensor::scalar(1.).to_dtype(dtype)).unwrap();
        for result in [&y, &z, &y.logical_and(&z).unwrap(), &y.logical_or(&z).unwrap(), &y.logical_xor(&z).unwrap()] {
            assert_eq!(result.strides(), &[1, 3]);
            assert_eq!(result.dtype(), DType::Bool);
            assert!(!result.requires_grad());
        }
        assert_eq!(y.to_vec_bool(), x.to_vec_f64().iter().map(|&v| v >= 0.).collect::<Vec<_>>());
        assert_eq!(y.logical_and(&z).unwrap().to_vec_bool(), x.to_vec_f64().iter().map(|&v| (0. ..=1.).contains(&v)).collect::<Vec<_>>());
    }
}

#[test]
fn ordinary_and_graph_selection_adjoints_use_condition_layout_and_keep_seed_edges() {
    for dtype in [DType::F32, DType::F64] { for graph in [false, true] {
        if dtype == DType::F64 && graph { continue; }
        let x = transpose(dtype).detach().requires_grad_(true).unwrap();
        let y = Tensor::ones(&[3, 2]).to_dtype(dtype).requires_grad_(true).unwrap();
        let c = x.ge_tensor(&Tensor::scalar(0.).to_dtype(dtype)).unwrap();
        let seed = Tensor::from_vec_f64(vec![f64::INFINITY, -0., f64::NAN, 1., 2., 3.], &[3, 2]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let output = Tensor::where_cond(&c, &x, &y).unwrap();
        let gradients = Tensor::vjp_many_wrt_optional(&[&output], &[&x, &y], &[&seed], graph).unwrap().into_iter().map(Option::unwrap).collect::<Vec<_>>();
        for (branch, gradient) in gradients.iter().enumerate() {
            assert_eq!(gradient.strides(), &[1, 3]);
            assert_eq!(gradient.requires_grad(), graph);
            for ((value, incoming), keep) in gradient.to_vec_f64().into_iter().zip(seed.to_vec_f64()).zip(c.to_vec_bool()) {
                let expected = if keep == (branch == 0) { incoming } else { 0. };
                assert!(value == expected || value.is_nan() && expected.is_nan());
                if expected == 0. { assert_eq!(value.is_sign_negative(), expected.is_sign_negative()); }
            }
        }
        if graph {
            assert!(!gradients[0].depends_on(&x));
            let mixed = gradients[0].sum().vjp_wrt_optional(&[&x, &y, &seed], &Tensor::scalar(1.).to_dtype(dtype), false).unwrap();
            assert!(mixed[0].is_none() && mixed[1].is_none());
            assert_eq!(mixed[2].as_ref().unwrap().to_vec_f64(), c.to_vec_bool().into_iter().map(|v| if v { 1. } else { 0. }).collect::<Vec<_>>());
        }
        assert!(x.grad().is_none() && y.grad().is_none() && seed.grad().is_none());
    } }
}

#[test]
fn clamp_direct_jvp_masks_nonfinite_directions_and_preserves_mixed_dtype_layout() {
    for wide in [false, true] { for same_layout in [false, true] { for value in [0.7, f64::INFINITY, f64::NAN, -0.] {
        let x = transpose(DType::F32).detach().requires_grad_(true).unwrap();
        let dtype = if wide { DType::F64 } else { DType::F32 };
        let direction = Tensor::from_vec_f64(vec![value; 6], if same_layout { &[2, 3] } else { &[3, 2] }).unwrap().to_dtype(dtype);
        let direction = if same_layout { direction.transpose(0, 1).unwrap() } else { direction };
        let direction = direction.requires_grad_(true).unwrap();
        let tangent_dtype = if wide && same_layout { DType::F64 } else { DType::F32 };
        let level = DualLevel::new(&[&x], &[&direction]).unwrap();
        let output = x.clamp(-0.5, 1.);
        let tangent = level.tangent(&output).unwrap().unwrap();
        drop(level);
        assert_eq!(tangent.strides(), &[1, 3]);
        assert_eq!(tangent.dtype(), tangent_dtype);
        for (got, input) in tangent.to_vec_f64().into_iter().zip(x.to_vec()) {
            let expected = if (-0.5..=1.).contains(&input) { if tangent_dtype == DType::F64 { value } else { value as f32 as f64 } } else { 0. };
            assert!(got == expected || got.is_nan() && expected.is_nan());
            if expected == 0. { assert_eq!(got.is_sign_negative(), expected.is_sign_negative()); }
        }
        let mixed = tangent.sum().vjp_wrt_optional(&[&x, &direction], &Tensor::scalar(1.).to_dtype(tangent_dtype), false).unwrap();
        assert!(mixed[0].is_none());
        assert_eq!(mixed[1].as_ref().unwrap().to_vec_f64(), vec![0., 1., 1., 1., 1., 0.]);
    } } }
}

#[test]
fn promoted_selection_preserves_cast_edges_and_empty_capacity() {
    let x = transpose(DType::F32).detach().requires_grad_(true).unwrap();
    let y = Tensor::ones(&[3, 2]).to_dtype(DType::F64).requires_grad_(true).unwrap();
    let c = x.ge_tensor(&Tensor::scalar(0.)).unwrap();
    let output = Tensor::where_cond(&c, &x, &y).unwrap();
    assert_eq!(output.dtype(), DType::F64); assert_eq!(output.strides(), &[1, 3]);
    let gradients = output.sum().grad_wrt(&[&x, &y], false).unwrap();
    assert_eq!(gradients[0].dtype(), DType::F32); assert_eq!(gradients[1].dtype(), DType::F64);
    assert_eq!(gradients[0].to_vec(), vec![0., 1., 0., 1., 1., 1.]);
    let huge = Tensor::scalar(1.).expand(&[1usize << 60, 1]).unwrap();
    let empty = Tensor::zeros(&[1, 0]).to_dtype(DType::F64);
    let condition = Tensor::from_vec_bool(vec![true], &[]).unwrap();
    let result = Tensor::where_cond(&condition, &huge, &empty).unwrap();
    assert_eq!(result.shape(), &[1usize << 60, 0]); assert_eq!(result.numel(), 0);
}

#[test]
fn efficient_zero_branches_materialize_real_where_values_without_losing_mask_layout() {
    let c = transpose(DType::F32).ge_tensor(&Tensor::scalar(0.)).unwrap();
    let zero = Tensor::structural_zero(&[3, 2], DType::F32, Device::Cpu).unwrap();
    let output = Tensor::where_cond(&c, &zero, &zero).unwrap();
    assert!(!output.is_structural_zero()); assert_eq!(output.strides(), &[1, 3]);
    assert_eq!(output.to_vec(), vec![0.; 6]);
}

#[test]
fn device_selection_remains_an_explicit_counted_host_fallback() {
    use std::any::Any;
    use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
    use crate::dispatch::{Backend, BinaryKind, DeviceBuffer, UnaryKind, register_backend};
    const DEV: Device = Device::Cuda(219);
    static SERIAL: Mutex<()> = Mutex::new(());
    struct Buffer(Vec<f32>);
    impl DeviceBuffer for Buffer {
        fn device(&self) -> Device { DEV }
        fn len(&self) -> usize { self.0.len() }
        fn as_any(&self) -> &dyn Any { self }
    }
    #[derive(Default)]
    struct Count { uploads: AtomicUsize, downloads: AtomicUsize }
    impl Backend for Count {
        fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("no host backend kernel") }
        fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("no host backend kernel") }
        fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("no host backend kernel") }
        fn alloc_from_host(&self, values: &[f32]) -> crate::Result<Box<dyn DeviceBuffer>> {
            self.uploads.fetch_add(1, Ordering::SeqCst); Ok(Box::new(Buffer(values.to_vec())))
        }
        fn copy_to_host(&self, buffer: &dyn DeviceBuffer) -> crate::Result<Vec<f32>> {
            self.downloads.fetch_add(1, Ordering::SeqCst); Ok(buffer.as_any().downcast_ref::<Buffer>().unwrap().0.clone())
        }
    }
    let _guard = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let backend = Arc::new(Count::default()); register_backend(DEV, backend.clone());
    let c = Tensor::from_vec(vec![1., 0., 1., 0., 1., 0.], &[2, 3]).unwrap().to_device(DEV).unwrap().transpose(0, 1).unwrap();
    let a = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap().to_device(DEV).unwrap().transpose(0, 1).unwrap().requires_grad_(true).unwrap();
    let b = Tensor::full(&[3, 2], -1.).to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let output = Tensor::where_cond(&c, &a, &b).unwrap();
    assert_eq!(output.device(), Device::Cpu); assert_eq!(output.strides(), &[2, 1]);
    assert_eq!(output.to_vec(), vec![1., -1., -1., 5., 3., -1.]);
    assert_eq!(backend.uploads.load(Ordering::SeqCst), 3);
    assert_eq!(backend.downloads.load(Ordering::SeqCst), 3);
    // The existing functional query contract requires one common device.
    assert!(Tensor::vjp_many_wrt_optional(&[&output], &[&a, &b], &[&Tensor::ones(&[3, 2])], false).is_err());
    assert_eq!(backend.downloads.load(Ordering::SeqCst), 3);
    output.sum().backward();
    let gradients = [a.grad(), b.grad()];
    // Publication owns a dense transpose for a; this backend has no device
    // materializer, so that layout copy adds one explicit download/upload.
    assert_eq!(backend.uploads.load(Ordering::SeqCst), 6);
    assert_eq!(backend.downloads.load(Ordering::SeqCst), 6);
    for gradient in gradients.iter().flatten() { assert_eq!(gradient.device(), DEV); }
    assert_eq!(gradients[0].as_ref().unwrap().strides(), &[1, 3]);
    assert_eq!(gradients[1].as_ref().unwrap().strides(), &[2, 1]);
    assert_eq!(gradients[0].as_ref().unwrap().storage_offset(), 0);
    assert_eq!(gradients[1].as_ref().unwrap().storage_offset(), 0);
    assert_ne!(gradients[0].as_ref().unwrap()._storage_ptr(), a._storage_ptr());
    assert_ne!(gradients[1].as_ref().unwrap()._storage_ptr(), b._storage_ptr());
    assert_eq!(gradients[0].as_ref().unwrap().to_vec(), vec![1., 0., 0., 1., 1., 0.]);
    let downloads = backend.downloads.load(Ordering::SeqCst);
    assert!(Tensor::where_cond(&Tensor::scalar(1.), &a, &b).is_err());
    assert_eq!(backend.downloads.load(Ordering::SeqCst), downloads);
}

#[test]
fn read_width_capacity_rejects_before_truth_conversion_or_promotion_materializes() {
    let count = isize::MAX as usize / 8 + 1;
    let wide_condition = Tensor::from_vec_f64(vec![1.], &[]).unwrap().expand(&[count]).unwrap();
    let boolean = Tensor::from_vec_bool(vec![true], &[]).unwrap();
    assert!(Tensor::where_cond(&wide_condition, &boolean, &boolean).is_err());
    let narrow = Tensor::scalar(1.).expand(&[count]).unwrap();
    assert!(narrow.logical_and(&Tensor::scalar(1.)).is_err());
    let integers = Tensor::from_vec_i64(vec![1], &[]).unwrap().expand(&[count]).unwrap();
    assert!(integers.ge_tensor(&narrow).is_err());
    let empty = Tensor::zeros(&[1, 0]);
    assert_eq!(Tensor::scalar(1.).expand(&[count, 1]).unwrap().logical_and(&empty).unwrap().shape(), &[count, 0]);
}
