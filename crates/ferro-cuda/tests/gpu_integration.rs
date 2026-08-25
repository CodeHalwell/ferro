//! Real-GPU integration tests: drive ferro_core tensors through the CUDA
//! backend on Device::Cuda(0). Every test no-ops when no usable device is
//! present, so `cargo test -p ferro-cuda` stays green on CPU-only boxes.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use ferro_core::{dispatch::Backend, Tensor};
use ferro_cuda::CudaBackend;

const DEV: ferro_core::Device = ferro_core::Device::Cuda(0);

fn lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn setup() -> Option<(MutexGuard<'static, ()>, Arc<CudaBackend>)> {
    let guard = lock();
    if !ferro_cuda::is_available() {
        return None;
    }
    let b = match CudaBackend::new(0) {
        Ok(b) => Arc::new(b),
        Err(_) => return None,
    };
    ferro_core::dispatch::register_backend(DEV, b.clone());
    Some((guard, b))
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let scale = tol.max(tol * y.abs());
        assert!(
            (x - y).abs() <= scale,
            "elem {i}: got {x} want {y} (tol {scale})"
        );
    }
}

// n < block, n == block, n = block+1, and sizes large enough to exercise the
// grid-stride loop and the REDUCE_MAX_BLOCKS cap (non-multiples of both 256
// and 2048*256).
#[test]
fn reduce_dev_matches_cpu_across_sizes() {
    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    use ferro_core::dispatch::ReduceKind;
    for &n in &[1usize, 255, 256, 257, 100_000, 1_000_003] {
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
        let xd = b.alloc_from_host(&x).unwrap();
        let (sum, mean) = (x.iter().sum::<f32>(), x.iter().sum::<f32>() / n as f32);
        let rs = b.reduce_dev(ReduceKind::Sum, xd.as_ref()).unwrap();
        close(
            &b.copy_to_host(rs.as_ref()).unwrap(),
            &[sum],
            1e-4 * (n as f32).sqrt().max(1.0),
        );
        let rm = b.reduce_dev(ReduceKind::Mean, xd.as_ref()).unwrap();
        close(
            &b.copy_to_host(rm.as_ref()).unwrap(),
            &[mean],
            1e-4 * (n as f32).sqrt().max(1.0),
        );
    }
}

#[test]
fn sum_dim_and_fill_on_device_match_reference() {
    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    let shape = [4usize, 5, 3];
    let x: Vec<f32> = (0..60).map(|i| i as f32 * 0.25 - 7.0).collect();
    let xd = b.alloc_from_host(&x).unwrap();
    for dim in 0..3 {
        let rd = b.sum_dim_dev(xd.as_ref(), &shape, dim).unwrap();
        let got = b.copy_to_host(rd.as_ref()).unwrap();
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();
        let mut want = Vec::new();
        for o in 0..outer {
            for i in 0..inner {
                want.push(
                    (0..shape[dim])
                        .map(|k| x[(o * shape[dim] + k) * inner + i])
                        .sum::<f32>(),
                );
            }
        }
        close(&got, &want, 1e-3);
    }
    let fd = b.fill_dev(-2.5, 77).unwrap();
    close(
        &b.copy_to_host(fd.as_ref()).unwrap(),
        &vec![-2.5f32; 77],
        0.0,
    );
}

// Tensor-level ops end to end: the same expression evaluated on Device::Cuda(0)
// and on CPU must agree elementwise.
#[test]
fn tensor_ops_on_device_match_cpu() {
    let _g = match setup() {
        Some((g, _)) => g,
        None => return,
    };
    use ferro_core::Tensor;
    let x_data: Vec<f32> = [-1.5, -0.5, 0.0, 0.75, 2.0, -3.25].to_vec();
    let y_data: Vec<f32> = [0.5, 1.0, -2.0, 4.0, -0.25, 0.125].to_vec();

    let eval = |dev: Option<ferro_core::Device>| -> Vec<f32> {
        let place = |t: Tensor| match dev {
            Some(d) => t.to_device(d).unwrap(),
            None => t,
        };
        let x = place(Tensor::from_vec(x_data.clone(), &[2, 3]).unwrap());
        let y = place(Tensor::from_vec(y_data.clone(), &[2, 3]).unwrap());
        let out = x
            .relu()
            .sigmoid()
            .exp()
            .add(&y)
            .unwrap()
            .sub(&y.mul(&y).unwrap())
            .unwrap()
            .div(
                &y.sigmoid()
                    .mul(&y.sigmoid())
                    .unwrap()
                    .add(&place(Tensor::scalar(1.0)))
                    .unwrap(),
            )
            .unwrap();
        out.to_device(ferro_core::Device::Cpu).unwrap().to_vec()
    };
    close(&eval(Some(DEV)), &eval(None), 1e-5);
}

#[test]
fn matmul_on_device_match_cpu_all_transpose_flags() {
    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    let (m, k, n) = (5usize, 3usize, 7usize);
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5 - 4.0).collect();
    let bb: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.125 + 1.0).collect();
    let want = ferro_core::dispatch::naive_matmul(&a, &bb, m, k, n);
    let tr = |v: &[f32], r: usize, c: usize| {
        (0..c)
            .flat_map(|j| (0..r).map(move |i| v[i * c + j]))
            .collect::<Vec<f32>>()
    };
    for ta in [false, true] {
        for tb in [false, true] {
            let abuf = if ta { tr(&a, m, k) } else { a.clone() };
            let bbuf = if tb { tr(&bb, k, n) } else { bb.clone() };
            let ab = b.alloc_from_host(&abuf).unwrap();
            let xb = b.alloc_from_host(&bbuf).unwrap();
            let rd = b
                .matmul_dev(ab.as_ref(), xb.as_ref(), m, k, n, ta, tb)
                .unwrap();
            close(&b.copy_to_host(rd.as_ref()).unwrap(), &want, 1e-3);
        }
    }
}

// Backward pass fully resident on GPU: relu -> mul(transpose) -> exp -> mean;
// gradients must land on Device::Cuda(0) and equal the CPU reference.
#[test]
fn backward_on_device_gradients_match_cpu() {
    let _g = match setup() {
        Some((g, _)) => g,
        None => return,
    };
    use ferro_core::Tensor;
    let x_data: Vec<f32> = [-1.0, 2.0, -3.0, 4.0, 0.5, -0.25].to_vec();
    let w_data: Vec<f32> = [1.0, -0.5, 0.25, 2.0, 0.75, -1.5].to_vec();

    let run = |dev: Option<ferro_core::Device>| -> (Vec<f32>, Vec<f32>) {
        let place = |t: Tensor| match dev {
            Some(d) => t.to_device(d).unwrap(),
            None => t,
        };
        let xd = place(Tensor::from_vec(x_data.clone(), &[2, 3]).unwrap())
            .requires_grad_(true)
            .unwrap();
        let wd = place(Tensor::from_vec(w_data.clone(), &[3, 2]).unwrap())
            .requires_grad_(true)
            .unwrap();
        let loss = xd
            .relu()
            .mul(&wd.transpose(0, 1).unwrap())
            .unwrap()
            .exp()
            .mean();
        loss.backward();
        let gx = xd.grad().unwrap();
        let gw = wd.grad().unwrap();
        if let Some(d) = dev {
            assert_eq!(gx.device(), d);
            assert_eq!(gw.device(), d);
        }
        (
            gx.to_device(ferro_core::Device::Cpu).unwrap().to_vec(),
            gw.to_device(ferro_core::Device::Cpu).unwrap().to_vec(),
        )
    };
    let (gx_gpu, gw_gpu) = run(Some(DEV));
    let (gx_cpu, gw_cpu) = run(None);
    close(&gx_gpu, &gx_cpu, 1e-5);
    close(&gw_gpu, &gw_cpu, 1e-5);
}

#[test]
fn foreign_device_buffers_are_rejected() {
    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    struct OtherDev;
    impl ferro_core::dispatch::DeviceBuffer for OtherDev {
        fn device(&self) -> ferro_core::Device {
            ferro_core::Device::Cuda(1)
        }
        fn len(&self) -> usize {
            1
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
    use ferro_core::{dispatch::ReduceKind, UnaryKind};
    let od = OtherDev;
    assert!(b.unary_dev(UnaryKind::Relu, &od).is_err());
    assert!(b.reduce_dev(ReduceKind::Sum, &od).is_err());
    // An f32 buffer must not be accepted where i64 storage is expected.
    let fd = b.fill_dev(1.0, 4).unwrap();
    assert!(b.copy_i64_to_host(fd.as_ref()).is_err());
}

// I64 index tensors are device-resident: upload/download round trip is exact.
#[test]
fn i64_to_device_roundtrip_is_exact() {
    let _g = match setup() {
        Some((g, _)) => g,
        None => return,
    };
    use ferro_core::Tensor;
    let data: Vec<i64> = vec![0, 7, -3, 1_000_000_007, i64::MAX / 2];
    let t = Tensor::from_vec_i64(data.clone(), &[data.len()]).unwrap();
    let dev = t.to_device(DEV).unwrap();
    assert_eq!(dev.device(), DEV);
    assert_eq!(dev.dtype(), ferro_core::DType::I64);
    assert_eq!(dev.to_vec_i64(), data);
    let back = dev.to_device(ferro_core::Device::Cpu).unwrap();
    assert_eq!(back.to_vec_i64(), data);
}

// Embedding with both weight and ids resident on Device::Cuda(0): forward
// matches the CPU reference bitwise (pure row copies), and the backward
// scatter-add gradient (with duplicate ids) matches too.
#[test]
fn embedding_on_device_matches_cpu_forward_and_grad() {
    let _g = match setup() {
        Some((g, _)) => g,
        None => return,
    };
    use ferro_core::Tensor;
    let w_data: Vec<f32> = (0..6 * 4).map(|i| (i as f32 * 0.5 - 5.0).sin()).collect();
    let ids_data: Vec<i64> = vec![2, 0, 5, 2, 3]; // duplicate id 2 exercises scatter-add
    let run = |dev: Option<ferro_core::Device>| -> (Vec<f32>, Vec<i64>, Vec<f32>) {
        let place = |t: Tensor| match dev {
            Some(d) => t.to_device(d).unwrap(),
            None => t,
        };
        let w = place(Tensor::from_vec(w_data.clone(), &[6, 4]).unwrap())
            .requires_grad_(true)
            .unwrap();
        let ids = place(Tensor::from_vec_i64(ids_data.clone(), &[ids_data.len()]).unwrap());
        let out = ferro_core::ops_ext::embedding(&w, &ids).unwrap();
        if dev.is_some() {
            assert_eq!(out.device(), DEV);
        }
        out.sum().backward();
        let gw = w.grad().unwrap();
        (
            out.to_device(ferro_core::Device::Cpu).unwrap().to_vec(),
            ids.to_device(ferro_core::Device::Cpu)
                .unwrap_or_else(|_| ids.clone())
                .to_vec_i64(),
            gw.to_device(ferro_core::Device::Cpu).unwrap().to_vec(),
        )
    };
    let (out_gpu, _, gw_gpu) = run(Some(DEV));
    let (out_cpu, _, gw_cpu) = run(None);
    // Pure gather of stored rows: exact equality expected.
    assert_eq!(out_gpu, out_cpu);
    assert_eq!(gw_gpu.len(), gw_cpu.len());
    // Scatter-add of ones over identical float paths is also exact.
    assert_eq!(gw_gpu, gw_cpu);
}
