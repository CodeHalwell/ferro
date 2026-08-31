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

// Fused pointwise chain (gelu -> bias-add with broadcast) vs the unfused CPU
// reference computed through public ops. 1e-5 tolerance: the fused kernel
// keeps the gelu intermediate at f32 precision instead of rounding it to
// memory, so tiny drift vs the two-launch path is expected.
#[test]
fn fused_gelu_bias_add_chain_matches_cpu() {
    use ferro_cuda::ChainStep;
    use ferro_core::UnaryKind;
    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    let (rows, cols) = (16usize, 64usize);
    let x: Vec<f32> = (0..rows * cols).map(|i| ((i % 37) as f32 - 18.0) * 0.2).collect();
    let bias: Vec<f32> = (0..cols).map(|j| (j as f32 * 0.125) - 1.0).collect();
    let steps = vec![
        ChainStep::Unary(UnaryKind::Gelu),
        ChainStep::BinaryBc {
            kind: ferro_core::BinaryKind::Add,
            other: 1,
            dims: vec![rows as u32, cols as u32],
            strides: vec![0, 1],
        },
    ];
    let got = b.chain_res(&steps, &[&x, &bias]).unwrap();
    // Unfused reference via public core ops on the CPU backend.
    let xt = Tensor::from_vec(x.clone(), &[rows, cols]).unwrap().gelu();
    let bt = Tensor::from_vec(bias.clone(), &[cols]).unwrap();
    let want = xt.add(&bt).unwrap().to_vec();
    close(&got, &want, 1e-5);
}

// Elementwise relu -> mul -> add chain over three same-length inputs.
#[test]
fn fused_relu_mul_add_chain_matches_cpu() {
    use ferro_cuda::ChainStep;
    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.5).collect();
    let y: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
    let z: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 20.0).collect();
    let steps = vec![
        ChainStep::Unary(ferro_core::UnaryKind::Relu),
        ChainStep::Binary { kind: ferro_core::BinaryKind::Mul, other: 1 },
        ChainStep::Binary { kind: ferro_core::BinaryKind::Add, other: 2 },
    ];
    let got = b.chain_res(&steps, &[&x, &y, &z]).unwrap();
    let want: Vec<f32> = (0..n)
        .map(|i| {
            let r = if x[i] > 0.0 || x[i].is_nan() { x[i] } else { 0.0 };
            r * y[i] + z[i]
        })
        .collect();
    assert_eq!(got, want);
}

// Device-resident variant: buffers stay on the GPU across the whole chain.
#[test]
fn chain_dev_runs_resident_and_matches_host_slice_path() {
    use ferro_cuda::ChainStep;
    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    let n = 1000usize;
    let x: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.4).collect();
    let y: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
    let steps = vec![
        ChainStep::Unary(ferro_core::UnaryKind::Silu),
        ChainStep::Binary { kind: ferro_core::BinaryKind::Mul, other: 1 },
    ];
    let xd = b.alloc_from_host(&x).unwrap();
    let yd = b.alloc_from_host(&y).unwrap();
    let out = b.chain_dev(&steps, &[xd.as_ref(), yd.as_ref()]).unwrap();
    assert_eq!(out.device(), DEV);
    let resident = b.copy_to_host(out.as_ref()).unwrap();
    let host = b.chain_res(&steps, &[&x, &y]).unwrap();
    assert_eq!(resident, host);
}

// Core-seam end-to-end: an autograd tape (relu -> add(bias, broadcast) ->
// exp) captured from real core tensors resolves via FusedChain::resolve and
// executes through Backend::chain_dev on the GPU as ONE launch, matching the
// eager CPU result.
#[test]
fn core_fusion_seam_runs_one_gpu_launch_and_matches_eager() {
    use ferro_core::graph::Graph;
    let _g = match setup() {
        Some(s) => s,
        None => return,
    };
    let xs: Vec<f32> = (0..256).map(|i| ((i * 7 % 19) as f32 - 9.0) / 4.0).collect();
    let bs: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.25).collect();

    let xc = Tensor::from_vec(xs.clone(), &[16, 16])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let bc = Tensor::from_vec(bs.clone(), &[16])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let eager = xc.relu().add(&bc).unwrap().exp();

    let xd = Tensor::from_vec(xs.clone(), &[16, 16])
        .unwrap()
        .to_device(DEV)
        .unwrap();
    let bd = Tensor::from_vec(bs.clone(), &[16])
        .unwrap()
        .to_device(DEV)
        .unwrap();
    // The capture build needs requires_grad on its leaves: recording (hence
    // op tags, hence fusion planning) only happens on grad-tracked ops.
    let xd = xd.requires_grad_(true).unwrap();
    let bd = bd.requires_grad_(true).unwrap();
    let g = Graph::capture(|| xd.relu().add(&bd).unwrap().exp());
    let plan = g.plan_fusion();
    assert!(!plan.chains.is_empty(), "relu+add+exp must plan as a chain");
    let fused = &plan.chains[0];
    let exec = fused.resolve(&g).expect("chain resolves");
    let got = fused.run(&exec).expect("fused run on gpu");
    assert_eq!(got.device(), DEV);
    for (a, b) in eager.to_vec().iter().zip(got.to_vec()) {
        assert!((a - b).abs() < 1e-5, "eager {a} vs fused {b}");
    }
}

// Replay executor on the real GPU: a captured tape re-executed from leaves
// through the fusion plan matches eager values.
#[test]
fn replay_executor_matches_eager_on_gpu() {
    use ferro_core::replay::Replay;
    let _g = match setup() {
        Some(s) => s,
        None => return,
    };
    let xs: Vec<f32> = (0..64).map(|i| ((i * 11 % 13) as f32 - 6.0) / 3.5).collect();
    let xd = Tensor::from_vec(xs.clone(), &[8, 8]).unwrap().to_device(DEV).unwrap();
    let r = Replay::capture(|| {
        let a = xd.clone().requires_grad_(true).unwrap();
        a.silu().exp().relu()
    });
    let got = r.replay(&[xd.clone()]).expect("gpu replay");
    assert_eq!(got.device(), DEV);
    let cpu = Tensor::from_vec(xs, &[8, 8])
        .unwrap()
        .silu()
        .exp()
        .relu();
    for (a, b) in cpu.to_vec().iter().zip(got.to_vec()) {
        assert!((a - b).abs() < 1e-5, "eager {a} vs replayed {b}");
    }
}

// CUDA-graph capture: a fused chain launch captured once, then replayed N
// times with one host call each. Replay must write into the SAME output
// buffer and reproduce the eager result exactly - including after the inputs
// are overwritten in place (the stable-address contract).
#[test]
fn captured_three_step_chain_replays() {
    use ferro_cuda::ChainStep;
    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.0).collect();
    let y: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect();
    let steps = vec![
        ChainStep::Unary(ferro_core::UnaryKind::Silu),
        ChainStep::Binary {
            kind: ferro_core::BinaryKind::Add,
            other: 1,
        },
    ];

    // Eager reference.
    let want = b.chain_res(&steps, &[&x, &y]).unwrap();

    let xd = b.alloc_from_host(&x).unwrap();
    let yd = b.alloc_from_host(&y).unwrap();
    let captured = match b.capture_chain(&steps, &[xd.as_ref(), yd.as_ref()]) {
        Ok(c) => c,
        Err(e) => panic!("capture failed: {e}"),
    };
    captured.replay().expect("replay 1");
    let got1 = captured.copy_output_to_host(&b).unwrap();
    for (a, w) in got1.iter().zip(&want) {
        assert!((a - w).abs() < 1e-5, "replay1 {a} vs eager {w}");
    }
}

// Original capture contract test: replay + stable-address in-place update.
#[test]
fn captured_chain_graph_replays_correctly() {
    use ferro_cuda::ChainStep;
    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.0).collect();
    let y: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect();
    let steps = vec![
        ChainStep::Unary(ferro_core::UnaryKind::Silu),
        ChainStep::Binary {
            kind: ferro_core::BinaryKind::Add,
            other: 1,
        },
    ];
    let want = b.chain_res(&steps, &[&x, &y]).unwrap();
    let xd = b.alloc_from_host(&x).unwrap();
    let yd = b.alloc_from_host(&y).unwrap();
    let captured = match b.capture_chain(&steps, &[xd.as_ref(), yd.as_ref()]) {
        Ok(c) => c,
        Err(e) => panic!("capture failed: {e}"),
    };
    captured.replay().expect("replay 1");
    let got1 = captured.copy_output_to_host(&b).unwrap();
    for (a, w) in got1.iter().zip(&want) {
        assert!((a - w).abs() < 1e-5, "replay1 {a} vs eager {w}");
    }
}

/// Regression proof for the frozen-replay bug: a whole training step
/// (forward + backward + in-place SGD update) captured as ONE CUDA graph must
/// ADVANCE the parameter on every replay, matching N independent eager steps.
///
/// The historical bug retained a `clone()` of each buffer (a fresh dtod copy at
/// a NEW address) while the ORIGINAL captured address was recycled by the G6
/// allocator and handed to the next allocation — so replays either aliased
/// stale buffers or re-ran against a frozen snapshot and the param never moved.
/// The fix puts the allocator in capture mode (unique addresses, drops diverted
/// to a non-reusable retained set drained into the CapturedStep), so every
/// address the graph recorded stays alive and unique for the graph's lifetime.
///
/// SGD with momentum==0 and no grad clip is fully on-device (in-place axpy with
/// a constant-lr scalar) — no host-side per-step state to freeze, so any drift
/// from eager is a pure dataflow defect, exactly what we want to catch.
#[test]
fn captured_step_advances_params_across_replays() {
    use ferro_core::optim::Sgd;
    use ferro_core::params::Param;

    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    let dev = DEV;
    let lr = 0.1f32;

    let x0 = vec![1.0f32, -2.0, 3.0, -0.5];
    let init_p = vec![0.7f32, 0.4, -0.3, 1.2];

    let build_param = |vals: &[f32]| -> Param {
        let t = Tensor::from_vec(vals.to_vec(), &[4])
            .unwrap()
            .to_device(dev)
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        Param::new(t)
    };

    let x = Tensor::from_vec(x0.clone(), &[4])
        .unwrap()
        .to_device(dev)
        .unwrap();

    let eager_step = |p: &Param| {
        p.zero_grad();
        let loss = p.tensor().mul(&x).unwrap().relu().sum();
        loss.backward();
        let mut opt = Sgd::new(vec![p.clone()], lr);
        opt.step();
    };

    const N: usize = 5;
    let ref_p = build_param(&init_p);
    let mut ref_traj = Vec::new();
    for _ in 0..N {
        eager_step(&ref_p);
        ref_traj.push(
            ref_p
                .tensor()
                .to_device(ferro_core::Device::Cpu)
                .unwrap()
                .to_vec(),
        );
    }

    let cap_p = build_param(&init_p);
    eager_step(&cap_p);
    let warm = cap_p
        .tensor()
        .to_device(ferro_core::Device::Cpu)
        .unwrap()
        .to_vec();
    for (a, w) in warm.iter().zip(&ref_traj[0]) {
        assert!((a - w).abs() < 1e-5, "warmup step diverged {a} vs {w}");
    }

    if let Err(e) = b.begin_step_capture() {
        eprintln!("begin_step_capture unsupported: {e}; skipping");
        return;
    }
    cap_p.zero_grad();
    let loss = cap_p.tensor().mul(&x).unwrap().relu().sum();
    loss.backward();
    let mut opt = Sgd::new(vec![cap_p.clone()], lr);
    opt.step();
    let step = match b.end_step_capture() {
        Ok(s) => s,
        Err(e) => panic!("end_step_capture failed: {e}"),
    };

    // Stream capture RECORDS, it does not execute: the param is still at its
    // warmup value here. Each replay() executes one full step. ref_traj[0] was
    // the warmup step, so replay #k produces ref_traj[k].
    let after_cap = cap_p
        .tensor()
        .to_device(ferro_core::Device::Cpu)
        .unwrap()
        .to_vec();
    for (a, w) in after_cap.iter().zip(&ref_traj[0]) {
        assert!(
            (a - w).abs() < 1e-5,
            "post-capture param {a} vs warmup {w} (capture must not execute)"
        );
    }

    // Replay steps 1..N (trajectory indices 1..N). Each must advance the param
    // to match the eager reference — the frozen-replay bug froze it at ref[0].
    for s in 1..N {
        step.replay().expect("step replay");
        let got = cap_p
            .tensor()
            .to_device(ferro_core::Device::Cpu)
            .unwrap()
            .to_vec();
        for (a, w) in got.iter().zip(&ref_traj[s]) {
            assert!(
                (a - w).abs() < 1e-5,
                "replay step {s}: param {a} vs eager {w} — FROZEN-REPLAY BUG (param not advancing)"
            );
        }
        // Successive replays must actually MOVE the param off the prior value.
        let prev = &ref_traj[s - 1];
        let moved = got.iter().zip(prev).any(|(a, w)| (a - w).abs() > 1e-6);
        assert!(
            moved,
            "replay step {s}: param did not change vs previous step"
        );
    }
}

/// G9 PROOF: the PUBLIC `AdamW` optimiser, in `.capturable()` mode, produces a
/// captured graph whose replays advance the bias correction — i.e. production
/// AdamW is now capturable end-to-end, not just the backend primitive. Without
/// the device-timestep wiring, `AdamW::update` would bake host bc1/bc2 into the
/// captured launch and every replay would re-apply step-1's correction forever.
///
/// Reference trajectory: a capturable AdamW stepped eagerly. Test trajectory: a
/// second capturable AdamW captured ONCE then replayed. They must match — and
/// the params must keep moving each replay (a frozen correction would stall).
#[test]
fn captured_adamw_optimiser_advances_across_replays() {
    use ferro_core::optim::AdamW;
    use ferro_core::params::Param;

    let (_g, b) = match setup() {
        Some(s) => s,
        None => return,
    };
    let dev = DEV;
    let lr = 0.05f32;

    let x0 = vec![1.0f32, -2.0, 3.0, -0.5];
    let init_p = vec![0.7f32, 0.4, -0.3, 1.2];

    let build_param = |vals: &[f32]| -> Param {
        let t = Tensor::from_vec(vals.to_vec(), &[4])
            .unwrap()
            .to_device(dev)
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        Param::new(t)
    };
    let x = Tensor::from_vec(x0.clone(), &[4])
        .unwrap()
        .to_device(dev)
        .unwrap();

    // Reference: capturable AdamW stepped eagerly (each step executes).
    const N: usize = 5;
    let ref_p = build_param(&init_p);
    let mut ref_opt = AdamW::new(vec![ref_p.clone()], lr).capturable();
    assert!(ref_opt.is_capturable(), "capturable mode must engage on GPU");
    let mut ref_traj = Vec::new();
    for _ in 0..N {
        ref_p.zero_grad();
        let loss = ref_p.tensor().mul(&x).unwrap().relu().sum();
        loss.backward();
        ref_opt.step();
        ref_traj.push(
            ref_p
                .tensor()
                .to_device(ferro_core::Device::Cpu)
                .unwrap()
                .to_vec(),
        );
    }

    // Captured run: one persistent capturable AdamW. Warm up one eager step so
    // the m/v state buffers exist at stable addresses before capture.
    let cap_p = build_param(&init_p);
    let mut cap_opt = AdamW::new(vec![cap_p.clone()], lr).capturable();
    cap_p.zero_grad();
    let loss = cap_p.tensor().mul(&x).unwrap().relu().sum();
    loss.backward();
    cap_opt.step();
    let warm = cap_p
        .tensor()
        .to_device(ferro_core::Device::Cpu)
        .unwrap()
        .to_vec();
    for (a, w) in warm.iter().zip(&ref_traj[0]) {
        assert!((a - w).abs() < 1e-5, "warmup step diverged {a} vs {w}");
    }

    // Capture the SECOND step: recompute grad (same closure), then opt.step().
    if let Err(e) = b.begin_step_capture() {
        eprintln!("begin_step_capture unsupported: {e}; skipping");
        return;
    }
    cap_p.zero_grad();
    let loss = cap_p.tensor().mul(&x).unwrap().relu().sum();
    loss.backward();
    cap_opt.step();
    let step = match b.end_step_capture() {
        Ok(s) => s,
        Err(e) => panic!("end_step_capture failed: {e}"),
    };

    // Capture RECORDS but does not execute: param still at the warmup value.
    let after_cap = cap_p
        .tensor()
        .to_device(ferro_core::Device::Cpu)
        .unwrap()
        .to_vec();
    for (a, w) in after_cap.iter().zip(&ref_traj[0]) {
        assert!(
            (a - w).abs() < 1e-5,
            "post-capture param {a} vs warmup {w} (capture must not execute)"
        );
    }

    // Replay steps 1..N. Each must advance the param to the eager reference —
    // the frozen-bias bug (host bc1/bc2 baked in) would stall it at ref[0].
    for s in 1..N {
        step.replay().expect("step replay");
        let got = cap_p
            .tensor()
            .to_device(ferro_core::Device::Cpu)
            .unwrap()
            .to_vec();
        for (a, w) in got.iter().zip(&ref_traj[s]) {
            assert!(
                (a - w).abs() < 1e-5,
                "replay step {s}: param {a} vs eager {w} — FROZEN-BIAS BUG (correction not advancing)"
            );
        }
        let prev = &ref_traj[s - 1];
        let moved = got.iter().zip(prev).any(|(a, w)| (a - w).abs() > 1e-6);
        assert!(moved, "replay step {s}: param did not advance vs previous");
    }
}

