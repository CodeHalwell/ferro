//! GPU acceptance for the resident spatial path (conv2d via unfold/GEMM/fold,
//! max_pool2d, batch_norm). Each body returns early without CUDA unless
//! FERRO_REQUIRE_CUDA is set, so a pass on a CPU-only host is not GPU evidence.

use ferro_core::nn::{cross_entropy, one_hot};
use ferro_core::{Device, Rng, Tensor};

const GPU: Device = Device::Cuda(0);

/// install(0) replaces the process-wide backend, so each test holds this for
/// its whole body: buffers from a replaced backend are rejected.
fn gpu() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<ferro_cuda::CudaBackend>)> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = ferro_cuda::install(0) {
        assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}");
        eprintln!("SKIP CUDA: {e}");
        return None;
    }
    Some((guard, ferro_cuda::cuda_backend()?))
}

fn downloads(b: &ferro_cuda::CudaBackend) -> usize {
    b.layer_norm_counts().2
}

fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{what}[{i}]: gpu {g} vs cpu {w}");
    }
}

fn leaf(t: &Tensor, d: Device) -> Tensor {
    t.to_device(d).unwrap().requires_grad_(true).unwrap()
}

#[test]
fn conv2d_matches_cpu_and_stays_resident() {
    let Some((_serial, b)) = gpu() else { return };
    let rng = Rng::new(1);
    let cases = [([4, 16, 32, 32], [32, 16, 3, 3], [1, 1], [1, 1], [1, 1]), ([2, 3, 17, 13], [8, 3, 5, 3], [2, 1], [2, 0], [1, 2]), ([8, 64, 8, 8], [64, 64, 1, 1], [1, 1], [0, 0], [1, 1])];
    for (xs, ws, s, p, d) in cases {
        let (x, w) = (Tensor::randn(&xs, &rng), Tensor::randn(&ws, &rng));
        let yc = { let (x, w) = (leaf(&x, Device::Cpu), leaf(&w, Device::Cpu)); let y = x.conv2d_with_options(&w, s, p, d, 1).unwrap(); y.sum().backward(); (y.to_vec(), x.grad().unwrap().to_vec(), w.grad().unwrap().to_vec()) };
        let (xg, wg) = (leaf(&x, GPU), leaf(&w, GPU));
        let before = downloads(&b);
        let y = xg.conv2d_with_options(&wg, s, p, d, 1).unwrap();
        assert_eq!(y.device(), GPU);
        y.sum().backward();
        assert_eq!(xg.grad().unwrap().device(), GPU);
        assert_eq!(downloads(&b), before, "conv2d forward/backward must not download");
        close(&y.to_vec(), &yc.0, 1e-4, "conv2d y");
        close(&xg.grad().unwrap().to_vec(), &yc.1, 1e-4, "conv2d dx");
        close(&wg.grad().unwrap().to_vec(), &yc.2, 1e-3, "conv2d dw");
    }
}

#[test]
fn max_pool2d_matches_cpu_bitwise_forward() {
    let Some((_serial, b)) = gpu() else { return };
    let rng = Rng::new(2);
    for (k, s) in [(2, 2), (3, 2), (3, 1)] {
        let x = Tensor::randn(&[3, 5, 15, 16], &rng);
        let (xc, xg) = (leaf(&x, Device::Cpu), leaf(&x, GPU));
        let yc = xc.max_pool2d(k, s).unwrap();
        yc.sum().backward();
        let before = downloads(&b);
        let yg = xg.max_pool2d(k, s).unwrap();
        yg.sum().backward();
        assert_eq!(downloads(&b), before);
        assert_eq!(yg.to_vec(), yc.to_vec(), "max_pool2d is a selection: values are exact");
        assert_eq!(xg.grad().unwrap().to_vec(), xc.grad().unwrap().to_vec(), "same windows, same summation order");
    }
}

#[test]
fn batch_norm_matches_cpu() {
    let Some((_serial, _)) = gpu() else { return };
    let rng = Rng::new(3);
    let x = Tensor::randn(&[8, 6, 9, 7], &rng);
    let (w, bias) = (Tensor::randn(&[6], &rng), Tensor::randn(&[6], &rng));
    for train in [true, false] {
        let run = |d: Device| {
            let (x, w, bias) = (leaf(&x, d), leaf(&w, d), leaf(&bias, d));
            let o = x.batch_norm(&w, &bias, &Tensor::zeros(&[6]).to_device(d).unwrap(), &Tensor::ones(&[6]).to_device(d).unwrap(), 1e-5, train, 0.1).unwrap();
            assert_eq!(o.output.device(), d);
            o.output.mul(&o.output).unwrap().sum().backward();
            [o.output.to_vec(), o.running_mean.to_vec(), o.running_var.to_vec(), x.grad().unwrap().to_vec(), w.grad().unwrap().to_vec(), bias.grad().unwrap().to_vec()]
        };
        for (i, (g, c)) in run(GPU).iter().zip(run(Device::Cpu)).enumerate() {
            close(g, &c, 1e-3, &format!("batch_norm train={train} output {i}"));
        }
    }
}

/// Thirty SGD steps of a small conv/bn/pool CNN on `d`; `check` runs after
/// each backward with the step index, before the loss is read.
fn train(d: Device, mut check: impl FnMut(usize)) -> Vec<f32> {
    let rng = Rng::new(4);
    let x = Tensor::randn(&[32, 3, 16, 16], &rng).to_device(d).unwrap();
    let labels = Tensor::from_vec_i64((0..32).map(|i| i % 10).collect(), &[32]).unwrap();
    let targets = one_hot(&labels, 10).unwrap().to_device(d).unwrap();
    let scale = |t: Tensor, s: f32| Tensor::from_vec(t.to_vec().iter().map(|v| v * s).collect(), t.shape()).unwrap();
    let param = |t: Tensor| ferro_core::Param::new(t.to_device(d).unwrap());
    let params = [param(scale(Tensor::randn(&[16, 3, 3, 3], &rng), 0.2)), param(scale(Tensor::randn(&[32, 16, 3, 3], &rng), 0.1)),
        param(Tensor::ones(&[16])), param(Tensor::zeros(&[16])), param(scale(Tensor::randn(&[32 * 4 * 4, 10], &rng), 0.05))];
    let (mut rm, mut rv) = (Tensor::zeros(&[16]).to_device(d).unwrap(), Tensor::ones(&[16]).to_device(d).unwrap());
    let mut opt = ferro_core::optim::Sgd::new(params.to_vec(), 0.05);
    (0..30).map(|step| {
        let [w1, w2, g1, b1, fc] = params.each_ref().map(|p| p.tensor());
        assert!(w1.device() == d && fc.device() == d);
        let bn = x.conv2d(&w1, 1, 1).unwrap().batch_norm(&g1, &b1, &rm, &rv, 1e-5, true, 0.1).unwrap();
        (rm, rv) = (bn.running_mean, bn.running_var);
        let h = bn.output.relu().max_pool2d(2, 2).unwrap();
        let h = h.conv2d(&w2, 1, 1).unwrap().relu().max_pool2d(2, 2).unwrap();
        let loss = cross_entropy(&h.reshape(&[32, 512]).unwrap().matmul(&fc).unwrap(), &targets).unwrap();
        opt.zero_grad();
        loss.backward();
        opt.step();
        check(step);
        loss.item()
    }).collect()
}

#[test]
fn cnn_training_reference_learns_on_cpu() {
    let losses = train(Device::Cpu, |_| {});
    assert!(losses[29] < 0.5 * losses[0], "loss {} -> {}", losses[0], losses[29]);
}

#[test]
fn cnn_training_step_is_resident_and_tracks_cpu() {
    let Some((_serial, b)) = gpu() else { return };
    let mut last = downloads(&b);
    let losses = train(GPU, |step| {
        let now = downloads(&b);
        // One scalar loss read per step happens after this check.
        if step > 0 { assert_eq!(now, last + 1, "step {step}: only the previous loss read may download"); }
        last = now;
    });
    let reference = train(Device::Cpu, |_| {});
    close(&losses, &reference, 2e-3, "loss trajectory");
}
