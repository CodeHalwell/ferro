//! GPU benchmark for the resident CNN path; examples/bench_gpu_torch.py is
//! its PyTorch twin (same case names, shapes, work counts and JSON schema).
//! Every timed iteration is bracketed by a device synchronize (a stream
//! fence, not a readback). Build with the `cuda` feature:
//!   cargo run --release --features cuda --bin bench_gpu -- --iters 50
//! CLI: --warmup 10 --iters 50 --filter <substr>

use std::hint::black_box;
use std::time::Instant;

use ferro_core::nn::{cross_entropy, one_hot};
use ferro_core::optim::Sgd;
use ferro_core::{Device, Param, Rng, Tensor};

type R<T> = Result<T, ferro_core::Error>;
const GPU: Device = Device::Cuda(0);

fn sync() {
    ferro_cuda::device_synchronize().expect("device synchronize");
}

fn main() -> R<()> {
    let (mut warmup, mut iters, mut filter) = (10usize, 50usize, String::new());
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_default();
        match k.as_str() {
            "--warmup" => warmup = v.parse().unwrap_or(warmup),
            "--iters" => iters = v.parse().unwrap_or(iters),
            "--filter" => filter = v,
            other => eprintln!("unknown arg {other}"),
        }
    }
    if let Err(e) = ferro_cuda::install(0) {
        eprintln!("CUDA unavailable: {e}");
        std::process::exit(2);
    }
    let run = |name: &str, shape: &str, work: f64, unit: &str, f: &mut dyn FnMut() -> R<()>| -> R<()> {
        if !name.contains(&filter) {
            return Ok(());
        }
        for _ in 0..warmup {
            f()?;
        }
        sync();
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            sync();
            let t = Instant::now();
            f()?;
            sync();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = ms[ms.len() / 2];
        println!("{{\"name\":\"{name}\",\"group\":\"gpu\",\"shape\":\"{shape}\",\"work\":{work},\"unit\":\"{unit}\",\"warmup\":{warmup},\"iters\":{iters},\"median_ms\":{median:.6},\"min_ms\":{:.6},\"p90_ms\":{:.6}}}",
            ms[0], ms[(ms.len() * 9 / 10).min(ms.len() - 1)]);
        Ok(())
    };
    let rng = Rng::new(42);
    let dev = |t: Tensor| t.to_device(GPU);
    let leaf = |t: Tensor| -> R<Tensor> { t.to_device(GPU)?.requires_grad_(true) };

    for n in [1024usize, 4096] {
        let (a, b) = (dev(Tensor::randn(&[n, n], &rng))?, dev(Tensor::randn(&[n, n], &rng))?);
        run(&format!("matmul_{n}"), &format!("{n}x{n}@{n}x{n}"), 2.0 * (n * n * n) as f64, "GFLOP/s", &mut || { black_box(a.matmul(&b)?); Ok(()) })?;
    }

    let (bn, cin, cout, hw, k) = (32usize, 64usize, 64usize, 56usize, 3usize);
    let flops = 2.0 * (bn * cout * hw * hw * cin * k * k) as f64;
    let shape = format!("x[{bn},{cin},{hw},{hw}] w[{cout},{cin},{k},{k}] pad=1");
    let (cx, cw) = (dev(Tensor::randn(&[bn, cin, hw, hw], &rng))?, dev(Tensor::randn(&[cout, cin, k, k], &rng))?);
    run("conv2d_fwd", &shape, flops, "GFLOP/s", &mut || { black_box(cx.conv2d(&cw, 1, 1)?); Ok(()) })?;
    let (cxg, cwg) = (leaf(Tensor::randn(&[bn, cin, hw, hw], &rng))?, leaf(Tensor::randn(&[cout, cin, k, k], &rng))?);
    run("conv2d_fwd_bwd", &shape, 3.0 * flops, "GFLOP/s", &mut || {
        cxg.zero_grad();
        cwg.zero_grad();
        cxg.conv2d(&cwg, 1, 1)?.sum().backward();
        Ok(())
    })?;

    let px = leaf(Tensor::randn(&[32, 64, 112, 112], &rng))?;
    let bytes = 4.0 * (32 * 64 * 112 * 112) as f64 * 2.5;
    run("max_pool2d_fwd_bwd", "x[32,64,112,112] k=2 s=2", bytes, "GB/s", &mut || {
        px.zero_grad();
        px.max_pool2d(2, 2)?.sum().backward();
        Ok(())
    })?;

    let (bx, bw, bb) = (leaf(Tensor::randn(&[32, 64, 56, 56], &rng))?, leaf(Tensor::ones(&[64]))?, leaf(Tensor::zeros(&[64]))?);
    let (rm, rv) = (dev(Tensor::zeros(&[64]))?, dev(Tensor::ones(&[64]))?);
    run("batch_norm_fwd_bwd", "x[32,64,56,56] train", 4.0 * (32 * 64 * 56 * 56) as f64 * 6.0, "GB/s", &mut || {
        for t in [&bx, &bw, &bb] { t.zero_grad(); }
        bx.batch_norm(&bw, &bb, &rm, &rv, 1e-5, true, 0.1)?.output.sum().backward();
        Ok(())
    })?;

    // conv3-32 bn relu pool, conv32-64 bn relu pool, fc 4096-10; bias-free convs.
    let batch = 256;
    let scale = |t: Tensor, s: f32| Tensor::from_vec(t.to_vec().iter().map(|v| v * s).collect(), t.shape());
    let param = |t: Tensor| -> R<Param> { Ok(Param::new(t.to_device(GPU)?)) };
    let ps = [param(scale(Tensor::randn(&[32, 3, 3, 3], &rng), 0.2)?)?, param(Tensor::ones(&[32]))?, param(Tensor::zeros(&[32]))?,
        param(scale(Tensor::randn(&[64, 32, 3, 3], &rng), 0.06)?)?, param(Tensor::ones(&[64]))?, param(Tensor::zeros(&[64]))?,
        param(scale(Tensor::randn(&[64 * 8 * 8, 10], &rng), 0.02)?)?];
    let mut opt = Sgd::new(ps.to_vec(), 0.01);
    let (x, y) = (dev(Tensor::randn(&[batch, 3, 32, 32], &rng))?, dev(one_hot(&Tensor::from_vec_i64((0..batch as i64).map(|i| i * 7 % 10).collect(), &[batch])?, 10)?)?);
    let stats = [dev(Tensor::zeros(&[32]))?, dev(Tensor::ones(&[32]))?, dev(Tensor::zeros(&[64]))?, dev(Tensor::ones(&[64]))?];
    run("cnn_train_step", "256x3x32x32, conv3-32 bn pool conv32-64 bn pool fc4096-10, CE, SGD", batch as f64, "samples/s", &mut || {
        let [w1, g1, b1, w2, g2, b2, fc] = ps.each_ref().map(|p| p.tensor());
        let h = x.conv2d(&w1, 1, 1)?.batch_norm(&g1, &b1, &stats[0], &stats[1], 1e-5, true, 0.1)?.output.relu().max_pool2d(2, 2)?;
        let h = h.conv2d(&w2, 1, 1)?.batch_norm(&g2, &b2, &stats[2], &stats[3], 1e-5, true, 0.1)?.output.relu().max_pool2d(2, 2)?;
        let loss = cross_entropy(&h.reshape(&[batch, 64 * 8 * 8])?.matmul(&fc)?, &y)?;
        opt.zero_grad();
        loss.backward();
        opt.step();
        Ok(())
    })?;
    Ok(())
}
