//! Op- and model-level CPU benchmark suite; examples/bench_suite_torch.py is
//! its PyTorch twin (same case names, shapes, work counts and JSON schema).
//! benchmarks/compare.py runs both and joins the results.
//!
//! Each case prints one JSON line to stdout:
//!   {"name","group","shape","work","unit","warmup","iters","median_ms","min_ms","p90_ms"}
//! `work` is per iteration: flops for GFLOP/s, bytes for GB/s, samples for
//! samples/s, so throughput = work / median.
//!
//! CLI: --warmup 5 --iters 30 --filter <substr> --backend fast|core
//!   fast: ferro_fastcpu::install_backend() (packed AVX2 matmul + vectorized
//!         elementwise); core: ferro_fastcpu::install() (matmul kernel only,
//!         the configuration bench_transformer uses).

use std::hint::black_box;
use std::time::Instant;

use ferro_core::modules::Conv2D;
use ferro_core::nn::{cross_entropy_indices, Linear, Module};
use ferro_core::optim::Sgd;
use ferro_core::{Param, Rng, Tensor};

type R<T> = Result<T, ferro_core::Error>;

struct Args {
    warmup: usize,
    iters: usize,
    filter: String,
    backend: String,
}

fn parse_args() -> Args {
    let mut a = Args { warmup: 5, iters: 30, filter: String::new(), backend: "fast".into() };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_default();
        match k.as_str() {
            "--warmup" => a.warmup = v.parse().unwrap_or(a.warmup),
            "--iters" => a.iters = v.parse().unwrap_or(a.iters),
            "--filter" => a.filter = v,
            "--backend" => a.backend = v,
            other => eprintln!("unknown arg {other}"),
        }
    }
    a
}

struct Bench<'a> {
    args: &'a Args,
}

impl Bench<'_> {
    fn run(&self, name: &str, group: &str, shape: &str, work: f64, unit: &str, mut f: impl FnMut() -> R<()>) -> R<()> {
        if !name.contains(&self.args.filter) {
            return Ok(());
        }
        for _ in 0..self.args.warmup {
            f()?;
        }
        let mut ms = Vec::with_capacity(self.args.iters);
        for _ in 0..self.args.iters {
            let t = Instant::now();
            f()?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| ms[((ms.len() as f64 * p) as usize).min(ms.len() - 1)];
        let median = if ms.len() % 2 == 1 { ms[ms.len() / 2] } else { (ms[ms.len() / 2 - 1] + ms[ms.len() / 2]) / 2.0 };
        println!(
            "{{\"name\":\"{name}\",\"group\":\"{group}\",\"shape\":\"{shape}\",\"work\":{work},\"unit\":\"{unit}\",\"warmup\":{},\"iters\":{},\"median_ms\":{median:.6},\"min_ms\":{:.6},\"p90_ms\":{:.6}}}",
            self.args.warmup, self.args.iters, ms[0], pct(0.9)
        );
        Ok(())
    }
}

fn params(mods: &[&dyn Module]) -> Vec<Param> {
    mods.iter().flat_map(|m| m.named_parameters()).map(|(_, p)| p).collect()
}

fn labels(n: usize, classes: usize) -> R<Tensor> {
    Tensor::from_vec_i64((0..n).map(|i| (i * 7 % classes) as i64).collect(), &[n])
}

fn main() -> R<()> {
    let args = parse_args();
    match args.backend.as_str() {
        "core" => ferro_fastcpu::install(),
        _ => ferro_fastcpu::install_backend(),
    }
    let b = Bench { args: &args };
    let rng = Rng::new(42);

    for n in [256usize, 512, 1024] {
        let x = Tensor::randn(&[n, n], &rng);
        let y = Tensor::randn(&[n, n], &rng);
        let flops = 2.0 * (n * n * n) as f64;
        b.run(&format!("matmul_{n}"), "op", &format!("{n}x{n}@{n}x{n}"), flops, "GFLOP/s", || {
            black_box(x.matmul(&y)?);
            Ok(())
        })?;
    }

    let n = 1 << 22;
    let x = Tensor::randn(&[n], &rng);
    let y = Tensor::randn(&[n], &rng);
    let shape = format!("{n}");
    let (bin, un) = (12.0 * n as f64, 8.0 * n as f64);
    b.run("add_4M", "op", &shape, bin, "GB/s", || { black_box(x.add(&y)?); Ok(()) })?;
    b.run("mul_4M", "op", &shape, bin, "GB/s", || { black_box(x.mul(&y)?); Ok(()) })?;
    b.run("relu_4M", "op", &shape, un, "GB/s", || { black_box(x.relu()); Ok(()) })?;
    b.run("exp_4M", "op", &shape, un, "GB/s", || { black_box(x.exp()); Ok(()) })?;
    b.run("gelu_tanh_4M", "op", &shape, un, "GB/s", || { black_box(x.gelu()); Ok(()) })?;
    b.run("sum_4M", "op", &shape, 4.0 * n as f64, "GB/s", || { black_box(x.sum()); Ok(()) })?;

    let s = Tensor::randn(&[1024, 1024], &rng);
    b.run("softmax_1024x1024", "op", "1024x1024 dim=1", 8.0 * (1 << 20) as f64, "GB/s", || {
        black_box(s.softmax(1)?);
        Ok(())
    })?;

    let (bn, cin, cout, hw, k) = (16usize, 32usize, 64usize, 32usize, 3usize);
    let conv_flops = 2.0 * (bn * cout * hw * hw * cin * k * k) as f64;
    let conv_shape = format!("x[{bn},{cin},{hw},{hw}] w[{cout},{cin},{k},{k}] pad=1");
    let cx = Tensor::randn(&[bn, cin, hw, hw], &rng);
    let cw = Tensor::randn(&[cout, cin, k, k], &rng);
    b.run("conv2d_fwd", "op", &conv_shape, conv_flops, "GFLOP/s", || {
        black_box(cx.conv2d(&cw, 1, 1)?);
        Ok(())
    })?;
    let cxg = cx.requires_grad_(true)?;
    let cwg = cw.requires_grad_(true)?;
    b.run("conv2d_fwd_bwd", "op", &conv_shape, 3.0 * conv_flops, "GFLOP/s", || {
        cxg.zero_grad();
        cwg.zero_grad();
        cxg.conv2d(&cwg, 1, 1)?.sum().backward();
        Ok(())
    })?;

    let batch = 128;
    let (l1, l2, l3) = (Linear::new(784, 512, &rng), Linear::new(512, 256, &rng), Linear::new(256, 10, &rng));
    let mut opt = Sgd::new(params(&[&l1, &l2, &l3]), 0.01);
    let mx = Tensor::randn(&[batch, 784], &rng);
    let my = labels(batch, 10)?;
    b.run("mlp_train_step", "model", "128x784 -> 512 -> 256 -> 10, relu, CE, SGD", batch as f64, "samples/s", || {
        let h = l2.forward(&l1.forward(&mx)?.relu())?.relu();
        let loss = cross_entropy_indices(&l3.forward(&h)?, &my)?;
        opt.zero_grad();
        loss.backward();
        opt.step();
        Ok(())
    })?;

    let batch = 32;
    let c1 = Conv2D::with_config(3, 16, 3, 1, 1, &rng);
    let c2 = Conv2D::with_config(16, 32, 3, 1, 1, &rng);
    let fc = Linear::new(32 * 8 * 8, 10, &rng);
    let mut opt = Sgd::new(params(&[&c1, &c2, &fc]), 0.01);
    let ix = Tensor::randn(&[batch, 3, 32, 32], &rng);
    let iy = labels(batch, 10)?;
    let cnn_shape = "32x3x32x32, conv3-16 pool conv16-32 pool fc2048-10, CE, SGD";
    b.run("cnn_train_step", "model", cnn_shape, batch as f64, "samples/s", || {
        let h = c1.forward(&ix)?.relu().max_pool2d(2, 2)?;
        let h = c2.forward(&h)?.relu().max_pool2d(2, 2)?;
        let loss = cross_entropy_indices(&fc.forward(&h.flatten(1, 3)?)?, &iy)?;
        opt.zero_grad();
        loss.backward();
        opt.step();
        Ok(())
    })?;
    Ok(())
}
