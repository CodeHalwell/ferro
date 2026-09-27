//! Serial production-path probe; exact scalar oracle, median raw samples.
use ferro_core::{Tensor, CpuBackend, dispatch::Backend};
use std::{hint::black_box, time::Instant};

fn scalar(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch*m*n];
    for bi in 0..batch { for i in 0..m { for j in 0..n {
        let mut s = 0.0;
        for p in 0..k { s += a[(bi*m+i)*k+p] * b[(bi*k+p)*n+j]; }
        out[(bi*m+i)*n+j] = s;
    } } }
    out
}
fn main() {
    ferro_fastcpu::install();
    let reverse = std::env::args().any(|s| s == "--reverse");
    for [batch,m,k,n] in [[1,4,8,8], [2,49,257,17], [16,128,128,128], [4,64,256,256], [2,128,128,1]] {
        let a: Vec<_> = (0..batch*m*k).map(|i| ((i*17%251) as f32-125.0)/127.0).collect();
        let b: Vec<_> = (0..batch*k*n).map(|i| ((i*31%257) as f32-128.0)/129.0).collect();
        let x = Tensor::from_vec(a.clone(), &[batch,m,k]).unwrap();
        let y = Tensor::from_vec(b.clone(), &[batch,k,n]).unwrap();
        let want = scalar(&a,&b,batch,m,k,n);
        let call = |route| match route {
            "scalar" => scalar(black_box(&a),black_box(&b),batch,m,k,n),
            "direct" => ferro_fastcpu::matmul_batch(black_box(&a),black_box(&b),batch,m,k,n),
            "registry" => CpuBackend.matmul_batch(black_box(&a),black_box(&b),batch,m,k,n),
            "tensor_registry" => x.bmm(&y).unwrap().to_vec(),
            _ => unreachable!(),
        };
        let mut routes = ["scalar","direct","registry","tensor_registry"];
        if reverse { routes.reverse(); }
        for route in routes {
            let got = call(route);
            assert_eq!(got.len(),want.len());
            for (i,(g,w)) in got.iter().zip(&want).enumerate() { assert_eq!(g.to_bits(),w.to_bits(),"{route} {i}"); }
            for _ in 0..4 { black_box(call(route)); }
            let mut ns = Vec::new();
            for _ in 0..15 {
                let start = Instant::now();
                let out = call(route);
                ns.push(start.elapsed().as_nanos());
                black_box(out);
            }
            println!("{{\"route\":\"{route}\",\"shape\":[{batch},{m},{k},{n}],\"all_output_max_ulp\":0,\"ns\":{ns:?}}}");
        }
    }
}
