use ferro_core::{Device, Tensor};
use ferro_core::testkit::grad_check;

#[test]
#[ignore = "serial CPU cost probe; not a historical BMM performance claim"]
fn serial_cpu_cost_probe() {
    use std::hint::black_box;
    use std::time::Instant;
    let (batch,m,k,n) = (16,128,128,128);
    let a = vec![0.5; batch*m*k];
    let b = vec![0.25; batch*k*n];
    let scalar = || ferro_fastcpu::matmul_batch(black_box(&a), black_box(&b), batch,m,k,n);
    let packed_per_slab = || {
        let mut out = Vec::with_capacity(batch*m*n);
        for bi in 0..batch {
            out.extend(ferro_fastcpu::matmul_with_threads(black_box(&a[bi*m*k..(bi+1)*m*k]),
                black_box(&b[bi*k*n..(bi+1)*k*n]),m,k,n,1));
        }
        out
    };
    assert_eq!(scalar(), packed_per_slab());
    for _ in 0..3 { black_box(scalar()); black_box(packed_per_slab()); }
    for run in 0..2 {
        for implementation in if run == 0 { ["scalar_bmm", "packed_serial_per_slab"] } else { ["packed_serial_per_slab", "scalar_bmm"] } {
            let mut ns = Vec::new();
            for _ in 0..9 {
                let start = Instant::now();
                let out = if implementation == "scalar_bmm" { scalar() } else { packed_per_slab() };
                ns.push(start.elapsed().as_nanos());
                black_box(out);
            }
            println!("{{\"run\":{run},\"implementation\":\"{implementation}\",\"shape\":[16,128,128,128],\"nanoseconds\":{ns:?}}}");
        }
    }
}

// One test owns process-global installation; no competing registry test here.
#[test]
fn installed_cpu_bmm_shapes_transposes_and_gradients() {
    ferro_fastcpu::install();
    ferro_fastcpu::install_backend();
    for (batch, m, k, n) in [(0,3,4,5), (2,0,4,5), (2,3,0,5), (2,3,4,0),
        (1,1,1,1), (2,5,7,17), (3,7,17,5), (2,49,257,17)] {
        let a: Vec<_> = (0..batch*m*k).map(|x| (x%7) as f32 - 3.0).collect();
        let b: Vec<_> = (0..batch*k*n).map(|x| (x%5) as f32 - 2.0).collect();
        let mut want = vec![0.0; batch*m*n];
        for bi in 0..batch { for row in 0..m { for p in 0..k { for col in 0..n {
            want[(bi*m+row)*n+col] += a[(bi*m+row)*k+p]*b[(bi*k+p)*n+col];
        } } } }
        for ta in [false, true] { for tb in [false, true] {
            let mut at = vec![0.0; a.len()];
            let mut bt = vec![0.0; b.len()];
            for bi in 0..batch { for row in 0..m { for p in 0..k {
                at[(bi*k+p)*m+row] = a[(bi*m+row)*k+p];
            } } }
            for bi in 0..batch { for p in 0..k { for col in 0..n {
                bt[(bi*n+col)*k+p] = b[(bi*k+p)*n+col];
            } } }
            let x = if ta { Tensor::from_vec(at, &[batch,k,m]).unwrap().transpose(1,2).unwrap() }
                else { Tensor::from_vec(a.clone(), &[batch,m,k]).unwrap() };
            let y = if tb { Tensor::from_vec(bt, &[batch,n,k]).unwrap().transpose(1,2).unwrap() }
                else { Tensor::from_vec(b.clone(), &[batch,k,n]).unwrap() };
            let z = x.bmm(&y).unwrap();
            assert_eq!(z.device(), Device::Cpu);
            assert_eq!(z.shape(), &[batch,m,n]);
            assert_eq!(z.to_vec(), want);
        } }
    }
    let x = Tensor::from_vec(vec![0.2; 12], &[2,2,3]).unwrap();
    let y = Tensor::from_vec(vec![0.3; 12], &[2,3,2]).unwrap();
    grad_check(&[x,y], |v| v[0].bmm(&v[1]).unwrap().sum());
    let x = Tensor::from_vec(vec![0.0; 6], &[1,2,3]).unwrap();
    let bad = Tensor::from_vec(vec![0.0; 8], &[1,4,2]).unwrap();
    assert!(x.bmm(&bad).is_err());
}
