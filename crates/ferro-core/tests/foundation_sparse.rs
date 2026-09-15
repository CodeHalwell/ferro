use ferro_core::{segment, Device, Error, Result, Tensor};
use ferro_core::sparse::{Coo, Csr};
use ferro_core::testkit::grad_check;
fn t(v: &[f32], shape: &[usize]) -> Tensor { Tensor::from_vec(v.to_vec(), shape).unwrap() }

#[test]
fn validated_integer_topology_and_stable_csr_conversion() {
    let coo = Coo::new(4, 3, vec![2,0,2,0], vec![1,2,1,0]).unwrap();
    assert_eq!(coo.shape(), [4,3]);
    assert_eq!(coo.nnz(), 4);
    let large = Coo::from_i64(16_777_219,1,&[16_777_217,16_777_218],&[0,0]).unwrap();
    assert_eq!(large.rows(), &[16_777_217,16_777_218]);
    assert!(Coo::new(usize::MAX,0,vec![],vec![]).unwrap().to_csr().is_err());
    let (csr, order) = coo.to_csr().unwrap();
    assert_eq!(csr.row_offsets(), &[0,2,2,4,4]);
    assert_eq!(csr.col_indices(), &[0,2,1,1]);
    assert_eq!(order, vec![3,1,0,2]);
    let back = csr.to_coo();
    assert_eq!(back.rows(), &[0,0,2,2]);
    assert_eq!(back.cols(), csr.col_indices());
    assert!(Coo::new(1,1,vec![1],vec![0]).is_err());
    assert!(Coo::new(1,1,vec![0],vec![1]).is_err());
    assert!(Coo::new(1,1,vec![0],vec![]).is_err());
    assert!(Coo::from_i64(1,1,&[-1],&[0]).is_err());
    assert!(Coo::from_i64(1,1,&[i64::MAX],&[0]).is_err());
    assert_eq!(Coo::from_i64(1,1,&[0],&[0]).unwrap().rows(), &[0]);
    for offsets in [vec![1,1,1],vec![0,2,1],vec![0,0],vec![0,0,2]] {
        assert!(Csr::new(2,1,offsets,vec![0]).is_err());
    }
    assert!(Csr::new(1,1,vec![0,1],vec![1]).is_err());
    assert!(Csr::new(usize::MAX,0,vec![],vec![]).is_err());
    let empty = Coo::new(0,0,vec![],vec![]).unwrap();
    assert_eq!(empty.to_csr().unwrap().0.row_offsets(), &[0]);
    let unsorted = Csr::new(1,3,vec![0,3],vec![2,0,2]).unwrap();
    assert_eq!(unsorted.to_coo().cols(), &[2,0,2]);
}


#[test]
fn duplicate_spmm_and_coalescing_have_independent_dense_adjoints() {
    let a = Coo::new(3,2,vec![1,0,1],vec![0,1,0]).unwrap();
    let w = t(&[2.,3.,-1.], &[3]).requires_grad_(true).unwrap();
    let x = t(&[1.,2.,4.,5.], &[2,2]).requires_grad_(true).unwrap();
    let y = a.spmm(&w,&x).unwrap();
    assert_eq!(y.device(), Device::Cpu);
    assert_eq!(y.to_vec(), vec![12.,15.,1.,2.,0.,0.]);
    let g = t(&[1.,2.,3.,4.,5.,6.], &[3,2]);
    y.mul(&g).unwrap().sum().backward();
    assert_eq!(w.grad().unwrap().to_vec(), vec![11.,14.,11.]);
    assert_eq!(x.grad().unwrap().to_vec(), vec![3.,4.,3.,6.]);
    let w = t(&[2.,3.,-1.], &[3]).requires_grad_(true).unwrap();
    let (c, cw) = a.coalesce(&w).unwrap();
    assert_eq!(c.rows(), &[0,1]);
    assert_eq!(c.cols(), &[1,0]);
    assert_eq!(cw.to_vec(), vec![3.,1.]);
    assert_eq!(c.spmm(&cw,&x.detach_copy()).unwrap().to_vec(), y.to_vec());
    cw.mul(&t(&[7.,11.], &[2])).unwrap().sum().backward();
    assert_eq!(w.grad().unwrap().to_vec(), vec![11.,7.,11.]);
    grad_check(&[t(&[0.3,0.7,-0.2], &[3]),t(&[0.6,-0.4,0.2,0.9], &[2,2])], |xs| {
        let (c,w) = a.coalesce(&xs[0]).unwrap();
        let y = c.spmm(&w,&xs[1]).unwrap();
        y.mul(&y).unwrap().sum()
    });
    let (csr,order) = a.to_csr().unwrap();
    assert_eq!(csr.shape(), a.shape());
    let weights = t(&order.iter().map(|&e| [2.,3.,-1.][e]).collect::<Vec<_>>(), &[3]);
    assert_eq!(csr.spmm(&weights,&x.detach_copy()).unwrap().to_vec(), y.to_vec());
}


#[test]
fn sddmm_retains_duplicate_edges_and_accumulates_both_adjoints() {
    let a = Coo::new(2,3,vec![1,0,1],vec![0,2,0]).unwrap();
    let l = t(&[1.,2.,3.,4.], &[2,2]).requires_grad_(true).unwrap();
    let r = t(&[2.,-1.,5.,6.,1.,3.], &[3,2]).requires_grad_(true).unwrap();
    let y = a.sddmm(&l,&r).unwrap();
    assert_eq!(y.to_vec(), vec![2.,7.,2.]);
    y.mul(&t(&[2.,3.,-1.], &[3])).unwrap().sum().backward();
    assert_eq!(l.grad().unwrap().to_vec(), vec![3.,9.,2.,-1.]);
    assert_eq!(r.grad().unwrap().to_vec(), vec![3.,4.,0.,0.,3.,6.]);
    grad_check(&[t(&[0.1,0.2,0.3,0.4], &[2,2]),t(&[0.2,-0.1,0.5,0.6,0.1,0.3], &[3,2])], |xs| {
        let y = a.sddmm(&xs[0],&xs[1]).unwrap();
        y.mul(&y).unwrap().sum()
    });
    // Shared operand appears twice in the autograd node.
    let square = Coo::new(2,2,vec![0,1],vec![1,0]).unwrap();
    grad_check(&[t(&[0.1,0.2,0.3,0.4], &[2,2])], |xs| square.sddmm(&xs[0],&xs[0]).unwrap().sum());
}

#[test]
fn sparse_empty_validation_and_zero_width() {
    let a = Coo::new(3,2,vec![],vec![]).unwrap();
    let w = t(&[], &[0]).requires_grad_(true).unwrap();
    let x = t(&[1.,2.,3.,4.], &[2,2]).requires_grad_(true).unwrap();
    let y = a.spmm(&w,&x).unwrap();
    assert_eq!(y.to_vec(), vec![0.;6]);
    y.sum().backward();
    assert_eq!(w.grad().unwrap().to_vec(), vec![]);
    assert_eq!(x.grad().unwrap().to_vec(), vec![0.;4]);
    assert_eq!(a.coalesce(&w).unwrap().1.shape(), &[0]);
    assert_eq!(a.sddmm(&t(&[], &[3,0]),&t(&[], &[2,0])).unwrap().to_vec(), vec![]);
    let a = Coo::new(1,1,vec![0],vec![0]).unwrap();
    assert_eq!(a.spmm(&t(&[1.], &[1]),&t(&[], &[1,0])).unwrap().shape(), &[1,0]);
    assert_eq!(a.sddmm(&t(&[], &[1,0]),&t(&[], &[1,0])).unwrap().to_vec(), vec![0.]);
    assert!(a.spmm(&t(&[1.], &[1,1]),&t(&[1.], &[1,1])).is_err());
    assert!(a.spmm(&t(&[1.], &[1]),&t(&[1.], &[1])).is_err());
    assert!(a.spmm(&Tensor::from_vec_i64(vec![1], &[1]).unwrap(),&t(&[1.], &[1,1])).is_err());
    assert!(a.coalesce(&t(&[], &[0])).is_err());
    assert!(a.sddmm(&t(&[1.], &[1]),&t(&[1.], &[1,1])).is_err());
    assert!(a.sddmm(&t(&[1.,2.], &[1,2]),&t(&[1.], &[1,1])).is_err());
}


fn run(a: &Coo, w: &[f32], x: &[f32]) -> (Vec<f32>,Vec<f32>,Vec<f32>) {
    let w = t(w, &[w.len()]).requires_grad_(true).unwrap();
    let x = t(x, &[a.shape()[1],2]).requires_grad_(true).unwrap();
    let y = a.spmm(&w,&x).unwrap();
    y.mul(&y).unwrap().sum().backward();
    (y.to_vec(),w.grad().unwrap().to_vec(),x.grad().unwrap().to_vec())
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(),b.len());
    for (a,b) in a.iter().zip(b) { assert!((a-b).abs() < 2e-5, "{a} != {b}"); }
}

#[test]
fn spmm_matches_independent_f64_dense_matrix_and_adjoint() {
    let a = Coo::new(3,3,vec![2,0,2,1,0],vec![1,2,1,0,1]).unwrap();
    let w = [0.4,-0.7,0.2,0.5,0.9];
    let x = [0.2,0.3,0.7,-0.4,0.8,0.1];
    let (y,dw,dx) = run(&a,&w,&x);
    let mut dense = [[0f64;3];3];
    for e in 0..w.len() { dense[a.rows()[e]][a.cols()[e]] += w[e] as f64; }
    let mut expected = vec![0f64;6];
    for i in 0..3 { for j in 0..3 { for f in 0..2 { expected[i*2+f] += dense[i][j] * x[j*2+f] as f64; } } }
    let mut grad_x = vec![0f64;6];
    for j in 0..3 { for i in 0..3 { for f in 0..2 { grad_x[j*2+f] += dense[i][j] * 2. * expected[i*2+f]; } } }
    let grad_w: Vec<f32> = (0..w.len()).map(|e| (0..2).map(|f| 2. * expected[a.rows()[e]*2+f] * x[a.cols()[e]*2+f] as f64).sum::<f64>() as f32).collect();
    close(&y,&expected.iter().map(|&v| v as f32).collect::<Vec<_>>());
    close(&dx,&grad_x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    close(&dw,&grad_w);
}

#[test]
fn node_edge_permutation_preserves_values_and_both_gradients() {
    let rows = [2,0,2,1,0]; let cols = [1,2,1,0,1];
    let a = Coo::new(3,3,rows.to_vec(),cols.to_vec()).unwrap();
    let w = [0.4,-0.7,0.2,0.5,0.9]; let x = [0.2,0.3,0.7,-0.4,0.8,0.1];
    let (y,dw,dx) = run(&a,&w,&x);
    let nodes = [2,0,1]; // old -> new
    let edges = [4,2,0,3,1]; // new -> old
    let b = Coo::new(3,3,edges.iter().map(|&e| nodes[rows[e]]).collect(),edges.iter().map(|&e| nodes[cols[e]]).collect()).unwrap();
    let mut px = vec![0.;6];
    for i in 0..3 { for f in 0..2 { px[nodes[i]*2+f] = x[i*2+f]; } }
    let pw: Vec<_> = edges.iter().map(|&e| w[e]).collect();
    let (py,pdw,pdx) = run(&b,&pw,&px);
    for i in 0..3 {
        close(&py[nodes[i]*2..nodes[i]*2+2],&y[i*2..i*2+2]);
        close(&pdx[nodes[i]*2..nodes[i]*2+2],&dx[i*2..i*2+2]);
    }
    close(&pdw,&edges.iter().map(|&e| dw[e]).collect::<Vec<_>>());
}

#[test]
fn disjoint_batch_matches_single_graphs_forward_and_gradients() {
    let a = Coo::new(3,2,vec![1,0,1],vec![0,1,0]).unwrap();
    let b = Coo::new(2,3,vec![1,0],vec![2,0]).unwrap();
    let wa = [0.2,0.3,-0.1]; let wb = [0.7,0.6];
    let xa = [0.1,0.2,0.3,0.4]; let xb = [0.5,0.6,0.7,0.8,0.9,1.];
    let (ya,dwa,dxa) = run(&a,&wa,&xa); let (yb,dwb,dxb) = run(&b,&wb,&xb);
    let batch = Coo::new(5,5,vec![1,0,1,4,3],vec![0,1,0,4,2]).unwrap();
    let (y,dw,dx) = run(&batch,&[wa.as_slice(),wb.as_slice()].concat(),&[xa.as_slice(),xb.as_slice()].concat());
    close(&y,&[ya,yb].concat()); close(&dw,&[dwa,dwb].concat()); close(&dx,&[dxa,dxb].concat());
}


#[test]
fn differentiable_graph_attention_composition_checks_all_operands() {
    let a = Coo::new(3,3,vec![2,0,2,1,0],vec![1,2,1,0,1]).unwrap();
    grad_check(&[t(&[0.1,0.2,0.3,0.4,0.5,0.6], &[3,2]),
        t(&[0.6,-0.1,0.2,0.3,-0.2,0.7], &[3,2]),
        t(&[0.4,0.3,-0.2,0.1,0.5,0.6], &[3,2])], |xs| {
        let logits = a.sddmm(&xs[0],&xs[1]).unwrap();
        let weights = segment::softmax(&logits,a.rows(),a.shape()[0]).unwrap();
        let y = a.spmm(&weights,&xs[2]).unwrap();
        y.mul(&y).unwrap().sum()
    });
}


#[test]
fn non_cpu_inputs_are_rejected_before_any_host_download() {
    use ferro_core::dispatch::{Backend, BinaryKind, DeviceBuffer, UnaryKind};
    use std::sync::{Arc, Mutex};
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Sentinel storage only: no CUDA runtime or device computation is exercised.
    const DEV: Device = Device::Cuda(231);
    struct Buffer(usize);
    impl DeviceBuffer for Buffer {
        fn device(&self) -> Device { DEV }
        fn len(&self) -> usize { self.0 }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }
    struct RefuseCompute;
    impl Backend for RefuseCompute {
        fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
        fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
        fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected compute") }
        fn alloc_from_host(&self, data: &[f32]) -> Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buffer(data.len()))) }
        fn copy_to_host(&self, _: &dyn DeviceBuffer) -> Result<Vec<f32>> { panic!("must reject before download") }
    }
    ferro_core::register_backend(DEV, Arc::new(RefuseCompute));
    let w = t(&[1.], &[1]); let x = t(&[1.], &[1,1]);
    let wd = w.to_device(DEV).unwrap(); let xd = x.to_device(DEV).unwrap();
    for op in [segment::sum,segment::mean,segment::max,segment::softmax] {
        assert!(matches!(op(&wd,&[0],1), Err(Error::Unsupported { .. })));
    }
    let a = Coo::new(1,1,vec![0],vec![0]).unwrap();
    assert!(matches!(a.coalesce(&wd), Err(Error::Unsupported { .. })));
    assert!(matches!(a.spmm(&wd,&x), Err(Error::Unsupported { .. })));
    assert!(matches!(a.spmm(&w,&xd), Err(Error::Unsupported { .. })));
    assert!(matches!(a.sddmm(&xd,&x), Err(Error::Unsupported { .. })));
    assert!(matches!(a.sddmm(&x,&xd), Err(Error::Unsupported { .. })));
}


#[test]
fn vector_coalescing_and_csr_gradients() {
    let a = Coo::new(2,2,vec![1,0,1],vec![0,1,0]).unwrap();
    let w = t(&[1.,2.,3.,4.,5.,6.], &[3,2]).requires_grad_(true).unwrap();
    let (c, v) = a.coalesce(&w).unwrap();
    assert_eq!(v.to_vec(),vec![3.,4.,6.,8.]);
    assert_eq!(c.nnz(),2);
    v.mul(&t(&[2.,3.,5.,7.], &[2,2])).unwrap().sum().backward();
    assert_eq!(w.grad().unwrap().to_vec(),vec![5.,7.,2.,3.,5.,7.]);
    let csr = a.to_csr().unwrap().0;
    grad_check(&[t(&[0.2,0.3,-0.1], &[3]),t(&[0.1,0.4,0.3,0.6], &[2,2])], |xs| {
        let y = csr.spmm(&xs[0],&xs[1]).unwrap();
        y.mul(&y).unwrap().sum()
    });
}
