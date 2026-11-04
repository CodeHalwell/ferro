use ferro_core::{DType, Device, Error, Tensor, capture};
use ferro_core::gnn::{EdgeIndex, edge_attention};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountAlloc;
thread_local! {
    static WATCH: Cell<bool> = const { Cell::new(false) };
    static TOTAL: Cell<usize> = const { Cell::new(0) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}
#[global_allocator]
static ALLOC: CountAlloc = CountAlloc;

fn allocation(bytes: usize) {
    let _ = WATCH.try_with(|watch| {
        if watch.get() {
            let _ = TOTAL.try_with(|total| total.set(total.get().saturating_add(bytes)));
            let _ = LARGEST.try_with(|largest| largest.set(largest.get().max(bytes)));
        }
    });
}

unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 { allocation(layout.size()); unsafe { System.alloc(layout) } }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 { allocation(layout.size()); unsafe { System.alloc_zeroed(layout) } }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocation(size); unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) { unsafe { System.dealloc(ptr, layout) } }
}

fn features(n: usize, d: usize, offset: f32) -> Tensor {
    Tensor::from_vec((0..n*d).map(|i| ((i * 7 % 13) as f32 - 6.0) / 7.0 + offset).collect(), &[n,d]).unwrap()
}

fn graph() -> EdgeIndex { EdgeIndex::new(5, vec![0,1,1,2,3,1,0,2], vec![1,2,2,1,3,0,2,0]).unwrap() }

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) { assert!((a-b).abs() < 3e-6, "{actual:?} != {expected:?}"); }
}

fn dense(q: &Tensor, k: &Tensor, v: &Tensor, edges: &EdgeIndex, bias: Option<&Tensor>) -> Vec<f32> {
    let (n, d, dv) = (edges.num_nodes(), q.shape()[1], v.shape()[1]);
    let (q, k, v) = (q.to_vec(), k.to_vec(), v.to_vec());
    let bias = bias.map(|b| b.to_vec()).unwrap_or(vec![0.; edges.num_edges()]);
    let mut matrix = vec![0f64; n*n];
    for (e, (&src, &dst)) in edges.src().iter().zip(edges.dst()).enumerate() {
        let dot: f64 = (0..d).map(|j| q[dst*d+j] as f64 * k[src*d+j] as f64).sum();
        matrix[dst*n+src] += (dot / (d as f64).sqrt() + bias[e] as f64).exp();
    }
    let mut result = vec![0.; n*dv];
    for dst in 0..n {
        let denominator: f64 = matrix[dst*n..(dst+1)*n].iter().sum();
        if denominator == 0. { continue; }
        for j in 0..dv {
            result[dst*dv+j] = (0..n).map(|src| matrix[dst*n+src] / denominator * v[src*dv+j] as f64).sum::<f64>() as f32;
        }
    }
    result
}

#[test]
fn directed_attention_matches_dense_oracle_with_bias_and_isolated_rows() {
    let edges = graph();
    let (q, k, v, bias) = (features(5,3,0.1), features(5,3,-0.2), features(5,2,0.3), features(8,1,0.2).reshape(&[8]).unwrap());
    for bias in [None, Some(&bias)] {
        let output = edge_attention(&q,&k,&v,&edges,bias).unwrap();
        assert_eq!(output.shape(), &[5,2]);
        close(&output.to_vec(), &dense(&q,&k,&v,&edges,bias));
        assert_eq!(&output.to_vec()[8..], &[0.,0.]);
    }
}

#[test]
fn all_operands_pass_strict_gradient_check_with_duplicate_edges() {
    let edges = graph();
    let inputs = [features(5,2,0.1), features(5,2,-0.2), features(5,3,0.3), features(8,1,0.2).reshape(&[8]).unwrap()];
    let probe = features(5,3,-0.3);
    ferro_core::testkit::grad_check_strict(&inputs, |x| {
        edge_attention(&x[0],&x[1],&x[2],&edges,Some(&x[3])).unwrap().mul(&probe).unwrap().sum()
    });
}

#[test]
fn duplicate_edges_have_independent_bias_gradients() {
    let edges = EdgeIndex::new(3, vec![0,0,1], vec![2,2,2]).unwrap();
    let q = Tensor::zeros(&[3,1]);
    let v = Tensor::from_vec(vec![1.,4.,9.], &[3,1]).unwrap().requires_grad_(true).unwrap();
    let bias = Tensor::from_vec(vec![0.,2f32.ln(),3f32.ln()], &[3]).unwrap().requires_grad_(true).unwrap();
    let output = edge_attention(&q,&q,&v,&edges,Some(&bias)).unwrap();
    close(&output.to_vec(), &[0.,0.,2.5]);
    output.sum().backward();
    close(&v.grad().unwrap().to_vec(), &[0.5,0.5,0.]);
    close(&bias.grad().unwrap().to_vec(), &[-0.25,-0.5,0.75]);
}

#[test]
fn node_and_edge_permutations_and_disjoint_batch_preserve_outputs_and_gradients() {
    let edges = graph();
    let (q,k,v) = (features(5,2,0.1),features(5,2,-0.2),features(5,3,0.3));
    let bias = features(8,1,0.2).reshape(&[8]).unwrap();
    let expected = edge_attention(&q,&k,&v,&edges,Some(&bias)).unwrap();
    let order = [7,2,4,0,6,3,1,5];
    let permuted = EdgeIndex::new(5, order.iter().map(|&e| edges.src()[e]).collect(), order.iter().map(|&e| edges.dst()[e]).collect()).unwrap();
    close(&edge_attention(&q,&k,&v,&permuted,Some(&bias.index_select(0,&order).unwrap())).unwrap().to_vec(), &expected.to_vec());
    let order = [2,4,0,3,1];
    let mut inverse = [0;5];
    for (new,&old) in order.iter().enumerate() { inverse[old] = new; }
    let permuted = EdgeIndex::new(5, edges.src().iter().map(|&s| inverse[s]).collect(), edges.dst().iter().map(|&s| inverse[s]).collect()).unwrap();
    close(&edge_attention(&q.index_select(0,&order).unwrap(),&k.index_select(0,&order).unwrap(),&v.index_select(0,&order).unwrap(),&permuted,Some(&bias)).unwrap().to_vec(),
        &expected.index_select(0,&order).unwrap().to_vec());

    let second = EdgeIndex::new(2,vec![0,1,1],vec![1,0,1]).unwrap();
    let union = EdgeIndex::try_batch(&[EdgeIndex::new(0,vec![],vec![]).unwrap(),edges.clone(),second.clone()]).unwrap().0;
    let inputs = [q,k,v,bias];
    let tail = [features(2,2,-0.4),features(2,2,0.5),features(2,3,0.1),features(3,1,-0.2).reshape(&[3]).unwrap()];
    let joined: Vec<_> = inputs.iter().zip(&tail).map(|(a,b)| Tensor::cat(&[a.clone(),b.clone()],0).unwrap().requires_grad_(true).unwrap()).collect();
    let output = edge_attention(&joined[0],&joined[1],&joined[2],&union,Some(&joined[3])).unwrap();
    let mut expected = Vec::new();
    let mut gradients = vec![Vec::new();4];
    for (parts, edges) in [(&inputs,&edges),(&tail,&second)] {
        let parts: Vec<_> = parts.iter().map(|t| t.requires_grad_(true).unwrap()).collect();
        let result = edge_attention(&parts[0],&parts[1],&parts[2],edges,Some(&parts[3])).unwrap();
        expected.extend(result.to_vec());
        result.mul(&result).unwrap().sum().backward();
        for (grad,t) in gradients.iter_mut().zip(parts) { grad.extend(t.grad().unwrap().to_vec()); }
    }
    close(&output.to_vec(), &expected);
    output.mul(&output).unwrap().sum().backward();
    for (input, expected) in joined.iter().zip(gradients) { close(&input.grad().unwrap().to_vec(), &expected); }
}

#[test]
fn empty_topology_nodes_and_value_width_keep_zero_adjoint_shapes() {
    for (nodes, width, nonempty) in [(3,2,false),(0,2,false),(3,0,true)] {
        let edges = if nonempty { EdgeIndex::new(nodes,vec![0,1,1],vec![1,2,2]).unwrap() }
            else { EdgeIndex::new(nodes,vec![],vec![]).unwrap() };
        let inputs: Vec<_> = [features(nodes,2,0.1),features(nodes,2,-0.2),features(nodes,width,0.3),Tensor::zeros(&[edges.num_edges()])]
            .into_iter().map(|x| x.requires_grad_(true).unwrap()).collect();
        let y = edge_attention(&inputs[0],&inputs[1],&inputs[2],&edges,Some(&inputs[3])).unwrap();
        assert_eq!(y.shape(), &[nodes,width]);
        assert!(y.to_vec().iter().all(|&x| x == 0.));
        y.sum().backward();
        for input in inputs {
            let grad = input.grad().unwrap();
            assert_eq!(grad.shape(), input.shape());
            assert!(grad.to_vec().iter().all(|&x| x == 0.));
        }
    }
}

#[test]
fn invalid_metadata_capture_nonfinite_logits_and_higher_order_reject() {
    let edges = EdgeIndex::new(2,vec![0,1],vec![1,1]).unwrap();
    let q = features(2,2,0.1);
    let v = features(2,3,-0.1);
    let bias = Tensor::zeros(&[2]);
    for bad in [Tensor::scalar(1.),Tensor::zeros(&[2]),Tensor::zeros(&[1,2]),Tensor::zeros(&[2,0]),Tensor::zeros(&[2,3])] {
        assert!(matches!(edge_attention(&bad,&q,&v,&edges,None),Err(Error::InvalidShape{..})));
        assert!(matches!(edge_attention(&q,&bad,&v,&edges,None),Err(Error::InvalidShape{..})));
    }
    for bad in [Tensor::scalar(1.),Tensor::zeros(&[1,3]),Tensor::zeros(&[2,2,1])] {
        assert!(matches!(edge_attention(&q,&q,&bad,&edges,None),Err(Error::InvalidShape{..})));
    }
    for bad in [Tensor::scalar(0.),Tensor::zeros(&[2,1]),Tensor::zeros(&[3])] {
        assert!(matches!(edge_attention(&q,&q,&v,&edges,Some(&bad)),Err(Error::InvalidShape{..})));
    }
    for dtype in [DType::I64,DType::F64,DType::F16,DType::BF16] {
        for slot in 0..4 {
            let mut args = [q.clone(),q.clone(),v.clone(),bias.clone()];
            args[slot] = args[slot].to_dtype(dtype);
            assert!(matches!(edge_attention(&args[0],&args[1],&args[2],&edges,Some(&args[3])),Err(Error::DtypeMismatch{..})));
        }
    }
    let huge = Tensor::from_vec(vec![], &[0,isize::MAX as usize / 4 + 1]).unwrap();
    assert!(matches!(edge_attention(&huge,&huge,&Tensor::zeros(&[0,0]),&EdgeIndex::new(0,vec![],vec![]).unwrap(),None),Err(Error::InvalidShape{..})));
    assert!(matches!(capture(|| edge_attention(&q,&q,&v,&edges,None)),Err(Error::Unsupported{op:"edge_attention",..})));
    for value in [f32::INFINITY,f32::NAN] {
        assert!(edge_attention(&q,&q,&v,&edges,Some(&Tensor::full(&[2],value))).is_err());
    }
    let q = q.requires_grad_(true).unwrap();
    let result = edge_attention(&q,&q,&v,&edges,None).unwrap().sum();
    assert!(matches!(result.grad_wrt(&[&q],true),Err(Error::Unsupported{..})));
    assert!(q.grad().is_none());
    assert!(result.grad_wrt(&[&q],false).is_ok());
}

#[test]
fn device_inputs_reject_before_download_or_compute() {
    use ferro_core::dispatch::{Backend,BinaryKind,DeviceBuffer,UnaryKind};
    use std::sync::{Arc,Mutex};
    static LOCK: Mutex<()> = Mutex::new(());
    const DEV: Device = Device::Cuda(241);
    struct Buffer(usize);
    impl DeviceBuffer for Buffer {
        fn device(&self) -> Device { DEV }
        fn len(&self) -> usize { self.0 }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }
    struct Refuse;
    impl Backend for Refuse {
        fn unary(&self,_:UnaryKind,_:&[f32])->Vec<f32>{panic!("unexpected compute")}
        fn binary(&self,_:BinaryKind,_:&[f32],_:&[f32])->Vec<f32>{panic!("unexpected compute")}
        fn matmul(&self,_:&[f32],_:&[f32],_:usize,_:usize,_:usize)->Vec<f32>{panic!("unexpected compute")}
        fn alloc_from_host(&self,data:&[f32])->ferro_core::Result<Box<dyn DeviceBuffer>>{Ok(Box::new(Buffer(data.len())))}
        fn copy_to_host(&self,_:&dyn DeviceBuffer)->ferro_core::Result<Vec<f32>>{panic!("unexpected download")}
    }
    let _guard = LOCK.lock().unwrap_or_else(|e|e.into_inner());
    ferro_core::register_backend(DEV,Arc::new(Refuse));
    let edges = graph();
    for slot in 0..4 {
        let mut args = [features(5,2,0.1),features(5,2,0.2),features(5,3,0.3),Tensor::zeros(&[8])];
        args[slot] = args[slot].to_device(DEV).unwrap();
        assert!(matches!(edge_attention(&args[0],&args[1],&args[2],&edges,Some(&args[3])),Err(Error::Unsupported{op:"edge_attention",..})));
    }
}

#[test]
fn sparse_forward_and_backward_allocations_stay_far_below_a_dense_node_matrix() {
    struct Reset;
    impl Drop for Reset { fn drop(&mut self) { WATCH.with(|w|w.set(false)); } }
    const N: usize = 4096;
    let edges = EdgeIndex::new(N,(0..128).collect(),(0..128).map(|e|N-1-e%16).collect()).unwrap();
    let q = features(N,4,0.1).requires_grad_(true).unwrap();
    let k = features(N,4,-0.2).requires_grad_(true).unwrap();
    let v = features(N,3,0.3).requires_grad_(true).unwrap();
    TOTAL.with(|x|x.set(0)); LARGEST.with(|x|x.set(0)); WATCH.with(|x|x.set(true));
    let reset = Reset;
    let output = edge_attention(&q,&k,&v,&edges,None).unwrap();
    output.sum().backward();
    drop(reset);
    let (total,largest) = (TOTAL.with(Cell::get),LARGEST.with(Cell::get));
    assert!(largest <= 256*1024,"unexpected dense allocation: {largest} bytes");
    assert!(total <= 8*1024*1024,"unexpected aggregate allocation: {total} bytes");
    assert!(total < N*N*std::mem::size_of::<f32>() / 8);
    assert!(q.grad().is_some() && k.grad().is_some() && v.grad().is_some());
}
