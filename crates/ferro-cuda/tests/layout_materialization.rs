use ferro_core::dispatch::Backend;

// Set FERRO_REQUIRE_CUDA=1 for GPU verification: initialization failures must fail.
fn cuda_or_skip<T>(result: Result<T, String>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(),
                "CUDA required for layout verification: {error}");
            eprintln!("SKIP CUDA layout verification: {error}; set FERRO_REQUIRE_CUDA=1 to fail instead");
            None
        }
    }
}

#[test]
fn direct_layout_values_and_bounds() {
    let Some(b) = cuda_or_skip(ferro_cuda::CudaBackend::new(0)) else { return };
    assert_eq!(b.layout_counts(), (0, 0, 0));
    let x = b.alloc_from_host(&(0..24).map(|i| i as f32).collect::<Vec<_>>()).unwrap();
    let out = b.materialize_dev(x.as_ref(), &[3, 2, 4], &[4, 12, 1], 0).unwrap();
    assert_eq!(b.copy_to_host(out.as_ref()).unwrap(), vec![0.,1.,2.,3.,12.,13.,14.,15.,4.,5.,6.,7.,16.,17.,18.,19.,8.,9.,10.,11.,20.,21.,22.,23.]);
    let offset = b.materialize_dev(x.as_ref(), &[2, 2], &[1, 4], 3).unwrap();
    assert_eq!(b.copy_to_host(offset.as_ref()).unwrap(), vec![3.,7.,4.,8.]);
    let scalar = b.materialize_dev(x.as_ref(), &[], &[], 7).unwrap();
    assert_eq!(b.copy_to_host(scalar.as_ref()).unwrap(), vec![7.]);
    let broadcast = b.materialize_dev(x.as_ref(), &[2, 3], &[0, 1], 1).unwrap();
    assert_eq!(b.copy_to_host(broadcast.as_ref()).unwrap(), vec![1.,2.,3.,1.,2.,3.]);
    assert_eq!(b.materialize_dev(x.as_ref(), &[0, 3], &[3, 1], 24).unwrap().len(), 0);
    assert_eq!(b.layout_counts(), (0, 0, 4));
    let _idx = b.alloc_i64_from_host(&[0, 1]).unwrap();
    assert_eq!(b.layout_counts(), (1, 16, 4));
    assert_eq!(b.materialize_dev(x.as_ref(), &[1; 16], &[0; 16], 23).unwrap().len(), 1);
    assert_eq!(b.materialize_dev(x.as_ref(), &[usize::MAX, 0], &[usize::MAX, 1], 24).unwrap().len(), 0);
    assert!(b.materialize_dev(x.as_ref(), &[1; 17], &[0; 17], 0).is_err());
    assert!(b.materialize_dev(x.as_ref(), &[0], &[1], 25).is_err());
    for (s, st, off) in [(vec![2], vec![], 0), (vec![2], vec![24], 0), (vec![2], vec![usize::MAX], 1), (vec![usize::MAX, 2], vec![1, 1], 0), (vec![], vec![], 24)] {
        assert!(b.materialize_dev(x.as_ref(), &s, &st, off).is_err());
    }
}

#[test]
fn eager_and_compiled_model_layouts_have_zero_index_uploads() {
    use ferro_core::{Tensor, Device, capture};
    use ferro_core::graph::CompiledChain;
    // Only this test uses the process registry; the direct test owns its backend.
    let Some(()) = cuda_or_skip(ferro_cuda::install(0)) else { return };
    let b = ferro_cuda::cuda_backend().unwrap();
    for (shape, a, c, out_shape) in [
        (vec![128,8,32],0,1,vec![8,128,32]),
        (vec![128,8,32],0,1,vec![8,128,32]),
        (vec![128,8,32],0,1,vec![8,128,32]),
        (vec![8,128,32],1,2,vec![8,32,128]),
        (vec![8,128,32],0,1,vec![128,256]),
    ] {
        let values: Vec<f32> = (0..32768).map(|i| i as f32).collect();
        let cpu = Tensor::from_vec(values, &shape).unwrap();
        let x = cpu.to_device(Device::Cuda(0)).unwrap();
        let before = b.layout_counts();
        let build = || x.transpose(a,c).unwrap().reshape(&out_shape).unwrap();
        let root = capture(build);
        let graph = CompiledChain::compile(&root).unwrap();
        let expected = cpu.transpose(a,c).unwrap().reshape(&out_shape).unwrap().to_vec();
        assert_eq!(root.to_vec(), expected);
        assert_eq!(graph.replay().unwrap().to_vec(), expected);
        let after = b.layout_counts();
        assert_eq!((after.0-before.0,after.1-before.1,after.2-before.2),(0,0,2));
        println!("layout {shape:?}: eager+replay index uploads=0 bytes=0 direct enqueues=2");
    }
}
