use super::*;

#[cfg(test)]
mod failure_tests {
    use super::*;
    #[test]
    fn zero_work_is_a_real_empty_graph_and_zero_gemm_rewrites_output() {
        let _context = TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
        let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
        let leaf:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[]).unwrap());
        let runs=[StaticRun {inputs:vec![0],op:StaticOp::Layout {strides:vec![1]},shape:vec![0]}];
        let mut empty=b.prepare_model_graph(&runs,vec![leaf.clone()]).unwrap();
        assert!(empty.graph.is_some());assert_eq!(empty.kernel_nodes(),0);
        assert!(empty._model.as_ref().unwrap().commands.is_empty());
        empty.replay().unwrap();assert_eq!(empty.replay_count(),1);
        assert!(empty.copy_output_to_host().unwrap().is_empty());drop(empty);
        for op in [StaticOp::MatMul {m:2,k:0,n:3},StaticOp::Bmm {batch:2,m:1,k:0,n:3}] {
            let runs=[StaticRun {inputs:vec![0,0],op,shape:vec![2,3]}];
            let mut graph=b.prepare_model_graph(&runs,vec![leaf.clone()]).unwrap();
            assert_eq!(graph.kernel_nodes(),1);
            for count in 1..=3 {
                b.stream.memcpy_htod(&[99.;6],&mut graph.outputs[graph.output_index]).unwrap();
                graph.replay().unwrap();assert_eq!(graph.replay_count(),count);
                assert_eq!(graph.copy_output_to_host().unwrap(),vec![0.;6]);
            }
        }
    }

    #[test]
    fn contiguous_layouts_alias_private_slots_without_allocations_or_nodes() {
        let _context = TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
        let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
        let a:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[1.,2.,3.,4.]).unwrap());
        let runs=[
            StaticRun {inputs:vec![0],op:StaticOp::Layout {strides:vec![2,1]},shape:vec![2,2]},
            StaticRun {inputs:vec![1],op:StaticOp::Layout {strides:vec![1]},shape:vec![4]},
            StaticRun {inputs:vec![2,1],op:StaticOp::MatMul {m:2,k:2,n:2},shape:vec![2,2]},
        ];
        let before=b.alloc_stats().requests;
        let mut graph=b.prepare_model_graph(&runs,vec![a.clone()]).unwrap();
        assert_eq!(b.alloc_stats().requests-before,1,"metadata layouts must not allocate");
        assert_eq!(graph.outputs.len(),1);
        assert_eq!(graph._model.as_ref().unwrap().commands.len(),1);
        let direct=[StaticRun {inputs:vec![0,0],op:StaticOp::MatMul {m:2,k:2,n:2},shape:vec![2,2]}];
        let reference=b.prepare_model_graph(&direct,vec![a.clone()]).unwrap();
        assert_eq!(graph.kernel_nodes(),reference.kernel_nodes(),"aliases add no CUDA nodes");
        drop(reference);
        graph.replay().unwrap();
        assert_eq!(graph.copy_output_to_host().unwrap(),vec![7.,10.,15.,22.]);
        let saved=graph.snapshot().unwrap();
        b.copy_into(a.as_ref(),&[2.,0.,0.,3.]).unwrap();
        drop(a);
        let allocations=b.alloc_stats().requests;
        graph.replay().unwrap();
        assert_eq!(b.alloc_stats().requests,allocations);
        assert_eq!(graph.copy_output_to_host().unwrap(),vec![4.,0.,0.,9.]);
        drop(graph);
        assert_eq!(b.copy_to_host(saved.as_ref()).unwrap(),vec![7.,10.,15.,22.]);
    }

    #[test]
    fn layout_alias_mapping_preserves_transposes_scratch_and_final_copies() {
        let _context = TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
        let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
        let a:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[1.,2.,3.,4.]).unwrap());
        let runs=[
            StaticRun {inputs:vec![0],op:StaticOp::Layout {strides:vec![0,2,1]},shape:vec![1,2,2]},
            StaticRun {inputs:vec![1],op:StaticOp::Layout {strides:vec![1,2]},shape:vec![2,2]},
            StaticRun {inputs:vec![2],op:StaticOp::Softmax {rows:2,cols:2},shape:vec![2,2]},
            StaticRun {inputs:vec![3],op:StaticOp::Layout {strides:vec![1]},shape:vec![4]},
        ];
        let before=b.alloc_stats().requests;
        let mut graph=b.prepare_model_graph(&runs,vec![a.clone()]).unwrap();
        assert_eq!(b.alloc_stats().requests-before,4); // transpose, softmax, final copy, scratch
        assert_eq!(graph._model.as_ref().unwrap().commands.len(),4);
        assert_eq!(graph.kernel_nodes(),4);
        graph.replay().unwrap();
        let values=graph.copy_output_to_host().unwrap();
        for (v,e) in values.iter().zip([0.11920292,0.880797,0.11920292,0.880797]) { assert!((v-e).abs()<1e-6); }
        // An all-layout graph must still own an independent public output.
        let last=[StaticRun {inputs:vec![0],op:StaticOp::Layout {strides:vec![1]},shape:vec![4]}];
        let mut graph=b.prepare_model_graph(&last,vec![a.clone()]).unwrap();
        graph.replay().unwrap();
        b.copy_into(a.as_ref(),&[9.;4]).unwrap();
        assert_eq!(graph.copy_output_to_host().unwrap(),vec![1.,2.,3.,4.]);
        graph.replay().unwrap();
        assert_eq!(graph.copy_output_to_host().unwrap(),vec![9.;4]);
    }

    #[test]
    fn model_preparation_failure_releases_capture_and_resources() {
        let _context = TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
        let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
        for stage in ["warm", "enqueue", "end", "upload"] {
            let a:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[1.,2.,3.,4.]).unwrap());
            let runs=[StaticRun {inputs:vec![0,0],op:StaticOp::MatMul {m:2,k:2,n:2},shape:vec![2,2]}];
            INJECT.with(|f| f.set(Some(stage)));
            assert!(b.prepare_model_graph(&runs,vec![a.clone()]).is_err());
            INJECT.with(|f| f.set(None));
            assert!(b.ctx.is_event_tracking());
            assert_eq!(b.stream.capture_status().unwrap(),sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE);
            assert_eq!(*b.static_graphs.lock().unwrap(),0);
            b.ctx.check_err().unwrap();
            let mut graph=b.prepare_model_graph(&runs,vec![a]).unwrap();
            graph.replay().unwrap();
            assert_eq!(graph.copy_output_to_host().unwrap(),vec![7.,10.,15.,22.]);
        }
    }
}

use ferro_core::dispatch::{StaticOp, StaticRun};
#[cfg(test)]
thread_local! { static INJECT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
fn injected(stage: &'static str) -> Result<()> {
    if INJECT.with(|f| f.get()==Some(stage)) { Err(cuda_err("static_model_injected",stage)) } else { Ok(()) }
}

enum Arg { Pointer(usize), U32(u32), U64(u64), F32(f32) }
enum PreparedCommand {
    Kernel { function: CudaFunction, args: Vec<Arg>, config: LaunchConfig },
    Gemm { a: usize, b: usize, out: usize, batch: i32, m: i32, k: i32, n: i32 },
}
impl PreparedCommand {
    fn enqueue(&self, stream: &CudaStream, blas: &CudaBlas, pointers: &[u64]) -> Result<()> {
        match self {
            Self::Kernel { function, args, config } => {
                let mut launch = stream.launch_builder(function);
                for arg in args {
                    match arg { Arg::Pointer(i) => { launch.arg(&pointers[*i]); }, Arg::U32(v) => { launch.arg(v); }, Arg::U64(v) => { launch.arg(v); }, Arg::F32(v) => { launch.arg(v); } }
                }
                unsafe { launch.launch(*config) }.map_err(|e| cuda_err("static_model_kernel", e))?;
            },
            Self::Gemm { a,b,out,batch,m,k,n } => {
                use cudarc::cublas::{result, sys::cublasOperation_t::CUBLAS_OP_N as N};
                // Row-major C = A B is column-major C^T = B^T A^T.
                // All extents, original allocations and lifetimes were validated.
                unsafe { result::sgemm_strided_batched(*blas.handle(), N,N,*n,*m,*k,&1.0,
                    pointers[*b] as *const f32,*n,(*k as i64)*(*n as i64),pointers[*a] as *const f32,*k,(*m as i64)*(*k as i64),
                    &0.0,pointers[*out] as *mut f32,*n,(*m as i64)*(*n as i64),*batch) }.map_err(|e| cuda_err("static_model_gemm", e))?;
            },
        }
        Ok(())
    }
}

pub(super) struct Resources {
    // Handle must be destroyed before its workspace and private stream.
    blas: CudaBlas,
    _workspace: CudaSlice<u8>,
    pub(super) stream: Arc<CudaStream>,
    commands: Vec<PreparedCommand>,
}

impl Drop for Resources {
    fn drop(&mut self) {
        // On a failed warmup, destinations are freed on the backend stream,
        // not the private stream. Fence private work before those owners drop.
        self.stream.context().record_err(self.stream.synchronize());
    }
}

impl CudaBackend {
    pub(crate) fn prepare_model_graph(self: &Arc<Self>, runs: &[StaticRun], leaves: Vec<Arc<dyn DeviceBuffer>>) -> Result<StaticPointwiseGraph> {
        let mut users = self.static_graphs.lock().unwrap_or_else(|e| e.into_inner());
        let bad = |msg| cuda_err("prepare_model_graph", msg);
        if !self.ctx.is_event_tracking() || self.stream.capture_status().map_err(|e| bad(e.to_string()))? != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(bad("another capture mode is active".into()));
        }
        if runs.is_empty() || leaves.is_empty() { return Err(bad("empty static model".into())); }
        let mut sizes = Vec::new();
        for leaf in &leaves { sizes.push(self.resident("prepare_model_graph", leaf.as_ref())?.len()); }
        for run in runs {
            if run.inputs.is_empty() || run.inputs.iter().any(|&i| i>=sizes.len()) { return Err(bad("invalid input slot".into())); }
            let n = run.shape.iter().try_fold(1usize, |a,b| a.checked_mul(*b)).ok_or_else(|| bad("shape overflow".into()))?;
            as_u32("prepare_model_graph", n)?;
            for &extent in &run.shape { as_u32("prepare_model_graph",extent)?; }
            run.shape.iter().try_fold(1usize,|a,&d|a.checked_mul(d.max(1))).ok_or_else(||bad("shape stride overflow".into()))?;

            match &run.op {
                StaticOp::Pointwise(steps) => {
                    let steps = Self::convert_steps(steps);
                    if steps.iter().any(|s| match s { ChainStep::Unary(_)=>false,ChainStep::Binary {other,..}|ChainStep::BinaryBc {other,..}=>*other>=run.inputs.len() }) { return Err(bad("invalid pointwise operand".into())); }
                    if steps.is_empty() || kernels::chain_input_count(&steps)!=run.inputs.len() || sizes[run.inputs[0]]!=n { return Err(bad("invalid pointwise run".into())); }
                    for step in steps {
                        match step {
                            ChainStep::Unary(_) => {},
                            ChainStep::Binary {other,..} if other < run.inputs.len() && sizes[run.inputs[other]]==n => {},
                            ChainStep::BinaryBc {other,dims,strides,..} if other<run.inputs.len() && dims.len()==strides.len() && dims.iter().try_fold(1usize,|a,&d|a.checked_mul(d as usize))==Some(n) => {
                                if n==0 { continue; }
                                let max = dims.iter().zip(&strides).try_fold(0usize, |a,(&d,&s)| a.checked_add((d as usize-1).checked_mul(s as usize)?));
                                if max.is_none_or(|v|v>=sizes[run.inputs[other]]) { return Err(bad("broadcast out of bounds".into())); }
                            },
                            _ => return Err(bad("unsupported broadcast".into())),
                        }
                    }
                },
                StaticOp::MatMul {m,k,n: cols} => {
                    as_i32("prepare_model_graph", (*m).max(*k).max(*cols))?;
                    if run.inputs.len()!=2 || m.checked_mul(*k)!=Some(sizes[run.inputs[0]]) || k.checked_mul(*cols)!=Some(sizes[run.inputs[1]]) || m.checked_mul(*cols)!=Some(n) {
                        return Err(bad("invalid or empty static matmul".into()));
                    }
                },
                StaticOp::Bmm {batch,m,k,n:cols} => {
                    as_i32("prepare_model_graph",(*batch).max(*m).max(*k).max(*cols))?;
                    if run.inputs.len()!=2 || batch.checked_mul(*m).and_then(|v|v.checked_mul(*k))!=Some(sizes[run.inputs[0]]) || batch.checked_mul(*k).and_then(|v|v.checked_mul(*cols))!=Some(sizes[run.inputs[1]]) || batch.checked_mul(*m).and_then(|v|v.checked_mul(*cols))!=Some(n) { return Err(bad("invalid static bmm".into())); }
                },
                StaticOp::Layout {strides} | StaticOp::Broadcast {strides} => {
                    if run.inputs.len()!=1 || strides.len()!=run.shape.len() || strides.len()>16 || (matches!(run.op,StaticOp::Layout {..}) && sizes[run.inputs[0]]!=n) { return Err(bad("invalid static layout".into())); }
                    if n==0 { sizes.push(n); continue; }
                    let max = run.shape.iter().zip(strides).try_fold(0usize,|a,(&d,&s)|a.checked_add((d-1).checked_mul(s)?));
                    if max.is_none_or(|v|v>=sizes[run.inputs[0]]) { return Err(bad("layout out of bounds".into())); }
                },
                StaticOp::Softmax {rows,cols} => {
                    as_u32("prepare_model_graph",*rows)?; as_u32("prepare_model_graph",*cols)?;
                    if run.inputs.len()!=1 || rows.checked_mul(*cols)!=Some(n) || sizes[run.inputs[0]]!=n { return Err(bad("invalid static softmax".into())); }
                },
                StaticOp::Sum {red,inner} => {
                    if run.inputs.len()!=1 || n.checked_mul(*red)!=Some(sizes[run.inputs[0]]) || (n!=0 && (*inner==0 || n%inner!=0)) { return Err(bad("invalid static sum".into())); }
                    as_u32("prepare_model_graph",*red)?; as_u32("prepare_model_graph",*inner)?;
                },
                StaticOp::LayerNorm {rows,cols,weight,bias,..} => {
                    as_u32("prepare_model_graph",*rows)?; as_u32("prepare_model_graph",*cols)?;
                    if run.inputs.len()!=3 || rows.checked_mul(*cols)!=Some(n) || sizes[run.inputs[0]]!=n || *cols==0 || (*weight && sizes[run.inputs[1]]!=*cols) || (*bias && sizes[run.inputs[2]]!=*cols) { return Err(bad("invalid static normalization".into())); }
                },
            }
            sizes.push(n);
        }
        let stream = self.ctx.new_stream().map_err(|e| bad(e.to_string()))?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| bad(e.to_string()))?;
        let mut workspace = stream.alloc_zeros::<u8>(4<<20).map_err(|e| bad(e.to_string()))?;
        {
            let (ptr, guard) = workspace.device_ptr_mut(&stream);
            unsafe { cudarc::cublas::sys::cublasSetWorkspace_v2(*blas.handle(), ptr as _, 4<<20).result() }.map_err(|e| bad(e.to_string()))?;
            drop(guard);
        }
        let mut outputs = Vec::new();
        // Logical run slots map to unique owned allocations. Only interior identity
        // layouts alias: the public output remains a private destination/lease.
        let mut slots: Vec<usize> = (0..leaves.len()).collect();
        let mut aliases = Vec::new();
        for (i, run) in runs.iter().enumerate() {
            let alias = i+1<runs.len() && match &run.op {
                StaticOp::Layout {strides} => {
                    let mut stride = 1;
                    run.shape.iter().zip(strides).rev().all(|(&d,&s)| {
                        let same = d==1 || s==stride;
                        stride *= d;
                        same
                    })
                },
                _ => false,
            };
            aliases.push(alias);
            if alias { slots.push(slots[run.inputs[0]]); }
            else {
                slots.push(leaves.len()+outputs.len());
                outputs.push(self.alloc_zeros("prepare_model_graph", sizes[leaves.len()+i])?);
            }
        }
        let output_index = slots[leaves.len()+runs.len()-1]-leaves.len();
        let mut commands = Vec::new();
        for (i, run) in runs.iter().enumerate() {
            if aliases[i] { continue; }
            let out = slots[leaves.len()+i];
            let n = sizes[leaves.len()+i] as u32;
            if n==0 { continue; }
            let inputs: Vec<_> = run.inputs.iter().map(|&slot|slots[slot]).collect();
            let kernel = |source: String, args, config| -> Result<PreparedCommand> {
                Ok(PreparedCommand::Kernel { function:self.get_kernel("prepare_model_graph", &source)?, args, config })
            };
            let command = match &run.op {
                StaticOp::Pointwise(steps) => {
                    let function = self.get_kernel("prepare_model_graph", &kernels::chain_source(&Self::convert_steps(steps)))?;
                    let mut args: Vec<_> = inputs.iter().map(|&i| Arg::Pointer(i)).collect();
                    args.extend(kernels::chain_bc_args(&Self::convert_steps(steps)).into_iter().map(Arg::U32));
                    args.push(Arg::Pointer(out)); args.push(Arg::U32(n));
                    PreparedCommand::Kernel { function, args, config: LaunchConfig::for_num_elems(n) }
                },
                StaticOp::MatMul {k:0,..} | StaticOp::Bmm {k:0,..} => kernel(kernels::fill_source(),vec![Arg::Pointer(out),Arg::U32(n),Arg::F32(0.)],LaunchConfig::for_num_elems(n))?,
                StaticOp::MatMul {m,k,n} => PreparedCommand::Gemm { a:inputs[0],b:inputs[1],out,batch:1,m:*m as i32,k:*k as i32,n:*n as i32 },
                StaticOp::Bmm {batch,m,k,n} => PreparedCommand::Gemm { a:inputs[0],b:inputs[1],out,batch:*batch as i32,m:*m as i32,k:*k as i32,n:*n as i32 },
                StaticOp::Layout {strides} | StaticOp::Broadcast {strides} => {
                    let mut args = vec![Arg::Pointer(inputs[0]),Arg::Pointer(out),Arg::U32(n),Arg::U32(run.shape.len() as u32),Arg::U64(0)];
                    for d in 0..16 { args.push(Arg::U64(*run.shape.get(d).unwrap_or(&1) as u64)); args.push(Arg::U64(*strides.get(d).unwrap_or(&0) as u64)); }
                    kernel(kernels::materialize_source(),args,LaunchConfig::for_num_elems(n))?
                },
                StaticOp::Sum {red,inner} => kernel(kernels::sum_dim_source(), vec![Arg::Pointer(inputs[0]),Arg::Pointer(out),Arg::U32(n),Arg::U32(*red as u32),Arg::U32(*inner as u32)],LaunchConfig::for_num_elems(n))?,
                StaticOp::LayerNorm {rows,cols,eps,weight,bias} => kernel(concat!("#define OUTPUT_ONLY\n",include_str!("../layer_norm.cu")).into(),vec![Arg::Pointer(inputs[0]),Arg::Pointer(inputs[1]),Arg::Pointer(inputs[2]),Arg::Pointer(out),Arg::U32(*cols as u32),Arg::F32(*eps),Arg::U32(*weight as u32),Arg::U32(*bias as u32)],LaunchConfig {grid_dim:(*rows as u32,1,1),block_dim:(256,1,1),shared_mem_bytes:0})?,
                StaticOp::Softmax {rows,cols} => {
                    let scratch = leaves.len()+outputs.len();
                    outputs.push(self.alloc_zeros("prepare_model_graph",2*rows)?);
                    commands.push(kernel(kernels::softmax_row_stats_source(),vec![Arg::Pointer(inputs[0]),Arg::Pointer(scratch),Arg::U32(*cols as u32)],LaunchConfig {grid_dim:(*rows as u32,1,1),block_dim:(kernels::REDUCE_BLOCK,1,1),shared_mem_bytes:0})?);
                    kernel(kernels::softmax_apply_source(false),vec![Arg::Pointer(inputs[0]),Arg::Pointer(scratch),Arg::Pointer(out),Arg::U32(n),Arg::U32(*cols as u32)],LaunchConfig::for_num_elems(n))?
                },
            };
            commands.push(command);
        }
        let resources = Resources { blas, _workspace: workspace, stream, commands };
        let mut flight = native::Flight { streams: vec![self.stream.cu_stream(), resources.stream.cu_stream()], calls: Arc::new(native::Driver(resources.stream.clone())), data: Some((outputs, leaves, resources, self.clone())), users: &mut users };
        let (outputs, leaves, resources, _) = flight.data.as_mut().unwrap();
        let mut pointers = Vec::new();
        let mut guards = Vec::new();
        for leaf in leaves.iter() {
            let (ptr,g) = self.resident("prepare_model_graph", leaf.as_ref())?.data.device_ptr(&self.stream);
            pointers.push(ptr); guards.push(g);
        }
        for output in outputs.iter_mut() {
            let (ptr,g) = output.device_ptr_mut(&self.stream);
            pointers.push(ptr); guards.push(g);
        }
        self.ctx.check_err().map_err(|e| bad(e.to_string()))?;
        self.stream.synchronize().map_err(|e| bad(e.to_string()))?;
        for command in &resources.commands { command.enqueue(&resources.stream, &resources.blas, &pointers)?; }
        #[cfg(test)] injected("warm")?;
        native::Driver(resources.stream.clone()).warm_fence().map_err(|e| bad(e.to_string()))?;
        drop(guards);
        let capture = CaptureSession::begin(resources.stream.clone())?;
        for command in &resources.commands { command.enqueue(&resources.stream, &resources.blas, &pointers)?; }
        #[cfg(test)] injected("enqueue")?;
        let graph = capture.finish()?;
        #[cfg(test)] injected("end")?;
        let mut nodes = 0;
        unsafe { sys::cuGraphGetNodes(graph.graph, std::ptr::null_mut(), &mut nodes).result() }.map_err(|e| bad(e.to_string()))?;
        graph.upload(self.stream.cu_stream())?;
        #[cfg(test)] injected("upload")?;
        self.ctx.check_err().map_err(|e| bad(e.to_string()))?;
        let (outputs, leaves, resources, _) = flight.take();
        *users += 1;
        Ok(StaticPointwiseGraph { graph: Some(graph), backend: self.clone(), leaves, output_index, outputs, _commands: Vec::new(), _model: Some(resources), nodes, replays: 0, poisoned: false })
    }
}
