use super::*;

#[cfg(test)]
mod failure_tests {
    use super::*;
    #[test]
    fn model_preparation_failure_releases_capture_and_resources() {
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
    stream: Arc<CudaStream>,
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
            if n==0 { return Err(bad("empty static destinations unsupported".into())); }
            match &run.op {
                StaticOp::Pointwise(steps) => {
                    let steps = Self::convert_steps(steps);
                    if steps.iter().any(|s| match s { ChainStep::Unary(_)=>false,ChainStep::Binary {other,..}|ChainStep::BinaryBc {other,..}=>*other>=run.inputs.len() }) { return Err(bad("invalid pointwise operand".into())); }
                    if steps.is_empty() || kernels::chain_input_count(&steps)!=run.inputs.len() || sizes[run.inputs[0]]!=n { return Err(bad("invalid pointwise run".into())); }
                    for step in steps {
                        match step {
                            ChainStep::Unary(_) => {},
                            ChainStep::Binary {other,..} if other < run.inputs.len() && sizes[run.inputs[other]]==n => {},
                            ChainStep::BinaryBc {other,dims,strides,..} if other<run.inputs.len() && dims.len()==strides.len() && dims.iter().try_fold(1usize,|a,&d|a.checked_mul(d as usize))==Some(n) && dims.iter().all(|&d|d>0) => {
                                let max = dims.iter().zip(&strides).try_fold(0usize, |a,(&d,&s)| a.checked_add((d as usize-1).checked_mul(s as usize)?));
                                if max.is_none_or(|v|v>=sizes[run.inputs[other]]) { return Err(bad("broadcast out of bounds".into())); }
                            },
                            _ => return Err(bad("unsupported broadcast".into())),
                        }
                    }
                },
                StaticOp::MatMul {m,k,n: cols} => {
                    as_i32("prepare_model_graph", (*m).max(*k).max(*cols))?;
                    if run.inputs.len()!=2 || *k==0 || m.checked_mul(*k)!=Some(sizes[run.inputs[0]]) || k.checked_mul(*cols)!=Some(sizes[run.inputs[1]]) || m.checked_mul(*cols)!=Some(n) {
                        return Err(bad("invalid or empty static matmul".into()));
                    }
                },
                StaticOp::Bmm {batch,m,k,n:cols} => {
                    as_i32("prepare_model_graph",(*batch).max(*m).max(*k).max(*cols))?;
                    if run.inputs.len()!=2 || *k==0 || batch.checked_mul(*m).and_then(|v|v.checked_mul(*k))!=Some(sizes[run.inputs[0]]) || batch.checked_mul(*k).and_then(|v|v.checked_mul(*cols))!=Some(sizes[run.inputs[1]]) || batch.checked_mul(*m).and_then(|v|v.checked_mul(*cols))!=Some(n) { return Err(bad("invalid static bmm".into())); }
                },
                StaticOp::Layout {strides} => {
                    if run.inputs.len()!=1 || strides.len()!=run.shape.len() || strides.len()>16 || sizes[run.inputs[0]]!=n { return Err(bad("invalid static layout".into())); }
                    let max = run.shape.iter().zip(strides).try_fold(0usize,|a,(&d,&s)|a.checked_add((d-1).checked_mul(s)?));
                    if max.is_none_or(|v|v>=sizes[run.inputs[0]]) { return Err(bad("layout out of bounds".into())); }
                },
                StaticOp::Softmax {rows,cols} => {
                    if run.inputs.len()!=1 || rows.checked_mul(*cols)!=Some(n) || sizes[run.inputs[0]]!=n || *cols==0 { return Err(bad("invalid static softmax".into())); }
                },
                StaticOp::Sum {red,inner} => {
                    if run.inputs.len()!=1 || n.checked_mul(*red)!=Some(sizes[run.inputs[0]]) || *inner==0 || n%inner!=0 { return Err(bad("invalid static sum".into())); }
                    as_u32("prepare_model_graph",*red)?; as_u32("prepare_model_graph",*inner)?;
                },
                StaticOp::LayerNorm {rows,cols,weight,bias,..} => {
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
        for &n in &sizes[leaves.len()..] { outputs.push(self.alloc_zeros("prepare_model_graph", n)?); }
        let mut commands = Vec::new();
        for (i, run) in runs.iter().enumerate() {
            let out = leaves.len()+i;
            let n = sizes[out] as u32;
            let kernel = |source: String, args, config| -> Result<PreparedCommand> {
                Ok(PreparedCommand::Kernel { function:self.get_kernel("prepare_model_graph", &source)?, args, config })
            };
            let command = match &run.op {
                StaticOp::Pointwise(steps) => {
                    let function = self.get_kernel("prepare_model_graph", &kernels::chain_source(&Self::convert_steps(steps)))?;
                    let mut args: Vec<_> = run.inputs.iter().map(|&i| Arg::Pointer(i)).collect();
                    args.extend(kernels::chain_bc_args(&Self::convert_steps(steps)).into_iter().map(Arg::U32));
                    args.push(Arg::Pointer(out)); args.push(Arg::U32(n));
                    PreparedCommand::Kernel { function, args, config: LaunchConfig::for_num_elems(n) }
                },
                StaticOp::MatMul {m,k,n} => PreparedCommand::Gemm { a:run.inputs[0],b:run.inputs[1],out,batch:1,m:*m as i32,k:*k as i32,n:*n as i32 },
                StaticOp::Bmm {batch,m,k,n} => PreparedCommand::Gemm { a:run.inputs[0],b:run.inputs[1],out,batch:*batch as i32,m:*m as i32,k:*k as i32,n:*n as i32 },
                StaticOp::Layout {strides} => {
                    let mut args = vec![Arg::Pointer(run.inputs[0]),Arg::Pointer(out),Arg::U32(n),Arg::U32(run.shape.len() as u32),Arg::U64(0)];
                    for d in 0..16 { args.push(Arg::U64(*run.shape.get(d).unwrap_or(&1) as u64)); args.push(Arg::U64(*strides.get(d).unwrap_or(&0) as u64)); }
                    kernel(kernels::materialize_source(),args,LaunchConfig::for_num_elems(n))?
                },
                StaticOp::Sum {red,inner} => kernel(kernels::sum_dim_source(), vec![Arg::Pointer(run.inputs[0]),Arg::Pointer(out),Arg::U32(n),Arg::U32(*red as u32),Arg::U32(*inner as u32)],LaunchConfig::for_num_elems(n))?,
                StaticOp::LayerNorm {rows,cols,eps,weight,bias} => kernel(concat!("#define OUTPUT_ONLY\n",include_str!("../layer_norm.cu")).into(),vec![Arg::Pointer(run.inputs[0]),Arg::Pointer(run.inputs[1]),Arg::Pointer(run.inputs[2]),Arg::Pointer(out),Arg::U32(*cols as u32),Arg::F32(*eps),Arg::U32(*weight as u32),Arg::U32(*bias as u32)],LaunchConfig {grid_dim:(*rows as u32,1,1),block_dim:(256,1,1),shared_mem_bytes:0})?,
                StaticOp::Softmax {rows,cols} => {
                    let scratch = leaves.len()+outputs.len();
                    outputs.push(self.alloc_zeros("prepare_model_graph",2*rows)?);
                    commands.push(kernel(kernels::softmax_row_stats_source(),vec![Arg::Pointer(run.inputs[0]),Arg::Pointer(scratch),Arg::U32(*cols as u32)],LaunchConfig {grid_dim:(*rows as u32,1,1),block_dim:(kernels::REDUCE_BLOCK,1,1),shared_mem_bytes:0})?);
                    kernel(kernels::softmax_apply_source(false),vec![Arg::Pointer(run.inputs[0]),Arg::Pointer(scratch),Arg::Pointer(out),Arg::U32(n),Arg::U32(*cols as u32)],LaunchConfig::for_num_elems(n))?
                },
            };
            commands.push(command);
        }
        let resources = Resources { blas, _workspace: workspace, stream, commands };
        let mut pointers = Vec::new();
        let mut guards = Vec::new();
        for leaf in &leaves {
            let (ptr,g) = self.resident("prepare_model_graph", leaf.as_ref())?.data.device_ptr(&self.stream);
            pointers.push(ptr); guards.push(g);
        }
        for output in &mut outputs {
            let (ptr,g) = output.device_ptr_mut(&self.stream);
            pointers.push(ptr); guards.push(g);
        }
        self.ctx.check_err().map_err(|e| bad(e.to_string()))?;
        self.stream.synchronize().map_err(|e| bad(e.to_string()))?;
        for command in &resources.commands { command.enqueue(&resources.stream, &resources.blas, &pointers)?; }
        #[cfg(test)] injected("warm")?;
        resources.stream.synchronize().map_err(|e| bad(e.to_string()))?;
        drop(guards);
        let capture = CaptureSession::begin(resources.stream.clone())?;
        for command in &resources.commands { command.enqueue(&resources.stream, &resources.blas, &pointers)?; }
        #[cfg(test)] injected("enqueue")?;
        let graph = capture.finish()?;
        #[cfg(test)] injected("end")?;
        let mut nodes = 0;
        unsafe { sys::cuGraphGetNodes(graph.graph, std::ptr::null_mut(), &mut nodes).result() }.map_err(|e| bad(e.to_string()))?;
        unsafe { result::graph::upload(graph.exec, self.stream.cu_stream()) }.map_err(|e| bad(e.to_string()))?;
        #[cfg(test)] injected("upload")?;
        self.stream.synchronize().map_err(|e| bad(e.to_string()))?;
        self.ctx.check_err().map_err(|e| bad(e.to_string()))?;
        *users += 1;
        Ok(StaticPointwiseGraph { graph: Some(graph), backend: self.clone(), leaves, output_index:runs.len()-1, outputs, _commands: Vec::new(), _model: Some(resources), nodes, replays: 0, poisoned: false })
    }
}
