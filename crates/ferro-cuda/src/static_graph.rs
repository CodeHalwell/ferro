//! Owned CUDA graphs for prepared pointwise DAGs and supported static model inference.
use super::*;
use cudarc::driver::{result, sys, DevicePtrMut};
mod model;
// These capture/failure tests share CUDA primary-context native state.
#[cfg(test)]
static TEST_CONTEXT: Mutex<()> = Mutex::new(());
#[cfg(test)]
thread_local! { static REPLAY_GUARDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

/// Slots 0..leaves.len() are owned leaves; each run appends one output slot.
/// Pointwise runs preserve the seed's shape; trailing operands may broadcast.
pub struct StaticPointwiseRun {
    pub inputs: Vec<usize>,
    pub steps: Vec<ChainStep>,
}

struct Command {
    function: CudaFunction,
    inputs: Vec<usize>,
    output: usize,
    n: u32,
    extra: Vec<u32>,
}

impl Command {
    fn enqueue(&self, stream: &CudaStream, pointers: &[u64]) -> Result<()> {
        let mut launch = stream.launch_builder(&self.function);
        for &i in &self.inputs { launch.arg(&pointers[i]); }
        for value in &self.extra { launch.arg(value); }
        launch.arg(&pointers[self.output]).arg(&self.n);
        // Scalar u64 arguments carry validated device addresses, not forged CudaSlice
        // aliases. Original owners and usage guards are retained by the caller.
        unsafe { launch.launch(LaunchConfig::for_num_elems(self.n)) }
            .map_err(|e| cuda_err("static_graph_enqueue", e))?;
        Ok(())
    }
}

mod native;
use native::{CaptureSession, GraphOwner};

/// One chain backed by the same ownership and failure boundary as static graphs.
/// Original input allocations stay pinned; caller-side in-place updates remain
/// visible. A live output borrow prevents replay from overwriting its allocation.
/// This owner is neither Send nor Sync, like StaticPointwiseGraph.
pub struct CapturedChain {
    pub(super) graph: StaticPointwiseGraph,
}

impl CapturedChain {
    /// Replay requires mutable access. Bind the handle with `let mut captured`.
    /// ```compile_fail
    /// fn invalid(chain: &mut ferro_cuda::CapturedChain) {
    ///     let output = chain.output();
    ///     chain.replay().unwrap();
    ///     let _ = output.len();
    /// }
    /// ```
    pub fn replay(&mut self) -> Result<()> { self.graph.replay() }
    /// Borrow the stable output allocation. Panics after a failed replay;
    /// copy_output_to_host reports the poisoned state as an error instead.
    pub fn output(&self) -> &CudaSlice<f32> { self.graph.output().expect("captured chain is poisoned") }
    pub fn copy_output_to_host(&self, backend: &CudaBackend) -> Result<Vec<f32>> {
        backend.dtoh("captured_output", self.graph.output()?)
    }
    pub fn len(&self) -> usize { self.output().len() }
    pub fn is_empty(&self) -> bool { self.output().is_empty() }
}

fn validate_runs(runs: &[StaticPointwiseRun], mut sizes: Vec<usize>) -> Result<Vec<usize>> {
    let invalid = || cuda_err("prepare_static_graph", "invalid pointwise shape, operand or broadcast metadata");
    if sizes.is_empty() || runs.is_empty() { return Err(invalid()); }
    for run in runs {
        if run.inputs.is_empty() || run.steps.is_empty() || run.inputs.iter().any(|&i| i >= sizes.len()) {
            return Err(invalid());
        }
        let n = sizes[run.inputs[0]];
        as_u32("prepare_static_graph", n)?;
        if n == 0 { return Err(invalid()); }
        let mut arity = 1;
        for step in &run.steps {
            match step {
                ChainStep::Unary(_) => {},
                ChainStep::Binary { other, .. } => {
                    if *other >= run.inputs.len() || sizes[run.inputs[*other]] != n { return Err(invalid()); }
                    arity = arity.max(other + 1);
                },
                ChainStep::BinaryBc { other, dims, strides, .. } => {
                    if *other >= run.inputs.len() || dims.len() != strides.len() { return Err(invalid()); }
                    let elements = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d as usize)).ok_or_else(invalid)?;
                    if elements != n { return Err(invalid()); }
                    let last = dims.iter().zip(strides).try_fold(0usize, |acc, (&d, &s)| {
                        (d as usize).checked_sub(1)?.checked_mul(s as usize)?.checked_add(acc)
                    }).ok_or_else(invalid)?;
                    if last > u32::MAX as usize || last >= sizes[run.inputs[*other]] { return Err(invalid()); }
                    arity = arity.max(other + 1);
                },
            }
        }
        if arity != run.inputs.len() { return Err(invalid()); }
        sizes.push(n);
    }
    Ok(sizes)
}

/// A CUDA graph owner shared by pointwise DAG and static model preparation.
/// The model path lowers supported pointwise, matrix, layout, normalization,
/// softmax and reduction runs; this is inference-only, not training capture.
/// Owns original leaves, every destination, functions and backend, plus the
/// private cuBLAS handle, workspace and capture stream used by model commands.
/// It intentionally does not expose Tensor mutation or autograd entry points.
///
/// Replay requires exclusive host access. `output()` borrows the reusable output;
/// use `snapshot()` to retain an independent device result across later replays.
/// Raw CUDA graph handles make this type neither Send nor Sync.
pub struct StaticPointwiseGraph {
    graph: Option<GraphOwner>,
    backend: Arc<CudaBackend>,
    leaves: Vec<Arc<dyn DeviceBuffer>>,
    outputs: Vec<CudaSlice<f32>>,
    output_index: usize,
    _commands: Vec<Command>,
    _model: Option<model::Resources>,
    nodes: usize,
    replays: usize,
    poisoned: bool,
}

impl StaticPointwiseGraph {
    pub fn replay(&mut self) -> Result<()> {
        if self.poisoned { return Err(cuda_err("static_graph_replay", "graph handle is poisoned")); }
        self.poisoned = true;
        let b = &self.backend;
        b.ctx.bind_to_thread().map_err(|e| cuda_err("static_graph_replay", e))?;
        let mut guards = Vec::with_capacity(self.leaves.len() + 1);
        for leaf in &self.leaves {
            guards.push(b.resident("static_graph_replay", leaf.as_ref())?.data.device_ptr(&b.stream).1);
        }
        // Scratch never escapes this !Send/!Sync owner. Preparation fences private
        // warmup; all replays use this one backend stream; Drop fences it before
        // freeing scratch. Stream order therefore covers private accesses. Keep
        // guards for ALL original leaves and the output lease (which can be read
        // on a foreign stream); context-wide tracking remains enabled.
        guards.push(self.outputs[self.output_index].device_ptr_mut(&b.stream).1);
        #[cfg(test)] REPLAY_GUARDS.with(|n|n.set(guards.len()));
        b.ctx.check_err().map_err(|e| cuda_err("static_graph_dependencies", e))?;
        let launch = unsafe { result::graph::launch(self.graph.as_ref().unwrap().exec, b.stream.cu_stream()) };
        drop(guards);
        launch.map_err(|e| cuda_err("static_graph_replay", e))?;
        b.ctx.check_err().map_err(|e| cuda_err("static_graph_completion", e))?;
        self.replays += 1;
        self.poisoned = false;
        Ok(())
    }

    /// A lease: another replay cannot overwrite a borrowed live output.
    /// ```compile_fail
    /// fn invalid(graph: &mut ferro_cuda::StaticPointwiseGraph) {
    ///     let output = graph.output().unwrap();
    ///     graph.replay().unwrap();
    ///     let _ = output.len();
    /// }
    /// ```
    pub fn output(&self) -> Result<&CudaSlice<f32>> {
        if self.poisoned { return Err(cuda_err("static_graph_output", "graph handle is poisoned")); }
        Ok(&self.outputs[self.output_index])
    }
    pub fn kernel_nodes(&self) -> usize { self.nodes }
    pub fn replay_count(&self) -> usize { self.replays }
    pub fn copy_output_to_host(&self) -> Result<Vec<f32>> { self.backend.dtoh("static_graph_output", self.output()?) }

    /// Independent copy-out: one allocation, device copy and producer-stream fence.
    /// The copy is complete on return, including for untracked DLPack consumers.
    /// Synchronization is outside graph replay; no consumer stream handoff is required.
    pub fn snapshot(&self) -> Result<Box<dyn DeviceBuffer>> {
        let b = &self.backend;
        let output = self.output()?;
        let mut out = unsafe { b.alloc_uninit("static_graph_snapshot", output.len())? };
        b.stream.memcpy_dtod(output, &mut out).map_err(|e| cuda_err("static_graph_snapshot", e))?;
        // DLPack export currently has no consumer-stream handoff. Finish the copy
        // before publishing its pointer; keep cudarc dependency tracking enabled.
        b.stream.synchronize().map_err(|e| cuda_err("static_graph_snapshot", e))?;
        Ok(b.wrap(out))
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;

    #[test]
    fn snapshot_completes_copy_before_returning_to_an_untracked_consumer() {
        let _context = TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
        let b = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required CUDA: {e}"),
            Err(_) => return,
        };
        let leaf: Arc<dyn DeviceBuffer> = Arc::from(b.alloc_from_host(&[7.0; 1024]).unwrap());
        let runs = [StaticPointwiseRun { inputs: vec![0], steps: vec![ChainStep::Unary(UnaryKind::Relu)] }];
        let mut graph = b.prepare_pointwise_graph(&runs, vec![leaf]).unwrap();
        let ptx = compile_ptx("extern \"C\" __global__ void delay() { unsigned long long start = clock64(); while (clock64() - start < 200000000ULL) {} }").unwrap();
        let module = b.ctx.load_module(ptx).unwrap();
        let delay = module.load_function("delay").unwrap();
        b.stream.synchronize().unwrap();
        // Keep the producer busy so an asynchronous snapshot cannot pass by luck.
        unsafe { b.stream.launch_builder(&delay).launch(LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 }).unwrap(); }
        graph.replay().unwrap();
        assert_eq!(unsafe { sys::cuStreamQuery(b.stream.cu_stream()) }, sys::CUresult::CUDA_ERROR_NOT_READY);
        let snapshot = graph.snapshot().unwrap();
        let ready = unsafe { sys::cuStreamQuery(b.stream.cu_stream()) };
        // Clean up even on RED; the assertion observes readiness before this fence.
        b.stream.synchronize().unwrap();
        assert_eq!(ready, sys::CUresult::CUDA_SUCCESS, "snapshot returned with producer work pending");
        assert_eq!(b.copy_to_host(snapshot.as_ref()).unwrap(), vec![7.0; 1024]);
    }
}

impl ferro_core::dispatch::StaticExecution for StaticPointwiseGraph {
    fn replay(&mut self) -> Result<()> { StaticPointwiseGraph::replay(self) }
    fn snapshot(&self) -> Result<Box<dyn DeviceBuffer>> { StaticPointwiseGraph::snapshot(self) }
    fn replay_count(&self) -> usize { self.replays }
}

impl Drop for StaticPointwiseGraph {
    fn drop(&mut self) {
        let mut streams = vec![self.backend.stream.cu_stream()];
        if let Some(model) = &self._model { streams.push(model.stream.cu_stream()); }
        let mut users = self.backend.static_graphs.lock().unwrap_or_else(|e| e.into_inner());
        *users -= 1;
        // Flight fences before tuple destruction (exec, graph, then addresses).
        // If completion remains unknown it retains the tuple and exclusion slot.
        let graph = self.graph.take();
        let calls = graph.as_ref().unwrap().calls.clone();
        drop(native::Flight { graph,
            data: Some((std::mem::take(&mut self.outputs), std::mem::take(&mut self.leaves), std::mem::take(&mut self._commands), self._model.take(), self.backend.clone())),
            calls, streams, users: &mut users,
        });
    }
}

impl CudaBackend {
    pub fn prepare_pointwise_graph(self: &Arc<Self>, runs: &[StaticPointwiseRun], leaves: Vec<Arc<dyn DeviceBuffer>>) -> Result<StaticPointwiseGraph> {
        let mut users = self.static_graphs.lock().unwrap_or_else(|e| e.into_inner());
        if !self.ctx.is_event_tracking() || self.stream.capture_status().map_err(|e| cuda_err("prepare_static_graph", e))? != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(cuda_err("prepare_static_graph", "another capture mode is active"));
        }
        let mut sizes = Vec::new();
        for leaf in &leaves { sizes.push(self.resident("prepare_static_graph", leaf.as_ref())?.len()); }
        // Validate the entire DAG before allocating, compiling or beginning capture.
        let sizes = validate_runs(runs, sizes)?;
        let mut outputs = Vec::new();
        let mut commands = Vec::new();
        for (i, run) in runs.iter().enumerate() {
            let n = sizes[leaves.len() + i];
            let function = self.get_kernel("prepare_static_graph", &kernels::chain_source(&run.steps))?;
            outputs.push(self.alloc_zeros("prepare_static_graph", n)?);
            commands.push(Command { function, inputs: run.inputs.clone(), output: leaves.len() + i, n: n as u32, extra: kernels::chain_bc_args(&run.steps) });
        }
        let capture_stream = self.ctx.new_stream().map_err(|e| cuda_err("prepare_static_graph", e))?;
        let mut flight = native::Flight { graph: None, data: Some((outputs, leaves, commands, self.clone())), calls: Arc::new(native::Driver::new(capture_stream.clone())), streams: vec![self.stream.cu_stream(), capture_stream.cu_stream()], users: &mut users };
        let (outputs, leaves, commands, _) = flight.data.as_mut().unwrap();
        let mut pointers = Vec::new();
        let mut guards = Vec::new();
        for leaf in leaves.iter() {
            let (p, g) = self.resident("prepare_static_graph", leaf.as_ref())?.data.device_ptr(&self.stream);
            pointers.push(p); guards.push(g);
        }
        for output in outputs.iter_mut() {
            let (p, g) = output.device_ptr_mut(&self.stream);
            pointers.push(p); guards.push(g);
        }
        self.ctx.check_err().map_err(|e| cuda_err("prepare_static_graph", e))?;
        for command in commands.iter() { command.enqueue(&self.stream, &pointers)?; }
        drop(guards);
        self.stream.synchronize().map_err(|e| cuda_err("prepare_static_graph", e))?;
        let capture = CaptureSession::begin(flight.calls.clone())?;
        for command in commands.iter() { command.enqueue(&capture_stream, &pointers)?; }
        let graph = capture.finish()?;
        let mut nodes = 0;
        unsafe { sys::cuGraphGetNodes(graph.graph, std::ptr::null_mut(), &mut nodes).result() }
            .map_err(|e| cuda_err("static_graph_nodes", e))?;
        graph.upload(self.stream.cu_stream())?;
        self.ctx.check_err().map_err(|e| cuda_err("prepare_static_graph", e))?;
        let (outputs, leaves, commands, _) = flight.take();
        *users += 1;
        Ok(StaticPointwiseGraph { graph: Some(graph), backend: self.clone(), leaves, output_index: outputs.len()-1, outputs, _commands: commands, _model: None, nodes, replays: 0, poisoned: false })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointwise_validation_checks_all_addresses_before_cuda_work() {
        let bc = |dims, strides| ChainStep::BinaryBc { kind: BinaryKind::Add, other: 1, dims, strides };
        let valid = StaticPointwiseRun { inputs: vec![0, 1], steps: vec![bc(vec![2, 3], vec![0, 1])] };
        assert_eq!(validate_runs(&[valid], vec![6, 3]).unwrap(), vec![6, 3, 6]);
        for step in [
            ChainStep::Binary { kind: BinaryKind::Add, other: usize::MAX },
            ChainStep::Binary { kind: BinaryKind::Add, other: 1 },
            bc(vec![2, 3], vec![1]),
            bc(vec![3], vec![1]),
            bc(vec![0, 3], vec![0, 1]),
            bc(vec![2, 3], vec![3, 1]),
            bc(vec![2, 3], vec![u32::MAX, u32::MAX]),
        ] {
            let run = StaticPointwiseRun { inputs: vec![0, 1], steps: vec![step] };
            assert!(validate_runs(&[run], vec![6, 3]).is_err());
        }
        for (inputs, steps, sizes) in [
            (vec![], vec![ChainStep::Unary(UnaryKind::Relu)], vec![2]),
            (vec![1], vec![ChainStep::Unary(UnaryKind::Relu)], vec![2]),
            (vec![0], vec![], vec![2]),
            (vec![0], vec![ChainStep::Unary(UnaryKind::Relu)], vec![0]),
            (vec![0, 1], vec![ChainStep::Unary(UnaryKind::Relu)], vec![2, 2]),
        ] {
            assert!(validate_runs(&[StaticPointwiseRun { inputs, steps }], sizes).is_err());
        }
    }

    #[test]
    fn pointwise_validation_allows_scalar_broadcast_and_checks_later_runs() {
        let unary = StaticPointwiseRun { inputs: vec![0], steps: vec![ChainStep::Unary(UnaryKind::Relu)] };
        assert_eq!(validate_runs(&[unary], vec![6, 3]).unwrap(), vec![6, 3, 6], "unused trailing capture inputs remain valid");
        let runs = [
            StaticPointwiseRun { inputs: vec![0, 1], steps: vec![ChainStep::BinaryBc { kind: BinaryKind::Mul, other: 1, dims: vec![2, 3], strides: vec![0, 0] }] },
            StaticPointwiseRun { inputs: vec![2], steps: vec![ChainStep::Unary(UnaryKind::Relu)] },
        ];
        assert_eq!(validate_runs(&runs, vec![6, 1]).unwrap(), vec![6, 1, 6, 6]);
        assert!(validate_runs(&runs, vec![6, 0]).is_err());
        assert!(validate_runs(&runs, vec![usize::MAX, 1]).is_err());
    }

    #[test]
    fn replay_tracks_leaves_and_exported_output_not_private_scratch() {
        let _context = TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
        let b = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required GPU: {e}"),
            Err(_) => return,
        };
        let x: Arc<dyn DeviceBuffer> = Arc::from(b.alloc_from_host(&[1.0,2.0]).unwrap());
        let runs: Vec<_> = (0..3).map(|i|StaticPointwiseRun {inputs:vec![i],steps:vec![ChainStep::Unary(UnaryKind::Relu)]}).collect();
        let mut graph = b.prepare_pointwise_graph(&runs, vec![x]).unwrap();
        graph.replay().unwrap();
        assert_eq!(REPLAY_GUARDS.with(|n|n.get()),2);
        assert!(b.ctx.is_event_tracking());
        assert_eq!(graph.copy_output_to_host().unwrap(),vec![1.0,2.0]);
    }

    #[test]
    fn private_scratch_replay_preserves_foreign_stream_output_read_dependencies() {
        let _context = TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
        let b = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required GPU: {e}"),
            Err(_) => return,
        };
        let x: Arc<dyn DeviceBuffer> = Arc::from(b.alloc_from_host(&[1.0;1024]).unwrap());
        let runs: Vec<_> = (0..3).map(|i|StaticPointwiseRun {inputs:vec![i],steps:vec![ChainStep::Unary(UnaryKind::Relu)]}).collect();
        let mut graph=b.prepare_pointwise_graph(&runs,vec![x.clone()]).unwrap();
        graph.replay().unwrap();
        b.stream.synchronize().unwrap();
        let foreign=b.ctx.new_stream().unwrap();
        let mut saved=foreign.alloc_zeros::<f32>(1024).unwrap();
        let module=b.ctx.load_module(compile_ptx("extern \"C\" __global__ void delay() { unsigned long long start=clock64(); while(clock64()-start<200000000ULL) {} }").unwrap()).unwrap();
        let delay=module.load_function("delay").unwrap();
        unsafe { foreign.launch_builder(&delay).launch(LaunchConfig {grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).unwrap(); }
        foreign.memcpy_dtod(graph.output().unwrap(),&mut saved).unwrap();
        b.copy_into(x.as_ref(),&[2.0;1024]).unwrap();
        graph.replay().unwrap();
        assert_eq!(foreign.clone_dtoh(&saved).unwrap(),vec![1.0;1024]);
        assert_eq!(graph.copy_output_to_host().unwrap(),vec![2.0;1024]);
    }

    #[test]
    fn failed_graph_boundary_poison_is_sticky() {
        let _context = TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
        let b = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required GPU unavailable: {e}"),
            Err(_) => return,
        };
        let x: Arc<dyn DeviceBuffer> = Arc::from(b.alloc_from_host(&[1.0]).unwrap());
        let runs = [StaticPointwiseRun { inputs: vec![0], steps: vec![ChainStep::Unary(UnaryKind::Relu)] }];
        let mut graph = b.prepare_pointwise_graph(&runs, vec![x]).unwrap();
        b.ctx.record_err::<()>(Err(cudarc::driver::DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE)));
        assert!(graph.replay().is_err());
        assert!(graph.replay().is_err(), "an uncertain graph boundary must poison this handle");
        assert!(graph.snapshot().is_err());
    }

    #[test]
    fn private_capture_unwind_restores_stream_with_tracking_enabled() {
        let _context = TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
        let b = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required GPU unavailable: {e}"),
            Err(_) => return,
        };
        assert_eq!(b._blas_workspace.len(),4<<20);
        assert!(Arc::ptr_eq(b._blas_workspace.stream(),&b.stream));
        let stream = b.ctx.new_stream().unwrap();
        assert!(b.ctx.is_event_tracking());
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _capture = CaptureSession::begin(Arc::new(native::Driver::new(stream.clone()))).unwrap();
            assert!(b.ctx.is_event_tracking());
            assert!(CaptureSession::begin(Arc::new(native::Driver::new(stream.clone()))).is_err());
            panic!("injected failure after begin");
        }));
        assert!(unwind.is_err());
        assert_eq!(stream.capture_status().unwrap(), sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE);
        assert!(b.ctx.is_event_tracking());
        b.ctx.check_err().unwrap();
        let x = b.alloc_from_host(&[-1.0, 2.0]).unwrap();
        let y = b.unary_dev(UnaryKind::Relu, x.as_ref()).unwrap();
        assert_eq!(b.copy_to_host(y.as_ref()).unwrap(), vec![0.0, 2.0]);
    }
}
