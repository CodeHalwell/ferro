//! Owned CUDA graphs for prepared pointwise DAGs and supported static model inference.
use super::*;
use cudarc::driver::{result, sys, DevicePtrMut};
mod model;

/// Slots 0..leaves.len() are owned leaves; each run appends one output slot.
/// Only shape-preserving pointwise runs are supported by this first primitive.
pub struct StaticPointwiseRun {
    pub inputs: Vec<usize>,
    pub steps: Vec<ChainStep>,
}

struct Command {
    function: CudaFunction,
    inputs: Vec<usize>,
    output: usize,
    n: u32,
}

impl Command {
    fn enqueue(&self, stream: &CudaStream, pointers: &[u64]) -> Result<()> {
        let mut launch = stream.launch_builder(&self.function);
        for &i in &self.inputs { launch.arg(&pointers[i]); }
        launch.arg(&pointers[self.output]).arg(&self.n);
        // Scalar u64 arguments carry validated device addresses, not forged CudaSlice
        // aliases. Original owners and usage guards are retained by the caller.
        unsafe { launch.launch(LaunchConfig::for_num_elems(self.n)) }
            .map_err(|e| cuda_err("static_graph_enqueue", e))?;
        Ok(())
    }
}

struct GraphOwner {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    ctx: Arc<CudaContext>,
}

impl Drop for GraphOwner {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        if !self.exec.is_null() { self.ctx.record_err(unsafe { result::graph::exec_destroy(self.exec) }); }
        if !self.graph.is_null() { self.ctx.record_err(unsafe { result::graph::destroy(self.graph) }); }
    }
}

// Every successful begin is paired with end, including unwinding. The private
// capture stream cannot receive unrelated backend work while the window is open.
struct CaptureSession {
    stream: Arc<CudaStream>,
    active: bool,
}

impl CaptureSession {
    fn begin(stream: Arc<CudaStream>) -> Result<Self> {
        if stream.capture_status().map_err(|e| cuda_err("static_capture", e))? != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(cuda_err("static_capture", "stream is already capturing"));
        }
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| cuda_err("static_capture", e))?;
        Ok(Self { stream, active: true })
    }

    fn finish(mut self) -> Result<GraphOwner> {
        let graph = unsafe { result::stream::end_capture(self.stream.cu_stream()) };
        self.active = false;
        let graph = graph.map_err(|e| cuda_err("static_capture_end", e))?;
        let mut owner = GraphOwner { graph, exec: std::ptr::null_mut(), ctx: self.stream.context().clone() };
        if graph.is_null() { return Err(cuda_err("static_capture_end", "empty graph")); }
        // Own the raw graph BEFORE fallible instantiation (cudarc 0.19.9's
        // CudaStream::end_capture leaks it on this failure branch).
        unsafe { sys::cuGraphInstantiateWithFlags(&mut owner.exec, graph, 0).result() }
            .map_err(|e| cuda_err("static_graph_instantiate", e))?;
        Ok(owner)
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) {
        if self.active {
            let ctx = self.stream.context();
            ctx.record_err(ctx.bind_to_thread());
            match unsafe { result::stream::end_capture(self.stream.cu_stream()) } {
                Ok(graph) if !graph.is_null() => ctx.record_err(unsafe { result::graph::destroy(graph) }),
                Ok(_) => {},
                Err(e) => ctx.record_err(Err::<(), _>(e)),
            }
        }
    }
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
        let mut guards = Vec::with_capacity(self.leaves.len() + self.outputs.len());
        for leaf in &self.leaves {
            guards.push(b.resident("static_graph_replay", leaf.as_ref())?.data.device_ptr(&b.stream).1);
        }
        for output in &mut self.outputs { guards.push(output.device_ptr_mut(&b.stream).1); }
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
        // Fence before destroying exec and releasing any captured address. This
        // conservative drop fence is deliberately outside steady-state replay.
        self.backend.ctx.record_err(self.backend.stream.synchronize());
        drop(self.graph.take());
        self.outputs.clear();
        self.leaves.clear();
        self._commands.clear();
        drop(self._model.take());
        let mut users = self.backend.static_graphs.lock().unwrap_or_else(|e| e.into_inner());
        *users -= 1;
    }
}

impl CudaBackend {
    pub fn prepare_pointwise_graph(self: &Arc<Self>, runs: &[StaticPointwiseRun], leaves: Vec<Arc<dyn DeviceBuffer>>) -> Result<StaticPointwiseGraph> {
        let mut users = self.static_graphs.lock().unwrap_or_else(|e| e.into_inner());
        if !self.ctx.is_event_tracking() || self.stream.capture_status().map_err(|e| cuda_err("prepare_static_graph", e))? != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(cuda_err("prepare_static_graph", "another capture mode is active"));
        }
        if leaves.is_empty() || runs.is_empty() { return Err(cuda_err("prepare_static_graph", "empty DAG")); }
        let mut sizes = Vec::new();
        for leaf in &leaves { sizes.push(self.resident("prepare_static_graph", leaf.as_ref())?.len()); }
        // Validate the entire DAG before allocating, compiling or beginning capture.
        for run in runs {
            if run.inputs.is_empty() || run.steps.is_empty() || run.inputs.iter().any(|&i| i >= sizes.len()) {
                return Err(cuda_err("prepare_static_graph", "invalid DAG input or empty run"));
            }
            let n = sizes[run.inputs[0]];
            as_u32("prepare_static_graph", n)?;
            if n == 0 { return Err(cuda_err("prepare_static_graph", "empty runs are not supported")); }
            for step in &run.steps {
                match step {
                    ChainStep::Unary(_) => {},
                    ChainStep::Binary { other, .. } if *other < run.inputs.len() && sizes[run.inputs[*other]] == n => {},
                    _ => return Err(cuda_err("prepare_static_graph", "unsupported broadcast or invalid operand")),
                }
            }
            if kernels::chain_input_count(&run.steps) != run.inputs.len() {
                return Err(cuda_err("prepare_static_graph", "input arity differs from kernel signature"));
            }
            sizes.push(n);
        }
        let mut outputs = Vec::new();
        let mut commands = Vec::new();
        for (i, run) in runs.iter().enumerate() {
            let n = sizes[leaves.len() + i];
            let function = self.get_kernel("prepare_static_graph", &kernels::chain_source(&run.steps))?;
            outputs.push(self.alloc_zeros("prepare_static_graph", n)?);
            commands.push(Command { function, inputs: run.inputs.clone(), output: leaves.len() + i, n: n as u32 });
        }
        let mut pointers = Vec::new();
        let mut guards = Vec::new();
        for leaf in &leaves {
            let (p, g) = self.resident("prepare_static_graph", leaf.as_ref())?.data.device_ptr(&self.stream);
            pointers.push(p); guards.push(g);
        }
        for output in &mut outputs {
            let (p, g) = output.device_ptr_mut(&self.stream);
            pointers.push(p); guards.push(g);
        }
        self.ctx.check_err().map_err(|e| cuda_err("prepare_static_graph", e))?;
        for command in &commands { command.enqueue(&self.stream, &pointers)?; }
        drop(guards);
        self.stream.synchronize().map_err(|e| cuda_err("prepare_static_graph", e))?;
        let capture_stream = self.ctx.new_stream().map_err(|e| cuda_err("prepare_static_graph", e))?;
        let capture = CaptureSession::begin(capture_stream.clone())?;
        for command in &commands { command.enqueue(&capture_stream, &pointers)?; }
        let graph = capture.finish()?;
        let mut nodes = 0;
        unsafe { sys::cuGraphGetNodes(graph.graph, std::ptr::null_mut(), &mut nodes).result() }
            .map_err(|e| cuda_err("static_graph_nodes", e))?;
        unsafe { result::graph::upload(graph.exec, self.stream.cu_stream()) }
            .map_err(|e| cuda_err("static_graph_upload", e))?;
        self.stream.synchronize().map_err(|e| cuda_err("static_graph_upload", e))?;
        self.ctx.check_err().map_err(|e| cuda_err("prepare_static_graph", e))?;
        *users += 1;
        Ok(StaticPointwiseGraph { graph: Some(graph), backend: self.clone(), leaves, output_index: outputs.len()-1, outputs, _commands: commands, _model: None, nodes, replays: 0, poisoned: false })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_graph_boundary_poison_is_sticky() {
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
            let _capture = CaptureSession::begin(stream.clone()).unwrap();
            assert!(b.ctx.is_event_tracking());
            assert!(CaptureSession::begin(stream.clone()).is_err());
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
