//! Private preparation-only native boundary; replay has no adapter dispatch.
use super::*;
use cudarc::driver::DriverError;
type DriverResult<T> = std::result::Result<T, DriverError>;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod gpu_tests;

// Out-parameters start null. A non-null output at this boundary must be an
// owned live handle, including injected Err/panic after successful creation.
// Fake opaque handles never enter Driver methods.
pub(super) trait Calls {
    fn upload(&self, exec: sys::CUgraphExec, stream: sys::CUstream) -> DriverResult<()>;
    fn fence(&self, stream: sys::CUstream) -> DriverResult<()>;
    fn capturing(&self) -> DriverResult<bool>;
    fn end(&self, graph: &mut sys::CUgraph) -> DriverResult<()>;
    fn instantiate(&self, graph: sys::CUgraph, exec: &mut sys::CUgraphExec) -> DriverResult<()>;
    fn bind(&self) -> DriverResult<()>;
    fn destroy_exec(&self, exec: sys::CUgraphExec) -> DriverResult<()>;
    fn destroy_graph(&self, graph: sys::CUgraph) -> DriverResult<()>;
    fn record(&self, error: DriverResult<()>);
}

// Temporary preparation ownership, including private warmup work. Persistent
// fence errors retain the complete resource tuple until process exit and retain
// a legacy-capture exclusion slot. There is no claimed fatal-context recovery.
pub(super) struct Flight<'a, T, C: Calls = Driver> {
    pub(super) data: Option<T>,
    pub(super) calls: Arc<C>,
    pub(super) streams: Vec<sys::CUstream>,
    pub(super) users: &'a mut usize,
}
impl<T, C: Calls> Flight<'_, T, C> {
    pub(super) fn take(mut self) -> T { self.data.take().unwrap() }
}
impl<T, C: Calls> Drop for Flight<'_, T, C> {
    fn drop(&mut self) {
        if self.data.is_some() {
            let mut complete = true;
            for &stream in &self.streams {
                if let Err(error) = self.calls.fence(stream) {
                    let retry = self.calls.fence(stream);
                    self.calls.record(Err(error));
                    self.calls.record(retry);
                    complete &= retry.is_ok();
                }
            }
            if !complete {
                std::mem::forget(self.data.take());
                std::mem::forget(self.calls.clone());
                *self.users += 1;
            }
        }
    }
}

pub(super) struct Driver(pub Arc<CudaStream>);
impl Driver {
    pub(super) fn warm_fence(&self) -> DriverResult<()> {
        #[cfg(test)] if gpu_tests::take("warm-fence-skip") { return Err(gpu_tests::error()); }
        self.0.synchronize()
    }
}
impl Calls for Driver {
    fn upload(&self, exec: sys::CUgraphExec, stream: sys::CUstream) -> DriverResult<()> {
        #[cfg(test)] { gpu_tests::event("upload"); if gpu_tests::take("upload-skip") { return Err(gpu_tests::error()); } }
        unsafe { result::graph::upload(exec, stream) }?;
        #[cfg(test)] if gpu_tests::take("upload-panic") { panic!("injected upload boundary panic"); }
        #[cfg(test)] if gpu_tests::take("upload-error") { return Err(gpu_tests::error()); }
        Ok(())
    }
    fn fence(&self, stream: sys::CUstream) -> DriverResult<()> {
        #[cfg(test)] if gpu_tests::take("fence-panic") { panic!("injected fence boundary panic"); }
        #[cfg(test)] { gpu_tests::event("fence"); if gpu_tests::take("fence-skip") { return Err(gpu_tests::error()); } }
        self.bind()?; unsafe { result::stream::synchronize(stream) }
    }
    fn capturing(&self) -> DriverResult<bool> { self.0.capture_status().map(|s| s != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE) }
    fn end(&self, graph: &mut sys::CUgraph) -> DriverResult<()> {
        #[cfg(test)] { gpu_tests::event("end"); if gpu_tests::take("end-skip") { return Err(gpu_tests::error()); } }
        let mut output = std::ptr::null_mut();
        unsafe { sys::cuStreamEndCapture(self.0.cu_stream(), &mut output).result() }?;
        *graph = output;
        #[cfg(test)] {
            if gpu_tests::take("end-panic") { panic!("injected end boundary panic"); }
            if gpu_tests::take("end-error") { return Err(gpu_tests::error()); }
            if gpu_tests::take("end-null") { self.destroy_graph(*graph)?; *graph=std::ptr::null_mut(); }
        }
        Ok(())
    }
    fn instantiate(&self, graph: sys::CUgraph, exec: &mut sys::CUgraphExec) -> DriverResult<()> {
        #[cfg(test)] { gpu_tests::event("instantiate"); if gpu_tests::take("instantiate-skip") { return Err(gpu_tests::error()); } }
        let mut output = std::ptr::null_mut();
        unsafe { sys::cuGraphInstantiateWithFlags(&mut output, graph, 0).result() }?;
        *exec = output;
        #[cfg(test)] if gpu_tests::take("instantiate-panic") { panic!("injected instantiate boundary panic"); }
        #[cfg(test)] if gpu_tests::take("instantiate-error") { return Err(gpu_tests::error()); }
        Ok(())
    }
    fn bind(&self) -> DriverResult<()> { self.0.context().bind_to_thread() }
    fn destroy_exec(&self, exec: sys::CUgraphExec) -> DriverResult<()> {
        #[cfg(test)] gpu_tests::event("exec");
        unsafe { result::graph::exec_destroy(exec) }
    }
    fn destroy_graph(&self, graph: sys::CUgraph) -> DriverResult<()> {
        #[cfg(test)] gpu_tests::event("graph");
        unsafe { result::graph::destroy(graph) }
    }
    fn record(&self, error: DriverResult<()>) { self.0.context().record_err(error); }
}

pub(super) struct GraphOwner<C: Calls = Driver> {
    pub(super) graph: sys::CUgraph,
    pub(super) exec: sys::CUgraphExec,
    calls: Arc<C>,
    pending: std::cell::Cell<Option<sys::CUstream>>,
}
impl<C: Calls> GraphOwner<C> {
    pub(super) fn upload(&self, stream: sys::CUstream) -> Result<()> {
        self.pending.set(Some(stream));
        let upload = self.calls.upload(self.exec, stream);
        let fence = self.calls.fence(stream);
        if fence.is_ok() { self.pending.set(None); }
        if upload.is_err() { self.calls.record(fence); }
        upload.and(fence).map_err(|e|cuda_err("static_graph_upload",e))
    }
}
impl<C: Calls> Drop for GraphOwner<C> {
    fn drop(&mut self) {
        if let Some(stream) = self.pending.take() {
            let fence = self.calls.fence(stream);
            self.calls.record(fence);
            if fence.is_err() {
                // Completion is unknown. Keep native handles/context alive until
                // process exit; callers quarantine their address owners too.
                std::mem::forget(self.calls.clone());
                return;
            }
        }
        self.calls.record(self.calls.bind());
        if !self.exec.is_null() { self.calls.record(self.calls.destroy_exec(self.exec)); }
        if !self.graph.is_null() { self.calls.record(self.calls.destroy_graph(self.graph)); }
    }
}

pub(super) struct CaptureSession<C: Calls = Driver> { calls: Arc<C>, active: bool }
impl CaptureSession {
    pub(super) fn begin(stream: Arc<CudaStream>) -> Result<Self> {
        if stream.capture_status().map_err(|e| cuda_err("static_capture", e))? != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(cuda_err("static_capture", "stream is already capturing"));
        }
        #[cfg(test)] gpu_tests::stream(&stream);
        let begin = || {
            #[cfg(test)] if gpu_tests::take("begin-skip") { return Err(gpu_tests::error()); }
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        };
        begin().map_err(|e| cuda_err("static_capture", e))?;
        Ok(Self { calls: Arc::new(Driver(stream)), active: true })
    }
}
impl<C: Calls> CaptureSession<C> {
    pub(super) fn finish(mut self) -> Result<GraphOwner<C>> {
        let mut owner = GraphOwner { graph: std::ptr::null_mut(), exec: std::ptr::null_mut(), calls: self.calls.clone(), pending: std::cell::Cell::new(None) };
        let ended = self.calls.end(&mut owner.graph);
        self.active = ended.is_err();
        let graph = owner.graph;
        ended.map_err(|e| cuda_err("static_capture_end", e))?;
        if graph.is_null() { return Err(cuda_err("static_capture_end", "null graph")); }
        self.calls.instantiate(graph, &mut owner.exec).map_err(|e| cuda_err("static_graph_instantiate", e))?;
        if owner.exec.is_null() { return Err(cuda_err("static_graph_instantiate", "null executable")); }
        Ok(owner)
    }
}
impl<C: Calls> Drop for CaptureSession<C> {
    fn drop(&mut self) {
        if self.active {
            match self.calls.capturing() {
                Ok(false) => return,
                Ok(true) => {},
                Err(error) => {
                    self.calls.record(Err(error));
                    std::mem::forget(self.calls.clone());
                    return;
                },
            }
            self.calls.record(self.calls.bind());
            let mut graph = std::ptr::null_mut();
            self.calls.record(self.calls.end(&mut graph));
            if !graph.is_null() { self.calls.record(self.calls.destroy_graph(graph)); }
        }
    }
}
