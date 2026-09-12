use super::*;
use std::cell::{Cell, RefCell};
fn error() -> DriverError { DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE) }
#[derive(Default)]
struct Fake { events: RefCell<Vec<&'static str>>, end_error: Cell<bool>, partial: Cell<bool>, skip: Cell<bool>, active: Cell<bool>, upload_error: Cell<bool>, null_exec: Cell<bool>, instantiate_error: Cell<bool>, null_graph: Cell<bool>, cleanup_error: Cell<bool>, fence_error: Cell<bool>, panic_end: Cell<bool>, ended: Cell<bool>, persistent_fence: Cell<bool>, status_error: Cell<bool> }
impl Calls for Fake {
    fn upload(&self, _: sys::CUgraphExec, _: sys::CUstream) -> DriverResult<()> { self.events.borrow_mut().push("upload"); if self.upload_error.get() { Err(error()) } else { Ok(()) } }
    fn fence(&self, _: sys::CUstream) -> DriverResult<()> { self.events.borrow_mut().push("fence"); if self.persistent_fence.get() || self.fence_error.replace(false) { Err(error()) } else { Ok(()) } }
    fn capturing(&self) -> DriverResult<bool> { if self.status_error.get() { Err(error()) } else { Ok(!self.ended.get() || self.active.get()) } }
    fn end(&self, graph: &mut sys::CUgraph) -> DriverResult<()> {
        self.events.borrow_mut().push("end");
        if self.skip.replace(false) { self.active.set(true); return Err(error()); }
        self.active.set(false); self.ended.set(true);
        if (!self.end_error.get() || self.partial.get()) && !self.null_graph.get() { *graph = 1usize as _; }
        if self.panic_end.replace(false) { panic!("injected native boundary unwind"); }
        if self.end_error.get() { Err(error()) } else { Ok(()) }
    }
    fn instantiate(&self, _: sys::CUgraph, exec: &mut sys::CUgraphExec) -> DriverResult<()> { self.events.borrow_mut().push("instantiate"); if !self.null_exec.get() { *exec = 2usize as _; } if self.instantiate_error.get() { Err(error()) } else { Ok(()) } }
    fn bind(&self) -> DriverResult<()> { self.events.borrow_mut().push("bind"); if self.cleanup_error.get() { Err(error()) } else { Ok(()) } }
    fn destroy_exec(&self, exec: sys::CUgraphExec) -> DriverResult<()> { assert_eq!(exec as usize, 2); self.events.borrow_mut().push("exec"); if self.cleanup_error.get() { Err(error()) } else { Ok(()) } }
    fn destroy_graph(&self, graph: sys::CUgraph) -> DriverResult<()> { assert_eq!(graph as usize, 1); self.events.borrow_mut().push("graph"); if self.cleanup_error.get() { Err(error()) } else { Ok(()) } }
    fn record(&self, result: DriverResult<()>) { if result.is_err() { self.events.borrow_mut().push("error"); } }
}
#[test]
fn null_graph_error_is_distinct_from_a_valid_zero_node_graph() {
    let calls=Arc::new(Fake::default()); calls.null_graph.set(true);
    let error=CaptureSession {calls,active:true}.finish().err().unwrap().to_string();
    assert!(error.contains("null graph"),"{error}");
}
#[test]
fn unknown_capture_status_does_not_blindly_end_twice() {
    let calls=Arc::new(Fake::default()); calls.end_error.set(true); calls.status_error.set(true);
    assert!(CaptureSession {calls:calls.clone(),active:true}.finish().is_err());
    assert_eq!(*calls.events.borrow(),["end","bind","error"]);
}
struct Probe(Arc<Fake>);
impl Drop for Probe { fn drop(&mut self) { self.0.events.borrow_mut().push("buffers"); } }
#[test]
fn preparation_fences_before_buffers_and_releases_only_its_exclusion() {
    for fail_once in [false,true] {
        let calls=Arc::new(Fake::default()); calls.fence_error.set(fail_once);
        let mut users=3;
        drop(Flight {data:Some(Probe(calls.clone())),calls:calls.clone(),streams:vec![std::ptr::null_mut()],users:&mut users});
        assert_eq!(*calls.events.borrow(),if fail_once {vec!["fence","fence","error","buffers"]} else {vec!["fence","buffers"]});
        assert_eq!(users,3);
    }
}
#[test]
fn uncertain_preparation_completion_retains_buffers_and_exclusion() {
    let calls=Arc::new(Fake::default()); calls.persistent_fence.set(true);
    let mut users=0;
    drop(Flight {data:Some(Probe(calls.clone())),calls:calls.clone(),streams:vec![std::ptr::null_mut()],users:&mut users});
    assert!(!calls.events.borrow().contains(&"buffers"),"released unfenced addresses");
    assert_eq!(users,1,"quarantine must continue excluding legacy capture");
}
#[test]
fn persistent_upload_fence_failure_quarantines_handles() {
    let calls=Arc::new(Fake::default()); calls.persistent_fence.set(true);
    let graph=CaptureSession {calls:calls.clone(),active:true}.finish().unwrap();
    assert!(graph.upload(std::ptr::null_mut()).is_err()); drop(graph);
    assert_eq!(*calls.events.borrow(),["end","instantiate","upload","fence","fence","error"]);
}
#[test]
fn panic_after_end_output_destroys_once_without_ending_twice() {
    let calls=Arc::new(Fake::default()); calls.panic_end.set(true);
    let panic=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { let _=CaptureSession {calls:calls.clone(),active:true}.finish(); }));
    assert!(panic.is_err());
    assert_eq!(*calls.events.borrow(),["end","bind","graph"]);
}
#[test]
fn failed_upload_completion_is_fenced_again_before_release() {
    let calls=Arc::new(Fake::default()); calls.fence_error.set(true);
    let graph=CaptureSession {calls:calls.clone(),active:true}.finish().unwrap();
    assert!(graph.upload(std::ptr::null_mut()).is_err()); drop(graph);
    assert_eq!(*calls.events.borrow(),["end","instantiate","upload","fence","fence","bind","exec","graph"]);
}
#[test]
fn existing_null_and_partial_failure_ownership_matrix() {
    for partial in [false,true] {
        let calls=Arc::new(Fake::default()); calls.instantiate_error.set(true); calls.null_exec.set(!partial); calls.cleanup_error.set(true);
        let err=CaptureSession {calls:calls.clone(),active:true}.finish().err().unwrap().to_string();
        assert!(err.contains("static_graph_instantiate"));
        let expected=if partial {vec!["end","instantiate","bind","error","exec","error","graph","error"]} else {vec!["end","instantiate","bind","error","graph","error"]};
        assert_eq!(*calls.events.borrow(),expected);
    }
    for end_error in [false,true] {
        let calls=Arc::new(Fake::default()); calls.null_graph.set(true); calls.end_error.set(end_error);
        assert!(CaptureSession {calls:calls.clone(),active:true}.finish().is_err());
        assert_eq!(*calls.events.borrow(),["end","bind"]);
    }
}
#[test]
fn successful_instantiate_with_null_exec_is_rejected() {
    let calls = Arc::new(Fake::default()); calls.null_exec.set(true);
    let result = CaptureSession { calls: calls.clone(), active: true }.finish();
    assert!(result.is_err(), "null exec must not be published");
    assert_eq!(*calls.events.borrow(), ["end", "instantiate", "bind", "graph"]);
}
#[test]
fn upload_error_fences_before_exec_and_graph_destruction() {
    let calls = Arc::new(Fake::default()); calls.upload_error.set(true);
    let graph = CaptureSession { calls: calls.clone(), active: true }.finish().unwrap();
    assert!(graph.upload(std::ptr::null_mut()).is_err()); drop(graph);
    assert_eq!(*calls.events.borrow(), ["end", "instantiate", "upload", "fence", "bind", "exec", "graph"]);
}
#[test]
fn unwind_end_error_destroys_partial_graph() {
    let calls = Arc::new(Fake::default()); calls.end_error.set(true); calls.partial.set(true);
    drop(CaptureSession { calls: calls.clone(), active: true });
    assert_eq!(*calls.events.borrow(), ["bind", "end", "error", "graph"]);
}
#[test]
fn skipped_end_is_closed_on_error_unwind() {
    let calls = Arc::new(Fake::default()); calls.skip.set(true);
    assert!(CaptureSession { calls: calls.clone(), active: true }.finish().is_err());
    assert!(!calls.active.get(), "failed end left capture active");
    assert_eq!(calls.events.borrow().iter().filter(|&&e|e=="end").count(), 2);
}
#[test]
fn end_error_owns_partial_graph_without_instantiating() {
    let calls = Arc::new(Fake::default());
    calls.end_error.set(true); calls.partial.set(true);
    let result = CaptureSession { calls: calls.clone(), active: true }.finish();
    assert!(result.is_err());
    assert_eq!(*calls.events.borrow(), ["end", "bind", "graph"]);
}
