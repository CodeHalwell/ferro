use super::*;
use std::cell::RefCell;
thread_local! { static MODE: RefCell<Option<&'static str>> = const { RefCell::new(None) }; static EVENTS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) }; static STREAM: RefCell<Option<Arc<CudaStream>>> = const { RefCell::new(None) }; }
struct Override(Option<&'static str>);
impl Override { fn new(mode: &'static str) -> Self { EVENTS.with(|v|v.borrow_mut().clear()); Self(MODE.with(|v|v.replace(Some(mode)))) } }
impl Drop for Override { fn drop(&mut self) { MODE.with(|v|v.replace(self.0)); } }
thread_local! { static SEQUENCE: RefCell<std::collections::VecDeque<&'static str>> = RefCell::new(std::collections::VecDeque::new()); }
struct Sequence;
impl Drop for Sequence { fn drop(&mut self) { SEQUENCE.with(|v|v.borrow_mut().clear()); } }
pub(super) fn take(mode: &str) -> bool {
    if SEQUENCE.with(|v| { let mut v=v.borrow_mut(); if v.front().copied()==Some(mode) { v.pop_front(); true } else { false } }) { return true; }
    MODE.with(|v| { let mut v=v.borrow_mut(); if *v==Some(mode) { *v=None; true } else { false } }) }
pub(super) fn event(stage: &'static str) { EVENTS.with(|v|v.borrow_mut().push(stage)); }
pub(super) fn stream(stream: &Arc<CudaStream>) { STREAM.with(|v|v.replace(Some(stream.clone()))); }
pub(super) fn error() -> DriverError { DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE) }
#[test]
fn published_graph_drop_retains_owners_with_poisoned_exclusion_mutex() {
    let _context=TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    let leaf:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[1.,2.]).unwrap());
    let runs=[StaticPointwiseRun {inputs:vec![0],steps:vec![ChainStep::Unary(UnaryKind::Relu)]}];
    let mut graph=b.prepare_pointwise_graph(&runs,vec![leaf.clone()]).unwrap();
    graph.replay().unwrap();
    let calls=graph.graph.as_ref().unwrap().calls.clone();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _lock=b.static_graphs.lock().unwrap(); panic!("poison exclusion mutex");
    })).is_err());
    let _scope=Override::new("exec-skip");
    drop(graph);
    assert_eq!(*b.static_graphs.lock().unwrap_or_else(|e|e.into_inner()),1);
    assert_eq!(Arc::strong_count(&leaf),2);
    let owners=calls.quarantine().owners.lock().unwrap();
    assert_eq!(owners.len(),1); assert!(!owners[0].0.is_null()); assert!(!owners[0].1.is_null());
    assert_eq!(EVENTS.with(|v|v.borrow().clone()),["fence","exec"]);
    drop(owners);
    let _=b.ctx.check_err();
    assert!(b.begin_step_capture().is_err());
    let mut next=b.prepare_pointwise_graph(&runs,vec![leaf]).unwrap();
    next.replay().unwrap(); assert_eq!(next.copy_output_to_host().unwrap(),vec![1.,2.]);
    drop(next);
    assert_eq!(*b.static_graphs.lock().unwrap_or_else(|e|e.into_inner()),1);
    STREAM.with(|s|s.borrow_mut().take());
}
#[test]
fn real_resources_keep_sticky_exclusion_after_simulated_cleanup_failures() {
    let _context=TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
    for sequence in [
        vec!["upload-error","fence-skip","fence-skip"],
        vec!["end-error","bind-skip"],
        vec!["instantiate-error","exec-skip"],
        vec!["end-error","graph-skip"],
        vec!["end-error","status-skip"],
    ] {
        for model in [false,true] {
            let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
            let leaf:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[1.,2.,3.,4.]).unwrap());
            let runs=[ferro_core::dispatch::StaticRun {inputs:vec![0,0],op:ferro_core::dispatch::StaticOp::MatMul {m:2,k:2,n:2},shape:vec![2,2]}];
            let point=[StaticPointwiseRun {inputs:vec![0],steps:vec![ChainStep::Unary(UnaryKind::Relu)]}];
            let prepare=|| if model { b.prepare_model_graph(&runs,vec![leaf.clone()]) } else { b.prepare_pointwise_graph(&point,vec![leaf.clone()]) };
            let _scope=Override::new("unused");
            let _sequence=Sequence;
            SEQUENCE.with(|v|*v.borrow_mut()=sequence.clone().into());
            assert!(prepare().is_err(),"{sequence:?}");
            SEQUENCE.with(|v|assert!(v.borrow().is_empty(),"unconsumed injection: {:?}",v.borrow()));
            assert_eq!(*b.static_graphs.lock().unwrap_or_else(|e|e.into_inner()),1,"{sequence:?}: lost quarantine exclusion");
            assert_eq!(Arc::strong_count(&leaf),2,"{sequence:?}: quarantined address owner lost");
            assert!(b.begin_step_capture().is_err(),"legacy capture must remain excluded");
            assert!(b.ctx.is_event_tracking());
            // Simulated skips do not corrupt CUDA. Drain recorded adapter errors,
            // then prove unrelated static preparation/replay still works. The
            // quarantined tuple itself is NOT reclaimed, including at test exit.
            let _=b.ctx.check_err();
            let mut next=prepare().unwrap(); next.replay().unwrap();
            assert_eq!(next.copy_output_to_host().unwrap(),if model {vec![7.,10.,15.,22.]} else {vec![1.,2.,3.,4.]});
            drop(next);
            assert_eq!(*b.static_graphs.lock().unwrap_or_else(|e|e.into_inner()),1);
            STREAM.with(|s|assert_eq!(s.borrow_mut().take().unwrap().capture_status().unwrap(),sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE));
        }
    }
}
#[test]
fn private_warmup_fence_error_releases_only_after_cleanup_fence() {
    let _context=TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    let leaf:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[1.,2.,3.,4.]).unwrap());
    let runs=[ferro_core::dispatch::StaticRun {inputs:vec![0,0],op:ferro_core::dispatch::StaticOp::MatMul {m:2,k:2,n:2},shape:vec![2,2]}];
    let scope=Override::new("warm-fence-skip");
    let failed=b.prepare_model_graph(&runs,vec![leaf.clone()]);
    let rejected=failed.is_err(); drop(failed); drop(scope);
    assert!(rejected,"warmup fence native error not returned");
    EVENTS.with(|v|assert_eq!(*v.borrow(),["fence","fence"]));
    assert_eq!(*b.static_graphs.lock().unwrap(),0); assert!(b.ctx.is_event_tracking()); b.ctx.check_err().unwrap();
    let mut graph=b.prepare_model_graph(&runs,vec![leaf]).unwrap();
    graph.replay().unwrap(); assert_eq!(graph.copy_output_to_host().unwrap(),vec![7.,10.,15.,22.]);
}
#[test]
fn native_boundary_panics_restore_override_and_recover() {
    let _context=TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    let leaf:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[1.,2.]).unwrap());
    let runs=[StaticPointwiseRun {inputs:vec![0],steps:vec![ChainStep::Unary(UnaryKind::Relu)]}];
    for mode in ["end-panic","instantiate-panic","upload-panic","fence-panic"] {
        let caught=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope=Override::new(mode);
            let _=b.prepare_pointwise_graph(&runs,vec![leaf.clone()]);
        }));
        assert!(caught.is_err(),"{mode}: boundary did not unwind");
        assert!(MODE.with(|v|v.borrow().is_none()));
        let expected: &[&str]=match mode {
            "end-panic"=>&["end","graph","fence","fence"],
            "instantiate-panic"=>&["end","instantiate","exec","graph","fence","fence"],
            _=>&["end","instantiate","upload","fence","exec","graph","fence","fence"],
        };
        EVENTS.with(|v|assert_eq!(v.borrow().as_slice(),expected,"{mode}"));
        STREAM.with(|s|assert_eq!(s.borrow_mut().take().unwrap().capture_status().unwrap(),sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE));
        assert_eq!(*b.static_graphs.lock().unwrap_or_else(|e|e.into_inner()),0);
        assert!(b.ctx.is_event_tracking()); b.ctx.check_err().unwrap();
        let mut graph=b.prepare_pointwise_graph(&runs,vec![leaf.clone()]).unwrap();
        graph.replay().unwrap(); assert_eq!(graph.copy_output_to_host().unwrap(),vec![1.,2.]);
    }
    STREAM.with(|s|s.borrow_mut().take());
}
#[test]
fn returned_native_errors_recover_next_model_and_pointwise_graph() {
    let _context=TEST_CONTEXT.lock().unwrap_or_else(|e|e.into_inner());
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    let leaf:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[1.,2.,3.,4.]).unwrap());
    let model=[ferro_core::dispatch::StaticRun { inputs:vec![0,0], op:ferro_core::dispatch::StaticOp::MatMul {m:2,k:2,n:2}, shape:vec![2,2] }];
    let point=[StaticPointwiseRun {inputs:vec![0],steps:vec![ChainStep::Unary(UnaryKind::Relu)]}];
    for is_model in [false,true] {
        for mode in ["begin-skip","end-skip","end-error","end-null","instantiate-skip","instantiate-error","upload-skip","upload-error","fence-skip"] {
            let prepare=|| if is_model { b.prepare_model_graph(&model,vec![leaf.clone()]) } else { b.prepare_pointwise_graph(&point,vec![leaf.clone()]) };
            let scope=Override::new(mode);
            let failed=prepare();
            let rejected=failed.is_err(); drop(failed); drop(scope);
            assert!(rejected,"{mode}: injected native result was not propagated");
            EVENTS.with(|events| {
                let events=events.borrow();
                let expected: &[&str]=match mode {
                    "begin-skip"=>&[],
                    "end-skip"=>&["end","end","graph"],
                    "end-error"|"end-null"=>&["end","graph"],
                    "instantiate-skip"=>&["end","instantiate","graph"],
                    "instantiate-error"=>&["end","instantiate","exec","graph"],
                    "fence-skip"=>&["end","instantiate","upload","fence","fence","exec","graph"],
                    _=>&["end","instantiate","upload","fence","exec","graph"],
                };
                let mut expected=expected.to_vec(); expected.extend(["fence","fence"]);
                assert_eq!(*events,expected,"{mode}");
            });
            STREAM.with(|s| { let stream=s.borrow_mut().take().unwrap(); assert_eq!(stream.capture_status().unwrap(),sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE); });
            assert_eq!(Arc::strong_count(&leaf),1,"{mode}: original leaf owner leaked");
            assert_eq!(*b.static_graphs.lock().unwrap(),0); assert!(b.ctx.is_event_tracking()); b.ctx.check_err().unwrap();
            let mut graph=prepare().unwrap(); graph.replay().unwrap();
            assert_eq!(graph.copy_output_to_host().unwrap(),if is_model {vec![7.,10.,15.,22.]} else {vec![1.,2.,3.,4.]});
            drop(graph);
        }
    }
}
