use super::*;
use std::sync::mpsc;
use std::time::Duration;
use std::sync::atomic::{AtomicUsize, Ordering};

thread_local! {
    static SNAPSHOT_PAUSE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

pub(super) fn snapshot_pause() {
    let pause = SNAPSHOT_PAUSE.with(|slot| slot.borrow_mut().take());
    if let Some(pause) = pause { pause(); }
}

struct ReentrantDrop(Arc<AtomicUsize>);

impl Backend for ReentrantDrop {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { unreachable!() }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { unreachable!() }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { unreachable!() }
}

impl Drop for ReentrantDrop {
    fn drop(&mut self) {
        eprintln!("REENTRANT_DROP_ENTER");
        let _snapshot = cuda_backend();
        let _registered = ferro_core::dispatch::backend_for(Device::Cpu).unwrap();
        self.0.fetch_add(1, Ordering::SeqCst);
        eprintln!("REENTRANT_DROP_COMPLETE");
    }
}

#[test]
fn reentrant_drop_direct_registration_without_cuda() {
    if isolated_with_timeout("reentrant_drop_direct_registration_without_cuda", Duration::from_secs(15)) { return; }
    let _registry = registry_test_guard();
    assert!(cuda_backend().is_none());
    let drops = Arc::new(AtomicUsize::new(0));
    register_backend(Device::Cuda(0), Arc::new(ReentrantDrop(drops.clone())));
    register_backend(Device::Cuda(0), Arc::new(ferro_core::dispatch::CpuBackend));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn reentrant_drop_during_install() {
    if isolated_with_timeout("reentrant_drop_during_install", Duration::from_secs(15)) { return; }
    let _registry = registry_test_guard();
    if !cuda_required() { return; }
    let drops = Arc::new(AtomicUsize::new(0));
    register_backend(Device::Cuda(0), Arc::new(ReentrantDrop(drops.clone())));
    install(0).unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(agrees_with_registry());
}

#[test]
fn reentrant_drop_of_last_mismatch_snapshot() {
    if isolated_with_timeout("reentrant_drop_of_last_mismatch_snapshot", Duration::from_secs(15)) { return; }
    let _registry = registry_test_guard();
    if !cuda_required() { return; }
    install(0).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let custom = Arc::new(ReentrantDrop(drops.clone()));
    let weak = Arc::downgrade(&custom);
    register_backend(Device::Cuda(0), custom);
    SNAPSHOT_PAUSE.with(|slot| *slot.borrow_mut() = Some(Box::new(move || {
        // The reader owns the snapshot; a direct writer removes its registry
        // owner before the reader releases it. No timing assumption is needed.
        std::thread::spawn(|| {
            register_backend(Device::Cuda(0), Arc::new(ferro_core::dispatch::CpuBackend));
        }).join().unwrap();
        assert_eq!(weak.strong_count(), 1, "snapshot must be the last owner");
    })));
    assert!(cuda_backend().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(cuda_backend().is_none());
}

thread_local! {
    pub(super) static PUBLICATION_PAUSE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

pub(super) fn publication_pause() {
    PUBLICATION_PAUSE.with(|slot| {
        if let Some(pause) = slot.borrow_mut().take() { pause(); }
    });
}

// Repeated context/BLAS setup must not overlap unrelated legacy capture tests
// in the unit-test process. Each child still runs real concurrent installers.
fn isolated(name: &str) -> bool {
    isolated_with_timeout(name, Duration::from_secs(60))
}

fn isolated_with_timeout(name: &str, timeout: Duration) -> bool {
    let test = format!("install_tests::{name}");
    if std::env::var("FERRO_INSTALL_TEST_CHILD").ok().as_deref() == Some(&test) { return false; }
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test, "--nocapture", "--test-threads=1"])
        .env("FERRO_INSTALL_TEST_CHILD", &test)
        .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped())
        .spawn().unwrap();
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if child.try_wait().unwrap().is_some() { break; }
        if std::time::Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!("{test} timed out: {} {}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{test}: {} {}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    true
}

fn cuda_required() -> bool {
    match CudaBackend::new(0) {
        Ok(_) => true,
        Err(e) => {
            assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}");
            eprintln!("SKIP install tests: {e}");
            false
        }
    }
}

fn agrees_with_registry() -> bool {
    let typed = cuda_backend().expect("installed backend");
    let erased: Arc<dyn Backend> = typed.clone();
    Arc::ptr_eq(&erased, &ferro_core::dispatch::backend_for(typed.device).unwrap())
}

#[test]
fn paused_publication_does_not_expose_a_split_pair() {
    if isolated("paused_publication_does_not_expose_a_split_pair") { return; }
    let _registry = registry_test_guard();
    if !cuda_required() { return; }
    install(0).unwrap();
    let (paused_tx, paused_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let writer = std::thread::spawn(move || {
        PUBLICATION_PAUSE.with(|slot| *slot.borrow_mut() = Some(Box::new(move || {
            paused_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        })));
        install(0).unwrap();
    });
    paused_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    // This exact publication boundary must be protected even if the reader
    // thread happens not to be scheduled until after we release the writer.
    let boundary_protected = LAST_BACKEND.try_lock().is_err();
    let (started_tx, started_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        result_tx.send(agrees_with_registry()).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let early = result_rx.recv_timeout(Duration::from_millis(250)).ok();
    release_tx.send(()).unwrap();
    writer.join().unwrap();
    reader.join().unwrap();
    let coherent = early.unwrap_or_else(|| result_rx.recv().unwrap());
    assert!(coherent, "CUDA reader observed LAST_BACKEND and registry from different installs");
    assert!(boundary_protected, "publication boundary is unprotected");
    assert!(agrees_with_registry());
    device_synchronize().unwrap();
}

#[test]
fn selection_tracks_the_last_successful_ordinal() {
    if isolated("selection_tracks_the_last_successful_ordinal") { return; }
    let _registry = registry_test_guard();
    if !cuda_required() { return; }
    install(0).unwrap();
    let first = cuda_backend().unwrap();
    assert!(install(u32::MAX).is_err());
    assert!(Arc::ptr_eq(&cuda_backend().unwrap(), &first));
    assert!(agrees_with_registry());
    if let Err(e) = install(1) {
        assert!(Arc::ptr_eq(&cuda_backend().unwrap(), &first));
        assert!(agrees_with_registry());
        eprintln!("MULTI_GPU_NOT_EXERCISED: ordinal 1 unavailable: {e}");
        return;
    }
    let second = cuda_backend().unwrap();
    assert_eq!(second.device, Device::Cuda(1));
    let first_erased: Arc<dyn Backend> = first.clone();
    assert!(Arc::ptr_eq(&ferro_core::dispatch::backend_for(Device::Cuda(0)).unwrap(), &first_erased));
    assert!(agrees_with_registry());
    device_synchronize().unwrap();
    install(0).unwrap();
    let selected = cuda_backend().unwrap();
    assert_eq!(selected.device, Device::Cuda(0));
    // Updating a non-selected ordinal must not change the selected handle.
    register_backend(Device::Cuda(1), Arc::new(CudaBackend::new(1).unwrap()));
    assert!(Arc::ptr_eq(&cuda_backend().unwrap(), &selected));
    second.synchronize().unwrap();
    device_synchronize().unwrap();
}

#[test]
fn overlapping_publications_cannot_leave_different_owners() {
    if isolated("overlapping_publications_cannot_leave_different_owners") { return; }
    let _registry = registry_test_guard();
    if !cuda_required() { return; }
    let (paused_tx, paused_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first = std::thread::spawn(move || {
        PUBLICATION_PAUSE.with(|slot| *slot.borrow_mut() = Some(Box::new(move || {
            paused_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        })));
        install(0).unwrap();
    });
    paused_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let protected = LAST_BACKEND.try_lock().is_err();
    let (done_tx, done_rx) = mpsc::channel();
    let second = std::thread::spawn(move || {
        install(0).unwrap();
        done_tx.send(()).unwrap();
    });
    // On the old implementation B can finish before A publishes the registry:
    // force that schedule rather than relying on stress or a short sleep.
    // With a protected publication, let A finish so B can acquire the gate.
    if !protected { done_rx.recv_timeout(Duration::from_secs(10)).unwrap(); }
    release_tx.send(()).unwrap();
    first.join().unwrap();
    second.join().unwrap();
    assert!(agrees_with_registry(), "concurrent installs left different owners");
    device_synchronize().unwrap();
}

#[test]
fn concurrent_installs_leave_one_registered_owner() {
    if isolated("concurrent_installs_leave_one_registered_owner") { return; }
    let _registry = registry_test_guard();
    if !cuda_required() { return; }
    let start = Arc::new(std::sync::Barrier::new(3));
    let mut threads = Vec::new();
    for _ in 0..2 {
        let start = start.clone();
        threads.push(std::thread::spawn(move || {
            start.wait();
            for _ in 0..8 { install(0).unwrap(); }
        }));
    }
    start.wait();
    for thread in threads { thread.join().unwrap(); }
    assert!(agrees_with_registry());
    let registered = ferro_core::dispatch::backend_for(Device::Cuda(0)).unwrap();
    let buffer = registered.alloc_from_host(&[2.0, -3.0]).unwrap();
    cuda_backend().unwrap().unary_dev(UnaryKind::Neg, &*buffer).unwrap();
    device_synchronize().unwrap();
}

#[test]
fn direct_registry_replacement_cannot_return_or_fence_stale_owner() {
    if isolated("direct_registry_replacement_cannot_return_or_fence_stale_owner") { return; }
    let _registry = registry_test_guard();
    if !cuda_required() { return; }
    install(0).unwrap();
    let old = cuda_backend().unwrap();
    let replacement = Arc::new(CudaBackend::new(0).unwrap());
    register_backend(Device::Cuda(0), replacement);
    assert!(cuda_backend().is_none(), "direct registry replacement left a stale typed accessor");
    assert!(device_synchronize().is_err(), "must not report success after fencing stale owner");
    old.synchronize().unwrap();
    // Re-registering the exact same owner is valid; equality is identity,
    // not merely a device ordinal or a matching backend name.
    register_backend(Device::Cuda(0), old.clone());
    assert!(Arc::ptr_eq(&cuda_backend().unwrap(), &old));
    device_synchronize().unwrap();
    install(0).unwrap();
    assert!(agrees_with_registry());
    device_synchronize().unwrap();
}
