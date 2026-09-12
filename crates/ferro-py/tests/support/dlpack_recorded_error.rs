use super::*;
use std::cell::Cell;
use std::sync::{Arc, Mutex};
use cudarc::driver::{CudaContext, DriverError, sys};

thread_local! {
    static INJECT_STAGE: Cell<u8> = const { Cell::new(0) };
    pub(super) static EXPORT_CALLS: Cell<usize> = const { Cell::new(0) };
}
static CUDA_TEST: Mutex<()> = Mutex::new(());

// Exercise cudarc's real recorded-error channel, not a native CUDA failure.
pub(super) fn inject(stage: u8, ctx: &CudaContext) {
    INJECT_STAGE.with(|armed| {
        if armed.get() & stage != 0 {
            armed.set(armed.get() & !stage);
            ctx.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE)));
        }
    });
}

fn recorded_error_rejects_export(stage: u8) {
    // Existing binding tests install backends outside the GIL. Isolate the
    // registry/context in a child instead of imposing a private, ineffective lock.
    if std::env::var("FERRO_DLPACK_ERROR_CHILD").ok().as_deref() != Some(&stage.to_string()) {
        let name = match stage {
            1 => "dependency_recorded_error_rejects_capsule",
            2 => "guard_release_recorded_error_rejects_capsule",
            3 => "dependency_error_still_drains_guard_release_error",
            _ => unreachable!(),
        };
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &format!("dlpack_error_tests::{name}"), "--nocapture"])
            .env("FERRO_DLPACK_ERROR_CHILD", stage.to_string()).output().unwrap();
        assert!(output.status.success(), "child regression failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        return;
    }
    let _lock = CUDA_TEST.lock().unwrap_or_else(|e| e.into_inner());
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let backend = Arc::new(ferro_cuda::CudaBackend::new(0).expect("required CUDA"));
        ferro_core::dispatch::register_backend(Device::Cuda(0), backend);
        let tensor = PyTensor::wrap(CoreTensor::full(&[4], 3.).to_device(Device::Cuda(0)).unwrap());
        let view = tensor.inner.dlpack_device_view().unwrap();
        let buffer = view.device_buffer().as_any().downcast_ref::<ferro_cuda::CudaBuf>().unwrap();
        let ctx = buffer.stream().context();
        ctx.check_err().unwrap();
        EXPORT_CALLS.with(|n| n.set(0));
        INJECT_STAGE.with(|armed| armed.set(stage));
        let result = tensor.__dlpack__(py, None);
        let recorded = ctx.check_err();
        INJECT_STAGE.with(|armed| assert_eq!(armed.replace(0), 0, "injection was reached"));
        assert!(result.is_err(), "recorded CUDA error must reject DLPack export before capsule allocation");
        let error = result.unwrap_err();
        assert!(error.is_instance_of::<PyValueError>(py));
        assert!(error.to_string().contains("CUDA_ERROR_INVALID_VALUE"), "{error}");
        EXPORT_CALLS.with(|n| assert_eq!(n.get(), 0, "capsule allocator must not be entered"));
        assert!(recorded.is_ok(), "export must drain the recorded error channel");
        let capsule = tensor.__dlpack__(py, None).expect("clean export after drained error");
        assert!(capsule.is_instance_of::<pyo3::types::PyCapsule>());
        EXPORT_CALLS.with(|n| assert_eq!(n.get(), 1));
        drop(capsule);
        assert_eq!(tensor.inner.to_vec(), vec![3.; 4]);
    });
}

#[test]
fn dependency_recorded_error_rejects_capsule() { recorded_error_rejects_export(1); }

#[test]
fn guard_release_recorded_error_rejects_capsule() { recorded_error_rejects_export(2); }

#[test]
fn dependency_error_still_drains_guard_release_error() { recorded_error_rejects_export(3); }
