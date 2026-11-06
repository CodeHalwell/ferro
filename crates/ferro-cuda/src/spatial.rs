use super::*;
use ferro_core::dispatch::Window2dDev;

/// Kernel-side geometry; field order matches `struct Geom` in spatial.cu.
#[repr(C)]
#[derive(Clone, Copy)]
struct Geom([i32; 14]);
unsafe impl cudarc::driver::DeviceRepr for Geom {}

fn declined(op: &'static str) -> Error {
    Error::Unsupported { op, msg: "backend does not implement device-resident storage".into() }
}

pub(crate) fn source(mode: u32) -> String {
    format!("#define MODE {mode}\n{}", include_str!("spatial.cu"))
}

impl CudaBackend {
    /// Launch spatial MODE over `total` outputs. Shapes beyond i32 geometry
    /// decline (core then takes the host path) rather than fail.
    pub(crate) fn spatial(&self, op: &'static str, mode: u32, x: &dyn DeviceBuffer, aux: Option<&dyn DeviceBuffer>,
        g: &Window2dDev, input: usize, total: usize, with_arg: bool,
    ) -> Result<(Box<dyn DeviceBuffer>, Option<Box<dyn DeviceBuffer>>)> {
        let dims = [g.n, g.c, g.h, g.w, g.kernel[0], g.kernel[1], g.stride[0], g.stride[1], g.padding[0], g.padding[1], g.dilation[0], g.dilation[1], g.oh, g.ow];
        if dims.iter().any(|&d| d > i32::MAX as usize) || total > (u32::MAX as usize) * 256 || (mode >= 2 && g.h * g.w > 1 << 24) {
            return Err(declined(op));
        }
        let x = self.resident(op, x)?;
        if x.data.len() != input { return Err(Error::InvalidShape { op, msg: format!("expected {input} input elements, got {}", x.data.len()) }); }
        let aux = aux.map(|a| self.resident(op, a)).transpose()?;
        if aux.is_some_and(|a| a.data.len() != input) { return Err(Error::InvalidShape { op, msg: "argmax/cotangent length mismatch".into() }); }
        let mut out = unsafe { self.alloc_uninit(op, total)? };
        let mut arg = if with_arg { Some(unsafe { self.alloc_uninit(op, total)? }) } else { None };
        if total > 0 {
            let f = self.get_kernel(op, &source(mode))?;
            let geom = Geom(dims.map(|d| d as i32));
            let total_arg = total as u64;
            let dummy = 0u64;
            let mut launch = self.stream.launch_builder(&f);
            launch.arg(&*x.data).arg(&*aux.unwrap_or(x).data).arg(&mut out);
            match arg.as_mut() { Some(a) => launch.arg(a), None => launch.arg(&dummy) };
            launch.arg(&total_arg).arg(&geom);
            let blocks = total.div_ceil(256) as u32;
            unsafe { launch.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(|e| cuda_err(op, e))?;
        }
        Ok((self.wrap(out), arg.map(|a| self.wrap(a))))
    }
}

#[cfg(test)]
mod nvrtc {
    use crate::kernels::*;
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    use ferro_core::dispatch::ReduceKind;
    use ferro_core::{BinaryKind, UnaryKind};

    /// Every generated kernel source compiles to PTX for Ampere (sm_86, the
    /// RTX 3090) and the given extra arches. Needs only libnvrtc, no GPU or
    /// driver; skipped (and says so) when libnvrtc cannot be loaded unless
    /// FERRO_REQUIRE_NVRTC=1. Covers the catalog the backend generates, so a
    /// typo in a kernel string fails here rather than on the first launch.
    #[test]
    fn every_kernel_source_compiles_for_ampere() {
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            assert!(std::env::var("FERRO_REQUIRE_NVRTC").is_err(), "FERRO_REQUIRE_NVRTC=1 but libnvrtc is not loadable");
            eprintln!("libnvrtc not loadable: skipping NVRTC compile check");
            return;
        }
        let unary = [UnaryKind::Neg, UnaryKind::Relu, UnaryKind::Exp, UnaryKind::Sigmoid, UnaryKind::Tanh, UnaryKind::Sqrt, UnaryKind::Abs, UnaryKind::Sign,
            UnaryKind::Log, UnaryKind::Powf(-0.5), UnaryKind::Clamp { min: -1.0, max: 2.0 }, UnaryKind::Gtz, UnaryKind::Gelu, UnaryKind::GeluErf, UnaryKind::GeluErfGrad, UnaryKind::Silu];
        let binary = [BinaryKind::Add, BinaryKind::Sub, BinaryKind::Mul, BinaryKind::Div, BinaryKind::ReluBackward];
        let mut sources: Vec<(String, String)> = (0..4).map(|m| (format!("spatial mode {m}"), super::source(m))).collect();
        sources.extend(unary.iter().map(|&k| (format!("unary {k:?}"), unary_source(k))));
        sources.extend(binary.iter().flat_map(|&k| [(format!("binary {k:?}"), binary_source(k)), (format!("binary_bc {k:?}"), binary_bc_source(k, 4)), (format!("inplace {k:?}"), binary_inplace_source(k))]));
        for (name, src) in [("reduce_partial", reduce_partial_source()), ("reduce sum", reduce_finalize_source(ReduceKind::Sum)), ("reduce mean", reduce_finalize_source(ReduceKind::Mean)),
            ("sum_dim", sum_dim_source()), ("materialize", materialize_source()), ("gather", gather_source()), ("copy", copy_source()), ("fill", fill_source()),
            ("affine", affine_inplace_source()), ("axpy", axpy_inplace_source()), ("sgd", sgd_step_source()), ("adamw", adamw_step_source()),
            ("adamw capturable", adamw_step_capturable_source()), ("scalar increment", scalar_increment_source()), ("softmax stats", softmax_row_stats_source()),
            ("softmax", softmax_apply_source(false)), ("log_softmax", softmax_apply_source(true)), ("layer_norm", include_str!("layer_norm.cu").into()),
            ("layer_norm output", concat!("#define OUTPUT_ONLY\n", include_str!("layer_norm.cu")).into()), ("segment", include_str!("segment.cu").into())] {
            sources.push((name.into(), src));
        }
        let arches: Vec<&'static str> = match std::env::var("FERRO_NVRTC_ARCHES") {
            Ok(v) => v.split(',').map(|a| &*Box::leak(a.trim().to_string().into_boxed_str())).collect(),
            Err(_) => vec!["compute_86"],
        };
        for arch in arches {
            for (name, src) in &sources {
                let opts = CompileOptions { arch: Some(arch), ..Default::default() };
                if let Err(e) = compile_ptx_with_opts(src, opts) {
                    panic!("{name} failed to compile for {arch}: {e}\n{src}");
                }
            }
        }
    }
}
