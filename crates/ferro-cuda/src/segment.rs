use super::*;
use ferro_core::dispatch::{SegmentOp, SegmentPlan};

struct CudaSegments {
    csr: Box<dyn DeviceBuffer>,
    edges: usize,
    groups: usize,
}
impl SegmentPlan for CudaSegments {
    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl CudaBackend {
    pub(crate) fn prepare_segment_plan(&self, ids: &[usize], groups: usize) -> Result<Arc<dyn SegmentPlan>> {
        let invalid = || Error::InvalidShape { op: "prepare_segments", msg: "invalid or oversized segment topology".into() };
        let len = groups.checked_add(1).and_then(|n| n.checked_add(ids.len())).ok_or_else(invalid)?;
        if len > i32::MAX as usize || ids.iter().any(|&i| i >= groups) { return Err(invalid()); }
        let mut csr = vec![0i64; len];
        for &s in ids { csr[s + 1] += 1; }
        for s in 0..groups { csr[s + 1] += csr[s]; }
        let mut next = csr[..groups].to_vec();
        for (e, &s) in ids.iter().enumerate() {
            csr[groups + 1 + next[s] as usize] = e as i64;
            next[s] += 1;
        }
        Ok(Arc::new(CudaSegments { csr: self.alloc_i64_from_host(&csr)?, edges: ids.len(), groups }))
    }

    /// Successful nonempty sum forward/backward, softmax forward/backward enqueues,
    /// four-byte finite-validation control downloads, and control allocations. No graph replays.
    /// Feature transfer counters are layer_norm_counts; topology uses layout_counts.
    pub fn segment_counts(&self) -> [usize; 6] {
        std::array::from_fn(|i| self.segment_counters[i].load(Ordering::Relaxed))
    }

    pub(crate) fn run_segment(&self, plan: &dyn SegmentPlan, kind: SegmentOp,
        x: &dyn DeviceBuffer, saved: Option<&dyn DeviceBuffer>, width: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        let op = "segment_dev";
        let invalid = || Error::InvalidShape { op, msg: "invalid segment buffer or size".into() };
        let plan = plan.as_any().downcast_ref::<CudaSegments>().ok_or_else(invalid)?;
        let csr = plan.csr.as_any().downcast_ref::<CudaBufI64>().ok_or_else(invalid)?;
        if !Arc::ptr_eq(csr.data.stream(), &self.stream) { return Err(invalid()); }
        let x = self.resident(op, x)?;
        let edges = plan.edges.checked_mul(width).ok_or_else(invalid)?;
        let groups = plan.groups.checked_mul(width).ok_or_else(invalid)?;
        if edges > i32::MAX as usize || groups > i32::MAX as usize { return Err(invalid()); }
        let (mode, input, output) = match kind {
            SegmentOp::Sum => (0i32, edges, groups),
            SegmentOp::SumBackward => (1i32, groups, edges),
            SegmentOp::Softmax => (2i32, edges, edges),
            SegmentOp::SoftmaxBackward => (3i32, edges, edges),
        };
        if x.data.len() != input { return Err(invalid()); }
        let y = if mode == 3 {
            let y = self.resident(op, saved.ok_or_else(invalid)?)?;
            if y.data.len() != edges { return Err(invalid()); }
            y
        } else { x };
        let mut out = self.alloc_zeros(op, output)?;
        if groups != 0 && output != 0 {
            let mut status = if mode == 2 {
                let status = self.stream.alloc_zeros::<i32>(1).map_err(|e| cuda_err(op, e))?;
                self.segment_counters[5].fetch_add(1, Ordering::Relaxed);
                Some(status)
            } else { None };
            let f = self.get_kernel(op, include_str!("segment.cu"))?;
            let ng = plan.groups as i32;
            let w = width as i32;
            let mut launch = self.stream.launch_builder(&f);
            launch.arg(&*x.data).arg(&*y.data).arg(&csr.data).arg(&mut out);
            let null_status = 0u64;
            if let Some(status) = status.as_mut() { launch.arg(status); }
            else { launch.arg(&null_status); }
            launch.arg(&ng).arg(&w).arg(&mode);
            unsafe { launch.launch(LaunchConfig::for_num_elems(groups as u32)) }.map_err(|e| cuda_err(op, e))?;
            self.segment_counters[mode as usize].fetch_add(1, Ordering::Relaxed);
            if mode == 2 {
                // Preserve finite-logit errors without downloading feature data.
                // This explicit four-byte control read synchronizes eager softmax.
                let host = self.stream.clone_dtoh(status.as_ref().unwrap()).map_err(|e| cuda_err(op, e))?;
                self.segment_counters[4].fetch_add(1, Ordering::Relaxed);
                if host[0] != 0 {
                    return Err(Error::Unsupported { op: "segment_softmax", msg: "finite logits required".into() });
                }
            }
        }
        Ok(self.wrap(out))
    }
}
