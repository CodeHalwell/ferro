// Test-only evidence; no kernel/dispatch instrumentation or numerical fallback.
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const CAP: usize = 256;
const BYTE_CAP: usize = 32;
const DOT_CAP: usize = 512;
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"),
            c if c.is_control() => { write!(out, "\\u{:04x}", c as u32).unwrap(); }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn hash(bytes: impl IntoIterator<Item = u8>) -> String {
    let h = bytes.into_iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    format!("{h:016x}")
}
fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|v| v.to_bits()).collect() }
fn delta(before: &[u32], after: &[u32]) -> String {
    let mut changed = 0;
    let mut retained = Vec::new();
    for (i, (a, b)) in before.iter().flat_map(|x| x.to_le_bytes()).zip(after.iter().flat_map(|x| x.to_le_bytes())).enumerate() {
        if a != b {
            changed += 1;
            if retained.len() < BYTE_CAP { retained.push(format!("[{i},{a},{b}]")); }
        }
    }
    format!("{{\"before_hash\":\"{}\",\"after_hash\":\"{}\",\"before_elements\":{},\"after_elements\":{},\"changed_bytes\":{},\"byte_changes\":[{}],\"byte_cap\":{BYTE_CAP}}}",
        hash(before.iter().flat_map(|x| x.to_le_bytes())), hash(after.iter().flat_map(|x| x.to_le_bytes())),
        before.len(), after.len(), changed + before.len().abs_diff(after.len()) * 4, retained.join(","))
}
fn seeded(seed: u64, skip: usize, len: usize) -> Vec<u32> {
    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    (0..skip+len).filter_map(|i| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 4.0;
        (i >= skip).then(|| v.to_bits())
    }).collect()
}
#[derive(Clone)]
struct Snapshot { label: &'static str, a: Vec<u32>, b: Vec<u32>, parallelism: Option<usize> }
#[derive(Clone)]
pub struct Capture {
    shape: [usize; 4], seeds: [u64; 2], batch_offset: usize, explicit_threads: Option<usize>,
    stages: Vec<Snapshot>, reference: Option<Vec<u32>>,
}
impl Capture {
    pub fn new(a: &[f32], b: &[f32], shape: [usize; 4], seeds: [u64; 2], batch_offset: usize, explicit_threads: Option<usize>) -> Self {
        let mut out = Self { shape, seeds, batch_offset, explicit_threads, stages: Vec::new(), reference: None };
        out.stage("before_reference", a, b);
        out
    }
    #[allow(dead_code)]
    pub fn slab(&self, bi: usize, threads: usize) -> Self {
        let [_, m, k, n] = self.shape;
        let mut out = self.clone();
        out.shape[0] = 1;
        out.batch_offset += bi;
        out.explicit_threads = Some(threads);
        for stage in &mut out.stages {
            stage.a = stage.a[bi*m*k..(bi+1)*m*k].to_vec();
            stage.b = stage.b[bi*k*n..(bi+1)*k*n].to_vec();
        }
        out.reference = None;
        out
    }
    pub fn stage(&mut self, label: &'static str, a: &[f32], b: &[f32]) {
        self.stages.push(Snapshot { label, a: bits(a), b: bits(b), parallelism: std::thread::available_parallelism().ok().map(|v| v.get()) });
    }
    pub fn before_fast(&mut self, a: &[f32], b: &[f32], want: &[f32]) {
        self.reference = Some(bits(want));
        self.stage("before_fast", a, b);
    }
    pub fn on_failure(&mut self, a: &[f32], b: &[f32], got: &[f32], want: &[f32], context: &str) {
        self.capture_failure(a, b, got, want, context, "observed_test_failure");
    }
    pub fn capture_failure(&mut self, a: &[f32], b: &[f32], got: &[f32], want: &[f32], context: &str, origin: &str) -> Option<PathBuf> {
        if got.len() == want.len() && got.iter().zip(want).all(|(x,y)| x.to_bits() == y.to_bits()) { return None; }
        self.stage("after_fast", a, b);
        let report = self.report(got, want, context, origin);
        match persist(&report, if origin.starts_with("injected_") { "injected" } else { "failure" }) {
            Ok(path) => { eprintln!("BMM evidence origin={origin}: {}", path.display()); Some(path) }
            Err(e) => { eprintln!("BMM evidence write failed: {e}; bounded report follows\n{report}"); None }
        }
    }
    pub fn report(&self, got: &[f32], want: &[f32], context: &str, origin: &str) -> String {
        let [batch, m, k, n] = self.shape;
        let mismatch_count = got.iter().zip(want).filter(|(x,y)| x.to_bits() != y.to_bits()).count() + got.len().abs_diff(want.len());
        let indices: Vec<_> = (0..got.len().max(want.len())).filter(|&i| got.get(i).map(|v| v.to_bits()) != want.get(i).map(|v| v.to_bits())).take(CAP).collect();
        let mut out = format!("{{\"record\":\"header\",\"schema\":1,\"origin\":{},\"context\":{},\"shape_batch_m_k_n\":{:?},\"batch_offset\":{},\"seeds\":{:?},\"generator\":\"local_lcg_f32_v1\",\"scalar_reference_inputs\":\"last_captured_input_stage\",\"hash_algorithm\":\"fnv1a64_le_bytes_noncryptographic\",\"got_len\":{},\"want_len\":{},\"mismatch_count\":{mismatch_count},\"retained_count\":{},\"mismatch_cap\":{CAP},\"dot_cap\":{DOT_CAP},\"got_hash\":\"{}\",\"want_hash\":\"{}\"}}\n",
            quote(origin), quote(context), self.shape, self.batch_offset, self.seeds, got.len(), want.len(), indices.len(),
            hash(got.iter().flat_map(|v| v.to_bits().to_le_bytes())), hash(want.iter().flat_map(|v| v.to_bits().to_le_bytes())));
        let first = &self.stages[0];
        let last = self.stages.last().unwrap();
        let sa = seeded(self.seeds[0], self.batch_offset*m*k, first.a.len());
        let sb = seeded(self.seeds[1], self.batch_offset*k*n, first.b.len());
        writeln!(out, "{{\"record\":\"seed_to_first_snapshot\",\"a\":{},\"b\":{}}}", delta(&sa, &first.a), delta(&sb, &first.b)).unwrap();
        for (i, stage) in self.stages.iter().enumerate() {
            let previous = if i == 0 { first } else { &self.stages[i-1] };
            writeln!(out, "{{\"record\":\"input_stage\",\"label\":{},\"available_parallelism\":{},\"a\":{},\"b\":{}}}",
                quote(stage.label), stage.parallelism.map_or("null".into(), |v| v.to_string()), delta(&previous.a, &stage.a), delta(&previous.b, &stage.b)).unwrap();
        }
        if let Some(reference) = &self.reference {
            writeln!(out, "{{\"record\":\"reference_before_fast_to_failure\",\"change\":{}}}", delta(reference, &bits(want))).unwrap();
        }
        // Runtime observations bracket, but do not intercept, the kernel's own query.
        // Never label inferred groups as actually observed worker execution.
        let observed = self.stages.iter().find(|s| s.label == "before_fast").unwrap_or(first).parallelism;
        let rows = batch*m;
        let threads = self.explicit_threads.unwrap_or(observed.unwrap_or(1)).clamp(1, rows.max(1));
        let inline = self.explicit_threads.map_or(batch as u128*m as u128*k as u128*n as u128 <= (1 << 21) || threads <= 1, |_| threads <= 1);
        let rows_per = if inline { rows.max(1) } else { rows.div_ceil(threads) };
        writeln!(out, "{{\"record\":\"decomposition\",\"status\":{},\"actual_worker_execution\":null,\"explicit_threads\":{},\"inferred_inline\":{inline},\"inferred_rows_per\":{rows_per},\"inferred_groups\":{},\"constants\":{{\"MC\":48,\"MR\":6,\"NR\":16,\"KC\":256,\"batch_threshold\":2097152}}}}",
            quote(if self.explicit_threads.is_some() { "derived_from_explicit_argument_not_instrumented" } else { "conditional_on_bracket_parallelism_not_instrumented" }),
            self.explicit_threads.map_or("null".into(), |v| v.to_string()), rows.div_ceil(rows_per)).unwrap();
        for (ordinal, i) in indices.iter().copied().enumerate() {
            let gv = got.get(i).map_or("null".into(), |v| v.to_bits().to_string());
            let wv = want.get(i).map_or("null".into(), |v| v.to_bits().to_string());
            if m == 0 || n == 0 || i >= batch*m*n {
                writeln!(out, "{{\"record\":\"mismatch\",\"index\":{i},\"got_bits\":{gv},\"want_bits\":{wv},\"coordinate\":null}}").unwrap();
                continue;
            }
            let (bi, row, col) = (i/(m*n), (i/n)%m, i%n);
            let mut scalar = 0.0f32;
            let mut wide = 0.0f64;
            let mut omitted = 0.0f32;
            let mut operands = Vec::new();
            let mut p90 = "null".to_string();
            for p in 0..k {
                let (ab, bb) = (last.a[(bi*m+row)*k+p], last.b[(bi*k+p)*n+col]);
                let (a, b) = (f32::from_bits(ab), f32::from_bits(bb));
                scalar += a*b;
                wide += a as f64*b as f64;
                if p != 90 { omitted += a*b; }
                if p == 90 { p90 = format!("{{\"a_bits\":{ab},\"b_bits\":{bb},\"product_bits\":{}}}", (a*b).to_bits()); }
                if ordinal == 0 && p < DOT_CAP { operands.push(format!("[{ab},{bb}]")); }
            }
            writeln!(out, "{{\"record\":\"mismatch\",\"index\":{i},\"coordinate\":[{bi},{row},{col}],\"source_batch\":{},\"got_bits\":{gv},\"want_bits\":{wv},\"independent_scalar_bits\":{},\"independent_f64_bits\":{},\"independent_f64\":{},\"p90\":{p90},\"counterfactual_skip90_bits\":{},\"inferred_group\":{},\"first_dot_operand_bits\":[{}],\"dot_operand_count\":{k}}}",
                bi+self.batch_offset, scalar.to_bits(), wide.to_bits(), if wide.is_finite() { wide.to_string() } else { "null".into() }, omitted.to_bits(), (bi*m+row)/rows_per, operands.join(",")).unwrap();
        }
        out
    }
}

pub fn persist(report: &str, prefix: &str) -> std::io::Result<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../verification/foundation-wave/fastcpu-diagnostics");
    std::fs::create_dir_all(&dir)?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
    let id = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("{prefix}-{}-{stamp}-{id}.jsonl", std::process::id()));
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
    // Persist numerical evidence before optional executable IO/provenance.
    file.write_all(report.as_bytes())?;
    file.sync_all()?;
    let thread = std::thread::current();
    writeln!(file, "{{\"record\":\"reporting_thread\",\"id\":{},\"name\":{}}}",
        quote(&format!("{:?}", thread.id())), thread.name().map_or("null".into(), quote))?;
    let exe = std::env::current_exe()?;
    let mut input = std::fs::File::open(&exe)?;
    let mut h = 0xcbf29ce484222325u64;
    let mut buffer = [0u8; 65536];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 { break; }
        for b in &buffer[..count] { h = (h ^ *b as u64).wrapping_mul(0x100000001b3); }
    }
    // No env dump or command-line dump. Build toolchain identity cannot be
    // recovered reliably from a running binary; runtime hints are not proof.
    writeln!(file, "{{\"record\":\"process\",\"pid\":{},\"executable\":{},\"executable_fnv1a64\":\"{h:016x}\",\"os\":{},\"arch\":{},\"debug_assertions\":{},\"build_rustc\":null,\"build_flags\":null,\"toolchain_status\":\"not_embedded_see_cargo_run_evidence\",\"rustup_toolchain_hint\":{},\"rust_test_threads_hint\":{}}}",
        std::process::id(), quote(&exe.to_string_lossy()), quote(std::env::consts::OS), quote(std::env::consts::ARCH), cfg!(debug_assertions),
        std::env::var("RUSTUP_TOOLCHAIN").ok().map_or("null".into(), |v| quote(&v)),
        std::env::var("RUST_TEST_THREADS").ok().map_or("null".into(), |v| quote(&v)))?;
    file.sync_all()?;
    Ok(path)
}

#[allow(dead_code)] // Shared helper also included by lib tests.
pub fn injected_report(a: &[f32], b: &[f32], got: &[f32], want: &[f32]) -> String {
    let mut capture = Capture::new(a, b, [1, 1, 128, 300], [2556, 3556], 0, Some(1));
    capture.before_fast(a, b, want);
    let path = capture.capture_failure(a, b, got, want, "schema_test_only", "injected_omission_p90_test_buffer").unwrap();
    std::fs::read_to_string(path).unwrap()
}
