//! Owning training-state snapshots and immutable checkpoint generations.
//! A single checkpoint.json replacement publishes all arrays and metadata.
//! Legacy model.safetensors plus checkpoint.json directories remain readable.
//! Module snapshots include named buffers and exact scalar modes/counters.
//! Whole-training snapshots also bind optimizer slots and parameter ties.
//! Snapshot I/O preserves tensor dtype bits and device placement in memory.
//! Restore is transactional for supported whole-contiguous CPU f32 targets.
//! Model device restore is rejected: no backend transaction exists. Optimizer
//! buffer restore is separate and supports devices without a rollback guarantee.

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::safetensors::{load_safetensors, save_safetensors};
use crate::tensor::Tensor;

/// Bumped on incompatible layout changes; loaders reject newer versions.
pub const FORMAT_VERSION: u32 = 1;

const MODEL_FILE: &str = "model.safetensors";

const META_FILE: &str = "checkpoint.json";
/// Prefix under which `OptimizerState::snapshot` arrays are stored.
pub const OPTIM_PREFIX: &str = "optim.";

pub struct Checkpoint {
    pub version: u32,
    pub step: u64,
    /// Seed the training run was started from (`None` if not tracked).
    pub rng_seed: Option<u64>,
    /// Counter-based RNG stream position (e.g. the Philox `offset` passed to
    /// `Tensor::dropout`), so recomputed masks land on the same elements
    /// (`None` if not tracked).
    pub rng_offset: Option<u64>,
    /// Parameters and optimizer buffers, one entry per named array.
    pub tensors: Vec<(String, Tensor)>,
}

impl Clone for Checkpoint {
    fn clone(&self) -> Self {
        Self { version: self.version, step: self.step, rng_seed: self.rng_seed, rng_offset: self.rng_offset,
            tensors: self.tensors.iter().map(|(n,t)| (n.clone(), t.owned_detach_copy())).collect() }
    }
}

impl Checkpoint {
    pub fn new(step: u64) -> Checkpoint {
        Checkpoint {
            version: FORMAT_VERSION,
            step,
            rng_seed: None,
            rng_offset: None,
            tensors: Vec::new(),
        }
    }

    pub fn with_tensor(mut self, name: impl Into<String>, t: Tensor) -> Checkpoint {
        self.tensors.push((name.into(), t.owned_detach_copy()));
        self
    }

    /// Explicit named buffers/modes can be supplied by module owners. Values own storage.
    pub fn with_named_state(mut self, state: &[(String, Tensor)]) -> Result<Self> {
        for (name, t) in state {
            if self.tensors.iter().any(|(n, _)| n == name) { return Err(format_error("duplicate state name")); }
            self.tensors.push((name.clone(), t.try_owned_detach_copy()?));
        }
        Ok(self)
    }

    /// Strict CPU transaction over explicit parameter/buffer handles. No rebinding.
    pub fn load_named_state_into(&self, targets: &[(String, Tensor)]) -> Result<()> {
        let staged = self.prepare_named_state(targets)?;
        commit_tensors(staged);
        Ok(())
    }

    fn prepare_named_state(&self, targets: &[(String, Tensor)]) -> Result<Vec<(Tensor, Tensor)>> {
        self.validate_header()?;
        if self.tensors.len() != targets.len() { return Err(format_error("state key count mismatch")); }
        let mut names = std::collections::HashSet::new();
        let mut identities = std::collections::HashMap::<usize, Vec<u32>>::new();
        let mut staged = Vec::new();
        for (name, dst) in targets {
            if !names.insert(name) { return Err(format_error("duplicate target name")); }
            let src = self.tensor(name)?;
            validate_destination(dst, src)?;
            let identity = std::sync::Arc::as_ptr(&dst.0.storage) as usize;
            let bits: Vec<_> = src.to_vec().iter().map(|x| x.to_bits()).collect();
            if let Some(previous) = identities.get(&identity) {
                if previous != &bits { return Err(format_error("conflicting aliased state values")); }
                continue;
            }
            identities.insert(identity, bits);
            staged.push((dst.clone(), src.to_device(crate::device::Device::Cpu)?.owned_detach_copy()));
        }
        Ok(staged)
    }

    pub fn with_rng(self, rng: &crate::rng::Rng) -> Self {
        let words: Vec<f32> = rng.state().iter().flat_map(|x| (0..4).map(move |i| ((x >> (i * 16)) & 65535) as f32)).collect();
        self.with_tensor("rng.xorshift128", Tensor::from_vec(words, &[8]).unwrap())
    }

    pub fn load_rng_into(&self, rng: &crate::rng::Rng) -> Result<()> {
        let t = self.tensor("rng.xorshift128")?;
        if t.shape() != [8] || t.dtype() != crate::dtype::DType::F32 { return Err(format_error("invalid RNG state")); }
        let words = t.to_vec();
        if words.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 65535.0 || x.fract() != 0.0) { return Err(format_error("invalid RNG word")); }
        let mut state = [0u64; 2];
        for (i, x) in words.iter().enumerate() { state[i / 4] |= (*x as u64) << ((i % 4) * 16); }
        rng.restore_state(state)
    }

    pub fn with_optimizer(mut self, name: &str, opt: &dyn crate::optim::OptimizerState) -> Result<Self> {
        if name.is_empty() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') { return Err(format_error("invalid optimizer namespace")); }
        let prefix = format!("optim.{name}.");
        if self.tensors.iter().any(|(n, _)| n.starts_with(&prefix)) { return Err(format_error("duplicate optimizer namespace")); }
        for (key, t) in opt.snapshot() { self.tensors.push((format!("{prefix}{key}"), t.try_owned_detach_copy()?)); }
        let config = opt.configuration();
        let n = config.len();
        self.tensors.push((format!("{prefix}config"), Tensor::from_vec(config, &[n])?));
        Ok(self)
    }

    pub fn load_named_optim_into(&self, name: &str, opt: &mut dyn crate::optim::OptimizerState) -> Result<()> {
        let prefix = format!("optim.{name}.");
        let config = self.tensor(&format!("{prefix}config"))?;
        if config.dtype() != crate::dtype::DType::F32 || config.ndim() != 1 { return Err(format_error("invalid configuration tensor")); }
        let state: Vec<_> = self.tensors.iter().filter_map(|(n,t)| n.strip_prefix(&prefix).filter(|k| *k != "config").map(|k| (k.to_owned(), t.clone()))).collect();
        let old_config = opt.configuration();
        opt.restore_configuration(&config.to_vec())?;
        if let Err(e) = opt.restore(&state) {
            opt.restore_configuration(&old_config)?;
            return Err(e);
        }
        Ok(())
    }

    /// Quiescent model/optimizer/RNG snapshot for CPU restart. Optimizer order is validated
    /// against canonical model parameter names, never process-local pointers.
    pub fn from_training_state(step: u64, module: &dyn Module,
        optimizers: &[(&str, &dyn crate::optim::OptimizerState)], rng: &crate::rng::Rng) -> Result<Self> {
        validate_optimizer_isolation(module, optimizers.iter().map(|(_,opt)| *opt))?;
        let mut cp = Self::from_module(step, module).with_rng(rng);
        for (key, value) in parameter_schema(module) { cp.tensors.push((key, encode_u64(value))); }
        for (name, opt) in optimizers {
            cp = cp.with_optimizer(name, *opt)?;
            for (i, index) in optimizer_binding(module, *opt)?.into_iter().enumerate() {
                cp.tensors.push((format!("bindings.{name}.{i}.{index}"), encode_u64(1)));
            }
        }
        cp.validate_header()?;
        Ok(cp)
    }

    /// All fallible validation and staging precede live writes. Failed preparation
    /// drops every token, leaving parameters, buffers, modes, optimizers and RNG
    /// unchanged. The commit phase has no fallible CPU operations. Caller must
    /// pause training; this is not concurrent-reader isolation or panic recovery.
    pub fn load_training_state_into(&self, module: &dyn Module,
        optimizers: &mut [(&str, &mut dyn crate::optim::OptimizerState)], rng: &crate::rng::Rng) -> Result<()> {
        self.validate_header()?;
        let scalars = self.module_scalars()?;
        module.validate_scalars(&scalars)?;
        validate_optimizer_isolation(module, optimizers.iter().map(|(_,opt)| &**opt))?;
        let scratch_rng = crate::rng::Rng::new(0);
        self.load_rng_into(&scratch_rng)?;
        let rng_state = scratch_rng.state();
        let mut expected: std::collections::HashSet<String> = module_targets(module).into_iter().map(|(n,_)| n).collect();
        expected.extend(scalars.iter().map(|(n,_)| format!("scalars.{n}")));
        expected.insert("rng.xorshift128".into());
        let mut trainable = Vec::new();
        for (key, _) in parameter_schema(module) {
            let value = decode_u64(self.tensor(&key)?)?;
            if (key.starts_with("ties.") && value != 1) || value > 1 { return Err(format_error("invalid parameter schema")); }
            expected.insert(key);
        }
        let mut flags = std::collections::HashMap::new();
        for (name, p) in module.named_parameters() {
            let flag = decode_u64(self.tensor(&format!("trainable.{name}"))?)? != 0;
            if flags.insert(p.identity(), flag).is_some_and(|old| old != flag) { return Err(format_error("conflicting tied freeze flags")); }
            trainable.push((p, flag));
        }
        let mut namespaces = std::collections::HashSet::new();
        for (name, opt) in optimizers.iter() {
            if !namespaces.insert(*name) { return Err(format_error("duplicate optimizer target")); }
            expected.insert(format!("optim.{name}.config"));
            expected.extend(opt.snapshot().iter().map(|(n,_)| format!("optim.{name}.{n}")));
            for (i, index) in optimizer_binding(module, &**opt)?.into_iter().enumerate() {
                let key = format!("bindings.{name}.{i}.{index}");
                if decode_u64(self.tensor(&key)?)? != 1 { return Err(format_error("optimizer parameter order mismatch")); }
                expected.insert(key);
            }
        }
        if expected.len() != self.tensors.len() || self.tensors.iter().any(|(n,_)| !expected.contains(n)) {
            return Err(format_error("training state keys mismatch"));
        }
        let mut model = Self::new(self.step);
        model.tensors = self.tensors.iter().filter(|(n,_)| n.starts_with("model.") || n.starts_with("buffers.")).cloned().collect();
        let staged = model.prepare_named_state(&module_targets(module))?;
        let mut commits = Vec::new();
        for (name, opt) in optimizers.iter_mut() {
            let prefix = format!("optim.{name}.");
            let config = self.tensor(&format!("{prefix}config"))?;
            if config.dtype() != crate::dtype::DType::F32 || config.ndim() != 1 { return Err(format_error("invalid configuration tensor")); }
            let state: Vec<_> = self.tensors.iter().filter_map(|(n,t)| n.strip_prefix(&prefix).filter(|k| *k != "config").map(|k| (k.to_owned(),t.clone()))).collect();
            commits.push(opt.prepare_cpu_restore(&config.to_vec(), &state)?);
        }
        commit_tensors(staged);
        for commit in commits { commit(); }
        for (p, flag) in trainable { p.set_trainable(flag); }
        module.commit_scalars(&scalars);
        rng.restore_state(rng_state).expect("validated RNG state");
        Ok(())
    }

    pub fn with_rng_seed(mut self, seed: u64) -> Checkpoint {
        self.rng_seed = Some(seed);
        self
    }

    pub fn with_rng_offset(mut self, offset: u64) -> Checkpoint {
        self.rng_offset = Some(offset);
        self
    }

    /// Own parameters (`model.`), buffers (`buffers.`), and exact u64 scalar
    /// state (`scalars.` as four 16-bit f32 limbs). No optimizer freeze metadata.
    pub fn from_module(step: u64, module: &dyn Module) -> Checkpoint {
        let mut cp = Checkpoint::new(step);
        for (name, p) in module.named_parameters() {
            cp.tensors.push((format!("model.{name}"), p.tensor().owned_detach_copy()));
        }
        for (name, t) in module.named_buffers() {
            cp.tensors.push((format!("buffers.{name}"), t.owned_detach_copy()));
        }
        for (name, value) in module.snapshot_scalars() {
            cp.tensors.push((format!("scalars.{name}"), encode_u64(value)));
        }
        cp
    }

    /// Restore parameters into a module, strictly: every module parameter must
    /// be present with matching shape/dtype and nothing else may remain -
    /// except optimizer buffers (`optim.` prefix), which belong to
    /// `load_optim_into`.
    pub fn load_into_module(&self, module: &dyn Module) -> Result<()> {
        self.validate_header()?;
        let scalars = self.module_scalars()?;
        module.validate_scalars(&scalars)?;
        let targets = module_targets(module);
        let mut state = Checkpoint::new(self.step);
        state.tensors = self.tensors.iter().filter(|(n, _)| !n.starts_with(OPTIM_PREFIX) && !n.starts_with("rng.") && !n.starts_with("scalars.")).cloned().collect();
        state.load_named_state_into(&targets)?;
        module.commit_scalars(&scalars);
        Ok(())
    }

    /// Snapshot a module's parameters plus an optimizer's state buffers.
    /// Optimizer arrays share the generation payload under `optim.` names.
    /// This legacy format has no optimizer configuration or slot bindings.
    pub fn from_module_with_optim(
        step: u64,
        module: &dyn Module,
        opt: &dyn crate::optim::OptimizerState,
    ) -> Checkpoint {
        let mut cp = Checkpoint::from_module(step, module);
        for (name, t) in opt.snapshot() {
            cp.tensors.push((format!("{OPTIM_PREFIX}{name}"), t));
        }
        cp
    }

    /// Restore optimizer state, strictly: every snapshot array must be present
    /// with matching shape and nothing extra may remain.
    pub fn load_optim_into(&self, opt: &mut dyn crate::optim::OptimizerState) -> Result<()> {
        const OP: &str = "checkpoint_load";
        let mut remaining: Vec<(String, Tensor)> = self
            .tensors
            .iter()
            .filter(|(n, _)| n.starts_with(OPTIM_PREFIX))
            .map(|(n, t)| (n[OPTIM_PREFIX.len()..].to_string(), t.clone()))
            .collect();
        let want = opt.snapshot();
        let mut restored = Vec::with_capacity(remaining.len());
        for (name, _) in &want {
            let pos = remaining
                .iter()
                .position(|(n, _)| n == name)
                .ok_or_else(|| Error::Format {
                    op: OP,
                    msg: format!("optimizer state is missing {name:?}"),
                })?;
            restored.push(remaining.swap_remove(pos));
        }
        match remaining.first() {
            None => opt.restore(&restored),
            Some((n, _)) => Err(Error::Format {
                op: OP,
                msg: format!("checkpoint has unexpected optimizer tensor {n:?}"),
            }),
        }
    }

    fn validate_header(&self) -> Result<()> {
        if self.version != FORMAT_VERSION { return Err(format_error("unsupported checkpoint version")); }
        let mut names = std::collections::HashSet::new();
        if self.tensors.iter().any(|(n, _)| !names.insert(n)) { return Err(format_error("duplicate checkpoint key")); }
        Ok(())
    }

    fn module_scalars(&self) -> Result<Vec<(String, u64)>> {
        self.tensors.iter().filter_map(|(n, t)| n.strip_prefix("scalars.").map(|k| (k, t)))
            .map(|(k, t)| Ok((k.to_owned(), decode_u64(t)?))).collect()
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
            .ok_or_else(|| Error::Format {
                op: "checkpoint_load",
                msg: format!("no tensor named {name:?}"),
            })
    }

    /// An optimizer buffer as host f32 values (they live as plain `Vec<f32>`).
    pub fn f32_buffer(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.dtype() != crate::dtype::DType::F32 {
            return Err(Error::DtypeMismatch {
                op: "checkpoint_load",
                expected: crate::dtype::DType::F32,
                got: t.dtype(),
            });
        }
        Ok(t.to_vec())
    }

    pub fn save_to_dir<P: AsRef<Path>>(&self, dir: P) -> Result<()> {
        self.save_before_publish(dir.as_ref(), || Ok(()))
    }

    fn save_before_publish(&self, dir: &Path, before_publish: impl FnOnce() -> Result<()>) -> Result<()> {
        const OP: &str = "checkpoint_save";
        fs::create_dir_all(dir).map_err(|e| Error::Io {
            op: OP,
            msg: format!("{}: {e}", dir.display()),
        })?;
        let opt_field = |v: Option<u64>| match v {
            Some(s) => s.to_string(),
            None => "null".to_string(),
        };
        let meta = format!(
            "{{\n  \"version\": {},\n  \"step\": {},\n  \"rng_seed\": {},\n  \"rng_offset\": {}\n}}\n",
            self.version,
            self.step,
            opt_field(self.rng_seed),
            opt_field(self.rng_offset),
        );
        let generation = dir.join(format!("{}.safetensors", temp_path(dir, "generation").file_name().unwrap().to_string_lossy()));
        fs::create_dir(&generation).map_err(io_error)?;
        save_safetensors(generation.join(MODEL_FILE), &self.tensors.iter().map(|(n,t)| (n.as_str(),t)).collect::<Vec<_>>())?;
        let hash = checksum(&fs::read(generation.join(MODEL_FILE)).map_err(io_error)?);
        let meta = meta.replace("{\n", &format!("{{\n  \"checksum\": {hash},\n"));
        fs::write(generation.join(META_FILE), &meta).map_err(io_error)?;
        for file in [MODEL_FILE, META_FILE] {
            fs::OpenOptions::new().write(true).open(generation.join(file)).and_then(|f| f.sync_all()).map_err(io_error)?;
        }
        sync_directory(&generation)?;
        sync_directory(dir)?;
        let pointer = temp_path(dir, "CURRENT");
        let published = meta.replace("{\n", &format!("{{\n  \"generation\": \"{}\",\n", generation.file_name().unwrap().to_string_lossy()));
        fs::write(&pointer, published).map_err(io_error)?;
        fs::OpenOptions::new().write(true).open(&pointer).and_then(|f| f.sync_all()).map_err(io_error)?;
        before_publish()?;
        // The only publication point. Never overwrite files in a published generation.
        fs::rename(&pointer, dir.join(META_FILE)).map_err(io_error)?;
        sync_directory(dir)?;
        Ok(())
    }

    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Checkpoint> {
        const OP: &str = "checkpoint_load";
        let dir = dir.as_ref();
        let meta_bytes = fs::read(dir.join(META_FILE)).map_err(io_error)?;
        let meta = Meta::parse(&meta_bytes)?;
        if meta.version > FORMAT_VERSION {
            return Err(Error::Unsupported { op: OP, msg: format!("unsupported checkpoint version {}", meta.version) });
        }
        let text = std::str::from_utf8(&meta_bytes).map_err(|_| format_error("metadata UTF-8"))?;
        let generation = text.lines().find(|l| l.trim().starts_with("\"generation\""))
            .and_then(|l| l.split('"').nth(3));
        let selected;
        let dir = if let Some(name) = generation {
            if name.is_empty() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)) || name == "." || name == ".." { return Err(format_error("invalid generation pointer")); }
            selected = dir.join(name);
            selected.as_path()
        } else { dir };
        if let Some(line) = text.lines().find(|l| l.trim().starts_with("\"checksum\"")) {
            let hash = line.split(':').nth(1).and_then(|x| x.trim().trim_end_matches(',').parse::<u64>().ok()).ok_or_else(|| format_error("invalid checksum"))?;
            if checksum(&fs::read(dir.join(MODEL_FILE)).map_err(io_error)?) != hash { return Err(format_error("checkpoint checksum mismatch")); }
        }
        let tensors = load_safetensors(dir.join(MODEL_FILE))?;
        Ok(Checkpoint {
            version: meta.version,
            step: meta.step,
            rng_seed: meta.rng_seed,
            rng_offset: meta.rng_offset,
            tensors,
        })
    }
}

struct Meta {
    version: u32,
    step: u64,
    rng_seed: Option<u64>,
    rng_offset: Option<u64>,
}

impl Meta {
    /// The sidecar is written by this module, one `"key": value` per line;
    /// parse exactly that shape rather than carrying a general JSON parser.
    fn parse(bytes: &[u8]) -> Result<Meta> {
        const OP: &str = "checkpoint_load";
        let ferr = |msg: String| Error::Format { op: OP, msg };
        let text = std::str::from_utf8(bytes).map_err(|_| ferr("sidecar is not utf-8".into()))?;
        let field = |key: &str| -> Result<Option<&str>> {
            for line in text.lines() {
                let line = line.trim();
                let some = line
                    .strip_prefix('"')
                    .and_then(|r| r.split_once("\":"))
                    .filter(|(k, _)| *k == key);
                if let Some((_, v)) = some {
                    return Ok(Some(v.trim().trim_end_matches(',')));
                }
            }
            Ok(None)
        };
        let num = |key: &str| -> Result<u64> {
            field(key)?
                .and_then(|v| v.parse::<u64>().ok())
                .ok_or_else(|| ferr(format!("sidecar missing integer {key:?}")))
        };
        let opt_num = |key: &str| -> Result<Option<u64>> {
            match field(key)? {
                None | Some("null") => Ok(None),
                Some(v) => v
                    .parse::<u64>()
                    .map(Some)
                    .map_err(|_| ferr(format!("bad {key} {v:?}"))),
            }
        };
        let rng_seed = opt_num("rng_seed")?;
        let rng_offset = opt_num("rng_offset")?;
        Ok(Meta {
            version: u32::try_from(num("version")?).map_err(|_| ferr("version overflow".into()))?,
            step: num("step")?,
            rng_seed,
            rng_offset,
        })
    }
}

fn validate_optimizer_isolation<'a>(module: &dyn Module, optimizers: impl Iterator<Item = &'a dyn crate::optim::OptimizerState>) -> Result<()> {
    let mut occupied: std::collections::HashSet<_> = module_targets(module).iter().map(|(_,t)| std::sync::Arc::as_ptr(&t.0.storage) as usize).collect();
    for opt in optimizers {
        let buffers = opt.state_buffers().ok_or_else(|| format_error("optimizer does not expose buffer identity"))?;
        for t in buffers {
            if !occupied.insert(std::sync::Arc::as_ptr(&t.0.storage) as usize) {
                return Err(format_error("optimizer buffers must not alias model or other optimizer state; construct independent optimizers"));
            }
        }
    }
    Ok(())
}

fn optimizer_binding(module: &dyn Module, opt: &dyn crate::optim::OptimizerState) -> Result<Vec<String>> {
    let named = module.named_parameters();
    opt.state_parameters().ok_or_else(|| format_error("optimizer does not expose parameter identity"))?
        .iter().map(|p| named.iter().find(|(_, target)| target.identity() == p.identity())
            .map(|(n, _)| n.clone()).ok_or_else(|| format_error("optimizer parameter outside model"))).collect()
}

fn parameter_schema(module: &dyn Module) -> Vec<(String, u64)> {
    let named = module.named_parameters();
    named.iter().flat_map(|(name, p)| {
        let canonical = &named.iter().find(|(_, q)| p.identity() == q.identity()).unwrap().0;
        [(format!("ties.{name}.{canonical}"), 1), (format!("trainable.{name}"), p.is_trainable() as u64)]
    }).collect()
}

fn commit_tensors(staged: Vec<(Tensor, Tensor)>) {
    for (dst, src) in staged { crate::inplace::raw_copy_("checkpoint_load", &dst, &src).expect("prevalidated CPU copy"); }
}

fn module_targets(module: &dyn Module) -> Vec<(String, Tensor)> {
    module.named_parameters().into_iter().map(|(n, p)| (format!("model.{n}"), p.tensor()))
        .chain(module.named_buffers().into_iter().map(|(n, t)| (format!("buffers.{n}"), t))).collect()
}

fn encode_u64(value: u64) -> Tensor {
    Tensor::from_vec((0..4).map(|i| ((value >> (16 * i)) & 65535) as f32).collect(), &[4]).unwrap()
}

fn decode_u64(t: &Tensor) -> Result<u64> {
    if t.shape() != [4] || t.dtype() != crate::dtype::DType::F32 { return Err(format_error("invalid exact scalar")); }
    let mut value = 0;
    for (i, word) in t.to_vec().into_iter().enumerate() {
        if !word.is_finite() || !(0.0..=65535.0).contains(&word) || word.fract() != 0.0 { return Err(format_error("invalid scalar limb")); }
        value |= (word as u64) << (16 * i);
    }
    Ok(value)
}

pub(crate) fn validate_destination(dst: &Tensor, src: &Tensor) -> Result<()> {
    if dst.shape() != src.shape() || dst.dtype() != src.dtype() { return Err(format_error("state shape/dtype mismatch")); }
    if dst.device() != crate::device::Device::Cpu || dst.dtype() != crate::dtype::DType::F32 || dst.0.offset != 0 || !dst.is_contiguous() || dst.storage_len() != dst.numel() {
        return Err(Error::Unsupported { op: "checkpoint_load", msg: "transaction requires whole contiguous CPU f32 destinations; device commit protocol not implemented".into() });
    }
    Ok(())
}

// Accidental corruption detection, not cryptographic authentication.
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3))
}

fn format_error(msg: &str) -> Error { Error::Format { op: "checkpoint", msg: msg.into() } }
fn io_error(e: std::io::Error) -> Error { Error::Io { op: "checkpoint", msg: e.to_string() } }

fn sync_directory(path: &Path) -> Result<()> {
    // Windows std has no directory fsync; file data is flushed, but power-loss
    // durability of directory metadata is filesystem/OS dependent.
    #[cfg(unix)]
    fs::File::open(path).and_then(|f| f.sync_all()).map_err(io_error)?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    #[test]
    fn interruption_before_publication_keeps_previous_pair() {
        let dir = std::env::temp_dir().join(format!("ferro_interruption_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        Checkpoint::new(1).with_tensor("x", Tensor::scalar(1.0)).save_to_dir(&dir).unwrap();
        let previous = fs::read(dir.join(META_FILE)).unwrap();
        let cp = Checkpoint::new(2).with_tensor("x", Tensor::scalar(2.0));
        assert!(cp.save_before_publish(&dir, || Err(format_error("injected pre-publication failure"))).is_err());
        assert_eq!(fs::read(dir.join(META_FILE)).unwrap(), previous);
        let loaded = Checkpoint::load_from_dir(&dir).unwrap();
        assert_eq!(loaded.step, 1);
        assert_eq!(loaded.tensor("x").unwrap().item(), 1.0);
        fs::remove_dir_all(dir).unwrap();
    }
}

fn temp_path(dir: &Path, file: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    dir.join(format!(".{file}.tmp{}-{nanos}-{sequence}", std::process::id()))
}
