//! Training checkpoints: model parameters, optimizer buffers (momentum /
//! Adam moments), the global step, and the RNG seed, saved atomically as a
//! safetensors file for the arrays plus a `checkpoint.json` sidecar for the
//! scalar metadata. Resume works because every quantity the update rule reads
//! is captured; see `tests/checkpoint.rs` for the identical-loss-trajectory
//! proof.
//!
//! Note on scope: `Sgd`/`Adam` keep their buffers private and `Rng` does not
//! expose its internal state, so this module is a container over named
//! tensors - the caller snapshots and restores optimizer buffers through it,
//! and records the RNG seed rather than stream position.

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::safetensors::{load_safetensors, save_safetensors};
use crate::tensor::Tensor;

/// Bumped on incompatible layout changes; loaders reject newer versions.
pub const FORMAT_VERSION: u32 = 1;

const MODEL_FILE: &str = "model.safetensors";
const OPTIM_FILE: &str = "optimizer.safetensors";
const META_FILE: &str = "checkpoint.json";
/// Prefix under which `OptimizerState::snapshot` arrays are stored.
pub const OPTIM_PREFIX: &str = "optim.";

#[derive(Clone)]
pub struct Checkpoint {
    pub version: u32,
    pub step: u64,
    /// Seed the training run was started from (`None` if not tracked).
    pub rng_seed: Option<u64>,
    /// Parameters and optimizer buffers, one entry per named array.
    pub tensors: Vec<(String, Tensor)>,
}

impl Checkpoint {
    pub fn new(step: u64) -> Checkpoint {
        Checkpoint {
            version: FORMAT_VERSION,
            step,
            rng_seed: None,
            tensors: Vec::new(),
        }
    }

    pub fn with_tensor(mut self, name: impl Into<String>, t: Tensor) -> Checkpoint {
        self.tensors.push((name.into(), t));
        self
    }

    pub fn with_rng_seed(mut self, seed: u64) -> Checkpoint {
        self.rng_seed = Some(seed);
        self
    }

    /// Snapshot a module's parameters under `model.` names.
    pub fn from_module(step: u64, module: &dyn Module) -> Checkpoint {
        let mut cp = Checkpoint::new(step);
        for (name, p) in module.named_parameters() {
            cp.tensors.push((format!("model.{name}"), p.tensor()));
        }
        cp
    }

    /// Restore parameters into a module, strictly: every module parameter must
    /// be present with matching shape/dtype and nothing extra may remain.
    pub fn load_into_module(&self, module: &dyn Module) -> Result<()> {
        let mut remaining = self.tensors.clone();
        for (name, param) in module.named_parameters() {
            let full = format!("model.{name}");
            let pos = remaining
                .iter()
                .position(|(n, _)| n == &full)
                .ok_or_else(|| Error::Format {
                    op: "checkpoint_load",
                    msg: format!("checkpoint is missing parameter {full:?}"),
                })?;
            let (_, t) = remaining.swap_remove(pos);
            let want = param.tensor();
            if t.shape() != want.shape() || t.dtype() != want.dtype() {
                return Err(Error::Format {
                    op: "checkpoint_load",
                    msg: format!(
                        "parameter {full:?}: expected {} {:?}, checkpoint has {} {:?}",
                        want.dtype(),
                        want.shape(),
                        t.dtype(),
                        t.shape()
                    ),
                });
            }
            param.set(t);
        }
        match remaining.first() {
            None => Ok(()),
            Some((n, _)) => Err(Error::Format {
                op: "checkpoint_load",
                msg: format!("checkpoint has unexpected tensor {n:?}"),
            }),
        }
    }

    /// Snapshot a module's parameters plus an optimizer's state buffers.
    /// Optimizer arrays are stored under `optim.` names in a separate
    /// optimizer.safetensors so model-only consumers are unaffected.
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
        const OP: &str = "checkpoint_save";
        let dir = dir.as_ref();
        fs::create_dir_all(dir).map_err(|e| Error::Io {
            op: OP,
            msg: format!("{}: {e}", dir.display()),
        })?;
        let meta = format!(
            "{{\n  \"version\": {},\n  \"step\": {},\n  \"rng_seed\": {}\n}}\n",
            self.version,
            self.step,
            match self.rng_seed {
                Some(s) => s.to_string(),
                None => "null".to_string(),
            }
        );
        // Temp file + rename in the target directory: a crash mid-write leaves
        // the previous checkpoint intact rather than a truncated file.
        let model_tmp = temp_path(dir, MODEL_FILE);
        let meta_tmp = temp_path(dir, META_FILE);
        save_safetensors(
            &model_tmp,
            &self
                .tensors
                .iter()
                .map(|(n, t)| (n.as_str(), t))
                .collect::<Vec<_>>(),
        )?;
        let write_meta = || -> Result<()> {
            fs::write(&meta_tmp, meta.as_bytes()).map_err(|e| Error::Io {
                op: OP,
                msg: format!("{}: {e}", meta_tmp.display()),
            })
        };
        write_meta()?;
        let rename = |from: &PathBuf, to: &str| -> Result<()> {
            let dest = dir.join(to);
            fs::rename(from, &dest).map_err(|e| Error::Io {
                op: OP,
                msg: format!("{} -> {}: {e}", from.display(), dest.display()),
            })
        };
        rename(&meta_tmp, META_FILE)?;
        rename(&model_tmp, MODEL_FILE)?;
        Ok(())
    }

    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Checkpoint> {
        const OP: &str = "checkpoint_load";
        let dir = dir.as_ref();
        let meta_bytes = fs::read(dir.join(META_FILE)).map_err(|e| Error::Io {
            op: OP,
            msg: format!("{}: {e}", dir.join(META_FILE).display()),
        })?;
        let meta = Meta::parse(&meta_bytes)?;
        if meta.version > FORMAT_VERSION {
            return Err(Error::Unsupported {
                op: OP,
                msg: format!(
                    "checkpoint version {} is newer than supported version {FORMAT_VERSION}",
                    meta.version
                ),
            });
        }
        let tensors = load_safetensors(dir.join(MODEL_FILE))?;
        Ok(Checkpoint {
            version: meta.version,
            step: meta.step,
            rng_seed: meta.rng_seed,
            tensors,
        })
    }
}

struct Meta {
    version: u32,
    step: u64,
    rng_seed: Option<u64>,
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
        let rng_seed = match field("rng_seed")? {
            None | Some("null") => None,
            Some(v) => Some(
                v.parse::<u64>()
                    .map_err(|_| ferr(format!("bad rng_seed {v:?}")))?,
            ),
        };
        Ok(Meta {
            version: num("version")? as u32,
            step: num("step")?,
            rng_seed,
        })
    }
}

fn temp_path(dir: &Path, file: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    dir.join(format!(".{file}.tmp{}", std::process::id() as u128 ^ nanos))
}
