//! safetensors read/write (https://github.com/huggingface/safetensors): the
//! model-import path for milestone M3 and the state_dict save format. Layout:
//! an 8-byte little-endian header length, a JSON header mapping tensor names
//! to {dtype, shape, data_offsets} (offsets relative to the data section),
//! then the raw little-endian tensor bytes. The header parser is a minimal
//! from-scratch JSON reader because ferro-core takes no dependencies; it
//! accepts the full format (strings with escapes, nested arrays/objects,
//! unsigned integers) and ignores `__metadata__`. F32/F64/I64 tensors are
//! supported; other dtypes (f16/bf16, ...) error until storage exists.

use std::fs;
use std::path::Path;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

pub fn save_safetensors<P: AsRef<Path>>(path: P, tensors: &[(&str, &Tensor)]) -> Result<()> {
    let bytes = to_safetensors_bytes(tensors)?;
    fs::write(path.as_ref(), bytes).map_err(|e| Error::Io {
        op: "save_safetensors",
        msg: format!("{}: {e}", path.as_ref().display()),
    })
}

pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Tensor)>> {
    let bytes = fs::read(path.as_ref()).map_err(|e| Error::Io {
        op: "load_safetensors",
        msg: format!("{}: {e}", path.as_ref().display()),
    })?;
    from_safetensors_bytes(&bytes)
}

pub fn to_safetensors_bytes(tensors: &[(&str, &Tensor)]) -> Result<Vec<u8>> {
    let mut header = String::from("{");
    let mut data = Vec::new();
    for (i, (name, t)) in tensors.iter().enumerate() {
        if tensors[..i].iter().any(|(n, _)| n == name) {
            return Err(Error::Format { op: "save_safetensors", msg: format!("duplicate tensor name {name:?}") });
        }
        let start = data.len();
        match t.dtype() {
            DType::F32 => t.to_vec().iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes())),
            DType::F64 => t.to_vec_f64().iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes())),
            DType::I64 => t.to_vec_i64().iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes())),
        }
        if i > 0 {
            header.push(',');
        }
        let shape: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
            escape(name),
            dtype_tag(t.dtype()),
            shape.join(","),
            data.len()
        ));
    }
    header.push('}');
    // Pad to an 8-byte boundary with trailing spaces (valid JSON whitespace)
    // so the data section is aligned, matching the reference implementation.
    while (8 + header.len()) % 8 != 0 {
        header.push(' ');
    }

    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    Ok(out)
}

pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Vec<(String, Tensor)>> {
    const OP: &str = "load_safetensors";
    let ferr = |msg: String| Error::Format { op: OP, msg };
    if bytes.len() < 8 {
        return Err(ferr("shorter than the 8-byte header length".into()));
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + hlen {
        return Err(ferr(format!("header length {hlen} exceeds file size {}", bytes.len())));
    }
    let data = &bytes[8 + hlen..];

    let header = Json::parse(&bytes[8..8 + hlen]).map_err(ferr)?;
    let Json::Obj(entries) = header else {
        return Err(ferr("header is not a JSON object".into()));
    };

    let mut out = Vec::new();
    for (name, spec) in entries {
        if name == "__metadata__" {
            continue;
        }
        let Json::Obj(fields) = spec else {
            return Err(ferr(format!("entry {name:?} is not an object")));
        };
        let field = |key: &str| {
            fields.iter().find(|(k, _)| k == key).map(|(_, v)| v).ok_or_else(|| ferr(format!("entry {name:?} missing {key:?}")))
        };
        let Json::Str(dtype) = field("dtype")? else {
            return Err(ferr(format!("entry {name:?}: dtype is not a string")));
        };
        let shape = match field("shape")? {
            Json::Arr(dims) => dims
                .iter()
                .map(|d| match d {
                    Json::Num(n) => Ok(*n as usize),
                    _ => Err(ferr(format!("entry {name:?}: non-integer dim"))),
                })
                .collect::<Result<Vec<usize>>>()?,
            _ => return Err(ferr(format!("entry {name:?}: shape is not an array"))),
        };
        let (start, end) = match field("data_offsets")? {
            Json::Arr(o) => match o.as_slice() {
                [Json::Num(s), Json::Num(e)] => (*s as usize, *e as usize),
                _ => return Err(ferr(format!("entry {name:?}: data_offsets is not [start, end]"))),
            },
            _ => return Err(ferr(format!("entry {name:?}: data_offsets is not an array"))),
        };

        let dt = match dtype.as_str() {
            "F32" => DType::F32,
            "F64" => DType::F64,
            "I64" => DType::I64,
            other => {
                return Err(Error::Unsupported { op: OP, msg: format!("entry {name:?}: dtype {other} (storage not implemented)") })
            }
        };
        let numel: usize = shape.iter().product();
        let width = if dt == DType::F32 { 4 } else { 8 };
        if end < start || end - start != numel * width || end > data.len() {
            return Err(ferr(format!(
                "entry {name:?}: offsets [{start}, {end}] do not fit shape {shape:?} ({} dtype, {} data bytes)",
                dtype,
                data.len()
            )));
        }

        let raw = &data[start..end];
        let t = match dt {
            DType::F32 => {
                Tensor::from_vec(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(), &shape)?
            }
            DType::F64 => Tensor::from_vec_f64(
                raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
                &shape,
            )?,
            DType::I64 => Tensor::from_vec_i64(
                raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
                &shape,
            )?,
        };
        out.push((name, t));
    }
    Ok(out)
}

fn dtype_tag(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::I64 => "I64",
    }
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// The subset of JSON a safetensors header can contain. Numbers are unsigned
/// integers (shapes and offsets); anything else in a numeric position is a
/// format error.
enum Json {
    Obj(Vec<(String, Json)>),
    Arr(Vec<Json>),
    Str(String),
    Num(u64),
}

impl Json {
    fn parse(bytes: &[u8]) -> std::result::Result<Json, String> {
        let mut p = JsonParser { b: bytes, pos: 0 };
        let v = p.value()?;
        p.ws();
        if p.pos != p.b.len() {
            return Err(format!("trailing bytes after JSON value at offset {}", p.pos));
        }
        Ok(v)
    }
}

struct JsonParser<'a> {
    b: &'a [u8],
    pos: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while matches!(self.b.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, c: u8) -> std::result::Result<(), String> {
        self.ws();
        if self.b.get(self.pos) == Some(&c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at offset {}", c as char, self.pos))
        }
    }

    fn value(&mut self) -> std::result::Result<Json, String> {
        self.ws();
        match self.b.get(self.pos) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'0'..=b'9') => self.number(),
            Some(&c) => Err(format!("unexpected byte {:?} at offset {}", c as char, self.pos)),
            None => Err("unexpected end of header".into()),
        }
    }

    fn object(&mut self) -> std::result::Result<Json, String> {
        self.eat(b'{')?;
        let mut entries = Vec::new();
        self.ws();
        if self.b.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Json::Obj(entries));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.eat(b':')?;
            entries.push((key, self.value()?));
            self.ws();
            match self.b.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(entries));
                }
                _ => return Err(format!("expected ',' or '}}' at offset {}", self.pos)),
            }
        }
    }

    fn array(&mut self) -> std::result::Result<Json, String> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.b.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.b.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at offset {}", self.pos)),
            }
        }
    }

    fn string(&mut self) -> std::result::Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            match self.b.get(self.pos) {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    match self.b.get(self.pos) {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'b') => out.push('\u{8}'),
                        Some(b'f') => out.push('\u{c}'),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'u') => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                // Surrogate pair: a second \uXXXX must follow.
                                if self.b.get(self.pos + 1) != Some(&b'\\') || self.b.get(self.pos + 2) != Some(&b'u') {
                                    return Err("unpaired surrogate".into());
                                }
                                self.pos += 2;
                                let lo = self.hex4()?;
                                0x10000 + ((hi - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or("invalid unicode escape")?);
                        }
                        _ => return Err(format!("bad escape at offset {}", self.pos)),
                    }
                    self.pos += 1;
                }
                Some(_) => {
                    // Multi-byte UTF-8 passes through; validate at the end of
                    // the run to keep per-byte handling simple.
                    let start = self.pos;
                    while !matches!(self.b.get(self.pos), None | Some(b'"' | b'\\')) {
                        self.pos += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.b[start..self.pos]).map_err(|_| "invalid utf-8 in string")?);
                }
            }
        }
    }

    fn hex4(&mut self) -> std::result::Result<u32, String> {
        let hex = self.b.get(self.pos + 1..self.pos + 5).ok_or("truncated unicode escape")?;
        let s = std::str::from_utf8(hex).map_err(|_| "bad unicode escape")?;
        let v = u32::from_str_radix(s, 16).map_err(|_| "bad unicode escape")?;
        self.pos += 4;
        Ok(v)
    }

    fn number(&mut self) -> std::result::Result<Json, String> {
        let start = self.pos;
        while matches!(self.b.get(self.pos), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if matches!(self.b.get(self.pos), Some(b'.' | b'e' | b'E' | b'-' | b'+')) {
            return Err(format!("non-integer number at offset {start}"));
        }
        std::str::from_utf8(&self.b[start..self.pos])
            .unwrap()
            .parse::<u64>()
            .map(Json::Num)
            .map_err(|e| format!("bad number at offset {start}: {e}"))
    }
}
