//! Byte-level BPE tokenizer in the GPT-2 vocab.json/merges.txt format - the
//! tokenizer milestone M3 needs to feed real checkpoints. Pure std, no
//! dependencies, no dependency on ferro-core.
//!
//! The pipeline is exactly the reference implementation's:
//! 1. Pre-tokenize the text with GPT-2's split rule (hand-rolled here; the
//!    original is the regex `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|
//!    ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, whose observable behavior is:
//!    contractions split off; a single LITERAL SPACE attaches to a following
//!    letter/number/symbol run; a whitespace run followed by a token yields
//!    the run minus its last character; trailing whitespace stays whole).
//! 2. Map each piece's UTF-8 bytes through the byte<->unicode table (every
//!    byte becomes a printable char, so the vocab holds no raw control
//!    bytes and any byte sequence is representable).
//! 3. Run ranked BPE merges over the mapped piece and look the resulting
//!    tokens up in the vocab.
//!
//! Decoding inverts: ids -> vocab strings -> bytes -> UTF-8 (lossy on
//! invalid boundaries, matching the reference decoder's errors="replace").

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug)]
pub enum TokenizerError {
    Io(String),
    Format(String),
    /// A piece produced a BPE symbol absent from the vocab (a complete
    /// byte-level vocab contains all 256 single-byte tokens, so this only
    /// happens with truncated/custom vocabularies).
    UnknownToken(String),
    UnknownId(u32),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Io(m) => write!(f, "tokenizer io: {m}"),
            TokenizerError::Format(m) => write!(f, "tokenizer format: {m}"),
            TokenizerError::UnknownToken(t) => write!(f, "token {t:?} is not in the vocab"),
            TokenizerError::UnknownId(id) => write!(f, "id {id} is not in the vocab"),
        }
    }
}

impl std::error::Error for TokenizerError {}

type Result<T> = std::result::Result<T, TokenizerError>;

/// GPT-2's byte<->unicode table: printable latin-1 bytes map to themselves,
/// every other byte to U+0100 + counter, giving a 256-way bijection into
/// printable chars.
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    table
}

/// GPT-2's pre-tokenization split (see module docs). Public for tests and
/// for callers that want the raw pieces.
pub fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    let emit = |out: &mut Vec<String>, chars: &[char], a: usize, b: usize| {
        out.push(chars[a..b].iter().collect());
    };
    while i < n {
        // Contractions: '(?:[sdmt]|ll|ve|re), single-letter suffixes first
        // like the regex alternation.
        if chars[i] == '\'' && i + 1 < n {
            let c1 = chars[i + 1];
            if matches!(c1, 's' | 'd' | 'm' | 't') {
                emit(&mut out, &chars, i, i + 2);
                i += 2;
                continue;
            }
            if i + 2 < n {
                let two = [c1, chars[i + 2]];
                if two == ['l', 'l'] || two == ['v', 'e'] || two == ['r', 'e'] {
                    emit(&mut out, &chars, i, i + 3);
                    i += 3;
                    continue;
                }
            }
        }
        // ` ?` + letter/number/other runs. Only a literal space attaches.
        let (run_start, first) = if chars[i] == ' ' && i + 1 < n {
            (i + 1, chars[i + 1])
        } else {
            (i, chars[i])
        };
        let other = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
        let class: Option<fn(char) -> bool> = if first.is_alphabetic() {
            Some(|c| c.is_alphabetic())
        } else if first.is_numeric() {
            Some(|c| c.is_numeric())
        } else if other(first) {
            Some(other)
        } else {
            None
        };
        if let Some(in_class) = class {
            let mut j = run_start;
            while j < n && in_class(chars[j]) {
                // A contraction interrupts an "other" run only via the next
                // alternation round; the regex's greedy class run does not
                // look ahead, and neither do we.
                j += 1;
            }
            emit(&mut out, &chars, i, j);
            i = j;
            continue;
        }
        // Whitespace: `\s+(?!\S)` then `\s+`. A run followed by a token
        // yields the run minus its final character (the regex backtracks
        // exactly one step); at end of text the whole run matches.
        debug_assert!(chars[i].is_whitespace());
        let mut j = i;
        while j < n && chars[j].is_whitespace() {
            j += 1;
        }
        let end = if j < n && j - i > 1 { j - 1 } else { j };
        emit(&mut out, &chars, i, end);
        i = end;
    }
    out
}

/// Byte-level BPE tokenizer over a GPT-2-format vocab.
pub struct Bpe {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    merges: HashMap<(String, String), u32>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    /// Per-piece BPE results; text repeats pieces constantly and the merge
    /// loop is the expensive part.
    cache: Mutex<HashMap<String, Vec<u32>>>,
}

impl Bpe {
    /// Build from the contents of vocab.json and merges.txt.
    pub fn from_strs(vocab_json: &str, merges_txt: &str) -> Result<Bpe> {
        let encoder = parse_vocab_json(vocab_json)?;
        let mut decoder = HashMap::with_capacity(encoder.len());
        for (tok, &id) in &encoder {
            if decoder.insert(id, tok.clone()).is_some() {
                return Err(TokenizerError::Format(format!("duplicate id {id} in vocab")));
            }
        }
        let mut merges = HashMap::new();
        for (rank, line) in merges_txt
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with("#version"))
            .enumerate()
        {
            let mut it = line.split_whitespace();
            let (Some(a), Some(b), None) = (it.next(), it.next(), it.next()) else {
                return Err(TokenizerError::Format(format!(
                    "merges line {rank} is not exactly two symbols: {line:?}"
                )));
            };
            merges.insert((a.to_string(), b.to_string()), rank as u32);
        }
        let byte_enc = byte_to_unicode();
        let byte_dec = byte_enc
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        Ok(Bpe {
            encoder,
            decoder,
            merges,
            byte_enc,
            byte_dec,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn from_files<P: AsRef<Path>>(vocab: P, merges: P) -> Result<Bpe> {
        let read = |p: &Path| {
            std::fs::read_to_string(p)
                .map_err(|e| TokenizerError::Io(format!("{}: {e}", p.display())))
        };
        Bpe::from_strs(&read(vocab.as_ref())?, &read(merges.as_ref())?)
    }

    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Direct vocab lookup (e.g. `token_id("<|endoftext|>")` for the eot id).
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.encoder.get(token).copied()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for piece in pretokenize(text) {
            if let Some(ids) = self.cache.lock().unwrap().get(&piece) {
                out.extend_from_slice(ids);
                continue;
            }
            let mapped: String = piece.bytes().map(|b| self.byte_enc[b as usize]).collect();
            let mut ids = Vec::new();
            for sym in self.bpe(&mapped) {
                match self.encoder.get(&sym) {
                    Some(&id) => ids.push(id),
                    None => return Err(TokenizerError::UnknownToken(sym)),
                }
            }
            out.extend_from_slice(&ids);
            self.cache.lock().unwrap().insert(piece, ids);
        }
        Ok(out)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let tok = self
                .decoder
                .get(&id)
                .ok_or(TokenizerError::UnknownId(id))?;
            for c in tok.chars() {
                match self.byte_dec.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => {
                        return Err(TokenizerError::Format(format!(
                            "token {tok:?} contains {c:?}, outside the byte table"
                        )))
                    }
                }
            }
        }
        // Lossy: token boundaries can split UTF-8 sequences mid-character
        // (the reference decoder uses errors="replace" for the same reason).
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The ranked merge loop: repeatedly replace every occurrence of the
    /// lowest-ranked adjacent symbol pair until none remains.
    fn bpe(&self, mapped: &str) -> Vec<String> {
        let mut word: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();
        while word.len() > 1 {
            let mut best: Option<(u32, (String, String))> = None;
            for pair in word.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                if let Some(&rank) = self.merges.get(&key) {
                    if best.as_ref().is_none_or(|(r, _)| rank < *r) {
                        best = Some((rank, key));
                    }
                }
            }
            let Some((_, (a, b))) = best else { break };
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(std::mem::take(&mut word[i]));
                    i += 1;
                }
            }
            word = merged;
        }
        word
    }
}

/// Minimal parser for vocab.json: one flat JSON object mapping token strings
/// to unsigned integer ids. Handles the full string escape set including
/// \uXXXX with surrogate pairs; anything structurally different errors.
fn parse_vocab_json(src: &str) -> Result<HashMap<String, u32>> {
    let b: Vec<char> = src.chars().collect();
    let mut i = 0;
    let err = |m: &str, at: usize| TokenizerError::Format(format!("vocab.json at {at}: {m}"));
    let skip_ws = |i: &mut usize| {
        while *i < b.len() && b[*i].is_whitespace() {
            *i += 1;
        }
    };
    skip_ws(&mut i);
    if i >= b.len() || b[i] != '{' {
        return Err(err("expected '{'", i));
    }
    i += 1;
    let mut out = HashMap::new();
    skip_ws(&mut i);
    if i < b.len() && b[i] == '}' {
        return Ok(out);
    }
    loop {
        skip_ws(&mut i);
        let key = parse_string(&b, &mut i)?;
        skip_ws(&mut i);
        if i >= b.len() || b[i] != ':' {
            return Err(err("expected ':'", i));
        }
        i += 1;
        skip_ws(&mut i);
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if start == i {
            return Err(err("expected an unsigned integer id", i));
        }
        let id: u32 = b[start..i]
            .iter()
            .collect::<String>()
            .parse()
            .map_err(|_| err("id out of range", start))?;
        if out.insert(key.clone(), id).is_some() {
            return Err(TokenizerError::Format(format!("duplicate token {key:?}")));
        }
        skip_ws(&mut i);
        match b.get(i) {
            Some(',') => i += 1,
            Some('}') => return Ok(out),
            _ => return Err(err("expected ',' or '}'", i)),
        }
    }
}

fn parse_string(b: &[char], i: &mut usize) -> Result<String> {
    let err = |m: &str, at: usize| TokenizerError::Format(format!("vocab.json at {at}: {m}"));
    if *i >= b.len() || b[*i] != '"' {
        return Err(err("expected '\"'", *i));
    }
    *i += 1;
    let mut s = String::new();
    loop {
        let c = *b.get(*i).ok_or_else(|| err("unterminated string", *i))?;
        *i += 1;
        match c {
            '"' => return Ok(s),
            '\\' => {
                let e = *b.get(*i).ok_or_else(|| err("unterminated escape", *i))?;
                *i += 1;
                match e {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'b' => s.push('\u{0008}'),
                    'f' => s.push('\u{000C}'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    'u' => {
                        let hi = parse_hex4(b, i)?;
                        let cp = if (0xD800..=0xDBFF).contains(&hi) {
                            // Surrogate pair: require the low half.
                            if b.get(*i) != Some(&'\\') || b.get(*i + 1) != Some(&'u') {
                                return Err(err("unpaired high surrogate", *i));
                            }
                            *i += 2;
                            let lo = parse_hex4(b, i)?;
                            if !(0xDC00..=0xDFFF).contains(&lo) {
                                return Err(err("invalid low surrogate", *i));
                            }
                            0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                        } else {
                            hi
                        };
                        s.push(char::from_u32(cp).ok_or_else(|| err("invalid codepoint", *i))?);
                    }
                    _ => return Err(err("unknown escape", *i)),
                }
            }
            _ => s.push(c),
        }
    }
}

fn parse_hex4(b: &[char], i: &mut usize) -> Result<u32> {
    let err = |at: usize| TokenizerError::Format(format!("vocab.json at {at}: bad \\u escape"));
    let mut v = 0u32;
    for _ in 0..4 {
        let c = *b.get(*i).ok_or_else(|| err(*i))?;
        *i += 1;
        v = v * 16 + c.to_digit(16).ok_or_else(|| err(*i))?;
    }
    Ok(v)
}
