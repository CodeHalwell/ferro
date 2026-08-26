//! Hermetic tokenizer tests over hand-built vocabularies, plus an env-gated
//! check against a real GPT-2 vocab (set FERRO_GPT2_DIR to a directory
//! holding vocab.json and merges.txt to enable it).

use ferro_tokenizer::{pretokenize, Bpe};

#[test]
fn pretokenizer_matches_gpt2_splits() {
    let cases: &[(&str, &[&str])] = &[
        ("Hello world", &["Hello", " world"]),
        // Two spaces: the run-minus-last rule leaves one to attach.
        ("Hello  world", &["Hello", " ", " world"]),
        ("I'm sure they've said it's fine", &[
            "I", "'m", " sure", " they", "'ve", " said", " it", "'s", " fine",
        ]),
        ("we'll don't", &["we", "'ll", " don", "'t"]),
        ("abc123 def", &["abc", "123", " def"]),
        ("x!!! (y)", &["x", "!!!", " (", "y", ")"]),
        ("hi  ", &["hi", "  "]),
        ("\n\nHello", &["\n", "\n", "Hello"]),
        // A tab cannot attach to a word (the regex's optional prefix is a
        // literal space), so it stands alone.
        ("a\tb", &["a", "\t", "b"]),
        ("price: $5.99", &["price", ":", " $", "5", ".", "99"]),
        // Unicode letters and an emoji (symbol class).
        ("héllo wörld \u{1F980}", &["héllo", " wörld", " \u{1F980}"]),
        ("", &[]),
        ("   ", &["   "]),
        // Double apostrophe is not a contraction; it is an "other" run.
        ("isn''t", &["isn", "''", "t"]),
    ];
    for (text, want) in cases {
        assert_eq!(&pretokenize(text), want, "text {text:?}");
    }
}

/// A tiny byte-level vocab: all printable single chars used below, plus the
/// merged pieces. Ids are arbitrary but distinct.
fn tiny() -> Bpe {
    let vocab = r#"{
        "l": 0, "o": 1, "w": 2, "e": 3, "r": 4, "h": 5,
        "lo": 6, "low": 7, "er": 8, "lower": 9,
        "Ġ": 10, "Ġlow": 11, "!": 12
    }"#;
    // Merge ranks decide the path: l+o, then lo+w, then e+r, then low+er,
    // then space+low.
    let merges = "#version: 0.2\nl o\nlo w\ne r\nlow er\n\u{0120} low\n";
    Bpe::from_strs(vocab, merges).unwrap()
}

#[test]
fn merge_order_follows_ranks() {
    let t = tiny();
    assert_eq!(t.encode("lower").unwrap(), vec![9]);
    assert_eq!(t.encode("low").unwrap(), vec![7]);
    // "lowerlower": pieces merge fully then concatenate.
    assert_eq!(t.encode("lowerlower").unwrap(), vec![9, 9]);
    // " low" uses the byte-mapped space (Ġ = Ġ) merge.
    assert_eq!(t.encode("low low").unwrap(), vec![7, 11]);
    // Unmergeable leftovers fall back to single symbols.
    assert_eq!(t.encode("he").unwrap(), vec![5, 3]);
    assert_eq!(t.encode("low!").unwrap(), vec![7, 12]);
}

#[test]
fn decode_inverts_encode() {
    let t = tiny();
    for text in ["lower", "low low", "he", "low!"] {
        let ids = t.encode(text).unwrap();
        assert_eq!(t.decode(&ids).unwrap(), *text, "{text}");
    }
    assert!(t.decode(&[999]).is_err());
}

#[test]
fn unknown_symbols_error_loudly() {
    let t = tiny();
    // 'z' is not in the tiny vocab and no merge produces it.
    assert!(t.encode("z").is_err());
}

#[test]
fn vocab_json_escapes_and_lookup() {
    // Ġ (Ġ), a quoted quote, a surrogate pair (🀄 = U+1F004), and a
    // special token.
    let vocab = r#"{"Ġ": 0, "\"": 1, "🀄": 2, "<|endoftext|>": 3, "A": 4}"#;
    let t = Bpe::from_strs(vocab, "").unwrap();
    assert_eq!(t.vocab_size(), 5);
    assert_eq!(t.token_id("<|endoftext|>"), Some(3));
    assert_eq!(t.token_id("\u{0120}"), Some(0));
    assert_eq!(t.token_id("\u{1F004}"), Some(2));
    // Malformed inputs error instead of panicking.
    assert!(Bpe::from_strs("{", "").is_err());
    assert!(Bpe::from_strs(r#"{"a": -1}"#, "").is_err());
    assert!(Bpe::from_strs(r#"{"a": 0, "b": 0}"#, "").is_err());
    assert!(Bpe::from_strs(r#"{"a": 0}"#, "a b c\n").is_err());
    assert!(Bpe::from_strs(r#"{"\uD800x": 0}"#, "").is_err());
}

#[test]
fn full_byte_vocab_round_trips_arbitrary_text() {
    // A vocab of exactly the 256 byte symbols (no merges): encoding is one
    // token per byte and decode must reproduce any string, emoji included.
    let mut entries = Vec::new();
    for b in 0u32..256 {
        // Reproduce the byte->unicode table: printable bytes are
        // themselves, the rest map to 256 + counter in table order.
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let ch = if printable {
            char::from_u32(b).unwrap()
        } else {
            let before = (0..b)
                .filter(|&x| {
                    !((33..=126).contains(&x) || (161..=172).contains(&x) || (174..=255).contains(&x))
                })
                .count() as u32;
            char::from_u32(256 + before).unwrap()
        };
        let escaped = match ch {
            '"' => "\\\"".to_string(),
            '\\' => "\\\\".to_string(),
            c => c.to_string(),
        };
        entries.push(format!("\"{escaped}\": {b}"));
    }
    let vocab = format!("{{{}}}", entries.join(","));
    let t = Bpe::from_strs(&vocab, "").unwrap();

    let text = "Grüße, 世界! \u{1F980}\n\ttabs & newlines";
    let ids = t.encode(text).unwrap();
    assert_eq!(ids.len(), text.len(), "one token per byte");
    assert_eq!(t.decode(&ids).unwrap(), text);
}

#[test]
fn real_gpt2_vocab_if_available() {
    // Gated the way GPU tests are: runs only when the real files are
    // present. Validates against token ids produced by the reference
    // implementation.
    let Ok(dir) = std::env::var("FERRO_GPT2_DIR") else {
        return;
    };
    let t = Bpe::from_files(
        format!("{dir}/vocab.json"),
        format!("{dir}/merges.txt"),
    )
    .unwrap();
    assert_eq!(t.vocab_size(), 50257);
    assert_eq!(t.token_id("<|endoftext|>"), Some(50256));

    let cases: &[(&str, &[u32])] = &[
        ("Hello world", &[15496, 995]),
        ("Hello, world!", &[15496, 11, 995, 0]),
        ("The quick brown fox jumps over the lazy dog.", &[
            464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13,
        ]),
        ("I'm sure it's fine", &[40, 1101, 1654, 340, 338, 3734]),
    ];
    for (text, want) in cases {
        assert_eq!(&t.encode(text).unwrap(), want, "{text:?}");
        assert_eq!(t.decode(want).unwrap(), *text);
    }

    // Round-trip arbitrary unicode through the full vocab.
    let text = "Grüße 世界 \u{1F980} — mixed\ttext\n";
    assert_eq!(t.decode(&t.encode(text).unwrap()).unwrap(), text);
}
