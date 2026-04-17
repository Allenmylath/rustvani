//! Abbreviation-aware sentence splitter for TTS text aggregation.
//!
//! Designed to be used by any TTS service handler that accumulates
//! streaming LLM tokens and sends complete sentence chunks to a TTS backend.
//!
//! # Boundary characters
//!
//! | Character    | Rule                                                        |
//! |--------------|-------------------------------------------------------------|
//! | `?` `!`      | Always a boundary when followed by whitespace / end         |
//! | `।` `॥`      | Always a boundary (Devanagari danda — no follow-up needed)  |
//! | `\n`         | Always a boundary (paragraph / line break)                  |
//! | `…` `...`    | NEVER a boundary — marks hesitation / trailing-off          |
//! | `.`          | Boundary only after ellipsis + abbreviation checks          |
//!
//! # Abbreviation rules for `.`
//!
//! A period is NOT a sentence boundary if ANY of:
//!   1. It is part of `...` (ASCII ellipsis).
//!   2. Not followed by whitespace or end-of-string.
//!   3. Next non-space character is lowercase → continuation phrase.
//!   4. Word before is in the known abbreviation list.
//!   5. Word before is a single ASCII letter → initial (e.g. "A. P. J.").
//!   6. Word before is all-uppercase ASCII → acronym (e.g. "IPC.", "POCSO.").

// ---------------------------------------------------------------------------
// Known abbreviations
// ---------------------------------------------------------------------------

/// Words followed by `.` that do NOT end a sentence.
/// All entries lowercase — comparison is case-insensitive.
pub const ABBREVIATIONS: &[&str] = &[
    // Titles
    "mr", "mrs", "ms", "dr", "prof", "rev", "sr", "jr",
    // Indian honorifics
    "sh", "smt", "km", "adv",
    // Military / government ranks
    "lt", "col", "gen", "maj", "capt", "sgt", "cpl", "pvt",
    "dept", "govt", "min",
    // Academic / publishing
    "vol", "no", "fig", "ed", "pp", "ch",
    // Common shorthand
    "vs", "etc", "approx", "est", "cont", "misc", "ref",
    // Geographic
    "st", "ave", "blvd", "rd",
    // Legal (IPC, CrPC, POCSO context)
    "sec", "art", "cl", "sub",
];

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract the word immediately before byte position `pos` in `text`.
fn word_before(text: &str, pos: usize) -> &str {
    let before = &text[..pos];
    let end = before.trim_end_matches(|c: char| !c.is_alphanumeric()).len();
    let slice = &before[..end];
    let start = slice
        .rfind(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|i| i + 1)
        .unwrap_or(0);
    &slice[start..]
}

/// Return `true` if the `.` at byte index `pos` is an abbreviation dot.
///
/// Called ONLY for `.` that is not part of an ellipsis and IS followed by
/// whitespace or end-of-string.
fn is_abbreviation_dot(text: &str, pos: usize) -> bool {
    let bytes = text.as_bytes();

    // Rule 3: next non-space char is lowercase → continuation
    let mut j = pos + 1;
    while j < bytes.len() && bytes[j].is_ascii_whitespace() {
        j += 1;
    }
    if j < bytes.len() && bytes[j].is_ascii_lowercase() {
        return true;
    }

    let w = word_before(text, pos);
    if w.is_empty() {
        return false;
    }

    // Rule 4: known abbreviation list
    if ABBREVIATIONS.contains(&w.to_lowercase().as_str()) {
        return true;
    }

    // Rule 5: single letter initial ("A. P. J.")
    if w.len() == 1 && w.chars().next().map_or(false, |c| c.is_ascii_alphabetic()) {
        return true;
    }

    // Rule 6: all-uppercase acronym ("IPC.", "POCSO.")
    if w.len() > 1 && w.chars().all(|c| c.is_ascii_uppercase()) {
        return true;
    }

    false
}

/// Return `true` if the ASCII `.` at byte `pos` is part of a `...` sequence.
fn is_ascii_ellipsis(bytes: &[u8], pos: usize) -> bool {
    let prev = pos.checked_sub(1).map(|i| bytes[i]);
    let next = bytes.get(pos + 1).copied();
    prev == Some(b'.') || next == Some(b'.')
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find the **last byte index** of the terminal punctuation of the first
/// complete sentence in `text`.
///
/// Returns `Some(last_byte)` where `text[..=last_byte]` contains the full
/// boundary character (critical for multi-byte Unicode like `।`).
/// Returns `None` if no complete sentence has ended yet.
///
/// # Examples
///
/// ```
/// use rustvani::utils::sentence_splitter::find_sentence_end;
///
/// assert_eq!(find_sentence_end("Hello world. Next"),  Some(11)); // period
/// assert_eq!(find_sentence_end("How are you?"),        Some(11)); // question
/// assert_eq!(find_sentence_end("Mr. Smith arrived"),   None);    // abbreviation
/// assert_eq!(find_sentence_end("IPC. Section 302"),   None);    // acronym
/// assert_eq!(find_sentence_end("wait... ok"),          None);    // ellipsis
/// assert_eq!(find_sentence_end("I think… maybe"),      None);    // unicode ellipsis
/// ```
pub fn find_sentence_end(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let ch = match text[i..].chars().next() {
            Some(c) => c,
            None    => break,
        };
        let ch_len  = ch.len_utf8();
        let last    = i + ch_len - 1; // inclusive last byte of this char

        match ch {
            // ---- Unambiguous ASCII boundaries -------------------------
            '?' | '!' => {
                let next = text[i + 1..].chars().next();
                if next.is_none() || next.map_or(false, |n| n.is_whitespace()) {
                    return Some(last);
                }
            }

            // ---- Devanagari danda / double danda ----------------------
            // Docs §5: "If a sentence ends in Hindi or a regional language,
            // use ।"  No trailing whitespace requirement.
            '।' | '॥' => {
                return Some(last);
            }

            // ---- Line break — paragraph boundary ----------------------
            // Docs §1: "Use line breaks between paragraphs for natural
            // breathing pauses"
            '\n' => {
                return Some(last);
            }

            // ---- Unicode ellipsis — NEVER a boundary ------------------
            // Docs §1: "Use … to create a hesitation or trailing-off
            // effect — it signals the speaker is thinking or pausing
            // mid-thought. Use sparingly."
            '…' => {
                i += ch_len;
                continue;
            }

            // ---- ASCII period — needs ellipsis + abbreviation checks --
            '.' => {
                // Rule 1: part of ASCII ellipsis (... sequence)
                if is_ascii_ellipsis(bytes, i) {
                    i += 1;
                    continue;
                }

                // Rule 2: must be followed by whitespace or end-of-string
                let next = text[i + 1..].chars().next();
                let followed_by_ws =
                    next.is_none() || next.map_or(false, |n| n.is_whitespace());

                if followed_by_ws && !is_abbreviation_dot(text, i) {
                    return Some(last);
                }
            }

            _ => {}
        }

        i += ch_len;
    }

    None
}

/// Find the latest genuine sentence boundary ending at or before `max_len` bytes.
///
/// Returns the **exclusive** drain position (last_byte + 1) suitable for
/// `buffer.drain(..pos)`, or `None` if no boundary exists within the limit.
///
/// Passes full `text` to `is_abbreviation_dot` so words straddling the cutoff
/// are identified correctly.
pub fn find_sentence_boundary_before(text: &str, max_len: usize) -> Option<usize> {
    let search = &text[..max_len.min(text.len())];
    let bytes  = search.as_bytes();
    let mut last: Option<usize> = None;
    let mut i = 0;

    while i < search.len() {
        let ch = match search[i..].chars().next() {
            Some(c) => c,
            None    => break,
        };
        let ch_len     = ch.len_utf8();
        let excl_end   = i + ch_len; // exclusive end for drain

        match ch {
            '?' | '!' => {
                let next = search[i + 1..].chars().next();
                if next.is_none() || next.map_or(false, |n| n.is_whitespace()) {
                    last = Some(excl_end);
                }
            }

            '।' | '॥' => {
                last = Some(excl_end);
            }

            '\n' => {
                last = Some(excl_end);
            }

            '…' => {
                i += ch_len;
                continue;
            }

            '.' => {
                if is_ascii_ellipsis(bytes, i) {
                    i += 1;
                    continue;
                }

                let next = search[i + 1..].chars().next();
                let followed_by_ws =
                    next.is_none() || next.map_or(false, |n| n.is_whitespace());

                if followed_by_ws && !is_abbreviation_dot(text, i) {
                    last = Some(excl_end);
                }
            }

            _ => {}
        }

        i += ch_len;
    }

    last
}

/// Drain complete sentences from `buffer` into a `Vec<String>`.
///
/// Applied in a loop:
///   1. If `buffer.len() >= max_chunk_length`: force-split at the last sentence
///      boundary before the limit, or hard-split at `max_chunk_length` if none.
///   2. Otherwise: split at the first natural sentence boundary.
///
/// The unfinished tail remains in `buffer`.
pub fn extract_sentences(buffer: &mut String, max_chunk_length: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    loop {
        // --- Hard size limit ---
        if buffer.len() >= max_chunk_length {
            let split_at = find_sentence_boundary_before(buffer, max_chunk_length)
                .unwrap_or_else(|| max_chunk_length.min(buffer.len()));

            let chunk = buffer.drain(..split_at).collect::<String>();
            let chunk = chunk.trim().to_string();
            if !chunk.is_empty() {
                chunks.push(chunk);
            }
            continue;
        }

        // --- Natural boundary ---
        match find_sentence_end(buffer) {
            Some(last_byte) => {
                // drain inclusive of the full boundary character
                let chunk = buffer.drain(..=last_byte).collect::<String>();
                let chunk = chunk.trim().to_string();
                if !chunk.is_empty() {
                    chunks.push(chunk);
                }
                // strip any leading whitespace / blank lines left in buffer
                let trim_start = buffer.len() - buffer.trim_start().len();
                buffer.drain(..trim_start);
            }
            None => break,
        }
    }

    chunks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Basic ASCII ----

    #[test]
    fn test_basic_period() {
        assert_eq!(find_sentence_end("Hello world. How"), Some(11));
    }

    #[test]
    fn test_period_at_end() {
        assert_eq!(find_sentence_end("Hello world."), Some(11));
    }

    #[test]
    fn test_question_mark() {
        assert_eq!(find_sentence_end("How are you? Fine."), Some(11));
    }

    #[test]
    fn test_exclamation() {
        assert_eq!(find_sentence_end("Stop! Now."), Some(4));
    }

    #[test]
    fn test_no_boundary() {
        assert_eq!(find_sentence_end("Hello world"), None);
    }

    // ---- Devanagari danda ----

    #[test]
    fn test_danda_boundary() {
        let text = "नमस्ते। अगला वाक्य";
        let end = find_sentence_end(text).unwrap();
        assert_eq!(&text[..=end], "नमस्ते।");
    }

    #[test]
    fn test_double_danda() {
        let text = "श्लोक समाप्त॥ अगला";
        let end = find_sentence_end(text).unwrap();
        assert!(text[..=end].ends_with('॥'));
    }

    #[test]
    fn test_danda_no_trailing_space_needed() {
        // Danda at end of string — must be a boundary
        assert!(find_sentence_end("नमस्ते।").is_some());
    }

    // ---- Ellipsis — NOT a boundary ----

    #[test]
    fn test_unicode_ellipsis_not_boundary() {
        assert_eq!(find_sentence_end("मुझे लगता है… शायद"), None);
    }

    #[test]
    fn test_ascii_ellipsis_not_boundary() {
        assert_eq!(find_sentence_end("wait... ok"), None);
    }

    #[test]
    fn test_ascii_ellipsis_at_string_end() {
        assert_eq!(find_sentence_end("I mean..."), None);
    }

    #[test]
    fn test_sentence_after_ellipsis_splits_on_danda() {
        let text = "So basically… हम India की हर language को voice देते हैं। अगला।";
        let end = find_sentence_end(text).unwrap();
        assert!(text[..=end].ends_with('।'));
        // Must NOT have split at the ellipsis
        assert!(text[..=end].contains("basically"));
    }

    // ---- Line break ----

    #[test]
    fn test_newline_is_boundary() {
        let text = "First line\nSecond line";
        let end = find_sentence_end(text).unwrap();
        assert_eq!(&text[..=end], "First line\n");
    }

    #[test]
    fn test_danda_before_newline() {
        // Danda comes first — should split there
        let text = "नमस्ते।\nHello.";
        let end = find_sentence_end(text).unwrap();
        assert_eq!(&text[..=end], "नमस्ते।");
    }

    // ---- Decimal numbers ----

    #[test]
    fn test_decimal_not_boundary() {
        assert_eq!(find_sentence_end("pi is 3.14 approximately"), None);
    }

    #[test]
    fn test_decimal_then_sentence() {
        assert_eq!(find_sentence_end("pi is 3.14. Yes."), Some(10));
    }

    // ---- Abbreviations ----

    #[test]
    fn test_mr_not_boundary() {
        assert_eq!(find_sentence_end("Mr. Smith arrived"), None);
    }

    #[test]
    fn test_dr_not_boundary() {
        assert_eq!(find_sentence_end("Dr. Sharma said hello"), None);
    }

    #[test]
    fn test_sh_not_boundary() {
        assert_eq!(find_sentence_end("Sh. Rajan spoke"), None);
    }

    #[test]
    fn test_smt_not_boundary() {
        assert_eq!(find_sentence_end("Smt. Devi attended"), None);
    }

    #[test]
    fn test_adv_not_boundary() {
        assert_eq!(find_sentence_end("Adv. Kumar argued"), None);
    }

    #[test]
    fn test_initials_not_boundary() {
        assert_eq!(find_sentence_end("A. P. J. Abdul Kalam"), None);
    }

    #[test]
    fn test_ipc_not_boundary() {
        assert_eq!(find_sentence_end("under IPC. Section 302"), None);
    }

    #[test]
    fn test_pocso_not_boundary() {
        assert_eq!(find_sentence_end("POCSO. Act cases"), None);
    }

    #[test]
    fn test_etc_not_boundary() {
        assert_eq!(find_sentence_end("fruits etc. are good"), None);
    }

    #[test]
    fn test_sentence_after_title() {
        let text = "Dr. Smith arrived. He left.";
        let end = find_sentence_end(text).unwrap();
        assert_eq!(&text[..=end], "Dr. Smith arrived.");
    }

    // ---- extract_sentences ----

    #[test]
    fn test_extract_two_english() {
        let mut buf = "Hello world. How are you?".to_string();
        let chunks = extract_sentences(&mut buf, 200);
        assert_eq!(chunks, vec!["Hello world.", "How are you?"]);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_extract_hindi_with_danda() {
        let mut buf = "नमस्ते। Sarvam AI में आपका स्वागत है।".to_string();
        let chunks = extract_sentences(&mut buf, 200);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "नमस्ते।");
        assert!(chunks[1].ends_with('।'));
    }

    #[test]
    fn test_extract_ellipsis_not_split() {
        let mut buf =
            "So basically… हम India की हर language को voice देते हैं। अगला।".to_string();
        let chunks = extract_sentences(&mut buf, 400);
        // First chunk should run from "So basically" all the way to first danda
        assert!(chunks[0].starts_with("So basically"));
        assert!(chunks[0].ends_with('।'));
    }

    #[test]
    fn test_extract_newline_paragraph() {
        let mut buf = "First paragraph.\n\nSecond paragraph.".to_string();
        let chunks = extract_sentences(&mut buf, 200);
        assert_eq!(chunks[0], "First paragraph.");
        // Empty line between paragraphs consumed; second chunk extracted
        assert!(chunks.iter().any(|c| c.contains("Second paragraph.")));
    }

    #[test]
    fn test_extract_partial_stays() {
        let mut buf = "Hello world. How are".to_string();
        let chunks = extract_sentences(&mut buf, 200);
        assert_eq!(chunks, vec!["Hello world."]);
        assert_eq!(buf, "How are");
    }

    #[test]
    fn test_extract_with_title() {
        let mut buf = "Dr. Smith arrived. He left.".to_string();
        let chunks = extract_sentences(&mut buf, 200);
        assert_eq!(chunks[0], "Dr. Smith arrived.");
        assert_eq!(chunks[1], "He left.");
    }

    #[test]
    fn test_extract_force_split() {
        let mut buf = "a".repeat(160);
        let chunks = extract_sentences(&mut buf, 150);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 150);
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn test_extract_empty() {
        let mut buf = String::new();
        assert!(extract_sentences(&mut buf, 200).is_empty());
    }

    #[test]
    fn test_extract_no_sentence_yet() {
        let mut buf = "Hello world".to_string();
        assert!(extract_sentences(&mut buf, 200).is_empty());
        assert_eq!(buf, "Hello world");
    }

    // ---- find_sentence_boundary_before ----

    #[test]
    fn test_boundary_before_finds_last() {
        let text = "Hello world. How are you? Fine.";
        let pos = find_sentence_boundary_before(text, 25).unwrap();
        assert_eq!(&text[..pos], "Hello world. How are you?");
    }

    #[test]
    fn test_boundary_before_none() {
        assert!(find_sentence_boundary_before("Hello world", 50).is_none());
    }

    #[test]
    fn test_boundary_before_danda() {
        let text = "नमस्ते। अगला।";
        // max_len covers only the first danda
        let pos = find_sentence_boundary_before(text, 12).unwrap();
        assert!(text[..pos].ends_with('।'));
    }
}