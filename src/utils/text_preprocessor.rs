//! Text preprocessing for TTS — applied to text chunks before sending to Sarvam.
//!
//! Based on Sarvam TTS best-practices documentation:
//!   - Section 14: Numbers > 4 digits should use commas for correct pronunciation.
//!
//! # Usage
//!
//! ```
//! use rustvani::utils::text_preprocessor::preprocess_for_tts;
//!
//! assert_eq!(preprocess_for_tts("population is 10000"),    "population is 10,000");
//! assert_eq!(preprocess_for_tts("pi is 3.14"),             "pi is 3.14");   // unchanged
//! assert_eq!(preprocess_for_tts("1000 items"),             "1000 items");   // 4 digits — unchanged
//! assert_eq!(preprocess_for_tts("cost is 100000 rupees"),  "cost is 1,00,000 rupees");
//! ```
//!
//! Additional preprocessors can be added to this module over time and wired
//! through [`preprocess_for_tts`].

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/// Format numbers with 5+ digits using the **Indian numbering system**:
/// last 3 digits are grouped, then groups of 2 from the right.
///
/// Examples:
///   - 10000   → 10,000
///   - 100000  → 1,00,000
///   - 1234567 → 12,34,567
///
/// Numbers adjacent to a decimal point (e.g. `3.14159`) are left unchanged.
/// Numbers with exactly 4 digits are left unchanged per the docs ("greater
/// than 4 digits").
///
/// The Indian system is used because Sarvam's target languages are Indian,
/// and the docs show `10,000` style formatting which maps to Indian when
/// generalised (first comma after 2 digits for 5-digit numbers).
fn format_numbers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len() + 8);
    let mut i = 0;

    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            // Collect a full run of digits
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }

            let digits: String = chars[start..i].iter().collect();

            // Check for adjacent decimal point — don't format decimals
            let preceded_by_dot = start > 0 && chars[start - 1] == '.';
            let followed_by_dot = i < chars.len() && chars[i] == '.';

            if digits.len() > 4 && !preceded_by_dot && !followed_by_dot {
                result.push_str(&indian_comma_format(&digits));
            } else {
                result.push_str(&digits);
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Insert commas using Indian numbering convention.
///
/// Indian system: group last 3 digits, then groups of 2 from the right.
///   10000   → "10,000"
///   100000  → "1,00,000"
///   1234567 → "12,34,567"
fn indian_comma_format(digits: &str) -> String {
    let len = digits.len();

    if len <= 3 {
        return digits.to_string();
    }

    let mut result = String::with_capacity(len + len / 2);
    let bytes = digits.as_bytes();

    // Split: last 3 digits are the "hundreds group"
    let hundreds_start = len - 3;
    let leading = &digits[..hundreds_start];

    // Format the leading digits in groups of 2 from the right
    let leading_bytes = leading.as_bytes();
    let leading_len = leading_bytes.len();
    let first_group = leading_len % 2; // 0 or 1

    let mut pos = 0;
    if first_group > 0 {
        result.push(leading_bytes[0] as char);
        pos = 1;
        if pos < leading_len {
            result.push(',');
        }
    }
    while pos < leading_len {
        result.push(leading_bytes[pos] as char);
        result.push(leading_bytes[pos + 1] as char);
        pos += 2;
        if pos < leading_len {
            result.push(',');
        }
    }

    result.push(',');

    // Append last 3 digits
    for &b in &bytes[hundreds_start..] {
        result.push(b as char);
    }

    result
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Apply all TTS text preprocessors in order and return the transformed string.
///
/// Current transformations (in order):
///   1. [`format_numbers`] — insert commas into numbers with 5+ digits.
///
/// This function is the single entry point for the TTS handler. Adding a new
/// preprocessor means adding it here — the handler doesn't change.
pub fn preprocess_for_tts(text: &str) -> String {
    let text = format_numbers(text);
    // Future preprocessors go here:
    // let text = normalize_urls(&text);
    // let text = expand_abbreviations(&text);
    text
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- indian_comma_format ----

    #[test]
    fn test_5_digits() {
        // 10000 → 10,000
        assert_eq!(indian_comma_format("10000"), "10,000");
    }

    #[test]
    fn test_6_digits() {
        // 100000 → 1,00,000
        assert_eq!(indian_comma_format("100000"), "1,00,000");
    }

    #[test]
    fn test_7_digits() {
        // 1234567 → 12,34,567
        assert_eq!(indian_comma_format("1234567"), "12,34,567");
    }

    #[test]
    fn test_8_digits() {
        // 12345678 → 1,23,45,678
        assert_eq!(indian_comma_format("12345678"), "1,23,45,678");
    }

    #[test]
    fn test_3_digits_unchanged() {
        assert_eq!(indian_comma_format("999"), "999");
    }

    // ---- format_numbers ----

    #[test]
    fn test_5_digit_in_sentence() {
        assert_eq!(format_numbers("population is 10000"), "population is 10,000");
    }

    #[test]
    fn test_4_digits_unchanged() {
        // Docs: "greater than 4 digits" — 4-digit numbers left alone
        assert_eq!(format_numbers("1000 items"), "1000 items");
        assert_eq!(format_numbers("9999 rupees"), "9999 rupees");
    }

    #[test]
    fn test_decimal_unchanged() {
        // 3.14159 — integer part only 1 digit, decimal part 5 — neither is standalone
        assert_eq!(format_numbers("pi is 3.14159"), "pi is 3.14159");
    }

    #[test]
    fn test_decimal_5_digit_unchanged() {
        // 3.14159 — the "14159" follows a decimal point → don't format
        assert_eq!(format_numbers("value 3.14159 units"), "value 3.14159 units");
    }

    #[test]
    fn test_large_preceded_by_dot_unchanged() {
        // e.g. version "1.10000" — the 10000 follows a dot
        assert_eq!(format_numbers("v1.10000"), "v1.10000");
    }

    #[test]
    fn test_multiple_numbers() {
        let result = format_numbers("cost 50000 and count 200");
        assert_eq!(result, "cost 50,000 and count 200");
    }

    #[test]
    fn test_rupees_lakh() {
        assert_eq!(format_numbers("₹100000 salary"), "₹1,00,000 salary");
    }

    #[test]
    fn test_crore() {
        assert_eq!(format_numbers("10000000 population"), "1,00,00,000 population");
    }

    // ---- preprocess_for_tts ----

    #[test]
    fn test_preprocess_passthrough_small() {
        assert_eq!(preprocess_for_tts("hello world"), "hello world");
    }

    #[test]
    fn test_preprocess_formats_large_number() {
        assert_eq!(
            preprocess_for_tts("There are 1000000 cases pending."),
            "There are 10,00,000 cases pending."
        );
    }

    #[test]
    fn test_preprocess_leaves_decimal() {
        assert_eq!(
            preprocess_for_tts("rate is 3.14159"),
            "rate is 3.14159"
        );
    }

    #[test]
    fn test_preprocess_hindi_with_number() {
        assert_eq!(
            preprocess_for_tts("₹50000 का भुगतान करें"),
            "₹50,000 का भुगतान करें"
        );
    }
}