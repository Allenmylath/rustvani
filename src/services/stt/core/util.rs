//! Small helpers shared by every STT provider.
//!
//! Each of these existed as a private copy in two to four provider files before
//! the core was extracted.

/// PCM i16 LE bytes → samples. Any trailing odd byte is dropped.
pub fn bytes_to_i16(audio: &[u8]) -> Vec<i16> {
    audio
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Samples → PCM i16 LE bytes.
pub fn i16_to_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// Transcript timestamp: `"<unix_secs>.<millis>"`.
///
/// The provider files called this `time_now_iso8601`, which it never was — the
/// format is kept (it is what `TranscriptionData.timestamp` has always carried)
/// but the name no longer lies.
pub fn timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}.{:03}", d.as_secs(), d.subsec_millis())
}

/// Percent-encode a query-parameter value (RFC 3986 unreserved set).
pub fn percent_encode(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => vec![c],
            ':' => vec!['%', '3', 'A'],
            '/' => vec!['%', '2', 'F'],
            _ => format!("%{:02X}", c as u32).chars().collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i16_round_trip() {
        let samples = vec![0i16, 1, -1, i16::MAX, i16::MIN, 1234];
        assert_eq!(bytes_to_i16(&i16_to_bytes(&samples)), samples);
    }

    #[test]
    fn bytes_to_i16_drops_odd_trailing_byte() {
        assert_eq!(bytes_to_i16(&[0x01, 0x00, 0x7f]), vec![1i16]);
    }

    #[test]
    fn percent_encode_handles_special_chars() {
        assert_eq!(percent_encode("saaras:v3"), "saaras%3Av3");
        assert_eq!(percent_encode("en-IN"), "en-IN");
        assert_eq!(percent_encode("a b"), "a%20b");
    }

    #[test]
    fn timestamp_has_millisecond_suffix() {
        let t = timestamp();
        let (secs, millis) = t.split_once('.').expect("expected <secs>.<millis>");
        assert!(secs.parse::<u64>().is_ok(), "bad secs: {t}");
        assert_eq!(millis.len(), 3, "millis must be zero-padded to 3: {t}");
    }
}
