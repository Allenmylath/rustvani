use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single billable occurrence emitted by a service handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BillingEvent {
    SessionStart {
        session_id: Uuid,
        started_at: DateTime<Utc>,
        metadata: HashMap<String, String>,
    },
    SessionEnd {
        session_id: Uuid,
        ended_at: DateTime<Utc>,
        /// "end" | "stop" | "cancel"
        finish_reason: String,
    },
    LlmUsage {
        session_id: Uuid,
        provider: String,
        model: String,
        input_tokens: u32,
        output_tokens: u32,
        /// `true` when the provider does not return token counts and we estimated.
        estimated: bool,
        occurred_at: DateTime<Utc>,
    },
    TtsUsage {
        session_id: Uuid,
        provider: String,
        voice: String,
        char_count: usize,
        occurred_at: DateTime<Utc>,
    },
    SttUsage {
        session_id: Uuid,
        provider: String,
        audio_duration_ms: f64,
        occurred_at: DateTime<Utc>,
    },
}

impl BillingEvent {
    pub fn session_id(&self) -> Uuid {
        match self {
            Self::SessionStart { session_id, .. } => *session_id,
            Self::SessionEnd   { session_id, .. } => *session_id,
            Self::LlmUsage     { session_id, .. } => *session_id,
            Self::TtsUsage     { session_id, .. } => *session_id,
            Self::SttUsage     { session_id, .. } => *session_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn id() -> Uuid { Uuid::new_v4() }

    #[test]
    fn session_start_round_trips() {
        let sid = id();
        let ev = BillingEvent::SessionStart {
            session_id: sid,
            started_at: Utc::now(),
            metadata: [("env".into(), "prod".into())].into_iter().collect(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"type\":\"session_start\""), "tag missing: {json}");
        assert!(json.contains("env"), "metadata missing: {json}");
        let back: BillingEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id(), sid);
    }

    #[test]
    fn session_end_contains_finish_reason() {
        let ev = BillingEvent::SessionEnd {
            session_id: id(),
            ended_at: Utc::now(),
            finish_reason: "cancel".into(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"type\":\"session_end\""));
        assert!(json.contains("cancel"));
    }

    #[test]
    fn llm_usage_round_trips_with_estimated_flag() {
        let ev = BillingEvent::LlmUsage {
            session_id: id(),
            provider: "sarvam".into(),
            model: "sarvam-30b".into(),
            input_tokens: 80,
            output_tokens: 40,
            estimated: true,
            occurred_at: Utc::now(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: BillingEvent = serde_json::from_str(&json).unwrap();
        match back {
            BillingEvent::LlmUsage { input_tokens, output_tokens, estimated, provider, .. } => {
                assert_eq!(input_tokens, 80);
                assert_eq!(output_tokens, 40);
                assert!(estimated);
                assert_eq!(provider, "sarvam");
            }
            _ => panic!("wrong variant after round-trip"),
        }
    }

    #[test]
    fn tts_usage_round_trips() {
        let ev = BillingEvent::TtsUsage {
            session_id: id(),
            provider: "deepgram".into(),
            voice: "aura-2-helena-en".into(),
            char_count: 120,
            occurred_at: Utc::now(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: BillingEvent = serde_json::from_str(&json).unwrap();
        match back {
            BillingEvent::TtsUsage { char_count, voice, .. } => {
                assert_eq!(char_count, 120);
                assert_eq!(voice, "aura-2-helena-en");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn stt_usage_round_trips() {
        let ev = BillingEvent::SttUsage {
            session_id: id(),
            provider: "gnani".into(),
            audio_duration_ms: 3750.5,
            occurred_at: Utc::now(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: BillingEvent = serde_json::from_str(&json).unwrap();
        match back {
            BillingEvent::SttUsage { audio_duration_ms, provider, .. } => {
                assert!((audio_duration_ms - 3750.5).abs() < 0.001);
                assert_eq!(provider, "gnani");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn session_id_accessor_works_for_all_five_variants() {
        let sid = id();
        let now = Utc::now();
        let variants: &[BillingEvent] = &[
            BillingEvent::SessionStart { session_id: sid, started_at: now, metadata: Default::default() },
            BillingEvent::SessionEnd   { session_id: sid, ended_at: now, finish_reason: "end".into() },
            BillingEvent::LlmUsage     { session_id: sid, provider: "x".into(), model: "m".into(),
                                          input_tokens: 1, output_tokens: 1, estimated: false, occurred_at: now },
            BillingEvent::TtsUsage     { session_id: sid, provider: "x".into(), voice: "v".into(),
                                          char_count: 1, occurred_at: now },
            BillingEvent::SttUsage     { session_id: sid, provider: "x".into(), audio_duration_ms: 1.0, occurred_at: now },
        ];
        for ev in variants {
            assert_eq!(ev.session_id(), sid, "session_id() wrong for {:?}", ev);
        }
    }

    #[test]
    fn session_summary_default_is_zeroed() {
        let s = SessionSummary::default();
        assert_eq!(s.llm_input_tokens, 0);
        assert_eq!(s.llm_output_tokens, 0);
        assert_eq!(s.llm_calls, 0);
        assert_eq!(s.tts_chars, 0);
        assert_eq!(s.tts_calls, 0);
        assert_eq!(s.stt_audio_ms, 0.0);
        assert_eq!(s.stt_calls, 0);
        assert!(s.started_at.is_none());
        assert!(s.ended_at.is_none());
        assert!(s.duration_secs.is_none());
    }
}

/// Per-session aggregated totals written to `billing_sessions` at session end.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: Uuid,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub duration_secs: Option<f64>,
    pub finish_reason: Option<String>,

    pub llm_input_tokens: u32,
    pub llm_output_tokens: u32,
    pub llm_calls: u32,

    pub tts_chars: usize,
    pub tts_calls: u32,

    pub stt_audio_ms: f64,
    pub stt_calls: u32,

    pub metadata: HashMap<String, String>,
}
