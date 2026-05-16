//! ML-based end-of-turn detection — pure Rust, zero runtime dependencies.

mod whisper_features;
mod engine;
mod smart_turn;

pub use engine::SmartTurnEngine;
pub use smart_turn::{EndOfTurnState, SmartTurnAnalyzer, SmartTurnConfig, TurnMetrics};
pub use whisper_features::Precision;
