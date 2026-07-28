//! Services — STT, LLM and TTS backends.
//!
//! Every backend is behind the feature that supplies its dependency, so a
//! `--no-default-features` build compiles only the providers it asked for. The
//! submodule declarations carry the same gates; these re-exports mirror them.

pub mod llm;
pub mod stt;
pub mod tts;

#[cfg(feature = "llm-openai")]
pub use llm::openai::{OpenAILLMConfig, OpenAILLMHandler};
#[cfg(feature = "stt-sarvam")]
pub use llm::sarvam::{SarvamLLMConfig, SarvamLLMHandler};
#[cfg(feature = "stt-deepgram")]
pub use stt::deepgram::{DeepgramSttConfig, DeepgramSttHandler};
#[cfg(feature = "stt-gnani")]
pub use stt::gnani::{GnaniSttConfig, GnaniSttHandler};
#[cfg(feature = "stt-sarvam")]
pub use stt::sarvam::{NoiseBackend, SarvamSttConfig, SarvamSttHandler};
#[cfg(feature = "stt-60db")]
pub use stt::sixtydb::{
    SixtyDbEncoding,
    SixtyDbSttConfig, SixtyDbSttHandler,
};
#[cfg(feature = "tts-sarvam")]
pub use tts::sarvam::{SarvamTtsConfig, SarvamTtsHandler};
#[cfg(feature = "tts-deepgram")]
pub use tts::{DeepgramTtsConfig, DeepgramTtsHandler};
#[cfg(feature = "tts-piper")]
pub use tts::piper::{PiperModel, PiperQuality, PiperTtsConfig, PiperTtsHandler};
pub use llm::FunctionRegistry;
