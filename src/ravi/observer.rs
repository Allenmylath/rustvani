/// `RaviObserver` — converts pipeline frame events into outbound RAVI messages.
///
/// Attach to the pipeline via `Pipeline::set_observer` (or equivalent).
/// Holds a clone of the `FrameProcessor` that sits just before the output
/// transport so that `Frame::ravi_server_message` frames it pushes downstream
/// reach the output transport and are sent to the client as WebSocket text.
///
/// # Deduplication
///
/// Broadcast frames arrive twice (downstream + upstream).  We skip the
/// upstream copy to avoid sending duplicate messages.  Additionally, the set
/// of already-seen frame IDs prevents double-processing frames that travel
/// through multiple processors.
use std::collections::HashSet;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::frames::{
    ControlFrame, DataFrame, Frame, FrameDirection, FrameInner, FrameProcessor, SystemFrame,
};
use crate::observer::{BaseObserver, FrameProcessed, FramePushed};

use super::models;

// ---------------------------------------------------------------------------
// RaviObserverParams
// ---------------------------------------------------------------------------

/// Feature flags controlling which pipeline events produce RAVI messages.
#[derive(Debug, Clone)]
pub struct RaviObserverParams {
    pub bot_speaking_enabled:       bool,
    pub bot_llm_enabled:            bool,
    pub bot_tts_enabled:            bool,
    pub user_speaking_enabled:      bool,
    pub user_transcription_enabled: bool,
    pub user_mute_enabled:          bool,
    /// Accumulate LLM tokens into sentences and emit `bot-transcription`
    /// (deprecated upstream but kept for older client compatibility).
    pub bot_transcription_enabled:  bool,
}

impl Default for RaviObserverParams {
    fn default() -> Self {
        Self {
            bot_speaking_enabled:       true,
            bot_llm_enabled:            true,
            bot_tts_enabled:            true,
            user_speaking_enabled:      true,
            user_transcription_enabled: true,
            user_mute_enabled:          true,
            bot_transcription_enabled:  false,
        }
    }
}

// ---------------------------------------------------------------------------
// RaviObserver
// ---------------------------------------------------------------------------

pub struct RaviObserver {
    /// A clone of the `RaviProcessor`'s `FrameProcessor`.  Pushing a
    /// `ravi_server_message` frame downstream from here reaches the output
    /// transport.
    ravi_proc: FrameProcessor,
    params:    RaviObserverParams,

    /// Prevents double-processing the same frame (e.g. broadcast copies).
    seen:      Mutex<HashSet<u64>>,

    /// Accumulates LLM tokens for `bot-transcription` sentence detection.
    llm_accum: Mutex<String>,
}

impl RaviObserver {
    pub fn new(ravi_proc: FrameProcessor, params: RaviObserverParams) -> Self {
        Self {
            ravi_proc,
            params,
            seen:      Mutex::new(HashSet::new()),
            llm_accum: Mutex::new(String::new()),
        }
    }

    /// Serialise `payload` into a `RaviServerMessage` frame and push it
    /// downstream through the `ravi_proc` into the output transport.
    async fn send(&self, payload: String) {
        let frame = Frame::ravi_server_message(payload);
        if let Err(e) = self.ravi_proc
            .push_frame(frame, FrameDirection::Downstream)
            .await
        {
            log::error!("RaviObserver: failed to push server message: {}", e);
        }
    }
}

#[async_trait]
impl BaseObserver for RaviObserver {
    async fn on_process_frame(&self, _event: FrameProcessed) {
        // Not used — we react in on_push_frame.
    }

    async fn on_push_frame(&self, event: FramePushed) {
        let frame     = &event.frame;
        let direction = event.direction;

        // Skip the upstream copy of broadcast frames.
        if frame.sibling_id.is_some() && direction != FrameDirection::Downstream {
            return;
        }

        // Deduplicate: each frame ID is processed at most once.
        {
            let mut seen = self.seen.lock().await;
            if !seen.insert(frame.id) {
                return;
            }
            // Bound memory: clear once we accumulate 4096 IDs per session.
            // Sessions are short-lived so this is a coarse safety valve.
            if seen.len() > 4096 {
                seen.clear();
            }
        }

        match &frame.inner {
            // ---- Bot speaking ----
            FrameInner::System(SystemFrame::BotStartedSpeaking)
                if self.params.bot_speaking_enabled =>
            {
                self.send(models::msg_bot_started_speaking()).await;
            }
            FrameInner::System(SystemFrame::BotStoppedSpeaking)
                if self.params.bot_speaking_enabled =>
            {
                self.send(models::msg_bot_stopped_speaking()).await;
            }

            // ---- User speaking ----
            FrameInner::System(SystemFrame::UserStartedSpeaking { .. })
                if self.params.user_speaking_enabled =>
            {
                self.send(models::msg_user_started_speaking()).await;
            }
            FrameInner::System(SystemFrame::UserStoppedSpeaking { .. })
                if self.params.user_speaking_enabled =>
            {
                self.send(models::msg_user_stopped_speaking()).await;
            }

            // ---- User transcription ----
            FrameInner::Data(DataFrame::Transcription(t))
                if self.params.user_transcription_enabled =>
            {
                let ts = &t.timestamp;
                let json = models::msg_user_transcription(
                    &t.text,
                    &t.user_id,
                    ts,
                    t.finalized,
                );
                self.send(json).await;
            }

            // ---- LLM lifecycle ----
            FrameInner::Control(ControlFrame::LLMFullResponseStart)
                if self.params.bot_llm_enabled =>
            {
                self.send(models::msg_bot_llm_started()).await;
            }
            FrameInner::Control(ControlFrame::LLMFullResponseEnd)
                if self.params.bot_llm_enabled =>
            {
                // Flush any remaining accumulated transcription.
                if self.params.bot_transcription_enabled {
                    let leftover = {
                        let mut acc = self.llm_accum.lock().await;
                        let s = acc.trim().to_string();
                        acc.clear();
                        s
                    };
                    if !leftover.is_empty() {
                        self.send(models::msg_bot_transcription(&leftover)).await;
                    }
                }
                self.send(models::msg_bot_llm_stopped()).await;
            }

            // ---- LLM token ----
            FrameInner::Data(DataFrame::LLMText(text))
                if self.params.bot_llm_enabled =>
            {
                self.send(models::msg_bot_llm_text(text)).await;

                // Optional sentence-level transcription accumulation.
                if self.params.bot_transcription_enabled {
                    let mut acc = self.llm_accum.lock().await;
                    acc.push_str(text);
                    // Emit on simple sentence-end heuristic.
                    if acc.ends_with(['.', '!', '?']) && acc.len() > 1 {
                        let sentence = acc.trim().to_string();
                        acc.clear();
                        drop(acc);
                        self.send(models::msg_bot_transcription(&sentence)).await;
                    }
                }
            }

            // ---- TTS lifecycle ----
            // NOTE: rustvani does not yet have TTSStarted/Stopped frames.
            // Add arms here once they exist.

            // Anything else is silently ignored.
            _ => {}
        }
    }
}
