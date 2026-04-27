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

#[derive(Debug, Clone)]
pub struct RaviObserverParams {
    pub bot_speaking_enabled:       bool,
    pub bot_llm_enabled:            bool,
    pub bot_tts_enabled:            bool,
    pub user_speaking_enabled:      bool,
    pub user_transcription_enabled: bool,
    pub user_mute_enabled:          bool,
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
    ravi_proc: FrameProcessor,
    params:    RaviObserverParams,
    seen:      Mutex<HashSet<u64>>,
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
    async fn on_process_frame(&self, _event: FrameProcessed) {}

    async fn on_push_frame(&self, event: FramePushed) {
        let frame     = &event.frame;
        let direction = event.direction;

        // Skip upstream copy of broadcast frames.
        if frame.sibling_id.is_some() && direction != FrameDirection::Downstream {
            return;
        }

        // Deduplicate by frame ID.
        {
            let mut seen = self.seen.lock().await;
            if !seen.insert(frame.id) {
                return;
            }
            if seen.len() > 4096 {
                seen.clear();
            }
        }

        match &frame.inner {
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
            FrameInner::Data(DataFrame::Transcription(t))
                if self.params.user_transcription_enabled =>
            {
                let json = models::msg_user_transcription(
                    &t.text,
                    &t.user_id,
                    &t.timestamp,
                    t.finalized,
                );
                self.send(json).await;
            }
            FrameInner::Control(ControlFrame::LLMFullResponseStart)
                if self.params.bot_llm_enabled =>
            {
                self.send(models::msg_bot_llm_started()).await;
            }
            FrameInner::Control(ControlFrame::LLMFullResponseEnd)
                if self.params.bot_llm_enabled =>
            {
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
            FrameInner::Data(DataFrame::LLMText(text))
                if self.params.bot_llm_enabled =>
            {
                self.send(models::msg_bot_llm_text(text)).await;

                if self.params.bot_transcription_enabled {
                    let mut acc = self.llm_accum.lock().await;
                    acc.push_str(text);
                    if acc.ends_with(['.', '!', '?']) && acc.len() > 1 {
                        let sentence = acc.trim().to_string();
                        acc.clear();
                        drop(acc);
                        self.send(models::msg_bot_transcription(&sentence)).await;
                    }
                }
            }
            _ => {}
        }
    }
}
