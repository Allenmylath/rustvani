use std::collections::HashSet;

use async_trait::async_trait;
use serde_json::json;
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
    /// Forward function call frames (start/end, in-progress, result, raw result)
    /// to the client as `server-message` events.
    pub function_call_enabled:      bool,
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
            function_call_enabled:      true,
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

        // Deduplicate by frame ID. One exception: the STT TurnGate re-pushes
        // the stashed VadStop frame with a transcript bundled on — SAME id,
        // different content. Key on (id, has_transcript) so the bare pre-gate
        // sighting doesn't swallow the transcript-carrying release.
        let dedup_key = match &frame.inner {
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { transcript, .. }) => {
                (frame.id << 1) | transcript.is_some() as u64
            }
            _ => frame.id << 1,
        };
        {
            let mut seen = self.seen.lock().await;
            if !seen.insert(dedup_key) {
                return;
            }
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
                let json = models::msg_user_transcription(
                    &t.text,
                    &t.user_id,
                    &t.timestamp,
                    t.finalized,
                );
                self.send(json).await;
            }

            // Bundled closing transcript on the VadStop frame (TurnGate release path).
            FrameInner::System(SystemFrame::VADUserStoppedSpeaking { transcript: Some(t), .. })
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

            // ---- LLM response boundaries ----
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

            // ---- LLM text tokens ----
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

            // ---- Function call: batch start ----
            FrameInner::Control(ControlFrame::FunctionCallStart)
                if self.params.function_call_enabled =>
            {
                self.send(models::msg_server_message(json!({
                    "type": "function-call-start",
                }))).await;
            }

            // ---- Function call: batch end ----
            FrameInner::Control(ControlFrame::FunctionCallEnd)
                if self.params.function_call_enabled =>
            {
                self.send(models::msg_server_message(json!({
                    "type": "function-call-end",
                }))).await;
            }

            // ---- Function call: in progress (tool invocation) ----
            FrameInner::Data(DataFrame::FunctionCallInProgress(data))
                if self.params.function_call_enabled =>
            {
                // Parse arguments as JSON if possible, fall back to string
                let args_value = serde_json::from_str::<serde_json::Value>(&data.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(data.arguments.clone()));

                self.send(models::msg_server_message(json!({
                    "type":          "function-call-in-progress",
                    "id":            data.id,
                    "function_name": data.function_name,
                    "arguments":     args_value,
                }))).await;
            }

            // ---- Function call: result summary (what LLM sees) ----
            FrameInner::Data(DataFrame::FunctionCallResult(data))
                if self.params.function_call_enabled =>
            {
                // Parse result as JSON if possible, fall back to string
                let result_value = serde_json::from_str::<serde_json::Value>(&data.result)
                    .unwrap_or_else(|_| serde_json::Value::String(data.result.clone()));

                self.send(models::msg_server_message(json!({
                    "type":          "function-call-result",
                    "id":            data.id,
                    "function_name": data.function_name,
                    "result":        result_value,
                }))).await;
            }

            // ---- Function call: raw structured data (LLM never sees this) ----
            FrameInner::Data(DataFrame::FunctionCallRawResult(data))
                if self.params.function_call_enabled =>
            {
                self.send(models::msg_server_message(json!({
                    "type":          "function-call-raw-result",
                    "id":            data.id,
                    "function_name": data.function_name,
                    "data":          data.raw_data,
                }))).await;
            }

            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{PassthroughHandler, StartFrameData, TranscriptionData};

    fn pushed(frame: Frame) -> FramePushed {
        FramePushed {
            source_name: "vad".into(),
            destination_name: "stt".into(),
            frame,
            direction: FrameDirection::Downstream,
            timestamp: 0.0,
        }
    }

    /// Regression: the STT TurnGate re-pushes the stashed VadStop frame with
    /// a transcript bundled on — same frame id. The observer's dedup must
    /// not swallow the transcript-carrying release just because the bare
    /// pre-gate frame was already seen.
    #[tokio::test]
    async fn bundled_vadstop_transcript_survives_dedup() {
        let proc = FrameProcessor::new("ravi-test", Box::new(PassthroughHandler), true);
        proc.process_frame(
            Frame::start(StartFrameData::default()),
            FrameDirection::Downstream,
        )
        .await
        .unwrap();

        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let cap = captured.clone();
        proc.on_before_push_frame(move |frame| {
            if let FrameInner::System(SystemFrame::RaviServerMessage { payload }) = &frame.inner {
                cap.lock().unwrap().push(payload.clone());
            }
        });

        let observer = RaviObserver::new(proc.clone(), RaviObserverParams::default());

        // 1. Bare VadStop travels VAD → STT (pre-gate sighting).
        let stop = Frame::vad_user_stopped_speaking(0.5, 1.0);
        observer.on_push_frame(pushed(stop.clone())).await;

        // 2. The gate releases the SAME frame (same id) with the transcript.
        let released = stop.with_vad_stop_transcript(
            TranscriptionData::new("hello world", "user", "now").finalized(),
        );
        observer.on_push_frame(pushed(released.clone())).await;

        let msgs = captured.lock().unwrap().clone();
        assert!(
            msgs.iter()
                .any(|m| m.contains("user-transcription") && m.contains("hello world")),
            "bundled closing transcript was not forwarded to the client: {msgs:?}"
        );

        // 3. Further hops of the released frame are still deduped (no dupes).
        drop(msgs);
        observer.on_push_frame(pushed(released)).await;
        let msgs = captured.lock().unwrap();
        assert_eq!(
            msgs.iter().filter(|m| m.contains("user-transcription")).count(),
            1,
            "transcript was sent more than once"
        );
    }
}
