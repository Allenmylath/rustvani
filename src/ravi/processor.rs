/// `RaviProcessor` — RAVI protocol handler sitting in the pipeline.
///
/// Place it early in the pipeline (after the input transport, before the LLM
/// aggregators).  It intercepts `RaviClientMessage` system frames, handles
/// protocol handshaking and control messages, and passes everything else
/// through unchanged.
///
/// # Typical pipeline
///
/// ```text
/// [WS Input] → [RaviProcessor] → [LLMUserAggregator] → [LLM] →
///             [LLMAssistantAggregator] → [TTS] → [WS Output]
/// ```
///
/// # Sending outbound messages
///
/// To push a RAVI message from anywhere in your app:
/// ```rust
/// processor.push_frame(
///     Frame::ravi_server_message(models::msg_server_message(json!({...}))),
///     FrameDirection::Downstream,
/// ).await?;
/// ```
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use crate::context::LLMContext;
use crate::error::Result;
use crate::frames::{
    ControlFrame, DataFrame, Frame, FrameDirection, FrameHandler, FrameInner, FrameProcessor,
    SystemFrame,
};

use super::models::{self, ClientReadyData, SendTextData, SendTextOptions};
use super::observer::{RaviObserver, RaviObserverParams};

// ---------------------------------------------------------------------------
// RaviParams
// ---------------------------------------------------------------------------

/// Configuration for `RaviProcessor`.
#[derive(Debug, Clone)]
pub struct RaviParams {
    /// Static `about` metadata included in `bot-ready`.
    /// Defaults to `{"library":"rustvani"}`.
    pub about: Option<Value>,

    /// If true, the bot-ready message is sent automatically as soon as the
    /// client-ready handshake completes (version validated).
    /// Set to false if you need to delay bot-ready (e.g. until a DB call
    /// completes), and call `send_bot_ready()` manually instead.
    pub auto_bot_ready: bool,

    /// RTVI protocol version we advertise.  Defaults to `models::PROTOCOL_VERSION`.
    pub protocol_version: String,
}

impl Default for RaviParams {
    fn default() -> Self {
        Self {
            about: Some(serde_json::json!({ "library": "rustvani" })),
            auto_bot_ready: true,
            protocol_version: models::PROTOCOL_VERSION.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal mutable state (behind Mutex for &self access in FrameHandler)
// ---------------------------------------------------------------------------

struct State {
    client_ready:    bool,
    client_ready_id: String,
    bot_ready:       bool,
}

// ---------------------------------------------------------------------------
// Inner handler
// ---------------------------------------------------------------------------

struct RaviHandler {
    params: RaviParams,
    state:  Mutex<State>,
}

impl RaviHandler {
    fn new(params: RaviParams) -> Self {
        Self {
            params,
            state: Mutex::new(State {
                client_ready:    false,
                client_ready_id: String::new(),
                bot_ready:       false,
            }),
        }
    }

    // ---- Protocol handlers ----

    /// `client-ready` — validate version; optionally emit `bot-ready`.
    async fn handle_client_ready(
        &self,
        processor: &FrameProcessor,
        msg_id: &str,
        data_raw: Option<&str>,
    ) -> Result<()> {
        let client_data: Option<ClientReadyData> = data_raw
            .and_then(|s| serde_json::from_str(s).ok());

        let version = client_data.as_ref().map(|d| d.version.as_str()).unwrap_or("unknown");
        log::info!("RaviProcessor: client-ready (version={})", version);

        // Version compatibility check: major must match.
        let server_major = self.params.protocol_version
            .split('.')
            .next()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(1);

        if let Some(client_major) = version
            .split('.')
            .next()
            .and_then(|s| s.parse::<u32>().ok())
        {
            if client_major != server_major {
                let err = format!(
                    "RAVI version {} is not compatible with server protocol {}. \
                     Compatibility issues may occur.",
                    version, self.params.protocol_version
                );
                log::warn!("RaviProcessor: {}", err);
                let payload = models::msg_error_response(msg_id, &err);
                processor
                    .push_frame(Frame::ravi_server_response(msg_id, payload), FrameDirection::Downstream)
                    .await?;
            }
        }

        {
            let mut s = self.state.lock().unwrap();
            s.client_ready    = true;
            s.client_ready_id = msg_id.to_string();
        }

        if self.params.auto_bot_ready {
            self.send_bot_ready(processor).await?;
        }

        Ok(())
    }

    /// Send `bot-ready` to the client.  Called automatically if
    /// `auto_bot_ready = true`, or manually by the user otherwise.
    pub async fn send_bot_ready(&self, processor: &FrameProcessor) -> Result<()> {
        let (msg_id, already_sent) = {
            let mut s = self.state.lock().unwrap();
            let id   = s.client_ready_id.clone();
            let sent = s.bot_ready;
            s.bot_ready = true;
            (id, sent)
        };

        if already_sent {
            log::warn!("RaviProcessor: send_bot_ready called more than once — ignoring");
            return Ok(());
        }

        let payload = models::msg_bot_ready(&msg_id, self.params.about.clone());
        processor
            .push_frame(
                Frame::ravi_server_message(payload),
                FrameDirection::Downstream,
            )
            .await?;

        log::info!("RaviProcessor: bot-ready sent");
        Ok(())
    }

    /// `disconnect-bot` — gracefully end the pipeline.
    async fn handle_disconnect(&self, processor: &FrameProcessor) -> Result<()> {
        log::info!("RaviProcessor: disconnect-bot received — ending pipeline");
        processor.push_frame(Frame::end(), FrameDirection::Downstream).await
    }

    /// `send-text` — inject user text directly into the LLM context.
    async fn handle_send_text(
        &self,
        processor: &FrameProcessor,
        data_raw: Option<&str>,
        context: &std::sync::Arc<std::sync::Mutex<LLMContext>>,
    ) -> Result<()> {
        let data: SendTextData = match data_raw.and_then(|s| serde_json::from_str(s).ok()) {
            Some(d) => d,
            None => {
                log::warn!("RaviProcessor: send-text missing or invalid data");
                return Ok(());
            }
        };

        let opts = data.options.unwrap_or_default();

        if opts.run_immediately {
            processor.broadcast_interruption().await?;
        }

        context.lock().unwrap().add_message("user", &data.content);

        processor
            .push_frame(
                Frame::llm_context(context.clone()),
                FrameDirection::Downstream,
            )
            .await?;

        log::debug!("RaviProcessor: send-text injected: '{}'", data.content);
        Ok(())
    }

    /// `client-message` — re-emit for user-defined handlers further down the
    /// pipeline.  Other processors can match on `RaviClientMessage` frames
    /// with `msg_type` set to whatever custom type the client sent.
    async fn handle_client_message(
        &self,
        processor: &FrameProcessor,
        msg_id: &str,
        msg_type: &str,
        data: Option<String>,
    ) -> Result<()> {
        log::debug!("RaviProcessor: client-message type='{}' id='{}'", msg_type, msg_id);
        // Re-push with the inner type so downstream handlers can pattern-match.
        processor
            .push_frame(
                Frame::ravi_client_message(msg_id, msg_type, data),
                FrameDirection::Downstream,
            )
            .await
    }
}

// ---------------------------------------------------------------------------
// FrameHandler implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameHandler for RaviHandler {
    async fn on_process_frame(
        &self,
        processor: &FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
    ) -> Result<()> {
        match &frame.inner {
            FrameInner::System(SystemFrame::RaviClientMessage { msg_id, msg_type, data }) => {
                let msg_id   = msg_id.clone();
                let msg_type = msg_type.clone();
                let data     = data.clone();

                match msg_type.as_str() {
                    "client-ready" => {
                        self.handle_client_ready(processor, &msg_id, data.as_deref()).await?;
                    }
                    "disconnect-bot" => {
                        self.handle_disconnect(processor).await?;
                    }
                    "client-message" => {
                        // The actual inner type is in `data.t` per the RAVI spec.
                        // Re-emit as a new RaviClientMessage with the inner type
                        // so downstream processors have a clean frame to match.
                        if let Some(raw) = &data {
                            if let Ok(inner) = serde_json::from_str::<serde_json::Value>(raw) {
                                let inner_type = inner.get("t")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown")
                                    .to_string();
                                let inner_data = inner.get("d").map(|v| v.to_string());
                                self.handle_client_message(
                                    processor, &msg_id, &inner_type, inner_data
                                ).await?;
                            }
                        }
                    }
                    unknown => {
                        log::warn!("RaviProcessor: unsupported message type '{}'", unknown);
                        let payload = models::msg_error_response(
                            &msg_id,
                            &format!("Unsupported message type: {}", unknown),
                        );
                        processor
                            .push_frame(
                                Frame::ravi_server_response(&msg_id, payload),
                                FrameDirection::Downstream,
                            )
                            .await?;
                    }
                }
            }

            // Pass everything else through.
            _ => {
                processor.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public constructor
// ---------------------------------------------------------------------------

/// Create the `RaviProcessor` `FrameProcessor`.
///
/// # Example
/// ```rust
/// let ravi = RaviProcessor::new(RaviParams::default());
/// let observer = RaviProcessor::create_observer(&ravi, RaviObserverParams::default());
/// // wire ravi into your pipeline, then set observer on the pipeline task
/// ```
pub struct RaviProcessor;

impl RaviProcessor {
    /// Build and return a `FrameProcessor` wrapping the RAVI protocol handler.
    pub fn new(params: RaviParams) -> FrameProcessor {
        FrameProcessor::new("RaviProcessor", Box::new(RaviHandler::new(params)), false)
    }

    /// Create a `RaviObserver` that uses `proc` to push outbound messages.
    ///
    /// `proc` should be the same `FrameProcessor` returned by `RaviProcessor::new`.
    pub fn create_observer(proc: &FrameProcessor, params: RaviObserverParams) -> RaviObserver {
        RaviObserver::new(proc.clone(), params)
    }
}
