//! Twilio Media Streams voice server driven by a **standalone coordinator** —
//! a `CoordinatorProcessor` sitting in the LLM slot with *no agent swarm behind
//! it*: no `LocalAgentBus`, no `AgentRunner`, no `BaseAgent`, no peer agents.
//!
//! The coordinator's closure answers entirely from local logic. It still receives
//! a `CoordinatorCtx`, so the day you *do* want peers you only have to wrap this
//! pipeline in a `BaseAgent` and start calling `cx.call(...)` — the pipeline shape
//! does not change. Without a `BaseAgent` the bus handle is simply `None` and
//! `cx.call` would return an error, which this brain never triggers.
//!
//! Pipeline (one fully isolated instance per phone call):
//!
//! ```text
//!   Twilio WS ──► WebSocketTransport.input()   (µ-law 8k → PCM 16k, VAD)
//!                   → DeepgramStt
//!                   → LLMUserAggregator
//!                   → CoordinatorProcessor     ← the brain, no agents
//!                   → LLMAssistantAggregator
//!                   → DeepgramTts
//!                   → WebSocketTransport.output()
//!   Twilio WS ◄────────────────────────────────  (PCM 16k → µ-law 8k)
//! ```
//!
//! Wire protocol is Twilio's Media Streams JSON, handled end to end by
//! [`TwilioFrameSerializer`]: inbound `media` events are base64 µ-law at 8 kHz and
//! get resampled up to the 16 kHz pipeline rate; outbound TTS audio is resampled
//! back down and re-encoded; barge-in emits Twilio's `clear` event; inbound `dtmf`
//! events become `InputDTMFFrame`s.
//!
//! Routes:
//!   `POST|GET /twiml`  — the voice webhook. Returns TwiML that connects the call
//!                        to `/ws`. Point your Twilio phone number's "A call comes
//!                        in" webhook here.
//!   `GET  /ws`         — the Media Streams WebSocket Twilio dials back into.
//!
//! Because the TwiML uses `<Connect><Stream>`, the call ends as soon as this
//! socket closes — so ending the pipeline hangs up the caller. If
//! `TWILIO_AUTH_TOKEN` is set, the serializer *also* terminates the call over the
//! REST API on EndFrame (belt and braces); without it, auto hang-up is disabled
//! and the socket close does the work.
//!
//! Builds on the crate's default features — no `--features` flag needed.
//!
//! Environment variables:
//!   DEEPGRAM_API_KEY  — required (STT + TTS)
//!   TWILIO_AUTH_TOKEN — optional; enables REST auto hang-up
//!   PUBLIC_HOST       — optional; e.g. "abc123.ngrok.app". Defaults to the
//!                       inbound request's Host header, which is what you want
//!                       behind ngrok or Fly.
//!   PORT              — optional, default 8080
//!   RUST_LOG          — e.g. "info" or "rustvani=debug,info"
//!
//! Local run:
//!   cargo run --bin twilio_coordinator_server
//!   ngrok http 8080
//!   # set the number's voice webhook to https://<ngrok-host>/twiml

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::{
    Router,
    extract::{State, WebSocketUpgrade, ws::{Message, WebSocket}},
    http::{HeaderMap, header},
    response::IntoResponse,
    routing::get,
};
use futures::future::BoxFuture;
use tokio::sync::mpsc;

use rustvani::agents::{CoordinatorCtx, CoordinatorProcessor};
use rustvani::observer::{BaseObserver, FrameProcessed, FramePushed};
use rustvani::serializers::{TwilioFrameSerializer, TwilioInputParams, TwilioStart};
use rustvani::transport::TransportParams;
use rustvani::transport::websocket::{WebSocketParams, WebSocketTransport};
use rustvani::{
    DeepgramSttConfig, DeepgramSttHandler, DeepgramTtsConfig, DeepgramTtsHandler, Frame,
    FrameDirection, FrameInner, FrameKind, LLMAssistantAggregator, LLMContext, LLMUserAggregator,
    PipelineParams, PipelineTask, SileroVadNative, SystemFrame, VadParams, shared_context,
    system_clock,
};
use rustvani::context::Message as ContextMessage;

/// Pipeline-internal audio rate. Twilio is 8 kHz µ-law on the wire; the
/// serializer resamples in both directions so VAD and STT see clean 16 kHz.
const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// TTS synthesis rate. The Twilio serializer resamples this down to 8 kHz.
const TTS_SAMPLE_RATE: u32 = 16_000;

/// How long to wait for Twilio's `start` handshake before giving up.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Grace period for the pipeline to drain after the caller hangs up.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

static CALL_SEQ: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// The coordinator brain — plain local logic, zero agents
// ---------------------------------------------------------------------------

/// What the coordinator decided to say, and whether it wants the call over.
struct Reply {
    text: String,
    end_call: bool,
}

impl Reply {
    fn say(text: impl Into<String>) -> Self {
        Self { text: text.into(), end_call: false }
    }

    fn farewell(text: impl Into<String>) -> Self {
        Self { text: text.into(), end_call: true }
    }
}

/// Route one user utterance to an answer.
///
/// This is the entire "coordinator" — a match over what the caller said. An
/// empty utterance means the pipeline just started and nobody has spoken yet, so
/// it is the call-opening greeting.
fn decide(utterance: &str) -> Reply {
    let u = utterance.trim().to_lowercase();

    if u.is_empty() {
        return Reply::say(
            "Hi! You're through to the rustvani demo line. \
             Ask me for the time, say help to hear what I can do, or say goodbye to hang up.",
        );
    }

    if contains_any(&u, &["goodbye", "good bye", "hang up", "that's all", "thats all"])
        || has_word(&u, "bye")
    {
        return Reply::farewell("Thanks for calling. Goodbye!");
    }

    if contains_any(&u, &["time", "clock", "what hour"]) {
        let now = chrono::Local::now();
        return Reply::say(format!("It is {} local time.", now.format("%-I:%M %p")));
    }

    if contains_any(&u, &["date", "what day", "today"]) {
        let now = chrono::Local::now();
        return Reply::say(format!("Today is {}.", now.format("%A, %B %-d")));
    }

    if contains_any(&u, &["help", "what can you do", "options", "menu"]) {
        return Reply::say(
            "I can tell you the time or today's date, and I'll hang up when you say goodbye. \
             I'm a demo coordinator, so that's the whole menu.",
        );
    }

    if contains_any(&u, &["hello", "good morning", "good evening"])
        || has_word(&u, "hi")
        || has_word(&u, "hey")
    {
        return Reply::say("Hello! What can I do for you?");
    }

    if u.contains("thank") {
        return Reply::say("You're very welcome.");
    }

    Reply::say(format!(
        "I heard you say: {utterance}. I'm a simple demo, so try asking for the time, \
         or say help."
    ))
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// Whole-word match, so short triggers like "hi" don't fire on "this".
fn has_word(haystack: &str, word: &str) -> bool {
    haystack
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| w == word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_utterance_greets_without_ending() {
        let r = decide("");
        assert!(!r.end_call);
        assert!(r.text.contains("rustvani"));
    }

    #[test]
    fn goodbye_ends_the_call() {
        for u in ["Goodbye.", "bye", "Bye!", "please hang up", "That's all, thanks"] {
            assert!(decide(u).end_call, "{u:?} should end the call");
        }
    }

    #[test]
    fn short_triggers_are_word_bounded() {
        // "this" must not trip the "hi" greeting, nor "maybe" the "bye" farewell.
        assert!(!decide("maybe later").end_call);
        assert!(!decide("what is this").text.starts_with("Hello!"));
        assert!(decide("hi there").text.starts_with("Hello!"));
    }

    #[test]
    fn time_and_help_are_answered_locally() {
        assert!(decide("what time is it").text.contains("local time"));
        assert!(decide("help").text.contains("time"));
    }
}

/// Pull the most recent user turn out of the shared context.
fn last_user_message(context: &Arc<Mutex<LLMContext>>) -> String {
    context
        .lock()
        .unwrap()
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            ContextMessage::User { content } => Some(content.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// The closure handed to [`CoordinatorProcessor`]. Note it takes a
/// [`CoordinatorCtx`] it never uses — that is the seam where `cx.call("weather",
/// ...)` would go once peer agents exist.
fn coordinate(
    call_id: u64,
    hangup: Arc<AtomicBool>,
    _cx: CoordinatorCtx,
    context: Arc<Mutex<LLMContext>>,
) -> BoxFuture<'static, String> {
    Box::pin(async move {
        let utterance = last_user_message(&context);
        let reply = decide(&utterance);

        if reply.end_call {
            // Don't end the task here — let the farewell finish speaking first.
            // CallObserver fires EndTask on BotStoppedSpeaking.
            hangup.store(true, Ordering::SeqCst);
        }

        log::info!("[call={call_id}] user={utterance:?} → bot={:?}", reply.text);
        reply.text
    })
}

// ---------------------------------------------------------------------------
// CallObserver — per-call logging plus deferred hang-up
// ---------------------------------------------------------------------------

struct CallObserver {
    call_id: u64,
    hangup: Arc<AtomicBool>,
    push_tx: mpsc::Sender<(Frame, FrameDirection)>,
    ended: AtomicBool,
    /// The observer sees every frame once per processor it traverses; this keeps
    /// one-per-event logs (DTMF) from repeating down the whole pipeline.
    last_dtmf_frame: AtomicU64,
}

#[async_trait]
impl BaseObserver for CallObserver {
    async fn on_process_frame(&self, event: FrameProcessed) {
        match event.frame.kind() {
            // High-frequency audio — never log.
            FrameKind::InputAudioRaw | FrameKind::OutputAudioRaw | FrameKind::LLMText => {}

            FrameKind::InputDTMF => {
                let seen = self.last_dtmf_frame.swap(event.frame.id, Ordering::Relaxed);
                if seen != event.frame.id {
                    if let FrameInner::System(SystemFrame::InputDTMF { button }) =
                        &event.frame.inner
                    {
                        log::info!("[call={}] keypad: {}", self.call_id, button.as_str());
                    }
                }
            }

            FrameKind::BotStoppedSpeaking => {
                // The farewell has finished playing — now it's safe to hang up.
                if self.hangup.load(Ordering::SeqCst)
                    && !self.ended.swap(true, Ordering::SeqCst)
                {
                    log::info!("[call={}] farewell complete, ending call", self.call_id);
                    // EndTask is only honoured at the *upstream* boundary, where
                    // the task source converts it into a real EndFrame.
                    let _ = self
                        .push_tx
                        .send((Frame::end_task(), FrameDirection::Upstream))
                        .await;
                }
            }

            _ => {
                log::debug!(
                    "[call={}] {:>34}  @  {}",
                    self.call_id,
                    event.frame.name(),
                    event.processor_name,
                );
            }
        }
    }

    async fn on_push_frame(&self, _event: FramePushed) {}
}

// ---------------------------------------------------------------------------
// App state + TwiML webhook
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    deepgram_api_key: String,
    twilio_auth_token: Option<String>,
    public_host: Option<String>,
}

/// The voice webhook. `<Connect><Stream>` is bidirectional and ends the call when
/// the WebSocket closes, which is exactly the lifetime we want.
async fn twiml_handler(State(state): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let host = state
        .public_host
        .clone()
        .or_else(|| {
            headers
                .get(header::HOST)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "localhost:8080".to_string());

    let twiml = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/ws" />
  </Connect>
</Response>"#
    );

    log::info!("TwiML served, stream url = wss://{host}/ws");
    ([(header::CONTENT_TYPE, "text/xml; charset=utf-8")], twiml)
}

// ---------------------------------------------------------------------------
// Media Streams WebSocket
// ---------------------------------------------------------------------------

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_call(socket, state))
}

/// Read Twilio's opening messages until the `start` event arrives.
///
/// Twilio sends `connected` first, then `start` with the SIDs we need to build
/// the serializer — so this has to happen before `run_socket` takes the socket.
async fn await_twilio_start(socket: &mut WebSocket) -> Option<TwilioStart> {
    let deadline = tokio::time::Instant::now() + HANDSHAKE_TIMEOUT;
    loop {
        let msg = tokio::time::timeout_at(deadline, socket.recv()).await.ok()??;
        match msg {
            Ok(Message::Text(text)) => {
                if let Some(start) = TwilioStart::parse(&text) {
                    return Some(start);
                }
                log::debug!("pre-start Twilio message ignored: {text}");
            }
            Ok(Message::Close(_)) => return None,
            Ok(_) => {}
            Err(e) => {
                log::warn!("socket error during Twilio handshake: {e}");
                return None;
            }
        }
    }
}

async fn handle_call(mut socket: WebSocket, state: AppState) {
    let call_id = CALL_SEQ.fetch_add(1, Ordering::Relaxed);

    let Some(start) = await_twilio_start(&mut socket).await else {
        log::warn!("[call={call_id}] no Twilio start event — dropping connection");
        return;
    };
    log::info!(
        "[call={call_id}] stream={} call_sid={:?}",
        start.stream_sid,
        start.call_sid,
    );

    // ---- Serializer ----
    // REST auto hang-up needs the account credentials; without an auth token we
    // rely on the socket close (via <Connect>) to end the call instead.
    let auto_hang_up = state.twilio_auth_token.is_some()
        && start.call_sid.is_some()
        && start.account_sid.is_some();
    if !auto_hang_up {
        log::info!("[call={call_id}] REST auto hang-up disabled; socket close ends the call");
    }

    let serializer = match TwilioFrameSerializer::from_start(
        start,
        state.twilio_auth_token.clone(),
        TwilioInputParams { auto_hang_up, ..TwilioInputParams::default() },
    ) {
        Ok(s) => s,
        Err(e) => {
            log::error!("[call={call_id}] serializer init failed: {e}");
            return;
        }
    };

    // ---- VAD ----
    let vad = match SileroVadNative::new(PIPELINE_SAMPLE_RATE) {
        Ok(v) => Arc::new(v),
        Err(e) => {
            log::error!("[call={call_id}] VAD init failed: {e}");
            return;
        }
    };

    // ---- Transport ----
    let transport = WebSocketTransport::new(
        &format!("TwilioTransport-{call_id}"),
        WebSocketParams {
            transport: TransportParams {
                audio_in_enabled: true,
                audio_in_sample_rate: Some(PIPELINE_SAMPLE_RATE),
                audio_in_channels: 1,
                audio_in_passthrough: true,
                audio_in_stream_on_start: true,
                audio_out_enabled: true,
                // Must match what TTS emits — run_socket stamps outgoing chunks
                // with this rate before handing them to the serializer.
                audio_out_sample_rate: Some(TTS_SAMPLE_RATE),
                // 20 ms per chunk, Twilio's native media-event cadence.
                audio_out_10ms_chunks: 2,
                vad_analyzer: Some(vad),
                vad_params: VadParams {
                    // Telephony audio is narrowband and quiet after µ-law, so
                    // gate on VAD confidence rather than raw volume.
                    confidence: 0.45,
                    min_volume: 0.0,
                    ..VadParams::default()
                },
                ..TransportParams::default()
            },
        },
    );
    transport.set_serializer(Box::new(serializer));

    // ---- Processors ----
    let context = shared_context(None);
    let hangup = Arc::new(AtomicBool::new(false));

    let stt = DeepgramSttHandler::new(DeepgramSttConfig {
        api_key: state.deepgram_api_key.clone(),
        model: "nova-3".to_string(),
        language: "en-US".to_string(),
        // Must match the transport's input rate — the serializer has already
        // upsampled Twilio's 8 kHz µ-law by the time frames reach STT.
        sample_rate: PIPELINE_SAMPLE_RATE,
        ..DeepgramSttConfig::default()
    })
    .into_processor();

    let coordinator = {
        let hangup = hangup.clone();
        CoordinatorProcessor::new("coordinator", move |cx, ctx| {
            coordinate(call_id, hangup.clone(), cx, ctx)
        })
        .into_processor()
    };

    let tts = match DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: state.deepgram_api_key.clone(),
        voice: "aura-2-helena-en".to_string(),
        sample_rate: TTS_SAMPLE_RATE,
        ..DeepgramTtsConfig::default()
    }) {
        Ok(t) => t.into_processor(),
        Err(e) => {
            log::error!("[call={call_id}] TTS init failed: {e}");
            return;
        }
    };

    // ---- Pipeline ----
    let task = PipelineTask::new(
        vec![
            transport.input(),
            stt,
            LLMUserAggregator::new(context.clone()),
            coordinator,
            LLMAssistantAggregator::new(context.clone()),
            tts,
            transport.output(),
        ],
        PipelineParams { allow_interruptions: true, ..PipelineParams::default() },
    );

    let push_tx = task.push_sender();

    // Greet the caller: an LLMContextFrame is the coordinator's trigger, and with
    // no user turn in the context yet, `decide` returns the opening line.
    {
        let push_tx = push_tx.clone();
        let context = context.clone();
        task.add_on_pipeline_started(move |_frame| {
            let push_tx = push_tx.clone();
            let context = context.clone();
            Box::pin(async move {
                let _ = push_tx
                    .send((Frame::llm_context(context), FrameDirection::Downstream))
                    .await;
            })
        });
    }

    let observer = Arc::new(CallObserver {
        call_id,
        hangup,
        push_tx: push_tx.clone(),
        ended: AtomicBool::new(false),
        last_dtmf_frame: AtomicU64::new(u64::MAX),
    });

    // Either side can end the call: the caller hangs up (socket closes) or the
    // coordinator says goodbye (pipeline ends, we drop the socket).
    let socket_fut = transport.run_socket(socket, push_tx);
    let task_fut = async {
        if let Err(e) = task.run(system_clock(), Some(observer)).await {
            log::error!("[call={call_id}] pipeline error: {e}");
        }
    };
    tokio::pin!(socket_fut, task_fut);

    tokio::select! {
        _ = &mut socket_fut => {
            // Caller hung up. run_socket already pushed an EndFrame; give the
            // pipeline a moment to unwind cleanly.
            log::info!("[call={call_id}] caller disconnected");
            let _ = tokio::time::timeout(DRAIN_TIMEOUT, task_fut).await;
        }
        _ = &mut task_fut => {
            // Bot ended the call. Dropping socket_fut closes the WebSocket, and
            // <Connect><Stream> makes that hang up the caller.
            log::info!("[call={call_id}] pipeline ended, closing stream");
        }
    }

    log::info!("[call={call_id}] finished");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let deepgram_api_key =
        std::env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY env var not set");
    let twilio_auth_token = std::env::var("TWILIO_AUTH_TOKEN").ok();
    let public_host = std::env::var("PUBLIC_HOST").ok();

    let state = AppState { deepgram_api_key, twilio_auth_token, public_host };

    let app = Router::new()
        .route("/twiml", get(twiml_handler).post(twiml_handler))
        .route("/ws", get(ws_handler))
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{port}");

    log::info!("twilio_coordinator_server listening on {addr}");
    log::info!("  voice webhook : https://<public-host>/twiml");
    log::info!("  media stream  : wss://<public-host>/ws");
    log::info!(
        "Pipeline: Twilio µ-law → VAD → STT(Deepgram) → Coordinator (no agents) → TTS(Deepgram) → Twilio"
    );

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
