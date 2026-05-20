//! Dhara Pizza v2 — voice ordering server (Neon-backed), new Dhara JSON API.
//!
//! Flow definition: dhara/pizza_order/dhara.json
//! Handler logic:   dhara/pizza_order/functions.rs  (included below)
//!
//! Pipeline:
//!   WebSocketTransport.input()
//!     → RaviProcessor
//!     → SarvamStt
//!     → LLMUserAggregator
//!     → OpenAILLM (with dhara transition hook)
//!     → LLMAssistantAggregator
//!     → DeepgramTts
//!     → WebSocketTransport.output()
//!
//! Environment variables:
//!   PORT             — listen port (default: 10000)
//!   DATABASE_URL     — required (Neon connection string)
//!   SARVAM_API_KEY   — required (STT)
//!   OPENAI_API_KEY   — required (LLM)
//!   DEEPGRAM_API_KEY — required (TTS)

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    Router,
    extract::{State, WebSocketUpgrade, ws::WebSocket},
    response::IntoResponse,
    routing::get,
};
use tower_http::cors::CorsLayer;

use rustvani::{
    system_clock, SileroVadNative, VadParams,
    PipelineParams, PipelineTask,
};
use rustvani::context::LLMContext;
use rustvani::dhara::{Dhara, DharaContext, DharaFunctionRegistry};
use rustvani::observer::{BaseObserver, FrameProcessed, FramePushed};
use rustvani::processors::{
    llm_assistant_aggregator::LLMAssistantAggregator,
    llm_user_aggregator::LLMUserAggregator,
};
use rustvani::ravi::{
    RaviObserverParams,
    processor::{RaviParams, RaviProcessor},
};
use rustvani::services::{
    OpenAILLMConfig, OpenAILLMHandler,
    SarvamSttConfig, SarvamSttHandler,
    DeepgramTtsConfig, DeepgramTtsHandler,
};
use rustvani::services::llm::function_registry::FunctionRegistry;
use rustvani::services::llm::openai::TransitionHook;
use rustvani::transport::websocket::{WebSocketParams, WebSocketTransport};
use rustvani::transport::TransportParams;

// Domain state, OrderWriter, DharaPizzaState, register_handlers — all in functions.rs
include!("../../dhara/pizza_order/functions.rs");

// ---------------------------------------------------------------------------
// Embedded flow definition
// ---------------------------------------------------------------------------

const FLOW_JSON: &str = include_str!("../../dhara/pizza_order/dhara.json");

// ---------------------------------------------------------------------------
// Connection ID counter
// ---------------------------------------------------------------------------

static CONN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_conn_id() -> u64 {
    CONN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Shared app state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    database_url:     String,
    sarvam_api_key:   String,
    openai_api_key:   String,
    deepgram_api_key: String,
}

// ---------------------------------------------------------------------------
// ConnectionFlow
// ---------------------------------------------------------------------------

struct ConnectionFlow {
    context:         Arc<Mutex<LLMContext>>,
    registry:        Arc<Mutex<FunctionRegistry>>,
    transition_hook: TransitionHook,
    dhara_ctx:       DharaContext,
}

/// Build the Dhara flow for a single connection.
///
/// Parses the embedded JSON, registers all handlers, then calls `dhara.build()`
/// to get context, registry, and transition hook ready for the pipeline.
fn build_flow(order_writer: Arc<OrderWriter>, conn_id: u64) -> ConnectionFlow {
    let order = Arc::new(Mutex::new(OrderState::default()));

    let state: Arc<dyn std::any::Any + Send + Sync> = Arc::new(DharaPizzaState {
        order:  order.clone(),
        writer: order_writer,
    });

    let dhara = Dhara::from_json(FLOW_JSON)
        .expect("pizza_order dhara.json is invalid");

    let mut handlers = DharaFunctionRegistry::new();
    register_handlers(&mut handlers);

    let built = dhara.build(&handlers, state, conn_id)
        .expect("pizza_order handler registry is incomplete");

    ConnectionFlow {
        context:         built.context,
        registry:        built.llm_registry,
        transition_hook: built.hook,
        dhara_ctx:       built.dhara_ctx,
    }
}

// ---------------------------------------------------------------------------
// NullObserver
// ---------------------------------------------------------------------------

struct NullObserver;

#[async_trait]
impl BaseObserver for NullObserver {
    async fn on_process_frame(&self, _: FrameProcessed) {}
    async fn on_push_frame(&self, _: FramePushed) {}
}

// ---------------------------------------------------------------------------
// WebSocket handler
// ---------------------------------------------------------------------------

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_connection(socket, state))
}

async fn handle_connection(socket: WebSocket, app_state: AppState) {
    let conn_id = next_conn_id();
    log::info!("[conn={}] connected — starting pizza flow v2", conn_id);

    let vad_analyzer = match SileroVadNative::new(16_000) {
        Ok(v)  => Arc::new(v),
        Err(e) => { log::error!("[conn={}] VAD init failed: {}", conn_id, e); return; }
    };

    let transport = WebSocketTransport::new(
        &format!("WsTransport-{}", conn_id),
        WebSocketParams {
            transport: TransportParams {
                audio_in_enabled:         true,
                audio_in_sample_rate:     Some(16_000),
                audio_in_channels:        1,
                audio_in_passthrough:     true,
                audio_in_stream_on_start: true,
                vad_analyzer:             Some(vad_analyzer),
                vad_params:               VadParams { confidence: 0.4, min_volume: 0.1, ..VadParams::default() },
                ..TransportParams::default()
            },
        },
    );

    let order_writer = Arc::new(OrderWriter::new());
    if let Err(e) = order_writer.init(&app_state.database_url).await {
        log::error!("[conn={}] OrderWriter init failed: {}", conn_id, e);
        return;
    }

    let flow = build_flow(order_writer, conn_id);

    let ravi = RaviProcessor::new(RaviParams {
        context: Some(flow.context.clone()),
        ..RaviParams::default()
    });
    let ravi_observer: Arc<dyn BaseObserver> = Arc::new(
        RaviProcessor::create_observer(&ravi, RaviObserverParams::default()),
    );

    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    let user_agg      = LLMUserAggregator::new(flow.context.clone());
    let assistant_agg = LLMAssistantAggregator::new(flow.context.clone());

    let llm_handler = OpenAILLMHandler::with_shared_registry(
        OpenAILLMConfig {
            api_key:         app_state.openai_api_key.clone(),
            model:           "gpt-4o-mini".to_string(),
            max_tool_rounds: 10,
            ..OpenAILLMConfig::default()
        },
        flow.registry.clone(),
    );
    llm_handler.set_transition_hook(flow.transition_hook);
    let llm = llm_handler.into_processor();

    let tts = match DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: app_state.deepgram_api_key.clone(),
        ..DeepgramTtsConfig::default()
    }) {
        Ok(t)  => t.into_processor(),
        Err(e) => { log::error!("[conn={}] TTS init failed: {}", conn_id, e); return; }
    };

    let task = PipelineTask::new(
        vec![transport.input(), ravi, stt, user_agg, llm, assistant_agg, tts, transport.output()],
        PipelineParams { allow_interruptions: true, ..PipelineParams::default() },
    );

    flow.dhara_ctx.set_push_sender(task.push_sender());
    let push_tx = task.push_sender();

    tokio::join!(
        async { task.run(system_clock(), Some(ravi_observer)).await.ok(); },
        transport.run_socket(socket, push_tx),
    );

    log::info!("[conn={}] disconnected", conn_id);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let app_state = AppState {
        database_url:     std::env::var("DATABASE_URL").expect("DATABASE_URL not set"),
        sarvam_api_key:   std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY not set"),
        openai_api_key:   std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
        deepgram_api_key: std::env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY not set"),
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "10000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    log::info!("Dhara Pizza v2 on ws://{}/ws  (flow: greeting → menu → confirm → farewell)", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    axum::serve(listener, app).await.unwrap();
}
