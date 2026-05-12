//! rustvani RAVI WebSocket voice agent server — with tool calling.
//!
//! Demonstrates function calling with a `get_weather` tool.
//! When the user asks about weather, the LLM calls the tool, gets the result,
//! and speaks the answer.
//!
//! Pipeline:
//!   WebSocketTransport.input()
//!     → RaviProcessor
//!     → SarvamStt
//!     → LLMUserAggregator
//!     → OpenAILLM              ← calls get_weather when needed
//!     → LLMAssistantAggregator
//!     → SarvamTts
//!     → WebSocketTransport.output()

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    Router,
    extract::{State, WebSocketUpgrade, ws::WebSocket},
    response::IntoResponse,
    routing::get,
};
use serde_json::json;
use tower_http::cors::CorsLayer;

use rustvani::{
    shared_context, system_clock, SileroVadNative, VadParams,
    PipelineParams, PipelineTask, FunctionRegistry,
};
use rustvani::adapters::schemas::{FunctionSchema, ToolsSchema};
use rustvani::context::shared_context_with_tools;
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
    SarvamTtsConfig, SarvamTtsHandler,
};
use rustvani::transport::websocket::{WebSocketParams, WebSocketTransport};
use rustvani::transport::TransportParams;

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
    sarvam_api_key: String,
    openai_api_key: String,
    system_prompt:  String,
}

// ---------------------------------------------------------------------------
// Tool: get_weather
// ---------------------------------------------------------------------------

/// Fake weather lookup — returns hardcoded data for demo purposes.
/// In production this would call a real weather API.
async fn get_weather(args_json: String) -> String {
    // Parse the arguments
    let args: serde_json::Value = serde_json::from_str(&args_json).unwrap_or_default();
    let city = args["city"].as_str().unwrap_or("unknown");

    log::info!("[tool] get_weather called for city={}", city);

    // Simulate a tiny delay like a real API would have
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Return fake weather data
    json!({
        "city": city,
        "temperature_celsius": 31,
        "condition": "partly cloudy",
        "humidity_percent": 78,
        "wind_kph": 12,
    })
    .to_string()
}

/// Build the FunctionSchema for get_weather.
fn weather_tool_schema() -> FunctionSchema {
    FunctionSchema::new("get_weather", "Get the current weather for a city")
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Kochi' or 'New Delhi'"
                }
            },
            "required": ["city"]
        }))
}

/// Build the function registry with all tools.
fn build_registry() -> FunctionRegistry {
    let mut registry = FunctionRegistry::new();
    registry.register("get_weather", get_weather);
    registry
}

/// Build the tools schema (tells the LLM what tools are available).
fn build_tools_schema() -> ToolsSchema {
    ToolsSchema::new(vec![weather_tool_schema()])
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
// WebSocket upgrade handler
// ---------------------------------------------------------------------------

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_connection(socket, state))
}

async fn handle_connection(socket: WebSocket, app_state: AppState) {
    let conn_id = next_conn_id();
    log::info!("[conn={}] connected", conn_id);

    // ---- VAD ----
    let vad_analyzer = match SileroVadNative::new(16_000) {
        Ok(v) => Arc::new(v),
        Err(e) => {
            log::error!("[conn={}] VAD init failed: {}", conn_id, e);
            return;
        }
    };

    // ---- Transport ----
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
                vad_params:               VadParams {
                    confidence: 0.4,
                    min_volume: 0.1,
                    ..VadParams::default()
                },
                ..TransportParams::default()
            },
        },
    );

    // ---- Shared conversation context WITH TOOLS ----
    let context = shared_context_with_tools(
        Some(app_state.system_prompt.clone()),
        build_tools_schema(),
        None, // tool_choice = None → provider default ("auto")
    );

    // ---- Pipeline processors ----

    let ravi = RaviProcessor::new(RaviParams {
        context: Some(context.clone()),
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

    let user_agg = LLMUserAggregator::new(context.clone());

    // ---- LLM with function registry ----
    let llm = OpenAILLMHandler::with_registry(
        OpenAILLMConfig {
            api_key: app_state.openai_api_key.clone(),
            model:   "gpt-4o-mini".to_string(),
            ..OpenAILLMConfig::default()
        },
        build_registry(),
    )
    .into_processor();

    let assistant_agg = LLMAssistantAggregator::new(context.clone());

    let tts = match SarvamTtsHandler::new(SarvamTtsConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "bulbul:v3".to_string(),
        voice:    "aditya".to_string(),
        language: "en-IN".to_string(),
        ..SarvamTtsConfig::default()
    }) {
        Ok(t) => t.into_processor(),
        Err(e) => {
            log::error!("[conn={}] TTS init failed: {}", conn_id, e);
            return;
        }
    };

    // ---- Pipeline ----
    let task = PipelineTask::new(
        vec![
            transport.input(),
            ravi,
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            transport.output(),
        ],
        PipelineParams { allow_interruptions: true, ..PipelineParams::default() },
    );

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

    let sarvam_api_key = std::env::var("SARVAM_API_KEY")
        .expect("SARVAM_API_KEY env var not set");

    let openai_api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY env var not set");

    let system_prompt = std::env::var("SYSTEM_PROMPT").unwrap_or_else(|_| {
        "You are a helpful voice assistant with access to tools. \
         When the user asks about weather, use the get_weather tool. \
         Keep your answers concise and conversational."
            .to_string()
    });

    let app_state = AppState { sarvam_api_key, openai_api_key, system_prompt };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "10000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    log::info!("rustvani RAVI voice agent listening on ws://{}/ws", addr);
    log::info!("Pipeline: audio → VAD → RAVI → STT → LLM(+tools) → TTS → audio");
    log::info!("Tools: get_weather");

    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    axum::serve(listener, app).await.unwrap();
}
