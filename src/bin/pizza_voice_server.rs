//! Dhara Pizza — voice ordering server.
//!
//! WebSocket voice agent with dhara conversation flow:
//!   greeting → menu → confirm → farewell
//!
//! Pipeline:
//!   WebSocketTransport.input()
//!     → RaviProcessor
//!     → SarvamStt
//!     → LLMUserAggregator
//!     → OpenAILLM (with dhara transition hook)
//!     → LLMAssistantAggregator
//!     → SarvamTts
//!     → WebSocketTransport.output()
//!
//! Environment variables:
//!   PORT             — listen port (default: 10000)
//!   SARVAM_API_KEY   — required (STT + TTS)
//!   OPENAI_API_KEY   — required (LLM)

use std::sync::{Arc, Mutex};

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
    system_clock, SileroVad, VadParams,
    PipelineParams, PipelineTask,
};
use rustvani::adapters::schemas::{FunctionSchema, ToolsSchema};
use rustvani::context::LLMContext;
use rustvani::dhara::{ContextStrategy, DharaManager, NodeConfig, TransitionResult};
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
use rustvani::services::llm::function_registry::FunctionRegistry;
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
}

// ---------------------------------------------------------------------------
// Order state — one per connection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct OrderItem {
    pizza: String,
    size: String,
    toppings: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct OrderState {
    items: Vec<OrderItem>,
}

impl OrderState {
    fn total_price(&self) -> f64 {
        self.items.iter().map(|item| {
            let base = match item.size.as_str() {
                "small" => 8.99,
                "medium" => 12.99,
                "large" => 15.99,
                _ => 12.99,
            };
            base + (item.toppings.len() as f64 * 1.50)
        }).sum()
    }

    fn summary(&self) -> String {
        if self.items.is_empty() {
            return "No items in order".to_string();
        }
        let items: Vec<String> = self.items.iter().enumerate().map(|(i, item)| {
            let toppings = if item.toppings.is_empty() {
                "no extra toppings".to_string()
            } else {
                item.toppings.join(", ")
            };
            format!("{}. {} {} with {}", i + 1, item.size, item.pizza, toppings)
        }).collect();
        format!("{}\nTotal: ${:.2}", items.join("\n"), self.total_price())
    }
}

// ---------------------------------------------------------------------------
// Tool schemas
// ---------------------------------------------------------------------------

fn browse_menu_schema() -> FunctionSchema {
    FunctionSchema::new("browse_menu", "Customer wants to see the menu and start ordering")
        .with_parameters(json!({
            "type": "object",
            "properties": {},
            "required": []
        }))
}

fn add_to_order_schema() -> FunctionSchema {
    FunctionSchema::new("add_to_order", "Add a pizza to the customer's order")
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "pizza": {
                    "type": "string",
                    "enum": ["Margherita", "Pepperoni", "Hawaiian", "BBQ Chicken", "Veggie Supreme"],
                    "description": "Type of pizza"
                },
                "size": {
                    "type": "string",
                    "enum": ["small", "medium", "large"],
                    "description": "Size of the pizza"
                },
                "toppings": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Extra toppings"
                }
            },
            "required": ["pizza", "size"]
        }))
}

fn remove_from_order_schema() -> FunctionSchema {
    FunctionSchema::new("remove_from_order", "Remove an item from the order by its number")
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "item_number": {
                    "type": "integer",
                    "description": "Item number to remove (1-based)"
                }
            },
            "required": ["item_number"]
        }))
}

fn view_order_schema() -> FunctionSchema {
    FunctionSchema::new("view_order", "Show the current order summary")
        .with_parameters(json!({
            "type": "object",
            "properties": {},
            "required": []
        }))
}

fn confirm_order_schema() -> FunctionSchema {
    FunctionSchema::new("confirm_order", "Customer wants to confirm and finalize the order")
        .with_parameters(json!({
            "type": "object",
            "properties": {},
            "required": []
        }))
}

fn modify_order_schema() -> FunctionSchema {
    FunctionSchema::new("modify_order", "Customer wants to go back and change the order")
        .with_parameters(json!({
            "type": "object",
            "properties": {},
            "required": []
        }))
}

fn place_order_schema() -> FunctionSchema {
    FunctionSchema::new("place_order", "Finalize and place the order for delivery")
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "delivery_address": {
                    "type": "string",
                    "description": "Delivery address"
                }
            },
            "required": ["delivery_address"]
        }))
}

// ---------------------------------------------------------------------------
// Node configs
// ---------------------------------------------------------------------------

fn greeting_node() -> NodeConfig {
    NodeConfig::new("greeting")
        .with_system_prompt(
            "You are a friendly pizza ordering assistant at Dhara Pizza. \
             You speak naturally and conversationally — keep it brief, warm, \
             and fun. You're taking voice orders so keep responses short."
        )
        .with_task_message(
            "Greet the customer warmly. Ask if they'd like to see the menu. \
             When they say yes, use the browse_menu tool."
        )
        .with_tools(ToolsSchema::new(vec![browse_menu_schema()]))
        .with_respond_immediately(true)
}

fn menu_node() -> NodeConfig {
    NodeConfig::new("menu")
        .with_system_prompt(
            "You are a pizza ordering assistant at Dhara Pizza. \
             Help the customer build their order. Keep responses short for voice. \
             Our menu: Margherita, Pepperoni, Hawaiian, BBQ Chicken, Veggie Supreme. \
             Sizes: small ($8.99), medium ($12.99), large ($15.99). \
             Extra toppings $1.50 each."
        )
        .with_task_message(
            "Help the customer order pizzas. Use add_to_order when they choose, \
             view_order to read back the order, remove_from_order if they change their mind. \
             When they're done ordering, use confirm_order. \
             Keep responses brief — this is a voice conversation."
        )
        .with_tools(ToolsSchema::new(vec![
            add_to_order_schema(),
            remove_from_order_schema(),
            view_order_schema(),
            confirm_order_schema(),
        ]))
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

fn confirm_node() -> NodeConfig {
    NodeConfig::new("confirm")
        .with_task_message(
            "Read back the complete order to the customer briefly. Ask them to \
             confirm or if they want to make changes. Use modify_order to go back, \
             or place_order with their delivery address to finalize."
        )
        .with_tools(ToolsSchema::new(vec![
            modify_order_schema(),
            place_order_schema(),
        ]))
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

fn farewell_node() -> NodeConfig {
    NodeConfig::new("farewell")
        .with_task_message(
            "The order has been placed! Thank the customer briefly, mention \
             delivery will be 30-45 minutes, and say goodbye. Keep it short and warm."
        )
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

// ---------------------------------------------------------------------------
// Handler factories
// ---------------------------------------------------------------------------

fn make_browse_menu_handler() -> rustvani::dhara::DharaHandlerFn {
    Arc::new(|_args: String| {
        Box::pin(async move {
            let menu = json!({
                "menu": [
                    {"name": "Margherita", "description": "Classic tomato, mozzarella, basil"},
                    {"name": "Pepperoni", "description": "Loaded with pepperoni"},
                    {"name": "Hawaiian", "description": "Ham and pineapple"},
                    {"name": "BBQ Chicken", "description": "BBQ sauce, chicken, red onion"},
                    {"name": "Veggie Supreme", "description": "Bell peppers, mushrooms, olives, onions"}
                ],
                "sizes": ["small ($8.99)", "medium ($12.99)", "large ($15.99)"],
                "extra_toppings": "$1.50 each"
            });
            TransitionResult::transition(menu.to_string(), "menu")
        })
    })
}

fn make_add_to_order_handler(order: Arc<Mutex<OrderState>>) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let order = order.clone();
        Box::pin(async move {
            let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let pizza = parsed["pizza"].as_str().unwrap_or("Margherita").to_string();
            let size = parsed["size"].as_str().unwrap_or("medium").to_string();
            let toppings: Vec<String> = parsed["toppings"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();

            let mut state = order.lock().unwrap();
            state.items.push(OrderItem {
                pizza: pizza.clone(),
                size: size.clone(),
                toppings: toppings.clone(),
            });

            let result = json!({
                "status": "added",
                "pizza": pizza,
                "size": size,
                "toppings": toppings,
                "order_summary": state.summary(),
            });
            TransitionResult::stay(result.to_string())
        })
    })
}

fn make_remove_from_order_handler(order: Arc<Mutex<OrderState>>) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let order = order.clone();
        Box::pin(async move {
            let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let num = parsed["item_number"].as_u64().unwrap_or(0) as usize;

            let mut state = order.lock().unwrap();
            let result = if num >= 1 && num <= state.items.len() {
                let removed = state.items.remove(num - 1);
                json!({
                    "status": "removed",
                    "removed": format!("{} {}", removed.size, removed.pizza),
                    "order_summary": state.summary(),
                })
            } else {
                json!({"status": "error", "error": format!("No item #{}", num)})
            };
            TransitionResult::stay(result.to_string())
        })
    })
}

fn make_view_order_handler(order: Arc<Mutex<OrderState>>) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |_args: String| {
        let order = order.clone();
        Box::pin(async move {
            let state = order.lock().unwrap();
            let result = json!({
                "order_summary": state.summary(),
                "item_count": state.items.len(),
                "total_price": format!("${:.2}", state.total_price()),
            });
            TransitionResult::stay(result.to_string())
        })
    })
}

fn make_confirm_order_handler(order: Arc<Mutex<OrderState>>) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |_args: String| {
        let order = order.clone();
        Box::pin(async move {
            let state = order.lock().unwrap();
            if state.items.is_empty() {
                return TransitionResult::stay(
                    json!({"error": "Cannot confirm — order is empty"}).to_string()
                );
            }
            let result = json!({
                "status": "ready_to_confirm",
                "order_summary": state.summary(),
            });
            TransitionResult::transition(result.to_string(), "confirm")
        })
    })
}

fn make_modify_order_handler() -> rustvani::dhara::DharaHandlerFn {
    Arc::new(|_args: String| {
        Box::pin(async move {
            TransitionResult::transition(
                json!({"status": "returning_to_menu"}).to_string(),
                "menu",
            )
        })
    })
}

fn make_place_order_handler(order: Arc<Mutex<OrderState>>) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let order = order.clone();
        Box::pin(async move {
            let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let address = parsed["delivery_address"].as_str().unwrap_or("unknown").to_string();

            let state = order.lock().unwrap();
            let order_id = format!("DP-{}", {
                use std::time::{SystemTime, UNIX_EPOCH};
                (SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos() % 99999) + 10000
            });

            let result = json!({
                "status": "order_placed",
                "order_id": order_id,
                "delivery_address": address,
                "order_summary": state.summary(),
                "estimated_delivery": "30-45 minutes",
            });
            TransitionResult::transition(result.to_string(), "farewell")
        })
    })
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
// Build dhara flow for a single connection
// ---------------------------------------------------------------------------

struct ConnectionFlow {
    context: Arc<Mutex<LLMContext>>,
    registry: Arc<Mutex<FunctionRegistry>>,
    transition_hook: rustvani::services::llm::openai::TransitionHook,
}

fn build_flow() -> ConnectionFlow {
    let order = Arc::new(Mutex::new(OrderState::default()));
    let context = Arc::new(Mutex::new(LLMContext::new(None)));
    let registry = Arc::new(Mutex::new(FunctionRegistry::new()));

    let mut dhara = DharaManager::new(context.clone(), registry.clone());

    dhara.register_node("greeting", greeting_node(), vec![
        ("browse_menu", make_browse_menu_handler()),
    ]);

    dhara.register_node("menu", menu_node(), vec![
        ("add_to_order", make_add_to_order_handler(order.clone())),
        ("remove_from_order", make_remove_from_order_handler(order.clone())),
        ("view_order", make_view_order_handler(order.clone())),
        ("confirm_order", make_confirm_order_handler(order.clone())),
    ]);

    dhara.register_node("confirm", confirm_node(), vec![
        ("modify_order", make_modify_order_handler()),
        ("place_order", make_place_order_handler(order.clone())),
    ]);

    dhara.register_node("farewell", farewell_node(), vec![]);

    dhara.set_initial_node("greeting");

    let hook = dhara.create_transition_hook();

    ConnectionFlow {
        context,
        registry,
        transition_hook: hook,
    }
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
    log::info!("[conn={}] connected — starting pizza flow", conn_id);

    // ---- VAD ----
    let vad_analyzer = match SileroVad::new(16_000) {
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

    // ---- Dhara flow (fresh per connection) ----
    let flow = build_flow();

    // ---- RAVI ----
    let ravi = RaviProcessor::new(RaviParams {
        context: Some(flow.context.clone()),
        ..RaviParams::default()
    });

    let ravi_observer: Arc<dyn BaseObserver> = Arc::new(
        RaviProcessor::create_observer(&ravi, RaviObserverParams::default()),
    );

    // ---- STT ----
    let stt = SarvamSttHandler::new(SarvamSttConfig {
        api_key:  app_state.sarvam_api_key.clone(),
        model:    "saaras:v3".to_string(),
        language: Some("en-IN".to_string()),
        mode:     Some("transcribe".to_string()),
        ..SarvamSttConfig::default()
    })
    .into_processor();

    // ---- Aggregators ----
    let user_agg = LLMUserAggregator::new(flow.context.clone());
    let assistant_agg = LLMAssistantAggregator::new(flow.context.clone());

    // ---- LLM with dhara ----
    let mut llm_handler = OpenAILLMHandler::with_shared_registry(
        OpenAILLMConfig {
            api_key: app_state.openai_api_key.clone(),
            model:   "gpt-4o-mini".to_string(),
            max_tool_rounds: 10,
            ..OpenAILLMConfig::default()
        },
        flow.registry.clone(),
    );
    llm_handler.set_transition_hook(flow.transition_hook);
    let llm = llm_handler.into_processor();

    // ---- TTS ----
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

    let app_state = AppState { sarvam_api_key, openai_api_key };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "10000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    log::info!("🍕 Dhara Pizza voice server on ws://{}/ws", addr);
    log::info!("Flow: greeting → menu → confirm → farewell");
    log::info!("Tools: browse_menu, add_to_order, remove_from_order, view_order, confirm_order, modify_order, place_order");

    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    axum::serve(listener, app).await.unwrap();
}
