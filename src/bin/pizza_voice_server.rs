//! Dhara Pizza — voice ordering server (Neon-backed).
//!
//! WebSocket voice agent with dhara conversation flow:
//!   greeting → menu → confirm → farewell
//!
//! Pipeline:
//!   WebSocketTransport.input()
//!     → RaviProcessor
//!     → SarvamStt
//!     → LLMUserAggregator
//!     → OpenAILLM (with dhara transition hook + NeonPostgresTool)
//!     → LLMAssistantAggregator
//!     → SarvamTts
//!     → WebSocketTransport.output()
//!
//! Environment variables:
//!   PORT             — listen port (default: 10000)
//!   DATABASE_URL     — required (Neon connection string)
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
use tokio::sync::Mutex as AsyncMutex;
use native_tls::TlsConnector;
use postgres_native_tls::MakeTlsConnector;
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
use rustvani::tools::{BuiltinTool, NeonPostgresTool};
use rustvani::tools::postgres::NeonPostgresConfig;
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
    database_url:    String,
    sarvam_api_key:  String,
    openai_api_key:  String,
}

// ---------------------------------------------------------------------------
// Order state — one per connection (in-memory until confirmed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct OrderItem {
    pizza:    String,
    size:     String,
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
                "small"  => 8.99,
                "medium" => 12.99,
                "large"  => 15.99,
                _        => 12.99,
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
// OrderWriter — dedicated Neon connection for writing confirmed orders.
//
// Intentionally separate from NeonPostgresTool so write transactions don't
// interfere with LLM query traffic, and so the write path is simple and
// auditable without touching the tool abstraction.
// ---------------------------------------------------------------------------

struct OrderWriter {
    client: AsyncMutex<Option<tokio_postgres::Client>>,
}

impl OrderWriter {
    fn new() -> Self {
        Self {
            client: AsyncMutex::new(None),
        }
    }

    /// Connect to Neon. Must be called before `write_order`.
    async fn init(&self, db_url: &str) -> Result<(), String> {
        let connector = TlsConnector::builder().build().map_err(|e| format!("TLS build: {}", e))?;
        let tls = MakeTlsConnector::new(connector);
        let (client, connection) = tokio_postgres::connect(db_url, tls)
            .await
            .map_err(|e| format!("OrderWriter: connect failed: {}", e))?;

        tokio::spawn(async move {
            if let Err(e) = connection.await {
                log::error!("OrderWriter: connection dropped: {}", e);
            }
        });

        *self.client.lock().await = Some(client);
        log::info!("OrderWriter: connected to Neon");
        Ok(())
    }

    /// Write a confirmed order inside a transaction.
    ///
    /// Returns the human-readable order ID (e.g. "DP-00042").
    async fn write_order(&self, address: &str, order: &OrderState) -> Result<String, String> {
        let mut guard = self.client.lock().await;
        let client = guard.as_mut()
            .ok_or_else(|| "OrderWriter not initialized".to_string())?;

        let total = order.total_price();

        // Begin transaction
        let tx = client.transaction().await
            .map_err(|e| format!("Transaction start failed: {}", e))?;

        // INSERT orders row
        let order_row = tx.query_one(
            "INSERT INTO orders (delivery_address, status, payment_completed, total_price) \
             VALUES ($1, 'confirmed', false, $2) RETURNING id",
            &[&address, &total],
        ).await.map_err(|e| format!("Insert order failed: {}", e))?;

        let order_id: i32 = order_row.get(0);

        // INSERT one order_items row per item
        for item in &order.items {
            let line_price = match item.size.as_str() {
                "small"  => 8.99f64,
                "medium" => 12.99,
                "large"  => 15.99,
                _        => 12.99,
            } + (item.toppings.len() as f64 * 1.50);

            // Resolve pizza_id (best-effort — 0 if name not found)
            let pizza_id: i32 = tx.query_opt(
                "SELECT id FROM pizzas WHERE name = $1",
                &[&item.pizza],
            ).await
            .map_err(|e| format!("Pizza lookup failed: {}", e))?
            .map(|r| r.get::<_, i32>(0))
            .unwrap_or(0);

            tx.execute(
                "INSERT INTO order_items \
                    (order_id, pizza_id, pizza_name, size, extra_toppings, line_price) \
                 VALUES ($1, $2, $3, $4, $5, $6)",
                &[
                    &order_id,
                    &pizza_id,
                    &item.pizza,
                    &item.size,
                    &item.toppings,   // Vec<String> → TEXT[]
                    &line_price,
                ],
            ).await.map_err(|e| format!("Insert order item failed: {}", e))?;
        }

        tx.commit().await
            .map_err(|e| format!("Commit failed: {}", e))?;

        log::info!("OrderWriter: committed order {} ({})", order_id, address);
        Ok(format!("DP-{:05}", order_id))
    }
}

// ---------------------------------------------------------------------------
// Tool schemas — pizza-specific
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
    // Pizza name is no longer a hardcoded enum — LLM must use the name
    // exactly as returned by pg_query on the pizzas table.
    FunctionSchema::new("add_to_order", "Add a pizza to the customer's order")
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "pizza": {
                    "type": "string",
                    "description": "Pizza name exactly as returned from the pizzas table"
                },
                "size": {
                    "type": "string",
                    "enum": ["small", "medium", "large"],
                    "description": "Size of the pizza"
                },
                "toppings": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Extra topping names, exactly as in the toppings table"
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
    FunctionSchema::new("view_order", "Show the current in-memory order summary")
        .with_parameters(json!({
            "type": "object",
            "properties": {},
            "required": []
        }))
}

fn confirm_order_schema() -> FunctionSchema {
    FunctionSchema::new("confirm_order", "Customer wants to review and confirm the order")
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
    FunctionSchema::new("place_order", "Finalize and place the order — writes to database")
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "delivery_address": {
                    "type": "string",
                    "description": "Full delivery address"
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
             Speak naturally and conversationally — brief, warm, and fun. \
             You are taking voice orders so keep responses short."
        )
        .with_task_message(
            "Greet the customer warmly. Ask if they'd like to see the menu. \
             When they say yes, use the browse_menu tool."
        )
        .with_tools(ToolsSchema::new(vec![browse_menu_schema()]))
        .with_respond_immediately(true)
}

fn menu_node(pg_schemas: Vec<FunctionSchema>) -> NodeConfig {
    // Merge pizza ordering tools with the pg query tools
    let mut tools = vec![
        add_to_order_schema(),
        remove_from_order_schema(),
        view_order_schema(),
        confirm_order_schema(),
    ];
    tools.extend(pg_schemas);

    NodeConfig::new("menu")
        .with_system_prompt(
            "You are a pizza ordering assistant at Dhara Pizza. \
             The menu lives in a Neon database — use pg_schema to understand the tables, \
             then pg_query to fetch pizza and topping information when the customer asks. \
             Always query the DB before answering questions about the menu, prices, \
             dietary requirements, or availability — never guess from training data. \
             Keep voice responses short. \
             Sizes: small, medium, large. Extra toppings $1.50 each."
        )
        .with_task_message(
            "Help the customer build their order. \
             Use pg_query to look up menu items, prices, and dietary info on demand. \
             Use add_to_order when they choose a pizza, view_order to read back the order, \
             remove_from_order if they change their mind. \
             When they are done ordering, use confirm_order. \
             Keep responses brief — this is a voice conversation."
        )
        .with_tools(ToolsSchema::new(tools))
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

fn confirm_node(pg_schemas: Vec<FunctionSchema>) -> NodeConfig {
    let mut tools = vec![
        view_order_schema(),
        modify_order_schema(),
        place_order_schema(),
    ];
    tools.extend(pg_schemas);

    NodeConfig::new("confirm")
        .with_task_message(
            "Read back the complete order summary to the customer briefly (use view_order). \
             Ask them to confirm or if they want to make changes. \
             Use modify_order to return to the menu, \
             or place_order with their delivery address to finalise. \
             You can use pg_query if you need to re-check any item details."
        )
        .with_tools(ToolsSchema::new(tools))
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

fn farewell_node() -> NodeConfig {
    NodeConfig::new("farewell")
        .with_task_message(
            "The order has been placed and saved. Thank the customer briefly, \
             mention delivery in 30-45 minutes, and say goodbye warmly. Keep it short."
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
            // Signal the LLM to transition to menu node.
            // The LLM will use pg_query there to fetch the live menu.
            let result = json!({
                "status": "menu_ready",
                "instruction": "Use pg_query on the pizzas and pizza_sizes tables to show the customer the menu."
            });
            TransitionResult::transition(result.to_string(), "menu")
        })
    })
}

fn make_add_to_order_handler(order: Arc<Mutex<OrderState>>) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let order = order.clone();
        Box::pin(async move {
            let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let pizza = parsed["pizza"].as_str().unwrap_or("Unknown").to_string();
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

fn make_place_order_handler(
    order: Arc<Mutex<OrderState>>,
    writer: Arc<OrderWriter>,
) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let order = order.clone();
        let writer = writer.clone();
        Box::pin(async move {
            let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let address = match parsed["delivery_address"].as_str() {
                Some(a) if !a.trim().is_empty() => a.to_string(),
                _ => {
                    return TransitionResult::stay(
                        json!({"error": "delivery_address is required to place the order"}).to_string()
                    );
                }
            };

            let order_snapshot = order.lock().unwrap().clone();

            if order_snapshot.items.is_empty() {
                return TransitionResult::stay(
                    json!({"error": "Cannot place an empty order"}).to_string()
                );
            }

            match writer.write_order(&address, &order_snapshot).await {
                Ok(order_id) => {
                    let result = json!({
                        "status": "order_placed",
                        "order_id": order_id,
                        "delivery_address": address,
                        "order_summary": order_snapshot.summary(),
                        "estimated_delivery": "30-45 minutes",
                        "payment_completed": false,
                    });
                    TransitionResult::transition(result.to_string(), "farewell")
                }
                Err(e) => {
                    log::error!("OrderWriter: write_order failed: {}", e);
                    TransitionResult::stay(
                        json!({
                            "error": "Failed to save your order. Please try again.",
                            "detail": e,
                        }).to_string()
                    )
                }
            }
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
// ConnectionFlow
// ---------------------------------------------------------------------------

struct ConnectionFlow {
    context:          Arc<Mutex<LLMContext>>,
    registry:         Arc<Mutex<FunctionRegistry>>,
    transition_hook:  rustvani::services::llm::openai::TransitionHook,
}

/// Build the Dhara flow for a single connection.
///
/// `pg_tool`     — NeonPostgresTool; schemas go into menu/confirm nodes, and its
///                 register_all() is re-run after every Dhara node transition so
///                 that pg_* handlers survive the registry swap.
/// `order_writer` — shared writer for the `place_order` handler.
fn build_flow(
    pg_tool: Arc<NeonPostgresTool>,
    order_writer: Arc<OrderWriter>,
) -> ConnectionFlow {
    let pg_schemas = pg_tool.tool_schemas();

    let order    = Arc::new(Mutex::new(OrderState::default()));
    let context  = Arc::new(Mutex::new(LLMContext::new(None)));
    let registry = Arc::new(Mutex::new(FunctionRegistry::new()));

    let mut dhara = DharaManager::new(context.clone(), registry.clone());

    // greeting — no DB tools
    dhara.register_node("greeting", greeting_node(), vec![
        ("browse_menu", make_browse_menu_handler()),
    ]);

    // menu — pizza tools + pg tools (schemas only; handlers re-injected via hook)
    dhara.register_node("menu", menu_node(pg_schemas.clone()), vec![
        ("add_to_order",      make_add_to_order_handler(order.clone())),
        ("remove_from_order", make_remove_from_order_handler(order.clone())),
        ("view_order",        make_view_order_handler(order.clone())),
        ("confirm_order",     make_confirm_order_handler(order.clone())),
    ]);

    // confirm — view/modify/place + pg tools (schemas only; handlers re-injected via hook)
    dhara.register_node("confirm", confirm_node(pg_schemas), vec![
        ("view_order",  make_view_order_handler(order.clone())),
        ("modify_order", make_modify_order_handler()),
        ("place_order", make_place_order_handler(order.clone(), order_writer)),
    ]);

    dhara.register_node_no_tools("farewell", farewell_node());

    dhara.set_initial_node("greeting");

    // Dhara swaps the shared registry on every node transition, wiping any
    // handlers not registered in that node's list. We wrap the hook to
    // re-inject pg_* handlers after every swap so they are always present.
    let dhara_hook = dhara.create_transition_hook();
    let pg_for_hook = pg_tool.clone();
    let reg_for_hook = registry.clone();
    let transition_hook: rustvani::services::llm::openai::TransitionHook =
        Arc::new(move |ctx| {
            dhara_hook(ctx);
            pg_for_hook.register_all(&mut reg_for_hook.lock().unwrap());
        });

    ConnectionFlow { context, registry, transition_hook }
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

    // ---- NeonPostgresTool (LLM query path) ----
    // Cheap to construct — actual Neon connection happens on StartFrame
    // when on_start() is called by the pipeline.
    let pg_tool = Arc::new(
        NeonPostgresTool::new(
            NeonPostgresConfig::new(&app_state.database_url)
                .with_statement_timeout_ms(8_000),
        )
    );
    // ---- OrderWriter (write path) ----
    let order_writer = Arc::new(OrderWriter::new());
    if let Err(e) = order_writer.init(&app_state.database_url).await {
        log::error!("[conn={}] OrderWriter init failed: {}", conn_id, e);
        return;
    }

    // ---- Dhara flow (fresh per connection) ----
    // pg_tool is passed into build_flow so the transition hook can re-register
    // pg_* handlers after every Dhara node swap.
    let flow = build_flow(pg_tool.clone(), order_writer);

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
    let user_agg      = LLMUserAggregator::new(flow.context.clone());
    let assistant_agg = LLMAssistantAggregator::new(flow.context.clone());

    // ---- LLM with Dhara + NeonPostgresTool ----
    let mut llm_handler = OpenAILLMHandler::with_shared_registry(
        OpenAILLMConfig {
            api_key:         app_state.openai_api_key.clone(),
            model:           "gpt-4o-mini".to_string(),
            max_tool_rounds: 10,
            ..OpenAILLMConfig::default()
        },
        flow.registry.clone(),
    );
    llm_handler.add_tool(pg_tool);          // registers pg_* handlers + lifecycle
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

    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL env var not set");

    let sarvam_api_key = std::env::var("SARVAM_API_KEY")
        .expect("SARVAM_API_KEY env var not set");

    let openai_api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY env var not set");

    let app_state = AppState { database_url, sarvam_api_key, openai_api_key };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "10000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    log::info!("🍕 Dhara Pizza voice server on ws://{}/ws", addr);
    log::info!("Flow: greeting → menu → confirm → farewell");
    log::info!("DB tools: pg_schema, pg_query, pg_refine (menu + confirm nodes)");
    log::info!("Write: OrderWriter (place_order → orders + order_items)");

    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    axum::serve(listener, app).await.unwrap();
}
