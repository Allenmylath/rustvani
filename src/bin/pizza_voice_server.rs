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
//!     → OpenAILLM (with dhara transition hook + fetch_menu data tool)
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

use std::error::Error;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    Router,
    extract::{State, WebSocketUpgrade, ws::WebSocket},
    response::IntoResponse,
    routing::get,
};
use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;
use native_tls::TlsConnector;
use postgres_native_tls::MakeTlsConnector;
use tower_http::cors::CorsLayer;

use rustvani::{
    system_clock, SileroVadNative, VadParams,
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
    DeepgramTtsConfig, DeepgramTtsHandler,
};
use rustvani::frames::{Frame, FrameDirection};
use rustvani::ravi::models as ravi_models;
use rustvani::services::llm::function_registry::{FunctionRegistry, ToolCallOutput};
use rustvani::transport::websocket::{WebSocketParams, WebSocketTransport};
use rustvani::transport::TransportParams;
use rustvani::turn::SmartTurnConfig;

// ---------------------------------------------------------------------------
// Deferred push sender — set after PipelineTask::new(), used by handlers
// ---------------------------------------------------------------------------

type PushSender = tokio::sync::mpsc::Sender<(Frame, FrameDirection)>;
type DeferredPush = Arc<std::sync::OnceLock<PushSender>>;

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
// Order state — one per connection (in-memory until confirmed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct OrderItem {
    pizza_id: i32,
    pizza:    String,
    size:     String,
    toppings: Vec<String>,
    line_price: f64,
}

#[derive(Debug, Clone, Default)]
struct OrderState {
    items: Vec<OrderItem>,
}

impl OrderState {
    fn total_price(&self) -> f64 {
        self.items.iter().map(|item| item.line_price).sum()
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
            format!(
                "{}. {} {} with {} — ${:.2}",
                i + 1, item.size, item.pizza, toppings, item.line_price
            )
        }).collect();
        format!("{}\nTotal: ${:.2}", items.join("\n"), self.total_price())
    }

    /// Build structured cart payload for the client UI via Ravi server-message.
    fn cart_payload(&self) -> Value {
        json!({
            "type": "cart-updated",
            "items": self.items.iter().map(|item| json!({
                "pizza_id": item.pizza_id,
                "pizza": item.pizza,
                "size": item.size,
                "toppings": item.toppings,
                "line_price": item.line_price,
            })).collect::<Vec<_>>(),
            "total_price": self.total_price(),
            "item_count": self.items.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// OrderWriter — dedicated Neon connection for all DB operations.
//
// Handles menu reads, validation reads (pizza/size/topping lookups),
// and confirmed order writes (place_order).
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

    /// Connect to Neon. Must be called before any reads or writes.
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

    // -----------------------------------------------------------------------
    // Menu — full menu fetch for the data tool
    // -----------------------------------------------------------------------

    /// Fetch the complete menu: all available pizzas with sizes/prices,
    /// plus all available toppings with prices.
    ///
    /// Returns structured JSON ready for the UI.
    async fn fetch_menu(&self) -> Result<Value, String> {
        let guard = self.client.lock().await;
        let client = guard.as_ref()
            .ok_or_else(|| "OrderWriter not initialized".to_string())?;

        // Pizzas with sizes (grouped by pizza name since ORDER BY groups them)
        let rows = client.query(
            "SELECT p.name, p.description, p.is_vegetarian, p.image_url, ps.size, ps.price \
             FROM pizzas p \
             JOIN pizza_sizes ps ON p.id = ps.pizza_id \
             WHERE p.is_available = true \
             ORDER BY p.name, ps.price",
            &[],
        ).await.map_err(|e| format!("Menu query failed: {}", e))?;

        let mut pizzas: Vec<Value> = Vec::new();

        for row in &rows {
            let name: String = row.get(0);
            let description: Option<String> = row.get(1);
            let vegetarian: bool = row.get(2);
            let image_url: Option<String> = row.get(3);
            let size: String = row.get(4);
            let price: f64 = row.get(5);

            let size_entry = json!({"size": size, "price": price});

            // If this is the same pizza as the last entry, append the size
            let same_pizza = pizzas.last()
                .and_then(|p| p["name"].as_str())
                .map(|n| n == name)
                .unwrap_or(false);

            if same_pizza {
                if let Some(last) = pizzas.last_mut() {
                    if let Some(obj) = last.as_object_mut() {
                        if let Some(Value::Array(sizes)) = obj.get_mut("sizes") {
                            sizes.push(size_entry);
                        }
                    }
                }
            } else {
                pizzas.push(json!({
                    "name": name,
                    "description": description.unwrap_or_default(),
                    "vegetarian": vegetarian,
                    "image_url": image_url,
                    "sizes": [size_entry],
                }));
            }
        }

        // Toppings
        let topping_rows = client.query(
            "SELECT name, price_per_unit FROM toppings WHERE is_available = true ORDER BY name",
            &[],
        ).await.map_err(|e| format!("Toppings query failed: {}", e))?;

        let toppings: Vec<Value> = topping_rows.iter().map(|r| {
            json!({
                "name": r.get::<_, String>(0),
                "price": r.get::<_, f64>(1),
            })
        }).collect();

        Ok(json!({
            "pizzas": pizzas,
            "toppings": toppings,
        }))
    }

    // -----------------------------------------------------------------------
    // Read methods — used at add_to_order time for validation
    // -----------------------------------------------------------------------

    /// Fetch detailed info for a single pizza (description, vegetarian, sizes).
    ///
    /// Used by `fetch_item_detail` to push structured data to the client.
    async fn get_pizza_detail(&self, name: &str) -> Result<Option<Value>, String> {
        let (pizza_id, canonical_name) = match self.lookup_pizza(name).await? {
            Some(pair) => pair,
            None => return Ok(None),
        };

        // Get description, vegetarian flag, and image_url (separate lock scope)
        let (description, vegetarian, image_url) = {
            let guard = self.client.lock().await;
            let client = guard.as_ref()
                .ok_or_else(|| "OrderWriter not initialized".to_string())?;

            let row = client.query_opt(
                "SELECT description, is_vegetarian, image_url FROM pizzas WHERE id = $1",
                &[&pizza_id],
            ).await.map_err(|e| format!("Pizza detail query failed: {}", e))?;

            match row {
                Some(r) => (
                    r.get::<_, Option<String>>(0).unwrap_or_default(),
                    r.get::<_, bool>(1),
                    r.get::<_, Option<String>>(2),
                ),
                None => return Ok(None),
            }
        }; // guard dropped — safe to call get_sizes_for_pizza

        let sizes = self.get_sizes_for_pizza(pizza_id).await?;

        Ok(Some(json!({
            "name": canonical_name,
            "description": description,
            "vegetarian": vegetarian,
            "image_url": image_url,
            "sizes": sizes.iter().map(|(s, p)| json!({"size": s, "price": p})).collect::<Vec<_>>(),
        })))
    }

    /// Lookup a pizza by name (case-insensitive, fuzzy via ILIKE).
    ///
    /// Returns (id, canonical_name) or None if not found.
    async fn lookup_pizza(&self, name: &str) -> Result<Option<(i32, String)>, String> {
        let guard = self.client.lock().await;
        let client = guard.as_ref()
            .ok_or_else(|| "OrderWriter not initialized".to_string())?;

        let pattern = format!("%{}%", name.trim());
        let row = client.query_opt(
            "SELECT id, name FROM pizzas WHERE LOWER(name) ILIKE LOWER($1) AND is_available = true LIMIT 1",
            &[&pattern],
        ).await.map_err(|e| format!("Pizza lookup failed: {}", e))?;

        Ok(row.map(|r| (r.get::<_, i32>(0), r.get::<_, String>(1))))
    }

    /// Get all available pizza names (for suggestions when lookup fails).
    async fn list_pizza_names(&self) -> Result<Vec<String>, String> {
        let guard = self.client.lock().await;
        let client = guard.as_ref()
            .ok_or_else(|| "OrderWriter not initialized".to_string())?;

        let rows = client.query("SELECT name FROM pizzas WHERE is_available = true ORDER BY name", &[])
            .await
            .map_err(|e| format!("List pizzas failed: {}", e))?;

        Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
    }

    /// Get available sizes and prices for a pizza.
    ///
    /// Returns vec of (size_name, price).
    async fn get_sizes_for_pizza(&self, pizza_id: i32) -> Result<Vec<(String, f64)>, String> {
        let guard = self.client.lock().await;
        let client = guard.as_ref()
            .ok_or_else(|| "OrderWriter not initialized".to_string())?;

        let rows = client.query(
            "SELECT size, price FROM pizza_sizes WHERE pizza_id = $1 ORDER BY price",
            &[&pizza_id],
        ).await.map_err(|e| format!("Size lookup failed: {}", e))?;

        Ok(rows.iter().map(|r| {
            (r.get::<_, String>(0), r.get::<_, f64>(1))
        }).collect())
    }

    /// Validate topping names against the toppings table.
    ///
    /// Returns (valid_toppings, invalid_toppings) with canonical names.
    async fn validate_toppings(
        &self,
        topping_names: &[String],
    ) -> Result<(Vec<(String, f64)>, Vec<String>), String> {
        if topping_names.is_empty() {
            return Ok((vec![], vec![]));
        }

        let guard = self.client.lock().await;
        let client = guard.as_ref()
            .ok_or_else(|| "OrderWriter not initialized".to_string())?;

        let mut valid = Vec::new();
        let mut invalid = Vec::new();

        for name in topping_names {
            let pattern = format!("%{}%", name.trim());
            let row = client.query_opt(
                "SELECT name, price_per_unit FROM toppings WHERE LOWER(name) ILIKE LOWER($1) AND is_available = true LIMIT 1",
                &[&pattern],
            ).await.map_err(|e| format!("Topping lookup failed: {}", e))?;

            match row {
                Some(r) => valid.push((r.get::<_, String>(0), r.get::<_, f64>(1))),
                None => invalid.push(name.clone()),
            }
        }

        Ok((valid, invalid))
    }

    /// Get all available topping names (for suggestions).
    async fn list_topping_names(&self) -> Result<Vec<String>, String> {
        let guard = self.client.lock().await;
        let client = guard.as_ref()
            .ok_or_else(|| "OrderWriter not initialized".to_string())?;

        let rows = client.query("SELECT name FROM toppings WHERE is_available = true ORDER BY name", &[])
            .await
            .map_err(|e| format!("List toppings failed: {}", e))?;

        Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
    }

    // -----------------------------------------------------------------------
    // Write methods — used at place_order time
    // -----------------------------------------------------------------------

    /// Write a confirmed order inside a transaction.
    ///
    /// All pizza_ids and prices are pre-validated — no lookups needed.
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

        // INSERT one order_items row per item — pizza_id is pre-validated
        for item in &order.items {
            tx.execute(
                "INSERT INTO order_items \
                    (order_id, pizza_id, pizza_name, size, extra_toppings, line_price) \
                 VALUES ($1, $2, $3, $4, $5, $6)",
                &[
                    &order_id,
                    &item.pizza_id,
                    &item.pizza,
                    &item.size,
                    &item.toppings,
                    &item.line_price,
                ],
            ).await.map_err(|e| {
                // Surface the full postgres error detail
                let detail = e.source()
                    .map(|s| format!(" (detail: {})", s))
                    .unwrap_or_default();
                format!("Insert order item failed: {}{}", e, detail)
            })?;
        }

        tx.commit().await
            .map_err(|e| format!("Commit failed: {}", e))?;

        log::info!("OrderWriter: committed order {} ({})", order_id, address);
        Ok(format!("DP-{:05}", order_id))
    }
}

// ---------------------------------------------------------------------------
// Tool schemas
// ---------------------------------------------------------------------------

fn fetch_menu_schema() -> FunctionSchema {
    FunctionSchema::new(
        "fetch_menu",
        "Display the full pizza menu on the customer's screen. \
         Shows all available pizzas with sizes and prices, plus available toppings. \
         Data goes directly to the UI — do NOT read the entire menu aloud."
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {},
        "required": []
    }))
}

fn browse_menu_schema() -> FunctionSchema {
    FunctionSchema::new("browse_menu", "Customer wants to see the menu and start ordering")
        .with_parameters(json!({
            "type": "object",
            "properties": {},
            "required": []
        }))
}

fn add_to_order_schema() -> FunctionSchema {
    FunctionSchema::new(
        "add_to_order",
        "Add a pizza to the customer's order. Validates the pizza name, size, \
         and toppings against the database. Returns the validated details \
         including canonical name, actual price, and available options."
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "pizza": {
                "type": "string",
                "description": "Pizza name (fuzzy matched against database)"
            },
            "size": {
                "type": "string",
                "enum": ["small", "medium", "large"],
                "description": "Size of the pizza"
            },
            "toppings": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Extra topping names (fuzzy matched against database)"
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

fn fetch_item_detail_schema() -> FunctionSchema {
    FunctionSchema::new(
        "fetch_item_detail",
        "Show detailed information about a specific pizza on the customer's screen. \
         Use when the customer asks about a particular pizza — its description, \
         whether it's vegetarian, and available sizes with prices. \
         The detail goes to the UI; give a brief verbal summary only."
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "pizza": {
                "type": "string",
                "description": "Pizza name to look up"
            }
        },
        "required": ["pizza"]
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
        .with_tools(ToolsSchema::new(vec![
            browse_menu_schema(),
            fetch_menu_schema(),
        ]))
        .with_respond_immediately(true)
}

fn menu_node() -> NodeConfig {
    NodeConfig::new("menu")
        .with_system_prompt(
            "You are a pizza ordering assistant at Dhara Pizza. \
             CRITICAL: when a tool call returns a result and item count, \
             the data is already on the customer's screen. \
             Do NOT read the entire menu aloud — just say something like \
             'The menu is on your screen — what looks good?' \
             When the customer picks a pizza, use add_to_order — it will \
             validate the name against the database and return the actual \
             price and available options. If validation fails, tell the \
             customer what went wrong and suggest alternatives. \
             Keep all voice responses to one or two sentences."
        )
        .with_task_message(
            "Help the customer build their order. \
             If they want to see the menu again, call fetch_menu — the data \
             appears on their screen, just invite them to pick. \
             Use add_to_order when they choose (it validates against the DB), \
             view_order to recap the order, \
             remove_from_order to change it, confirm_order when done."
        )
        .with_tools(ToolsSchema::new(vec![
            fetch_menu_schema(),
            fetch_item_detail_schema(),
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
            "Read back the complete order summary to the customer briefly (use view_order). \
             Ask them to confirm or if they want to make changes. \
             Use modify_order to return to the menu, \
             or place_order with their delivery address to finalise."
        )
        .with_tools(ToolsSchema::new(vec![
            fetch_menu_schema(),
            fetch_item_detail_schema(),
            view_order_schema(),
            modify_order_schema(),
            place_order_schema(),
        ]))
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

fn farewell_node() -> NodeConfig {
    NodeConfig::new("farewell")
        .with_task_message(
            "The order has been placed and saved. Thank the customer briefly, \
             mention delivery in 30-45 minutes, and say goodbye warmly. Keep it short."
        )
        .with_tools(ToolsSchema::new(vec![
            fetch_menu_schema(),
        ]))
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

// ---------------------------------------------------------------------------
// Data tool registration — fetch_menu
// ---------------------------------------------------------------------------

/// Register the `fetch_menu` data tool into the function registry.
///
/// This is a data tool: the full menu JSON goes to the UI via
/// `FunctionCallRawResultFrame`, and only a short summary reaches the LLM.
///
/// Must be called after every Dhara node transition (the transition hook
/// replaces the registry, wiping non-Dhara handlers).
fn register_fetch_menu(registry: &mut FunctionRegistry, writer: Arc<OrderWriter>) {
    registry.register_data("fetch_menu", move |_args: String| {
        let writer = writer.clone();
        async move {
            match writer.fetch_menu().await {
                Ok(menu) => {
                    let pizza_count = menu["pizzas"]
                        .as_array().map(|a| a.len()).unwrap_or(0);
                    let topping_count = menu["toppings"]
                        .as_array().map(|a| a.len()).unwrap_or(0);
                    ToolCallOutput::with_data(
                        format!(
                            "Menu displayed on customer's screen: {} pizzas, {} toppings available. \
                             Do NOT read the menu aloud — the customer can see it. \
                             Just say something like 'the menu is on your screen' and help them pick.",
                            pizza_count, topping_count
                        ),
                        menu,
                    )
                }
                Err(e) => ToolCallOutput::summary_only(format!("Error fetching menu: {}", e)),
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Ravi client push helper
// ---------------------------------------------------------------------------

/// Push a structured JSON payload to the client as a Ravi `server-message`.
///
/// Uses the deferred push sender — safe to call from Dhara handlers since
/// the sender is guaranteed to be set before any handler runs.
async fn push_ravi_msg(push: &DeferredPush, data: Value) {
    if let Some(tx) = push.get() {
        let payload = ravi_models::msg_server_message(data);
        let frame = Frame::ravi_server_message(payload);
        if let Err(e) = tx.send((frame, FrameDirection::Downstream)).await {
            log::error!("push_ravi_msg: send failed: {}", e);
        }
    } else {
        log::warn!("push_ravi_msg: deferred sender not yet initialized");
    }
}

// ---------------------------------------------------------------------------
// Handler factories — Dhara handlers
// ---------------------------------------------------------------------------

fn make_browse_menu_handler(
    writer: Arc<OrderWriter>,
    push: DeferredPush,
) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |_args: String| {
        let writer = writer.clone();
        let push = push.clone();
        Box::pin(async move {
            // Fetch menu from DB and push directly to client UI
            let menu_result = match writer.fetch_menu().await {
                Ok(menu) => {
                    let pizza_count = menu["pizzas"]
                        .as_array().map(|a| a.len()).unwrap_or(0);
                    let topping_count = menu["toppings"]
                        .as_array().map(|a| a.len()).unwrap_or(0);

                    // Push full menu to client as server-message
                    push_ravi_msg(&push, json!({
                        "type": "menu",
                        "data": menu,
                    })).await;

                    json!({
                        "status": "menu_displayed",
                        "pizza_count": pizza_count,
                        "topping_count": topping_count,
                        "instruction": "The menu is now on the customer's screen. \
                                        Say something like 'Here's our menu!' and help them pick. \
                                        Do NOT read the entire menu aloud.",
                    })
                }
                Err(e) => {
                    log::error!("browse_menu: fetch_menu failed: {}", e);
                    json!({
                        "status": "error",
                        "error": format!("Could not load menu: {}", e),
                        "instruction": "Apologize and ask the customer to try again.",
                    })
                }
            };

            TransitionResult::transition(
                menu_result.to_string(),
                "menu",
            )
        })
    })
}

fn make_add_to_order_handler(
    order: Arc<Mutex<OrderState>>,
    writer: Arc<OrderWriter>,
    push: DeferredPush,
) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let order = order.clone();
        let writer = writer.clone();
        let push = push.clone();
        Box::pin(async move {
            let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let pizza_name = parsed["pizza"].as_str().unwrap_or("").to_string();
            let size = parsed["size"].as_str().unwrap_or("medium").to_string();
            let topping_names: Vec<String> = parsed["toppings"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();

            // 1. Validate pizza name against DB
            let (pizza_id, canonical_name) = match writer.lookup_pizza(&pizza_name).await {
                Ok(Some((id, name))) => (id, name),
                Ok(None) => {
                    let available = writer.list_pizza_names().await.unwrap_or_default();
                    let result = json!({
                        "status": "error",
                        "error": format!("Pizza '{}' not found in our menu", pizza_name),
                        "available_pizzas": available,
                        "instruction": "Tell the customer we don't have that pizza and suggest the available options."
                    });
                    return TransitionResult::stay(result.to_string());
                }
                Err(e) => {
                    let result = json!({
                        "status": "error",
                        "error": format!("Database error looking up pizza: {}", e),
                    });
                    return TransitionResult::stay(result.to_string());
                }
            };

            // 2. Validate size and get price from DB
            let sizes = match writer.get_sizes_for_pizza(pizza_id).await {
                Ok(s) => s,
                Err(e) => {
                    let result = json!({
                        "status": "error",
                        "error": format!("Could not look up sizes: {}", e),
                    });
                    return TransitionResult::stay(result.to_string());
                }
            };

            let size_lower = size.to_lowercase();
            let size_match = sizes.iter().find(|(s, _)| s.to_lowercase() == size_lower);

            let (canonical_size, base_price) = match size_match {
                Some((s, p)) => (s.clone(), *p),
                None => {
                    let available: Vec<String> = sizes.iter()
                        .map(|(s, p)| format!("{} (${:.2})", s, p))
                        .collect();
                    let result = json!({
                        "status": "error",
                        "error": format!("Size '{}' not available for {}", size, canonical_name),
                        "available_sizes": available,
                        "instruction": "Tell the customer that size isn't available and list the options with prices."
                    });
                    return TransitionResult::stay(result.to_string());
                }
            };

            // 3. Validate toppings against DB
            let (valid_toppings, invalid_toppings) = match writer.validate_toppings(&topping_names).await {
                Ok(result) => result,
                Err(e) => {
                    let result = json!({
                        "status": "error",
                        "error": format!("Could not validate toppings: {}", e),
                    });
                    return TransitionResult::stay(result.to_string());
                }
            };

            // If some toppings are invalid, report them
            if !invalid_toppings.is_empty() {
                let available = writer.list_topping_names().await.unwrap_or_default();
                let result = json!({
                    "status": "error",
                    "error": format!("These toppings are not available: {}", invalid_toppings.join(", ")),
                    "valid_toppings_added": valid_toppings.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
                    "available_toppings": available,
                    "instruction": "Tell the customer which toppings we don't have and suggest alternatives. \
                                    The valid toppings listed were NOT added yet — ask the customer if they \
                                    want to proceed with just the valid ones, or pick different toppings."
                });
                return TransitionResult::stay(result.to_string());
            }

            // 4. Calculate line price
            let topping_total: f64 = valid_toppings.iter().map(|(_, p)| p).sum();
            let line_price = base_price + topping_total;

            let canonical_topping_names: Vec<String> = valid_toppings.iter()
                .map(|(n, _)| n.clone())
                .collect();

            // 5. Add validated item to order
            {
                let mut state = order.lock().unwrap();
                state.items.push(OrderItem {
                    pizza_id,
                    pizza: canonical_name.clone(),
                    size: canonical_size.clone(),
                    toppings: canonical_topping_names.clone(),
                    line_price,
                });
            }

            // 6. Push cart update to client UI
            let cart_data = order.lock().unwrap().cart_payload();
            push_ravi_msg(&push, cart_data).await;

            let state = order.lock().unwrap();
            let result = json!({
                "status": "added",
                "pizza_id": pizza_id,
                "pizza": canonical_name,
                "size": canonical_size,
                "base_price": format!("${:.2}", base_price),
                "toppings": canonical_topping_names,
                "topping_total": format!("${:.2}", topping_total),
                "line_price": format!("${:.2}", line_price),
                "order_summary": state.summary(),
            });
            TransitionResult::stay(result.to_string())
        })
    })
}

fn make_remove_from_order_handler(
    order: Arc<Mutex<OrderState>>,
    push: DeferredPush,
) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let order = order.clone();
        let push = push.clone();
        Box::pin(async move {
            let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let num = parsed["item_number"].as_u64().unwrap_or(0) as usize;

            // Scope the MutexGuard so it's dropped before any .await
            let (result, cart_data) = {
                let mut state = order.lock().unwrap();
                if num >= 1 && num <= state.items.len() {
                    let removed = state.items.remove(num - 1);
                    let r = json!({
                        "status": "removed",
                        "removed": format!("{} {} (${:.2})", removed.size, removed.pizza, removed.line_price),
                        "order_summary": state.summary(),
                    });
                    (r, Some(state.cart_payload()))
                } else {
                    (json!({"status": "error", "error": format!("No item #{}", num)}), None)
                }
            }; // guard dropped here

            // Push cart update outside the lock
            if let Some(cart) = cart_data {
                push_ravi_msg(&push, cart).await;
            }

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
// fetch_item_detail — Dhara handler that pushes item data to client
// ---------------------------------------------------------------------------

fn make_fetch_item_detail_handler(
    writer: Arc<OrderWriter>,
    push: DeferredPush,
) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let writer = writer.clone();
        let push = push.clone();
        Box::pin(async move {
            let parsed: Value = serde_json::from_str(&args).unwrap_or_default();
            let pizza_name = parsed["pizza"].as_str().unwrap_or("").to_string();

            match writer.get_pizza_detail(&pizza_name).await {
                Ok(Some(detail)) => {
                    // Push structured detail to client UI
                    push_ravi_msg(&push, json!({
                        "type": "item-detail",
                        "pizza": detail.clone(),
                    })).await;

                    // Return short summary to LLM for verbal narration
                    let name = detail["name"].as_str().unwrap_or(&pizza_name);
                    let desc = detail["description"].as_str().unwrap_or("");
                    let veg = if detail["vegetarian"].as_bool().unwrap_or(false) {
                        " (vegetarian)"
                    } else {
                        ""
                    };
                    TransitionResult::stay(json!({
                        "status": "detail_sent",
                        "pizza": name,
                        "description": desc,
                        "vegetarian_note": veg,
                        "instruction": "The pizza details are on the customer's screen. \
                                        Give a brief verbal mention — don't read all sizes aloud."
                    }).to_string())
                }
                Ok(None) => {
                    let available = writer.list_pizza_names().await.unwrap_or_default();
                    TransitionResult::stay(json!({
                        "status": "error",
                        "error": format!("Pizza '{}' not found", pizza_name),
                        "available_pizzas": available,
                    }).to_string())
                }
                Err(e) => {
                    TransitionResult::stay(json!({
                        "status": "error",
                        "error": format!("Failed to look up pizza: {}", e),
                    }).to_string())
                }
            }
        })
    })
}

// ---------------------------------------------------------------------------
// ConnectionFlow
// ---------------------------------------------------------------------------

struct ConnectionFlow {
    context:          Arc<Mutex<LLMContext>>,
    registry:         Arc<Mutex<FunctionRegistry>>,
    transition_hook:  rustvani::services::llm::openai::TransitionHook,
    push_tx:          DeferredPush,
}

/// Build the Dhara flow for a single connection.
///
/// `order_writer` — shared writer for menu reads, add_to_order validation,
///                  and place_order writes.
fn build_flow(order_writer: Arc<OrderWriter>) -> ConnectionFlow {
    let order    = Arc::new(Mutex::new(OrderState::default()));
    let context  = Arc::new(Mutex::new(LLMContext::new(None)));
    let registry = Arc::new(Mutex::new(FunctionRegistry::new()));
    let push_tx: DeferredPush = Arc::new(std::sync::OnceLock::new());

    let mut dhara = DharaManager::new(context.clone(), registry.clone());

    // greeting — browse_menu fetches + pushes menu, then transitions to menu node
    dhara.register_node("greeting", greeting_node(), vec![
        ("browse_menu", make_browse_menu_handler(order_writer.clone(), push_tx.clone())),
    ]);

    // menu — pizza ordering tools + fetch_item_detail
    dhara.register_node("menu", menu_node(), vec![
        ("fetch_item_detail", make_fetch_item_detail_handler(order_writer.clone(), push_tx.clone())),
        ("add_to_order",      make_add_to_order_handler(order.clone(), order_writer.clone(), push_tx.clone())),
        ("remove_from_order", make_remove_from_order_handler(order.clone(), push_tx.clone())),
        ("view_order",        make_view_order_handler(order.clone())),
        ("confirm_order",     make_confirm_order_handler(order.clone())),
    ]);

    // confirm — view/modify/place + fetch_item_detail
    dhara.register_node("confirm", confirm_node(), vec![
        ("fetch_item_detail", make_fetch_item_detail_handler(order_writer.clone(), push_tx.clone())),
        ("view_order",    make_view_order_handler(order.clone())),
        ("modify_order",  make_modify_order_handler()),
        ("place_order",   make_place_order_handler(order.clone(), order_writer.clone())),
    ]);

    // farewell — no Dhara handlers needed
    dhara.register_node_no_tools("farewell", farewell_node());

    dhara.set_initial_node("greeting");

    // Dhara swaps the shared registry on every node transition, wiping any
    // handlers not registered in that node's list. We wrap the hook to
    // re-inject the fetch_menu data tool after every swap.
    let dhara_hook = dhara.create_transition_hook();
    let writer_for_hook = order_writer.clone();
    let reg_for_hook = registry.clone();
    let transition_hook: rustvani::services::llm::openai::TransitionHook =
        Arc::new(move |ctx| {
            dhara_hook(ctx);
            register_fetch_menu(&mut reg_for_hook.lock().unwrap(), writer_for_hook.clone());
        });

    // Also register fetch_menu for the initial node (before any transition)
    register_fetch_menu(&mut registry.lock().unwrap(), order_writer);

    ConnectionFlow { context, registry, transition_hook, push_tx }
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
                //turn_config:              Some(SmartTurnConfig::default()),
                ..TransportParams::default()
            },
        },
    );

    // ---- OrderWriter (single DB connection for all app operations) ----
    let order_writer = Arc::new(OrderWriter::new());
    if let Err(e) = order_writer.init(&app_state.database_url).await {
        log::error!("[conn={}] OrderWriter init failed: {}", conn_id, e);
        return;
    }

    // ---- Dhara flow (fresh per connection) ----
    let flow = build_flow(order_writer);

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

    // ---- LLM with Dhara transition hook ----
    let mut llm_handler = OpenAILLMHandler::with_shared_registry(
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

    // ---- TTS (Deepgram) ----
    let tts = match DeepgramTtsHandler::new(DeepgramTtsConfig {
        api_key: app_state.deepgram_api_key.clone(),
        ..DeepgramTtsConfig::default()
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

    // Wire the deferred push sender — handlers can now push Ravi messages
    let _ = flow.push_tx.set(task.push_sender());

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

    let deepgram_api_key = std::env::var("DEEPGRAM_API_KEY")
        .expect("DEEPGRAM_API_KEY env var not set");

    let app_state = AppState { database_url, sarvam_api_key, openai_api_key, deepgram_api_key };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "10000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    log::info!("🍕 Dhara Pizza voice server on ws://{}/ws", addr);
    log::info!("Flow: greeting → menu → confirm → farewell");
    log::info!("Tools: fetch_menu (data), fetch_item_detail/add/remove/view/confirm/modify/place (Dhara)");

    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    axum::serve(listener, app).await.unwrap();
}
