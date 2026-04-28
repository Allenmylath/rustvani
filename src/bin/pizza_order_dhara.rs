//! Pizza ordering demo — dhara conversation flow.
//!
//! Demonstrates multi-stage conversation with node transitions:
//!
//!   greeting → menu → confirm → farewell
//!                ↑       │
//!                └───────┘  (modify_order)
//!
//! Each node has its own system prompt, task messages, tools, and handlers.
//! Handlers return `TransitionResult::Stay` or `TransitionResult::Transition`
//! to control flow.
//!
//! Run:
//!   OPENAI_API_KEY=your-key cargo run --bin pizza_order_dhara

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use rustvani::{
    system_clock, ControlFrame, DataFrame, Frame, FrameDirection,
    FrameInner, FrameKind, PipelineParams, PipelineTask,
    FunctionCallData, FunctionCallResultData,
};
use rustvani::adapters::schemas::{FunctionSchema, ToolsSchema};
use rustvani::context::LLMContext;
use rustvani::dhara::{ContextStrategy, DharaManager, NodeConfig, TransitionResult};
use rustvani::observer::{BaseObserver, FrameProcessed, FramePushed};
use rustvani::processors::llm_assistant_aggregator::LLMAssistantAggregator;
use rustvani::services::llm::function_registry::FunctionRegistry;
use rustvani::services::{OpenAILLMConfig, OpenAILLMHandler};

// ---------------------------------------------------------------------------
// Shared order state
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
                    "description": "Extra toppings (e.g. mushrooms, olives, jalapenos)"
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
                    "description": "The item number to remove (1-based)"
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
             Greet the customer warmly and ask if they'd like to see the menu."
        )
        .with_task_message(
            "Greet the customer. If they want to order, use the browse_menu tool. \
             Keep it brief and friendly."
        )
        .with_tools(ToolsSchema::new(vec![browse_menu_schema()]))
        .with_respond_immediately(true)
}

fn menu_node() -> NodeConfig {
    NodeConfig::new("menu")
        .with_system_prompt(
            "You are a pizza ordering assistant at Dhara Pizza. Help the customer \
             build their order. Our menu: Margherita ($8.99-$15.99), Pepperoni, \
             Hawaiian, BBQ Chicken, Veggie Supreme. Sizes: small ($8.99), \
             medium ($12.99), large ($15.99). Extra toppings $1.50 each."
        )
        .with_task_message(
            "Help the customer build their pizza order. Use add_to_order to add pizzas, \
             view_order to show the current order, remove_from_order to remove items. \
             When they're happy with the order, use confirm_order. \
             Always confirm what they want before adding."
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
            "Read back the full order to the customer and ask them to confirm. \
             Use modify_order if they want changes, or place_order with their \
             delivery address to finalize."
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
            "The order has been placed! Thank the customer, tell them the \
             estimated delivery time (30-45 minutes), and say goodbye warmly."
        )
        .with_context_strategy(ContextStrategy::Append)
        .with_respond_immediately(true)
}

// ---------------------------------------------------------------------------
// Handler factories — each returns an Arc<dyn Fn> for the DharaManager
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

            let item = OrderItem { pizza: pizza.clone(), size: size.clone(), toppings: toppings.clone() };
            let mut state = order.lock().unwrap();
            state.items.push(item);

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
            let item_number = parsed["item_number"].as_u64().unwrap_or(0) as usize;

            let mut state = order.lock().unwrap();
            let result = if item_number >= 1 && item_number <= state.items.len() {
                let removed = state.items.remove(item_number - 1);
                json!({
                    "status": "removed",
                    "removed": format!("{} {}", removed.size, removed.pizza),
                    "order_summary": state.summary(),
                })
            } else {
                json!({
                    "status": "error",
                    "error": format!("No item #{}", item_number),
                })
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
                "message": "Please confirm or modify"
            });
            TransitionResult::transition(result.to_string(), "confirm")
        })
    })
}

fn make_modify_order_handler() -> rustvani::dhara::DharaHandlerFn {
    Arc::new(|_args: String| {
        Box::pin(async move {
            let result = json!({"status": "returning_to_menu"});
            TransitionResult::transition(result.to_string(), "menu")
        })
    })
}

fn make_place_order_handler(order: Arc<Mutex<OrderState>>) -> rustvani::dhara::DharaHandlerFn {
    Arc::new(move |args: String| {
        let order = order.clone();
        Box::pin(async move {
            let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let address = parsed["delivery_address"].as_str().unwrap_or("unknown");

            let state = order.lock().unwrap();
            let order_id = format!("DP-{}", rand_id());

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

fn rand_id() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    (SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos() % 99999) + 10000
}

// ---------------------------------------------------------------------------
// Observer — logs frames nicely
// ---------------------------------------------------------------------------

struct PizzaObserver;

#[async_trait]
impl BaseObserver for PizzaObserver {
    async fn on_process_frame(&self, event: FrameProcessed) {
        let is_agg = event.processor_name == "LLMAssistantAggregator";

        match &event.frame.inner {
            FrameInner::Control(ControlFrame::LLMFullResponseStart) if is_agg => {
                print!("\n🍕 Assistant: ");
            }
            FrameInner::Data(DataFrame::LLMText(text)) if is_agg => {
                print!("{}", text);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            FrameInner::Control(ControlFrame::LLMFullResponseEnd) if is_agg => {
                println!();
            }
            FrameInner::Data(DataFrame::FunctionCallInProgress(FunctionCallData {
                function_name,
                arguments,
                ..
            })) => {
                println!("\n⚙️  [tool] {}({})", function_name,
                    if arguments.len() > 80 { format!("{}…", &arguments[..80]) } else { arguments.clone() }
                );
            }
            FrameInner::Data(DataFrame::FunctionCallResult(FunctionCallResultData {
                function_name,
                result,
                ..
            })) => {
                println!("   [result] {} → {}",
                    function_name,
                    if result.len() > 100 { format!("{}…", &result[..100]) } else { result.clone() }
                );
            }
            FrameInner::Control(ControlFrame::FunctionCallStart) => {
                println!("\n🔧 [tool calls starting]");
            }
            FrameInner::Control(ControlFrame::FunctionCallEnd) => {
                println!("🔧 [tool calls done — re-invoking LLM]");
            }
            _ => {}
        }
    }

    async fn on_push_frame(&self, _: FramePushed) {}
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,rustvani=info"),
    )
    .init();

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    println!("╔══════════════════════════════════════════╗");
    println!("║     🍕 Dhara Pizza — Order System 🍕     ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Nodes: greeting → menu → confirm → bye  ║");
    println!("╚══════════════════════════════════════════╝\n");

    // ---- Shared state ----
    let order = Arc::new(Mutex::new(OrderState::default()));
    let context = Arc::new(Mutex::new(LLMContext::new(None)));
    let registry = Arc::new(Mutex::new(FunctionRegistry::new()));

    // ---- Build DharaManager ----
    let mut dhara = DharaManager::new(context.clone(), registry.clone());

    // Register nodes with handlers
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

    // Set initial node
    dhara.set_initial_node("greeting");

    // Create transition hook
    let hook = dhara.create_transition_hook();

    // ---- Build LLM handler with shared registry ----
    let mut llm_handler = OpenAILLMHandler::with_shared_registry(
        OpenAILLMConfig {
            api_key,
            model: "gpt-4o-mini".to_string(),
            max_tool_rounds: 10, // pizza ordering can involve several rounds
            ..OpenAILLMConfig::default()
        },
        registry.clone(),
    );
    llm_handler.set_transition_hook(hook);

    // ---- Pipeline ----
    let task = PipelineTask::new(
        vec![
            llm_handler.into_processor(),
            LLMAssistantAggregator::new(context.clone()),
        ],
        PipelineParams::default(),
    );

    let push_tx = task.push_sender();
    let context_for_input = context.clone();

    // ---- Interactive input loop ----
    let input_handle = tokio::spawn(async move {
        use std::io::{BufRead, Write};
        let stdin = std::io::stdin();

        // Initial trigger — let the greeting node speak first
        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = push_tx
            .send((Frame::llm_context(context_for_input.clone()), FrameDirection::Downstream))
            .await;

        loop {
            print!("\n👤 You: ");
            std::io::stdout().flush().ok();

            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                break; // EOF
            }
            let line = line.trim().to_string();

            if line.is_empty() {
                continue;
            }
            if line == "/quit" || line == "/exit" {
                println!("Goodbye!");
                let _ = push_tx.send((Frame::end(), FrameDirection::Downstream)).await;
                break;
            }
            if line == "/order" {
                let state = context_for_input.lock().unwrap();
                println!("\n📋 Context has {} messages", state.messages.len());
                continue;
            }

            // Add user message and trigger inference
            context_for_input.lock().unwrap().add_user_message(&line);
            let _ = push_tx
                .send((Frame::llm_context(context_for_input.clone()), FrameDirection::Downstream))
                .await;
        }

        // Give time for final response
        tokio::time::sleep(Duration::from_secs(5)).await;
        let _ = push_tx.send((Frame::end(), FrameDirection::Downstream)).await;
    });

    task.run(system_clock(), Some(Arc::new(PizzaObserver))).await?;
    let _ = input_handle.await;

    // ---- Final state ----
    println!("\n╔══════════════════════════════════════════╗");
    println!("║           📋 Final Order State            ║");
    println!("╚══════════════════════════════════════════╝");
    let final_order = order.lock().unwrap();
    if final_order.items.is_empty() {
        println!("  (no items ordered)");
    } else {
        println!("{}", final_order.summary());
    }

    Ok(())
}
