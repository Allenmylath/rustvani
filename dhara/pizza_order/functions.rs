// Domain state types and handler registration for the pizza_order flow.
//
// Included into pizza_voice_server_v2.rs via include!().
// All types here are visible in the binary's module scope.

// Imports provided by the including binary (pizza_voice_server_v2.rs):
//   std::sync::{Arc, Mutex}
//   serde_json::{json, Value}
//   rustvani::dhara::{DharaContext, DharaFunctionRegistry}

use std::error::Error;
use tokio::sync::Mutex as AsyncMutex;
use native_tls::TlsConnector;
use postgres_native_tls::MakeTlsConnector;
use serde_json::{json, Value};

use rustvani::dhara::HandlerResult;

// ---------------------------------------------------------------------------
// Order state — one per connection (in-memory until confirmed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct OrderItem {
    pizza_id:   i32,
    pizza:      String,
    size:       String,
    toppings:   Vec<String>,
    line_price: f64,
}

#[derive(Debug, Clone, Default)]
struct OrderState {
    items: Vec<OrderItem>,
}

impl OrderState {
    fn total_price(&self) -> f64 {
        self.items.iter().map(|i| i.line_price).sum()
    }

    fn summary(&self) -> String {
        if self.items.is_empty() {
            return "No items in order".to_string();
        }
        let lines: Vec<String> = self.items.iter().enumerate().map(|(i, item)| {
            let toppings = if item.toppings.is_empty() {
                "no extra toppings".to_string()
            } else {
                item.toppings.join(", ")
            };
            format!("{}. {} {} with {} — ${:.2}", i + 1, item.size, item.pizza, toppings, item.line_price)
        }).collect();
        format!("{}\nTotal: ${:.2}", lines.join("\n"), self.total_price())
    }

    fn cart_payload(&self) -> Value {
        json!({
            "type": "cart-updated",
            "items": self.items.iter().map(|item| json!({
                "pizza_id": item.pizza_id,
                "pizza":    item.pizza,
                "size":     item.size,
                "toppings": item.toppings,
                "line_price": item.line_price,
            })).collect::<Vec<_>>(),
            "total_price": self.total_price(),
            "item_count":  self.items.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// OrderWriter — dedicated Neon connection for all DB operations
// ---------------------------------------------------------------------------

struct OrderWriter {
    client: AsyncMutex<Option<tokio_postgres::Client>>,
}

impl OrderWriter {
    fn new() -> Self {
        Self { client: AsyncMutex::new(None) }
    }

    async fn init(&self, db_url: &str) -> Result<(), String> {
        let connector = TlsConnector::builder().build()
            .map_err(|e| format!("TLS build: {}", e))?;
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

    async fn fetch_menu(&self) -> Result<Value, String> {
        let guard  = self.client.lock().await;
        let client = guard.as_ref().ok_or("OrderWriter not initialized")?;

        let rows = client.query(
            "SELECT p.name, p.description, p.is_vegetarian, p.image_url, ps.size, ps.price \
             FROM pizzas p JOIN pizza_sizes ps ON p.id = ps.pizza_id \
             WHERE p.is_available = true ORDER BY p.name, ps.price",
            &[],
        ).await.map_err(|e| format!("Menu query failed: {}", e))?;

        let mut pizzas: Vec<Value> = Vec::new();
        for row in &rows {
            let name: String              = row.get(0);
            let description: Option<String> = row.get(1);
            let vegetarian: bool          = row.get(2);
            let image_url: Option<String> = row.get(3);
            let size: String              = row.get(4);
            let price: f64                = row.get(5);
            let size_entry = json!({"size": size, "price": price});
            let same = pizzas.last().and_then(|p| p["name"].as_str()).map(|n| n == name).unwrap_or(false);
            if same {
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

        let topping_rows = client.query(
            "SELECT name, price_per_unit FROM toppings WHERE is_available = true ORDER BY name",
            &[],
        ).await.map_err(|e| format!("Toppings query failed: {}", e))?;

        let toppings: Vec<Value> = topping_rows.iter().map(|r| json!({
            "name":  r.get::<_, String>(0),
            "price": r.get::<_, f64>(1),
        })).collect();

        Ok(json!({ "pizzas": pizzas, "toppings": toppings }))
    }

    async fn get_pizza_detail(&self, name: &str) -> Result<Option<Value>, String> {
        let (pizza_id, canonical_name) = match self.lookup_pizza(name).await? {
            Some(p) => p,
            None    => return Ok(None),
        };
        let (description, vegetarian, image_url) = {
            let guard  = self.client.lock().await;
            let client = guard.as_ref().ok_or("OrderWriter not initialized")?;
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
        };
        let sizes = self.get_sizes_for_pizza(pizza_id).await?;
        Ok(Some(json!({
            "name":        canonical_name,
            "description": description,
            "vegetarian":  vegetarian,
            "image_url":   image_url,
            "sizes": sizes.iter().map(|(s, p)| json!({"size": s, "price": p})).collect::<Vec<_>>(),
        })))
    }

    async fn lookup_pizza(&self, name: &str) -> Result<Option<(i32, String)>, String> {
        let guard  = self.client.lock().await;
        let client = guard.as_ref().ok_or("OrderWriter not initialized")?;
        let pattern = format!("%{}%", name.trim());
        let row = client.query_opt(
            "SELECT id, name FROM pizzas WHERE LOWER(name) ILIKE LOWER($1) AND is_available = true LIMIT 1",
            &[&pattern],
        ).await.map_err(|e| format!("Pizza lookup failed: {}", e))?;
        Ok(row.map(|r| (r.get::<_, i32>(0), r.get::<_, String>(1))))
    }

    async fn list_pizza_names(&self) -> Result<Vec<String>, String> {
        let guard  = self.client.lock().await;
        let client = guard.as_ref().ok_or("OrderWriter not initialized")?;
        let rows = client.query("SELECT name FROM pizzas WHERE is_available = true ORDER BY name", &[])
            .await.map_err(|e| format!("List pizzas failed: {}", e))?;
        Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
    }

    async fn get_sizes_for_pizza(&self, pizza_id: i32) -> Result<Vec<(String, f64)>, String> {
        let guard  = self.client.lock().await;
        let client = guard.as_ref().ok_or("OrderWriter not initialized")?;
        let rows = client.query(
            "SELECT size, price FROM pizza_sizes WHERE pizza_id = $1 ORDER BY price",
            &[&pizza_id],
        ).await.map_err(|e| format!("Size lookup failed: {}", e))?;
        Ok(rows.iter().map(|r| (r.get::<_, String>(0), r.get::<_, f64>(1))).collect())
    }

    async fn validate_toppings(&self, names: &[String]) -> Result<(Vec<(String, f64)>, Vec<String>), String> {
        if names.is_empty() { return Ok((vec![], vec![])); }
        let guard  = self.client.lock().await;
        let client = guard.as_ref().ok_or("OrderWriter not initialized")?;
        let mut valid   = Vec::new();
        let mut invalid = Vec::new();
        for name in names {
            let pattern = format!("%{}%", name.trim());
            let row = client.query_opt(
                "SELECT name, price_per_unit FROM toppings \
                 WHERE LOWER(name) ILIKE LOWER($1) AND is_available = true LIMIT 1",
                &[&pattern],
            ).await.map_err(|e| format!("Topping lookup failed: {}", e))?;
            match row {
                Some(r) => valid.push((r.get::<_, String>(0), r.get::<_, f64>(1))),
                None    => invalid.push(name.clone()),
            }
        }
        Ok((valid, invalid))
    }

    async fn list_topping_names(&self) -> Result<Vec<String>, String> {
        let guard  = self.client.lock().await;
        let client = guard.as_ref().ok_or("OrderWriter not initialized")?;
        let rows = client.query("SELECT name FROM toppings WHERE is_available = true ORDER BY name", &[])
            .await.map_err(|e| format!("List toppings failed: {}", e))?;
        Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
    }

    async fn write_order(&self, address: &str, order: &OrderState) -> Result<String, String> {
        let mut guard  = self.client.lock().await;
        let client = guard.as_mut().ok_or("OrderWriter not initialized")?;
        let total  = order.total_price();
        let tx = client.transaction().await
            .map_err(|e| format!("Transaction start failed: {}", e))?;
        let order_row = tx.query_one(
            "INSERT INTO orders (delivery_address, status, payment_completed, total_price) \
             VALUES ($1, 'confirmed', false, $2) RETURNING id",
            &[&address as &(dyn tokio_postgres::types::ToSql + Sync), &total],
        ).await.map_err(|e| format!("Insert order failed: {}", e))?;
        let order_id: i32 = order_row.get(0);
        for item in &order.items {
            tx.execute(
                "INSERT INTO order_items \
                    (order_id, pizza_id, pizza_name, size, extra_toppings, line_price) \
                 VALUES ($1, $2, $3, $4, $5, $6)",
                &[&order_id, &item.pizza_id, &item.pizza, &item.size, &item.toppings, &item.line_price],
            ).await.map_err(|e| {
                let detail = e.source().map(|s| format!(" (detail: {})", s)).unwrap_or_default();
                format!("Insert order item failed: {}{}", e, detail)
            })?;
        }
        tx.commit().await.map_err(|e| format!("Commit failed: {}", e))?;
        log::info!("OrderWriter: committed order {} ({})", order_id, address);
        Ok(format!("DP-{:05}", order_id))
    }
}

// ---------------------------------------------------------------------------
// Dhara flow state — passed as Arc<dyn Any + Send + Sync> to dhara.build()
// ---------------------------------------------------------------------------

struct DharaPizzaState {
    order:  Arc<Mutex<OrderState>>,
    writer: Arc<OrderWriter>,
}

// ---------------------------------------------------------------------------
// Handler registration
// ---------------------------------------------------------------------------

fn register_handlers(reg: &mut DharaFunctionRegistry) {
    // fetch_menu — push full menu to UI, return short summary to LLM. No transition.
    reg.register("fetch_menu", |_args, ctx| async move {
        let writer = ctx.state::<DharaPizzaState>().map(|s| s.writer.clone()).unwrap();
        match writer.fetch_menu().await {
            Ok(menu) => {
                let pizza_count   = menu["pizzas"].as_array().map(|a| a.len()).unwrap_or(0);
                let topping_count = menu["toppings"].as_array().map(|a| a.len()).unwrap_or(0);
                ctx.push_ravi_message(json!({ "type": "menu", "data": menu })).await;
                HandlerResult::ok(format!(
                    "Menu displayed: {} pizzas, {} toppings. The data is on screen — do NOT read it aloud.",
                    pizza_count, topping_count
                ))
            }
            Err(e) => HandlerResult::ok(format!("Error fetching menu: {}", e)),
        }
    });

    // browse_menu — fetch + push menu, then JSON "default" transition → menu node.
    reg.register("browse_menu", |_args, ctx| async move {
        let writer = ctx.state::<DharaPizzaState>().map(|s| s.writer.clone()).unwrap();
        match writer.fetch_menu().await {
            Ok(menu) => {
                let pizza_count   = menu["pizzas"].as_array().map(|a| a.len()).unwrap_or(0);
                let topping_count = menu["toppings"].as_array().map(|a| a.len()).unwrap_or(0);
                ctx.push_ravi_message(json!({ "type": "menu", "data": menu })).await;
                HandlerResult::ok(json!({
                    "status": "menu_displayed",
                    "pizza_count": pizza_count,
                    "topping_count": topping_count,
                    "instruction": "The menu is now on the customer's screen. \
                                    Say 'Here\u{2019}s our menu!' and help them pick. \
                                    Do NOT read the entire menu aloud.",
                }).to_string())
            }
            Err(e) => {
                log::error!("browse_menu: fetch_menu failed: {}", e);
                HandlerResult::ok(json!({
                    "status": "error",
                    "error": format!("Could not load menu: {}", e),
                    "instruction": "Apologize and ask the customer to try again.",
                }).to_string())
            }
        }
    });

    // add_to_order — validate pizza/size/toppings, add to in-memory order. No transition.
    reg.register("add_to_order", |args, ctx| async move {
        let (writer, order) = {
            let s = ctx.state::<DharaPizzaState>().unwrap();
            (s.writer.clone(), s.order.clone())
        };
        let parsed: Value      = serde_json::from_str(&args).unwrap_or_default();
        let pizza_name         = parsed["pizza"].as_str().unwrap_or("").to_string();
        let size               = parsed["size"].as_str().unwrap_or("medium").to_string();
        let topping_names: Vec<String> = parsed["toppings"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let (pizza_id, canonical_name) = match writer.lookup_pizza(&pizza_name).await {
            Ok(Some((id, name))) => (id, name),
            Ok(None) => {
                let available = writer.list_pizza_names().await.unwrap_or_default();
                return HandlerResult::ok(json!({
                    "status": "error",
                    "error": format!("Pizza '{}' not found in our menu", pizza_name),
                    "available_pizzas": available,
                    "instruction": "Tell the customer we don't have that pizza and suggest alternatives."
                }).to_string());
            }
            Err(e) => return HandlerResult::ok(json!({
                "status": "error", "error": format!("Database error: {}", e)
            }).to_string()),
        };

        let sizes = match writer.get_sizes_for_pizza(pizza_id).await {
            Ok(s) => s,
            Err(e) => return HandlerResult::ok(json!({
                "status": "error", "error": format!("Could not look up sizes: {}", e)
            }).to_string()),
        };
        let size_lower = size.to_lowercase();
        let (canonical_size, base_price) = match sizes.iter().find(|(s, _)| s.to_lowercase() == size_lower) {
            Some((s, p)) => (s.clone(), *p),
            None => {
                let available: Vec<String> = sizes.iter().map(|(s, p)| format!("{} (${:.2})", s, p)).collect();
                return HandlerResult::ok(json!({
                    "status": "error",
                    "error": format!("Size '{}' not available for {}", size, canonical_name),
                    "available_sizes": available,
                    "instruction": "Tell the customer that size isn't available and list the options with prices."
                }).to_string());
            }
        };

        let (valid_toppings, invalid_toppings) = match writer.validate_toppings(&topping_names).await {
            Ok(r) => r,
            Err(e) => return HandlerResult::ok(json!({
                "status": "error", "error": format!("Could not validate toppings: {}", e)
            }).to_string()),
        };
        if !invalid_toppings.is_empty() {
            let available = writer.list_topping_names().await.unwrap_or_default();
            return HandlerResult::ok(json!({
                "status": "error",
                "error": format!("These toppings are not available: {}", invalid_toppings.join(", ")),
                "valid_toppings_added": valid_toppings.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
                "available_toppings": available,
                "instruction": "Tell the customer which toppings we don't have and suggest alternatives. \
                                The valid toppings listed were NOT added yet."
            }).to_string());
        }

        let topping_total: f64       = valid_toppings.iter().map(|(_, p)| p).sum();
        let line_price               = base_price + topping_total;
        let canonical_topping_names: Vec<String> = valid_toppings.iter().map(|(n, _)| n.clone()).collect();

        {
            let mut state = order.lock().unwrap();
            state.items.push(OrderItem {
                pizza_id,
                pizza: canonical_name.clone(),
                size:  canonical_size.clone(),
                toppings: canonical_topping_names.clone(),
                line_price,
            });
        }
        let cart_data = order.lock().unwrap().cart_payload();
        ctx.push_ravi_message(cart_data).await;

        let state = order.lock().unwrap();
        HandlerResult::ok(json!({
            "status":        "added",
            "pizza_id":      pizza_id,
            "pizza":         canonical_name,
            "size":          canonical_size,
            "base_price":    format!("${:.2}", base_price),
            "toppings":      canonical_topping_names,
            "topping_total": format!("${:.2}", topping_total),
            "line_price":    format!("${:.2}", line_price),
            "order_summary": state.summary(),
        }).to_string())
    });

    // remove_from_order — remove item by 1-based number. No transition.
    reg.register("remove_from_order", |args, ctx| async move {
        let order  = ctx.state::<DharaPizzaState>().map(|s| s.order.clone()).unwrap();
        let parsed: Value = serde_json::from_str(&args).unwrap_or_default();
        let num = parsed["item_number"].as_u64().unwrap_or(0) as usize;
        let (result, cart_data) = {
            let mut state = order.lock().unwrap();
            if num >= 1 && num <= state.items.len() {
                let removed = state.items.remove(num - 1);
                let r = json!({
                    "status":        "removed",
                    "removed":       format!("{} {} (${:.2})", removed.size, removed.pizza, removed.line_price),
                    "order_summary": state.summary(),
                });
                (r, Some(state.cart_payload()))
            } else {
                (json!({"status": "error", "error": format!("No item #{}", num)}), None)
            }
        };
        if let Some(cart) = cart_data {
            ctx.push_ravi_message(cart).await;
        }
        HandlerResult::ok(result.to_string())
    });

    // view_order — return current order summary. No transition.
    reg.register("view_order", |_args, ctx| async move {
        let order = ctx.state::<DharaPizzaState>().map(|s| s.order.clone()).unwrap();
        let state = order.lock().unwrap();
        HandlerResult::ok(json!({
            "order_summary": state.summary(),
            "item_count":    state.items.len(),
            "total_price":   format!("${:.2}", state.total_price()),
        }).to_string())
    });

    // confirm_order — JSON transition "ready" → confirm (or stay if cart empty).
    reg.register("confirm_order", |_args, ctx| async move {
        let order = ctx.state::<DharaPizzaState>().map(|s| s.order.clone()).unwrap();
        let state = order.lock().unwrap();
        if state.items.is_empty() {
            return HandlerResult::ok(json!({"error": "Cannot confirm — order is empty"}).to_string());
        }
        HandlerResult::with_status(
            json!({"status": "ready_to_confirm", "order_summary": state.summary()}).to_string(),
            "ready",
        )
    });

    // modify_order — JSON "default" transition → menu.
    reg.register("modify_order", |_args, _ctx| async move {
        HandlerResult::ok(json!({"status": "returning_to_menu"}).to_string())
    });

    // place_order — write order to DB, JSON "placed" transition → farewell (or stay on error).
    reg.register("place_order", |args, ctx| async move {
        let (writer, order) = {
            let s = ctx.state::<DharaPizzaState>().unwrap();
            (s.writer.clone(), s.order.clone())
        };
        let parsed: Value = serde_json::from_str(&args).unwrap_or_default();
        let address = match parsed["delivery_address"].as_str() {
            Some(a) if !a.trim().is_empty() => a.to_string(),
            _ => return HandlerResult::ok(
                json!({"error": "delivery_address is required"}).to_string()
            ),
        };
        let snapshot = order.lock().unwrap().clone();
        if snapshot.items.is_empty() {
            return HandlerResult::ok(json!({"error": "Cannot place an empty order"}).to_string());
        }
        match writer.write_order(&address, &snapshot).await {
            Ok(order_id) => HandlerResult::with_status(
                json!({
                    "status":             "order_placed",
                    "order_id":           order_id,
                    "delivery_address":   address,
                    "order_summary":      snapshot.summary(),
                    "estimated_delivery": "30-45 minutes",
                    "payment_completed":  false,
                }).to_string(),
                "placed",
            ),
            Err(e) => {
                log::error!("OrderWriter: write_order failed: {}", e);
                HandlerResult::ok(json!({
                    "error":  "Failed to save your order. Please try again.",
                    "detail": e,
                }).to_string())
            }
        }
    });

    // fetch_item_detail — push pizza detail to UI, return short summary. No transition.
    reg.register("fetch_item_detail", |args, ctx| async move {
        let writer     = ctx.state::<DharaPizzaState>().map(|s| s.writer.clone()).unwrap();
        let parsed: Value = serde_json::from_str(&args).unwrap_or_default();
        let pizza_name = parsed["pizza"].as_str().unwrap_or("").to_string();
        match writer.get_pizza_detail(&pizza_name).await {
            Ok(Some(detail)) => {
                ctx.push_ravi_message(json!({"type": "item-detail", "pizza": detail.clone()})).await;
                let name     = detail["name"].as_str().map(String::from).unwrap_or_else(|| pizza_name.clone());
                let desc     = detail["description"].as_str().unwrap_or("").to_string();
                let veg_note = if detail["vegetarian"].as_bool().unwrap_or(false) { " (vegetarian)" } else { "" };
                HandlerResult::ok(json!({
                    "status":         "detail_sent",
                    "pizza":          name,
                    "description":    desc,
                    "vegetarian_note": veg_note,
                    "instruction":    "Pizza details are on the customer's screen. Brief verbal mention only.",
                }).to_string())
            }
            Ok(None) => {
                let available = writer.list_pizza_names().await.unwrap_or_default();
                HandlerResult::ok(json!({
                    "status": "error",
                    "error":  format!("Pizza '{}' not found", pizza_name),
                    "available_pizzas": available,
                }).to_string())
            }
            Err(e) => HandlerResult::ok(json!({
                "status": "error",
                "error":  format!("Failed to look up pizza: {}", e),
            }).to_string()),
        }
    });
}
