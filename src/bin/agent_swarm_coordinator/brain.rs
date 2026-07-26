//! The coordinator's decision logic — runs INSIDE the pipeline processor,
//! dispatching to peer agents via `cx.call(...)`.

use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

use rustvani::agents::{CoordinatorCtx, TaskResult};
use rustvani::context::Message;
use rustvani::LLMContext;

/// Tool agents the planner is allowed to select.
const KNOWN_AGENTS: [&str; 3] = ["clock", "dice", "weather"];

pub const ROUTER_SYS: &str = "\
You are a coordinator that can call helper agents. Available agents:
- \"clock\": current UTC time. No args.
- \"dice\": roll a random number. args: {\"sides\": <int, default 20>}
- \"weather\": (mock) current weather for a city. args: {\"city\": <string>}
Given the user's message, decide which helpers (if any) are needed.
Reply with ONLY a JSON object, no prose, no code fences:
{\"calls\": [{\"agent\": \"<name>\", \"args\": {..}}], \"reply\": \"<direct answer if no calls needed, else empty>\"}
For greetings or smalltalk, return an empty calls list and answer in \"reply\".";

pub const COMPOSE_SYS: &str = "\
You are a helpful assistant. You asked helper agents for data and got their
JSON results. Answer the user's original question naturally in 1-3 sentences,
using the results. Do not mention agents, tools, or JSON.";

/// The coordinator brain: ask the planner which agents to call, dispatch the
/// chosen ones, then ask the composer to write the final answer.
pub async fn coordinate(cx: CoordinatorCtx, context: Arc<Mutex<LLMContext>>) -> String {
    let question = last_user(&context);

    // 1. Ask the planner which agents to call.
    let plan = match cx.call("planner", "ask", json!({ "prompt": question })).await {
        Ok(r) => extract_json(&answer_str(Ok(r))),
        Err(e) => return format!("[planner error: {e}]"),
    };

    let chosen: Vec<(String, Value)> = plan
        .as_ref()
        .and_then(|p| p.get("calls"))
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|c| {
                    let name = c.get("agent").and_then(Value::as_str)?;
                    KNOWN_AGENTS
                        .contains(&name)
                        .then(|| (name.to_string(), c.get("args").cloned().unwrap_or(json!({}))))
                })
                .collect()
        })
        .unwrap_or_default();

    if chosen.is_empty() {
        return plan
            .and_then(|p| p.get("reply").and_then(Value::as_str).map(str::to_string))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Hi! How can I help?".to_string());
    }

    println!(
        "   [coordinator dispatched: {:?}]",
        chosen.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );

    // 2. Dispatch the chosen tools.
    let mut results = Map::new();
    for (agent, args) in chosen {
        if let Ok(r) = cx.call(&agent, "ask", args).await {
            results.insert(agent, r.response.unwrap_or(Value::Null));
        }
    }

    // 3. Ask the composer to write the final answer.
    let facts = format!(
        "The user asked: {question}\n\nHelper results (JSON):\n{}\n\nAnswer the user.",
        Value::Object(results)
    );
    match cx.call("composer", "ask", json!({ "prompt": facts })).await {
        Ok(r) => answer_str(Ok(r)),
        Err(e) => format!("[composer error: {e}]"),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn answer_str(res: rustvani::Result<TaskResult>) -> String {
    res.ok()
        .and_then(|r| r.response)
        .and_then(|v| v.get("answer").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_default()
}

/// Pull the first `{..}` object out of a model reply (tolerates prose / fences).
fn extract_json(s: &str) -> Option<Value> {
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    (end >= start).then(|| serde_json::from_str(&s[start..=end]).ok()).flatten()
}

fn last_user(context: &Arc<Mutex<LLMContext>>) -> String {
    context
        .lock()
        .unwrap()
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::User { content } => Some(content.clone()),
            _ => None,
        })
        .unwrap_or_default()
}
