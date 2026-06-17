//! Shared conversation context.
//!
//! Owned by both aggregators via `Arc<Mutex<LLMContext>>`.
//! The LLM service reads it; the aggregators write to it.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::adapters::schemas::{ToolChoice, ToolsSchema};

// ---------------------------------------------------------------------------
// ToolCall — a single function invocation requested by the model
// ---------------------------------------------------------------------------

/// A function call the model wants to execute.
///
/// Streamed as argument-string fragments during SSE; by the time this struct
/// is constructed, `arguments` is the fully accumulated JSON string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique call ID assigned by the model (e.g. `"call_abc123"`).
    pub id: String,
    /// Name of the function to invoke.
    pub function_name: String,
    /// Raw JSON string of the function arguments.
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Message — type-safe conversation turn
// ---------------------------------------------------------------------------

/// A single turn in the conversation.
///
/// Each variant enforces what fields are valid for that role, unlike the
/// Python dict approach where any key can appear on any message.
#[derive(Debug, Clone)]
pub enum Message {
    /// System-level instruction. Typically the first message.
    System { content: String },

    /// User turn — transcribed speech or typed text.
    User { content: String },

    /// Assistant turn — may be text, tool calls, or both.
    Assistant {
        /// `None` when the model responds with only tool calls.
        content: Option<String>,
        /// `None` for plain text responses.
        tool_calls: Option<Vec<ToolCall>>,
    },

    /// Result of a tool/function call, sent back to the model.
    ToolResult {
        /// Matches the `id` of the `ToolCall` this responds to.
        tool_call_id: String,
        /// Serialized result (typically JSON).
        content: String,
    },
}

// ---------------------------------------------------------------------------
// LLMContext
// ---------------------------------------------------------------------------

/// Shared conversation context passed between aggregators and the LLM service.
///
/// ## Turn transaction (see `doc/turn-acid.md`)
///
/// Beyond the committed `messages`, the context carries a small **staging
/// buffer** used to make a tool round atomic. The LLM stages an assistant
/// `tool_calls` message and its `ToolResult`s with [`stage_message`], then
/// [`commit`]s them into `messages` once the whole round is in. If the turn is
/// interrupted mid-round (barge-in), the in-flight future is dropped and
/// [`rollback`] discards the staged messages — so `messages` never retains an
/// assistant `tool_calls` without its matching results (which would make the
/// next request malformed).
///
/// Commit happens at the **round boundary**, not turn end: the Dhara transition
/// hook runs between rounds and manipulates `messages` directly, so each round's
/// content must be in `messages` by the time the hook sees it. Plain user and
/// assistant text messages are committed directly (no staging) — they carry no
/// orphan risk.
#[derive(Debug, Clone)]
pub struct LLMContext {
    /// System prompt — prepended as `Message::System` in `to_api_messages()`.
    pub system_prompt: Option<String>,
    /// Conversation history (user turns, assistant turns, tool results).
    pub messages: Vec<Message>,
    /// Available tools for this context. `None` = no function calling.
    pub tools: Option<ToolsSchema>,
    /// How the model should pick tools. `None` = provider default (usually "auto").
    pub tool_choice: Option<ToolChoice>,
    /// Uncommitted messages for the in-flight tool round. Empty between rounds.
    /// Not part of `to_api_messages()` — committed into `messages` at the round
    /// boundary, or discarded by `rollback()` on interruption.
    staged: Vec<Message>,
    /// Monotonic turn identity, bumped by `begin_turn()`. Seeds turn-level
    /// isolation; full epoch fencing across the agent bus is future work.
    epoch: u64,
}

impl LLMContext {
    pub fn new(system_prompt: Option<String>) -> Self {
        Self {
            system_prompt,
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            staged: Vec::new(),
            epoch: 0,
        }
    }

    /// Create a context with tools configured.
    pub fn with_tools(
        system_prompt: Option<String>,
        tools: ToolsSchema,
        tool_choice: Option<ToolChoice>,
    ) -> Self {
        Self {
            system_prompt,
            messages: Vec::new(),
            tools: Some(tools),
            tool_choice,
            staged: Vec::new(),
            epoch: 0,
        }
    }

    // ---- Convenience push methods ----

    /// Append any message variant.
    pub fn push_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Append a user turn.
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(Message::User {
            content: content.into(),
        });
    }

    /// Append a plain-text assistant turn (no tool calls).
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(Message::Assistant {
            content: Some(content.into()),
            tool_calls: None,
        });
    }

    /// Append an assistant turn that contains tool calls.
    pub fn add_assistant_tool_calls(
        &mut self,
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) {
        self.messages.push(Message::Assistant {
            content,
            tool_calls: Some(tool_calls),
        });
    }

    /// Append a tool result.
    pub fn add_tool_result(
        &mut self,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
    ) {
        self.messages.push(Message::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        });
    }

    // ---- Turn transaction (see doc/turn-acid.md) ----

    /// Open a new turn. Bumps the [`epoch`](Self::epoch) (turn identity) and
    /// discards any leftover staged messages — an implicit rollback of a prior,
    /// interrupted round. Returns the new epoch.
    pub fn begin_turn(&mut self) -> u64 {
        self.epoch = self.epoch.wrapping_add(1);
        self.staged.clear();
        self.epoch
    }

    /// Current turn epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Number of staged (uncommitted) messages. Exposed for tests/diagnostics.
    pub fn staged_len(&self) -> usize {
        self.staged.len()
    }

    /// Stage a message for the in-flight round. Not visible to
    /// `to_api_messages()` until [`commit`](Self::commit)ted.
    pub fn stage_message(&mut self, msg: Message) {
        self.staged.push(msg);
    }

    /// Stage an assistant turn that contains tool calls.
    pub fn stage_assistant_tool_calls(
        &mut self,
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) {
        self.staged.push(Message::Assistant {
            content,
            tool_calls: Some(tool_calls),
        });
    }

    /// Stage a tool result.
    pub fn stage_tool_result(
        &mut self,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
    ) {
        self.staged.push(Message::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        });
    }

    /// Commit the staged round into `messages` atomically. Drops any assistant
    /// `tool_calls` whose ids are not all answered by a staged `ToolResult`
    /// (consistency: the API rejects an assistant `tool_calls` without matching
    /// results), along with their dangling results. Returns the number of
    /// messages committed.
    pub fn commit(&mut self) -> usize {
        if self.staged.is_empty() {
            return 0;
        }
        let mut staged = std::mem::take(&mut self.staged);
        Self::repair_orphan_tool_calls(&mut staged);
        let n = staged.len();
        self.messages.append(&mut staged);
        n
    }

    /// Discard the staged round without touching committed `messages`.
    /// Idempotent. Called on interruption so an aborted tool round leaves no
    /// orphan behind.
    pub fn rollback(&mut self) {
        if !self.staged.is_empty() {
            log::debug!(
                "LLMContext: rolling back {} staged message(s)",
                self.staged.len()
            );
            self.staged.clear();
        }
    }

    /// Remove assistant `tool_calls` messages whose calls are not all answered
    /// by a `ToolResult` in the same staged batch, plus any `ToolResult` that
    /// then answers nothing. Keeps plain text messages untouched.
    fn repair_orphan_tool_calls(staged: &mut Vec<Message>) {
        use std::collections::HashSet;

        let answered: HashSet<&str> = staged
            .iter()
            .filter_map(|m| match m {
                Message::ToolResult { tool_call_id, .. } => Some(tool_call_id.as_str()),
                _ => None,
            })
            .collect();

        // Ids of tool calls that survive (all their calls are answered).
        let mut kept_call_ids: HashSet<String> = HashSet::new();
        let mut keep: Vec<bool> = Vec::with_capacity(staged.len());
        for m in staged.iter() {
            let k = match m {
                Message::Assistant { tool_calls: Some(tcs), .. } => {
                    let ok = tcs.iter().all(|tc| answered.contains(tc.id.as_str()));
                    if ok {
                        for tc in tcs {
                            kept_call_ids.insert(tc.id.clone());
                        }
                    } else {
                        log::warn!(
                            "LLMContext: dropping orphaned assistant tool_calls at commit \
                             (unanswered tool call)"
                        );
                    }
                    ok
                }
                _ => true,
            };
            keep.push(k);
        }
        // Second pass: drop tool results whose assistant message was dropped.
        let mut i = 0;
        staged.retain(|m| {
            let k = keep[i];
            i += 1;
            match m {
                Message::ToolResult { tool_call_id, .. } if k => {
                    kept_call_ids.contains(tool_call_id.as_str())
                }
                _ => k,
            }
        });
    }

    /// Build the full messages array for the API call.
    ///
    /// System prompt is prepended as the first message if present.
    /// The adapter then converts these `Message` variants into the
    /// provider's wire format.
    pub fn to_api_messages(&self) -> Vec<Message> {
        let mut result = Vec::new();
        if let Some(sys) = &self.system_prompt {
            result.push(Message::System {
                content: sys.clone(),
            });
        }
        result.extend(self.messages.clone());
        result
    }

    /// Rough token estimate: ~4 chars per token, covers all message fields.
    pub fn estimate_tokens(&self) -> usize {
        let mut chars: usize = self.system_prompt.as_deref().map_or(0, |s| s.len());
        for msg in &self.messages {
            chars += match msg {
                Message::System { content } => content.len(),
                Message::User { content } => content.len(),
                Message::Assistant { content, tool_calls } => {
                    content.as_deref().map_or(0, |c| c.len())
                        + tool_calls.as_ref().map_or(0, |tcs| {
                            tcs.iter()
                                .map(|tc| tc.function_name.len() + tc.arguments.len() + 20)
                                .sum()
                        })
                }
                Message::ToolResult { content, .. } => content.len(),
            };
        }
        chars.saturating_div(4)
    }

    /// Drop oldest conversation groups until the estimated token count fits
    /// within `context_window_tokens * 0.8` (reserves headroom for the reply).
    ///
    /// A "group" is everything from one User message up to (but not including)
    /// the next User message, so Assistant tool-call + ToolResult pairs are
    /// never orphaned. Stops if no safe drop point remains.
    pub fn trim_to_context_budget(&mut self, context_window_tokens: usize) {
        let budget = (context_window_tokens as f64 * 0.8) as usize;
        loop {
            if self.estimate_tokens() <= budget {
                break;
            }
            // Find the first User message that has another User message after it.
            let first_user = self
                .messages
                .iter()
                .position(|m| matches!(m, Message::User { .. }));
            let next_user = first_user.and_then(|i| {
                self.messages[i + 1..]
                    .iter()
                    .position(|m| matches!(m, Message::User { .. }))
                    .map(|j| i + 1 + j)
            });
            match (first_user, next_user) {
                (Some(start), Some(end)) => {
                    let dropped = end - start;
                    self.messages.drain(start..end);
                    log::warn!(
                        "LLMContext: trimmed {} messages to fit {}-token budget",
                        dropped,
                        context_window_tokens
                    );
                }
                _ => {
                    log::warn!(
                        "LLMContext: context near limit ({} estimated tokens) but cannot safely trim further",
                        self.estimate_tokens()
                    );
                    break;
                }
            }
        }
    }
}

/// Convenience: create a shared context ready for pipeline use.
pub fn shared_context(system_prompt: Option<String>) -> Arc<Mutex<LLMContext>> {
    Arc::new(Mutex::new(LLMContext::new(system_prompt)))
}

/// Convenience: create a shared context with tools configured.
pub fn shared_context_with_tools(
    system_prompt: Option<String>,
    tools: ToolsSchema,
    tool_choice: Option<ToolChoice>,
) -> Arc<Mutex<LLMContext>> {
    Arc::new(Mutex::new(LLMContext::with_tools(
        system_prompt,
        tools,
        tool_choice,
    )))
}

// ---------------------------------------------------------------------------
// Tests — turn transaction (see doc/turn-acid.md)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tc(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            function_name: name.into(),
            arguments: "{}".into(),
        }
    }

    fn assistant_text(messages: &[Message]) -> Vec<&str> {
        messages
            .iter()
            .filter_map(|m| match m {
                Message::Assistant { content: Some(c), tool_calls: None } => Some(c.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn staged_is_invisible_until_commit() {
        let mut ctx = LLMContext::new(None);
        ctx.add_user_message("hello");
        ctx.stage_assistant_tool_calls(None, vec![tc("call_1", "lookup")]);
        ctx.stage_tool_result("call_1", "ok");

        // Nothing committed yet — to_api_messages must not leak staged work.
        assert_eq!(ctx.staged_len(), 2);
        assert_eq!(ctx.to_api_messages().len(), 1); // just the user message
    }

    #[test]
    fn commit_splices_full_round() {
        let mut ctx = LLMContext::new(None);
        ctx.add_user_message("status of 4471?");
        ctx.stage_assistant_tool_calls(None, vec![tc("call_1", "lookup")]);
        ctx.stage_tool_result("call_1", "shipped");

        let n = ctx.commit();
        assert_eq!(n, 2);
        assert_eq!(ctx.staged_len(), 0);
        // user + assistant(tool_calls) + tool_result
        assert_eq!(ctx.messages.len(), 3);
        assert!(matches!(ctx.messages[1], Message::Assistant { tool_calls: Some(_), .. }));
        assert!(matches!(ctx.messages[2], Message::ToolResult { .. }));
    }

    #[test]
    fn rollback_discards_orphaned_round() {
        // Interrupt after staging the tool_calls but before its result.
        let mut ctx = LLMContext::new(None);
        ctx.add_user_message("status of 4471?");
        ctx.stage_assistant_tool_calls(None, vec![tc("call_1", "lookup")]);

        ctx.rollback();
        assert_eq!(ctx.staged_len(), 0);
        // Committed history is untouched — no orphaned tool_calls remain.
        assert_eq!(ctx.messages.len(), 1);
        assert!(matches!(ctx.messages[0], Message::User { .. }));
    }

    #[test]
    fn commit_drops_orphan_tool_calls_for_consistency() {
        // A defensive commit of a round missing one result must not splice an
        // assistant tool_calls without all its matching ToolResults.
        let mut ctx = LLMContext::new(None);
        ctx.stage_assistant_tool_calls(None, vec![tc("call_1", "a"), tc("call_2", "b")]);
        ctx.stage_tool_result("call_1", "done"); // call_2 unanswered

        let n = ctx.commit();
        assert_eq!(n, 0, "orphaned round must be dropped entirely");
        assert!(ctx.messages.is_empty());
    }

    #[test]
    fn commit_keeps_plain_text_assistant() {
        let mut ctx = LLMContext::new(None);
        ctx.stage_message(Message::Assistant {
            content: Some("hi there".into()),
            tool_calls: None,
        });
        assert_eq!(ctx.commit(), 1);
        assert_eq!(assistant_text(&ctx.messages), vec!["hi there"]);
    }

    #[test]
    fn begin_turn_bumps_epoch_and_clears_stale_staged() {
        let mut ctx = LLMContext::new(None);
        let e0 = ctx.epoch();
        ctx.stage_assistant_tool_calls(None, vec![tc("call_1", "lookup")]); // leftover

        let e1 = ctx.begin_turn();
        assert_eq!(e1, e0 + 1);
        assert_eq!(ctx.staged_len(), 0, "begin_turn discards a prior interrupted round");
    }
}
