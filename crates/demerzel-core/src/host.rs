//! The loop seam: one host beside the engine that sees each loop event
//! and can tighten or loosen a tool call's approval within fixed
//! limits.
//!
//! The engine calls the host at four events: `turn_start`,
//! `tool_before`, `tool_after`, and `turn_end`. Each method has a
//! default that returns at once and changes nothing, so a host
//! implements only the events it needs. Without a host the engine
//! takes one branch per event and does no other work.
//!
//! At `tool_before` the host receives the policy's decision and can
//! return a changed one. The engine accepts only these changes: ask to
//! allow, allow to ask, and any decision to deny. A host cannot turn a
//! refusal into a run.

use serde_json::Value;

use std::collections::VecDeque;

use crate::approval::{ApprovalDecision, ApprovalMode, ToolAccess};
use crate::engine::TurnOutcome;
use crate::message::{ContentBlock, Message, Role};
use crate::tool::{BoxFuture, ToolOutput};

/// The loop seam. Every method has a default that returns at once and
/// changes nothing.
pub trait LoopHost: Send + Sync {
    /// A user turn starts, before its message enters the history.
    fn turn_start<'a>(&'a self, _event: &'a TurnStart<'a>) -> BoxFuture<'a, HostReply> {
        Box::pin(std::future::ready(HostReply::default()))
    }

    /// A known tool call is about to dispatch, after the policy
    /// decided and before the user is asked. The returned decision
    /// passes through the engine's limits; see the module docs.
    fn tool_before<'a>(&'a self, event: &'a ToolBefore<'a>) -> BoxFuture<'a, ToolGate> {
        Box::pin(std::future::ready(ToolGate::keep(event.decision)))
    }

    /// A tool call ran and returned its output. Calls that the policy,
    /// the user, or an abort refused do not reach this method.
    ///
    /// The future is `'static`: an engine may hold it until the batch
    /// ends.
    fn tool_after(&self, _event: &ToolAfter<'_>) -> BoxFuture<'static, HostReply> {
        Box::pin(std::future::ready(HostReply::default()))
    }

    /// A user turn ended, after its turn record: a turn that returned
    /// an outcome, or one that a provider error ended. A dropped turn
    /// and a session failure do not reach this method.
    fn turn_end(&self, _event: &TurnEnd<'_>) -> BoxFuture<'static, HostReply> {
        Box::pin(std::future::ready(HostReply::default()))
    }
}

/// The `turn_start` event.
#[derive(Debug)]
#[non_exhaustive]
pub struct TurnStart<'a> {
    /// The user's text for the turn.
    pub prompt: &'a str,
}

/// The `tool_before` event: one known tool call and the policy's
/// decision for it.
#[derive(Debug)]
#[non_exhaustive]
pub struct ToolBefore<'a> {
    pub call_id: &'a str,
    pub tool: &'a str,
    pub input: &'a Value,
    pub access: ToolAccess,
    pub mode: ApprovalMode,
    /// The policy's decision for this call.
    pub decision: ApprovalDecision,
    pub turn: TurnView<'a>,
}

/// The `tool_after` event: one call that ran, with its output.
#[derive(Debug)]
#[non_exhaustive]
pub struct ToolAfter<'a> {
    pub call_id: &'a str,
    pub tool: &'a str,
    pub input: &'a Value,
    /// `None` for a name the tool host does not hold.
    pub access: Option<ToolAccess>,
    pub output: &'a ToolOutput,
    pub turn: TurnView<'a>,
}

/// The `turn_end` event.
#[derive(Debug)]
#[non_exhaustive]
pub struct TurnEnd<'a> {
    /// How the turn ended.
    pub outcome: TurnEndOutcome,
    pub turn: TurnView<'a>,
}

/// How a turn that reaches `turn_end` ended: an outcome of the engine,
/// or a provider failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TurnEndOutcome {
    /// The turn returned this outcome.
    Outcome(TurnOutcome),
    /// A provider error ended the turn.
    ProviderError,
}

impl TurnEndOutcome {
    /// The turn record's vocabulary: [`TurnOutcome::as_str`], or
    /// `error` for a provider failure.
    pub const fn as_str(self) -> &'static str {
        match self {
            TurnEndOutcome::Outcome(outcome) => outcome.as_str(),
            TurnEndOutcome::ProviderError => "error",
        }
    }
}

/// What a host returns at `turn_start`, `tool_after`, and `turn_end`.
/// It holds nothing yet; later fields stay additive.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct HostReply {}

/// What a host returns at `tool_before`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ToolGate {
    /// The decision the host asks for. The engine applies it within
    /// its limits.
    pub decision: ApprovalDecision,
}

impl ToolGate {
    /// A gate that keeps `decision`.
    pub fn keep(decision: ApprovalDecision) -> Self {
        Self { decision }
    }
}

/// The current turn as the engine holds it: its messages from the
/// turn's user message on, and the result blocks of the running batch
/// that are answered but not yet in the history.
#[derive(Debug, Clone, Copy)]
pub struct TurnView<'a> {
    messages: &'a [Message],
    answered: &'a [ContentBlock],
}

impl<'a> TurnView<'a> {
    pub fn new(messages: &'a [Message], answered: &'a [ContentBlock]) -> Self {
        Self { messages, answered }
    }

    /// The last text block of the turn's user message; empty when the
    /// view holds no user message.
    pub fn prompt(&self) -> &'a str {
        let Some(first) = self.messages.first().filter(|m| m.role == Role::User) else {
            return "";
        };
        first
            .content
            .iter()
            .rev()
            .find_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or("")
    }

    /// Each answered call of the turn, in the order of its answers:
    /// `tool_use` blocks paired with their results by id, the first
    /// open call first. A result whose call is outside the turn, such
    /// as a repair of a cut-off call, is not a call of the turn.
    pub fn calls(&self) -> impl Iterator<Item = CallSeen<'a>> + use<'a> {
        let mut open: VecDeque<(&'a str, &'a str, &'a Value)> = VecDeque::new();
        let mut seen = Vec::new();
        let blocks = self
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .chain(self.answered.iter());
        for block in blocks {
            match block {
                ContentBlock::ToolUse { id, name, input } => open.push_back((id, name, input)),
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    // Results usually answer the oldest open call:
                    // that match costs nothing, others a scan.
                    let position = match open.front() {
                        Some((id, ..)) if id == tool_use_id => Some(0),
                        _ => open.iter().position(|(id, ..)| id == tool_use_id),
                    };
                    if let Some((id, name, input)) = position.and_then(|p| open.remove(p)) {
                        seen.push(CallSeen {
                            id,
                            name,
                            input,
                            is_error: *is_error,
                            output: content,
                        });
                    }
                }
                _ => {}
            }
        }
        seen.into_iter()
    }

    /// The text of the turn's last assistant message, its text blocks
    /// joined by newlines; `None` when the turn has no assistant text.
    pub fn final_text(&self) -> Option<String> {
        let last = self
            .messages
            .iter()
            .rev()
            .find(|message| message.role == Role::Assistant)?;
        let mut texts = last.content.iter().filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        });
        let mut joined = texts.next()?.to_string();
        for text in texts {
            joined.push('\n');
            joined.push_str(text);
        }
        Some(joined)
    }
}

/// One answered call of the turn, as [`TurnView::calls`] gives it.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CallSeen<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub input: &'a Value,
    pub is_error: bool,
    /// The result text the model receives.
    pub output: &'a str,
}

/// The decision that stands after a host's gate: the host can turn an
/// ask into an allow, an allow into an ask, and any decision into a
/// deny. It cannot lift a deny.
pub(crate) fn constrain(policy: ApprovalDecision, host: ApprovalDecision) -> ApprovalDecision {
    use ApprovalDecision::{Allow, AskUser, Deny};
    match (policy, host) {
        (_, Deny) | (Deny, _) => Deny,
        (AskUser, Allow) | (Allow, AskUser) => host,
        _ => policy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use ApprovalDecision::{Allow, AskUser, Deny};

    #[test]
    fn constrain_accepts_the_allowed_changes_and_refuses_lifting_a_deny() {
        // (policy, host, result)
        let matrix = [
            (Allow, Allow, Allow),
            (Allow, AskUser, AskUser),
            (Allow, Deny, Deny),
            (AskUser, Allow, Allow),
            (AskUser, AskUser, AskUser),
            (AskUser, Deny, Deny),
            (Deny, Deny, Deny),
            // Refused: a host cannot turn a refusal into a run or an ask.
            (Deny, Allow, Deny),
            (Deny, AskUser, Deny),
        ];
        for (policy, host, result) in matrix {
            assert_eq!(constrain(policy, host), result, "{policy:?} -> {host:?}");
        }
    }

    fn text(role: Role, text: &str) -> Message {
        Message {
            role,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    fn call(id: &str, name: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input: json!({"id": id}),
        }
    }

    fn result(id: &str, output: &str, is_error: bool) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: id.into(),
            content: output.into(),
            is_error,
        }
    }

    #[test]
    fn a_view_pairs_calls_with_results_across_the_history_and_the_batch() {
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![
                    // A repair of a call from an earlier turn.
                    result("old", "interrupted", true),
                    ContentBlock::Text {
                        text: "first".into(),
                    },
                    ContentBlock::Text { text: "go".into() },
                ],
            },
            Message {
                role: Role::Assistant,
                content: vec![call("c1", "read")],
            },
            Message {
                role: Role::User,
                content: vec![result("c1", "one", false)],
            },
            // The running batch reuses an id; its first result is
            // answered, the second call is not yet.
            Message {
                role: Role::Assistant,
                content: vec![call("c2", "bash"), call("c2", "edit")],
            },
        ];
        let answered = [result("c2", "two", true)];
        let view = TurnView::new(&messages, &answered);
        assert_eq!(view.prompt(), "go");
        let calls: Vec<(&str, &str, bool, &str)> = view
            .calls()
            .map(|call| (call.id, call.name, call.is_error, call.output))
            .collect();
        assert_eq!(
            calls,
            [("c1", "read", false, "one"), ("c2", "bash", true, "two")]
        );
        assert_eq!(view.calls().nth(1).unwrap().input, &json!({"id": "c2"}));
        // No assistant text in the turn.
        assert_eq!(view.final_text(), None);
    }

    #[test]
    fn a_view_gives_the_last_assistant_text_joined() {
        let messages = vec![
            text(Role::User, "go"),
            text(Role::Assistant, "earlier"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text { text: "a".into() },
                    call("c1", "read"),
                    ContentBlock::Text { text: "b".into() },
                ],
            },
        ];
        let view = TurnView::new(&messages, &[]);
        assert_eq!(view.final_text().as_deref(), Some("a\nb"));
        assert_eq!(view.calls().count(), 0);
        // An empty view has no prompt, no calls, and no final text.
        let empty = TurnView::new(&[], &[]);
        assert_eq!(empty.prompt(), "");
        assert_eq!(empty.calls().count(), 0);
        assert_eq!(empty.final_text(), None);
    }

    /// A host with every default.
    struct Defaults;

    impl LoopHost for Defaults {}

    #[test]
    fn the_defaults_change_nothing() {
        let host: Box<dyn LoopHost> = Box::new(Defaults);
        let input = json!({});
        let event = ToolBefore {
            call_id: "c1",
            tool: "write",
            input: &input,
            access: ToolAccess::Write,
            mode: ApprovalMode::Ask,
            decision: AskUser,
            turn: TurnView::new(&[], &[]),
        };
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        let gate = host.tool_before(&event).as_mut().poll(&mut cx);
        assert_eq!(gate, std::task::Poll::Ready(ToolGate::keep(AskUser)));
    }
}
