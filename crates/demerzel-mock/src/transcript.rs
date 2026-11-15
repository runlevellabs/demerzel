//! The scripted model transcript that the `demerzel-mock` binary
//! replays.
//!
//! A transcript is one JSON document, `{"schema": 1, "replies": [...]}`.
//! Each reply is `{"text": "..."}` or
//! `{"tool_calls": [{"id": "...", "name": "...", "input": {...}}]}`,
//! and becomes one Chat Completions stream from the [`compat`]
//! builders. Several calls in one reply are parallel calls in one
//! assistant message. The parser refuses unknown fields.
//!
//! The reply after a `tool_calls` reply carries a body check: after its
//! last `assistant` message, the request must hold exactly one `tool`
//! message for each call id of that reply. A request that fails the
//! check gets status 400 and takes the reply out of the script.

use std::fmt;

use serde::Deserialize;
use serde_json::Value;

use crate::{MockResponse, compat};

/// The transcript schema version this parser reads.
pub const SCHEMA: u64 = 1;

/// Why a transcript was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptError {
    /// The zero-based index of the reply at fault, when one reply is.
    pub reply: Option<usize>,
    pub reason: String,
}

impl fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.reply {
            Some(index) => write!(f, "replies[{index}]: {}", self.reason),
            None => f.write_str(&self.reason),
        }
    }
}

impl std::error::Error for TranscriptError {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: u64,
    replies: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    #[serde(default, deserialize_with = "present")]
    text: Option<String>,
    #[serde(default, deserialize_with = "present")]
    tool_calls: Option<Vec<Call>>,
}

/// A field that may be absent but is never `null`.
fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Call {
    id: String,
    name: String,
    input: Value,
}

/// Parse a transcript into its replies, in order.
pub fn parse(text: &str) -> Result<Vec<MockResponse>, TranscriptError> {
    let whole = |reason: String| TranscriptError {
        reply: None,
        reason,
    };
    let document: Document = serde_json::from_str(text).map_err(|e| whole(e.to_string()))?;
    if document.schema != SCHEMA {
        return Err(whole(format!(
            "schema {} is not supported; expected {SCHEMA}",
            document.schema
        )));
    }
    if document.replies.is_empty() {
        return Err(whole("replies is empty".to_string()));
    }
    let mut responses = Vec::with_capacity(document.replies.len());
    // The call ids of the previous reply, which the next request must
    // answer.
    let mut answer: Vec<String> = Vec::new();
    for (index, value) in document.replies.into_iter().enumerate() {
        let (response, ids) = reply(value).map_err(|reason| TranscriptError {
            reply: Some(index),
            reason,
        })?;
        let expected = std::mem::replace(&mut answer, ids);
        responses.push(if expected.is_empty() {
            response
        } else {
            response.with_check(move |body| tool_results(body, &expected))
        });
    }
    Ok(responses)
}

/// One reply and the ids of the calls it makes.
fn reply(value: Value) -> Result<(MockResponse, Vec<String>), String> {
    let reply: Reply = serde_json::from_value(value).map_err(|e| e.to_string())?;
    match (reply.text, reply.tool_calls) {
        (Some(text), None) => Ok((compat::text_turn(&text), Vec::new())),
        (None, Some(calls)) => {
            if calls.is_empty() {
                return Err("tool_calls is empty".to_string());
            }
            if let Some(index) = calls.iter().position(|call| !call.input.is_object()) {
                return Err(format!("tool_calls[{index}].input is not a JSON object"));
            }
            let wire: Vec<(&str, &str, &Value)> = calls
                .iter()
                .map(|call| (call.id.as_str(), call.name.as_str(), &call.input))
                .collect();
            let response = compat::tool_calls_turn(&wire);
            Ok((response, calls.into_iter().map(|call| call.id).collect()))
        }
        (Some(_), Some(_)) => Err("a reply has both text and tool_calls".to_string()),
        (None, None) => Err("a reply has neither text nor tool_calls".to_string()),
    }
}

/// Check that a Chat Completions request answers each call in `ids`:
/// after its last `assistant` message, exactly one `tool` message has
/// each id as its `tool_call_id`.
fn tool_results(body: &Value, ids: &[String]) -> Result<(), String> {
    let messages = body["messages"].as_array().map_or(&[][..], Vec::as_slice);
    let Some(last) = messages
        .iter()
        .rposition(|message| message["role"] == "assistant")
    else {
        return Err("the request has no assistant message".to_string());
    };
    let answered: Vec<&str> = messages[last + 1..]
        .iter()
        .filter(|message| message["role"] == "tool")
        .filter_map(|message| message["tool_call_id"].as_str())
        .collect();
    let wrong: Vec<&str> = ids
        .iter()
        .map(String::as_str)
        .filter(|id| answered.iter().filter(|answer| *answer == id).count() != 1)
        .collect();
    if wrong.is_empty() {
        Ok(())
    } else {
        let wrong = serde_json::to_string(&wrong).unwrap_or_default();
        Err(format!(
            "these call ids need exactly one tool message after the last assistant message: {wrong}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::tool_results;

    fn ids(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    fn assistant(calls: &[&str]) -> serde_json::Value {
        let calls: Vec<_> = calls
            .iter()
            .map(|id| json!({"id": id, "type": "function", "function": {"name": "read", "arguments": "{}"}}))
            .collect();
        json!({"role": "assistant", "content": null, "tool_calls": calls})
    }

    fn tool(id: &str) -> serde_json::Value {
        json!({"role": "tool", "tool_call_id": id, "content": "ok"})
    }

    #[test]
    fn each_call_answered_once_after_the_last_assistant_message_passes() {
        let body = json!({"messages": [
            {"role": "user", "content": "go"},
            assistant(&["a", "b"]),
            tool("a"),
            tool("b"),
            {"role": "user", "content": "a note after the results"},
        ]});
        assert_eq!(tool_results(&body, &ids(&["a", "b"])), Ok(()));
    }

    #[test]
    fn a_missing_stale_or_repeated_result_fails() {
        for messages in [
            json!([{"role": "user", "content": "go"}]),
            json!([assistant(&["a", "b"]), tool("a")]),
            json!([assistant(&["a", "b"]), tool("a"), tool("b"), tool("b")]),
            json!([assistant(&["a", "b"]), tool("a"), tool("b"), {"role": "assistant", "content": "x"}]),
        ] {
            let body = json!({ "messages": messages });
            assert!(tool_results(&body, &ids(&["a", "b"])).is_err(), "{body}");
        }
    }
}
