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
//!
//! An optional `judge` member scripts the judge route:
//! `{"model": "...", "requests": <n>, "answers": {"<question id>": <answer>}}`.
//! Each answer has the wire shape of its type, as the [`systemone`]
//! builders make it: `noul`, `choice`, or `score`. `requests` is the
//! number of judge requests the task expects; the parser does not
//! enforce it.

use std::collections::BTreeMap;
use std::fmt;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::systemone::{self, JudgeScript};
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

/// A parsed transcript: the replies in order, and the judge script
/// when the transcript has a `judge` member.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub replies: Vec<MockResponse>,
    pub judge: Option<TranscriptJudge>,
}

/// The `judge` member of a transcript.
#[derive(Debug, Clone)]
pub struct TranscriptJudge {
    /// The model and the wire answer of each question id.
    pub script: JudgeScript,
    /// The number of judge requests the task expects.
    pub requests: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: u64,
    replies: Vec<Value>,
    #[serde(default, deserialize_with = "present")]
    judge: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JudgeDoc {
    model: String,
    requests: u64,
    #[serde(default)]
    answers: BTreeMap<String, Value>,
    #[serde(default)]
    sequence: Vec<BTreeMap<String, Value>>,
    /// Optional check overrides and rules read by the smoke runner;
    /// accepted here so the mock validates the same schema 1 document.
    #[serde(default)]
    #[allow(dead_code)]
    checks: BTreeMap<String, String>,
    #[serde(default)]
    #[allow(dead_code)]
    rules: Vec<Value>,
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

/// Parse a transcript into its replies, in order. The `judge` member,
/// when present, must be valid too; [`Transcript::parse`] returns it.
pub fn parse(text: &str) -> Result<Vec<MockResponse>, TranscriptError> {
    Transcript::parse(text).map(|transcript| transcript.replies)
}

impl Transcript {
    /// Parse a whole transcript: its replies and its judge script.
    pub fn parse(text: &str) -> Result<Self, TranscriptError> {
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
        let judge = document
            .judge
            .map(judge)
            .transpose()
            .map_err(|reason| whole(format!("judge: {reason}")))?;
        let replies = replies(document.replies)?;
        Ok(Self { replies, judge })
    }
}

/// The replies of a transcript, each with its body check.
fn replies(replies: Vec<Value>) -> Result<Vec<MockResponse>, TranscriptError> {
    if replies.is_empty() {
        return Err(TranscriptError {
            reply: None,
            reason: "replies is empty".to_string(),
        });
    }
    let mut responses = Vec::with_capacity(replies.len());
    // The call ids of the previous reply, which the next request must
    // answer.
    let mut answer: Vec<String> = Vec::new();
    for (index, value) in replies.into_iter().enumerate() {
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

/// The judge script of a `judge` member.
fn judge(value: Value) -> Result<TranscriptJudge, String> {
    let doc: JudgeDoc = serde_json::from_value(value).map_err(|e| e.to_string())?;
    if doc.model.is_empty() {
        return Err("model is empty".to_string());
    }
    if doc.answers.is_empty() && doc.sequence.is_empty() {
        return Err("answers is empty".to_string());
    }
    let answers = doc
        .answers
        .into_iter()
        .map(|(id, value)| match wire_answer(&value) {
            Ok(answer) => Ok((id, answer)),
            Err(reason) => Err(format!("answers.{id}: {reason}")),
        })
        .collect::<Result<_, String>>()?;
    let sequence = doc
        .sequence
        .into_iter()
        .enumerate()
        .map(|(step_index, map)| {
            map.into_iter()
                .map(|(id, value)| match wire_answer(&value) {
                    Ok(answer) => Ok((id, answer)),
                    Err(reason) => Err(format!("sequence[{step_index}].{id}: {reason}")),
                })
                .collect::<Result<BTreeMap<_, _>, String>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(TranscriptJudge {
        script: JudgeScript {
            model: doc.model,
            answers,
            sequence,
        },
        requests: doc.requests,
    })
}

/// The wire answer that `value` gives, rebuilt by the [`systemone`]
/// builders of its type. `value` must have exactly the members of the
/// builder's answer. A `score` answer keeps its own legend text.
fn wire_answer(value: &Value) -> Result<Value, String> {
    let Some(given) = value.as_object() else {
        return Err("expected a JSON object".to_string());
    };
    let built = match given.get("type").and_then(Value::as_str) {
        Some("noul") => systemone::noul(number(given, "noul")?),
        Some("choice") => {
            let choice = given
                .get("choice")
                .and_then(Value::as_str)
                .ok_or("choice is not a string")?;
            let probabilities = numbers(given.get("probabilities"))?;
            let pairs: Vec<(&str, f64)> = probabilities
                .iter()
                .map(|(name, p)| (name.as_str(), *p))
                .collect();
            systemone::choice(choice, &pairs, number(given, "confidence")?)
        }
        Some("score") => score_answer(given)?,
        _ => return Err("type is not noul, choice, or score".to_string()),
    };
    // Each builder makes an object.
    let members: &Map<String, Value> = built.as_object().expect("a builder makes an object");
    if let Some(unknown) = given.keys().find(|key| !members.contains_key(*key)) {
        return Err(format!("unknown field `{unknown}`"));
    }
    if let Some(missing) = members.keys().find(|key| !given.contains_key(*key)) {
        return Err(format!("missing field `{missing}`"));
    }
    Ok(built)
}

/// A `score` answer: the builder's answer with the given legend, which
/// names each level of `probabilities`, keyed from `"0"`.
fn score_answer(given: &Map<String, Value>) -> Result<Value, String> {
    let probabilities = numbers(given.get("probabilities"))?;
    let levels = (0..probabilities.len())
        .map(|level| {
            probabilities
                .get(&level.to_string())
                .copied()
                .ok_or_else(|| format!("probabilities has no level {level}; levels count from 0"))
        })
        .collect::<Result<Vec<f64>, String>>()?;
    let legend = given
        .get("legend")
        .and_then(Value::as_object)
        .filter(|legend| {
            legend.len() == levels.len()
                && legend
                    .iter()
                    .all(|(level, text)| probabilities.contains_key(level) && text.is_string())
        })
        .ok_or("legend does not hold one string for each level")?;
    let mut built = systemone::score(
        number(given, "score")?,
        &levels,
        number(given, "confidence")?,
    );
    built["legend"] = Value::Object(legend.clone());
    Ok(built)
}

/// The number in member `field`.
fn number(given: &Map<String, Value>, field: &str) -> Result<f64, String> {
    given
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{field} is not a number"))
}

/// An object of numbers, by name.
fn numbers(value: Option<&Value>) -> Result<BTreeMap<String, f64>, String> {
    let Some(object) = value.and_then(Value::as_object).filter(|o| !o.is_empty()) else {
        return Err("probabilities is not a non-empty object".to_string());
    };
    object
        .iter()
        .map(|(name, p)| match p.as_f64() {
            Some(p) => Ok((name.clone(), p)),
            None => Err(format!("probabilities.{name} is not a number")),
        })
        .collect()
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
