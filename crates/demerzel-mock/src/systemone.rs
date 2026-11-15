//! The judge route: `POST /v1/systemone`, answered by question id.
//!
//! A [`JudgeScript`] holds one wire answer per question id. Each
//! request gets the scripted answer of every question it asks, in one
//! reply, in the System One reply shape. A request without a bearer
//! key gets 401. A request with a question that has no scripted
//! answer, or whose answer has another type, gets 422. A request that
//! names another model than the script's, or that holds no question,
//! gets 422 too. Each of these counts as refused. The builders below give the wire answer of each type.

use std::collections::BTreeMap;

use axum::http::StatusCode;
use serde_json::{Map, Value, json};

/// The scripted judge of a mock server.
#[derive(Debug, Clone)]
pub struct JudgeScript {
    /// The `model` field of every reply.
    pub model: String,
    /// One wire answer per question id.
    pub answers: BTreeMap<String, Value>,
}

/// A yes-or-no answer: the probability of yes.
pub fn noul(p: f64) -> Value {
    json!({"type": "noul", "noul": p})
}

/// A choice answer: the chosen option, a probability for each option
/// by name, and the confidence.
pub fn choice(choice: &str, probabilities: &[(&str, f64)], confidence: f64) -> Value {
    let probabilities: Map<String, Value> = probabilities
        .iter()
        .map(|(name, p)| (name.to_string(), json!(p)))
        .collect();
    json!({
        "type": "choice",
        "choice": choice,
        "confidence": confidence,
        "probabilities": probabilities,
    })
}

/// A score answer: the position, a probability for each level by its
/// index from 0, and the confidence. The legend names each level by
/// its index.
pub fn score(score: f64, probabilities: &[f64], confidence: f64) -> Value {
    let mut by_level = Map::new();
    let mut legend = Map::new();
    for (level, p) in probabilities.iter().enumerate() {
        by_level.insert(level.to_string(), json!(p));
        legend.insert(level.to_string(), json!(format!("level {level}")));
    }
    json!({
        "type": "score",
        "score": score,
        "confidence": confidence,
        "legend": legend,
        "probabilities": by_level,
    })
}

/// The reply to one judge request: the 200 body, or the status and
/// body of a refusal. `authorized` says the request carried a bearer
/// key.
pub(crate) fn answer(
    script: &JudgeScript,
    authorized: bool,
    body: &Value,
) -> Result<Value, (StatusCode, Value)> {
    if !authorized {
        return Err((
            StatusCode::UNAUTHORIZED,
            json!({"detail": "the request carries no bearer key"}),
        ));
    }
    // A harness that sends another model than its config names must
    // not pass.
    let model = body.get("model").and_then(Value::as_str);
    if model != Some(script.model.as_str()) {
        return Err(invalid(
            &["body", "model"],
            "the request names another model than the judge script",
        ));
    }
    let questions = body
        .get("questions")
        .and_then(Value::as_object)
        .filter(|questions| !questions.is_empty())
        .ok_or_else(|| invalid(&["body", "questions"], "the request has no questions"))?;
    let mut answers = Map::new();
    for (id, question) in questions {
        let Some(answer) = script.answers.get(id) else {
            return Err(invalid(
                &["body", "questions", id],
                "the script has no answer for this question",
            ));
        };
        let asked = question.get("type").and_then(Value::as_str);
        if asked.is_none() || asked != answer.get("type").and_then(Value::as_str) {
            return Err(invalid(
                &["body", "questions", id, "type"],
                "the scripted answer has another type",
            ));
        }
        answers.insert(id.clone(), answer.clone());
    }
    let state_bytes = match body.get("state") {
        Some(Value::String(state)) => state.len(),
        Some(other) => other.to_string().len(),
        None => 0,
    };
    Ok(json!({
        "model": script.model,
        "answers": answers,
        "usage": {"input_tokens": state_bytes / 4, "output_tokens": 0},
    }))
}

/// A 422 body in the validation shape.
fn invalid(location: &[&str], message: &str) -> (StatusCode, Value) {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        json!({"detail": [{"loc": location, "msg": message, "type": "value_error"}]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MockResponse, MockServer};
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    fn script() -> JudgeScript {
        JudgeScript {
            model: "judge-9.9.9".into(),
            answers: BTreeMap::from([
                ("done".to_string(), noul(0.875)),
                (
                    "kind".to_string(),
                    choice("fix", &[("fix", 0.75), ("docs", 0.25)], 0.5),
                ),
                ("risk".to_string(), score(0.5, &[0.5, 0.5], 0.25)),
            ]),
        }
    }

    fn request(ids: &[(&str, &str)]) -> Value {
        let questions: Map<String, Value> = ids
            .iter()
            .map(|(id, kind)| (id.to_string(), json!({"type": kind, "instructions": "q"})))
            .collect();
        json!({"model": "judge-9.9.9", "state": "twelve bytes", "questions": questions})
    }

    #[test]
    fn each_question_gets_its_scripted_answer() {
        let body = request(&[("done", "noul"), ("kind", "choice"), ("risk", "score")]);
        let reply = answer(&script(), true, &body).unwrap();
        assert_eq!(reply["model"], "judge-9.9.9");
        assert_eq!(reply["answers"]["done"], noul(0.875));
        assert_eq!(reply["answers"]["kind"]["probabilities"]["docs"], 0.25);
        assert_eq!(reply["answers"]["risk"]["probabilities"]["1"], 0.5);
        assert_eq!(reply["usage"]["input_tokens"], 3);
        // Only the asked questions get an answer.
        let reply = answer(&script(), true, &request(&[("done", "noul")])).unwrap();
        assert_eq!(reply["answers"].as_object().unwrap().len(), 1);
    }

    #[test]
    fn a_missing_key_answer_or_type_is_refused() {
        let body = request(&[("done", "noul")]);
        let (status, _) = answer(&script(), false, &body).unwrap_err();
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        for body in [
            request(&[("done", "noul"), ("other", "noul")]),
            request(&[("done", "choice")]),
            request(&[]),
            json!({"state": "s"}),
            json!({"model": "judge-0.0.1", "state": "s", "questions": {"done": {"type": "noul"}}}),
            json!({"state": "s", "questions": {"done": {"type": "noul"}}}),
        ] {
            let (status, detail) = answer(&script(), true, &body).unwrap_err();
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
            assert!(detail["detail"][0]["loc"].is_array(), "{detail}");
        }
    }

    /// One HTTP/1.1 POST with an optional bearer key: the status and
    /// the body. Blocking, so the test runs it off the runtime.
    fn post(url: &str, path: &str, key: Option<&str>, body: &Value) -> (u16, Value) {
        let addr = url.strip_prefix("http://").unwrap();
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let body = body.to_string();
        let auth = key.map_or(String::new(), |key| {
            format!("Authorization: Bearer {key}\r\n")
        });
        write!(
            stream,
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\n{auth}Content-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, serde_json::from_str(body).unwrap_or(Value::Null))
    }

    async fn post_async(url: String, path: &'static str, key: Option<&'static str>) -> u16 {
        let body = request(&[("done", "noul")]);
        tokio::task::spawn_blocking(move || post(&url, path, key, &body).0)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn the_server_counts_answered_and_refused_requests() {
        // The chat server serves the judge under `/judge`; the scripted
        // chat reply stays untouched.
        let server = MockServer::start_chat(vec![MockResponse::error(500)]).await;
        let url = server.url();
        // Without a script the judge route is like any other path.
        assert_eq!(
            post_async(url.clone(), "/judge/v1/systemone", Some("fake-key")).await,
            404
        );
        assert_eq!((server.judged(), server.refused()), (0, 1));

        server.set_judge(script());
        assert_eq!(
            post_async(url.clone(), "/judge/v1/systemone", Some("fake-key")).await,
            200
        );
        assert_eq!(
            post_async(url.clone(), "/judge/v1/systemone", None).await,
            401
        );
        // The judge does not answer on the chat server's root.
        assert_eq!(
            post_async(url.clone(), "/v1/systemone", Some("fake-key")).await,
            404
        );
        assert_eq!(server.judged(), 1);
        assert_eq!(server.refused(), 3);
        assert_eq!(server.served(), 0);
        let received = server.received();
        assert_eq!(received.len(), 2);
        assert_eq!(received[0].path, "/judge/v1/systemone");
        assert_eq!(
            received[0].authorization.as_deref(),
            Some("Bearer fake-key")
        );
        assert_eq!(received[0].body["questions"]["done"]["type"], "noul");
    }

    #[tokio::test]
    async fn the_all_routes_server_replays_the_list_until_a_script_is_set() {
        let server = MockServer::start(
            vec![
                MockResponse::json(json!({"listed": true})),
                MockResponse::error(529),
            ],
            Vec::new(),
        )
        .await;
        let url = server.url();
        let reply = {
            let url = url.clone();
            tokio::task::spawn_blocking(move || {
                post(&url, "/v1/systemone", Some("fake-key"), &json!({}))
            })
            .await
            .unwrap()
        };
        assert_eq!(reply, (200, json!({"listed": true})));
        assert_eq!(
            post_async(url.clone(), "/v1/systemone", Some("fake-key")).await,
            529
        );
        assert_eq!(server.served(), 2);
        server.set_judge(script());
        assert_eq!(
            post_async(url.clone(), "/v1/systemone", Some("fake-key")).await,
            200
        );
        assert_eq!((server.judged(), server.served()), (1, 2));
    }
}
