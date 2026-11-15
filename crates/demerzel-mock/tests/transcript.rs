//! The transcript format: what parses, and what the parser refuses.

use demerzel_mock::SseEvent;
use demerzel_mock::systemone::{choice, noul, score};
use demerzel_mock::transcript::{Transcript, parse};
use serde_json::{Value, json};

fn doc(replies: Value) -> String {
    json!({"schema": 1, "replies": replies}).to_string()
}

fn chunks(events: &[SseEvent]) -> Vec<Value> {
    events
        .iter()
        .filter_map(|event| serde_json::from_str(&event.data).ok())
        .collect()
}

#[test]
fn text_one_call_and_two_calls_parse_to_one_response_each() {
    let responses = parse(&doc(json!([
        {"text": "hello"},
        {"tool_calls": [{"id": "call_1", "name": "read", "input": {"path": "a"}}]},
        {"tool_calls": [
            {"id": "call_2", "name": "read", "input": {"path": "b"}},
            {"id": "call_3", "name": "grep", "input": {"pattern": "c"}},
        ]},
    ])))
    .unwrap();
    assert_eq!(responses.len(), 3);
    assert!(responses.iter().all(|response| response.status == 200));
}

#[test]
fn a_two_call_reply_streams_wire_index_0_then_1() {
    let responses = parse(&doc(json!([{"tool_calls": [
        {"id": "call_a", "name": "read", "input": {"path": "a"}},
        {"id": "call_b", "name": "grep", "input": {"pattern": "b"}},
    ]}])))
    .unwrap();
    let indexes: Vec<u64> = chunks(&responses[0].events)
        .iter()
        .flat_map(|chunk| chunk["choices"].as_array().cloned().unwrap_or_default())
        .flat_map(|choice| {
            choice["delta"]["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|fragment| fragment["index"].as_u64())
        .collect();
    assert!(indexes.contains(&0) && indexes.contains(&1));
    assert!(indexes.is_sorted(), "calls stream one after the other");
}

#[test]
fn input_with_multibyte_characters_parses() {
    // The argument text splits mid-way; the midpoint here falls inside
    // a two-byte character.
    let responses = parse(&doc(json!([
        {"tool_calls": [{"id": "c", "name": "write", "input": {"s": "ééééé"}}]},
    ])))
    .unwrap();
    let arguments: String = chunks(&responses[0].events)
        .iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).unwrap(),
        json!({"s": "ééééé"})
    );
}

#[test]
fn malformed_documents_are_refused_without_a_reply_index() {
    for text in [
        "not json".to_string(),
        json!({"schema": 2, "replies": [{"text": "x"}]}).to_string(),
        json!({"replies": [{"text": "x"}]}).to_string(),
        json!({"schema": 1}).to_string(),
        json!({"schema": 1, "replies": []}).to_string(),
        json!({"schema": 1, "replies": [{"text": "x"}], "extra": true}).to_string(),
    ] {
        let error = parse(&text)
            .err()
            .unwrap_or_else(|| panic!("accepted {text}"));
        assert_eq!(error.reply, None, "{text}");
    }
}

#[test]
fn malformed_replies_are_refused_with_their_index() {
    let call = json!({"id": "c", "name": "read", "input": {}});
    for reply in [
        json!({}),
        json!({"text": "x", "tool_calls": [call]}),
        json!({"text": null}),
        json!({"text": "x", "extra": 1}),
        json!({"tool_calls": []}),
        json!({"tool_calls": [{"id": "c", "name": "read", "input": "{}"}]}),
        json!({"tool_calls": [{"id": "c", "name": "read", "input": [1]}]}),
        json!({"tool_calls": [{"id": "c", "name": "read"}]}),
        json!({"tool_calls": [{"id": "c", "name": "read", "input": {}, "extra": 1}]}),
        json!("text"),
    ] {
        let error = parse(&doc(json!([{"text": "ok"}, reply])))
            .err()
            .unwrap_or_else(|| panic!("accepted {reply}"));
        assert_eq!(error.reply, Some(1), "{reply}");
    }
}

#[test]
fn a_judge_member_parses_to_the_builders_wire_answers() {
    let mut legend_score = score(1.5, &[0.25, 0.25, 0.5], 0.75);
    legend_score["legend"] = json!({"0": "low", "1": "mid", "2": "high"});
    let text = json!({
        "schema": 1,
        "replies": [{"text": "x"}],
        "judge": {
            "model": "judge-9.9.9",
            "requests": 0,
            "answers": {
                "done": {"type": "noul", "noul": 1},
                "kind": choice("fix", &[("fix", 0.75), ("docs", 0.25)], 0.5),
                "risk": legend_score,
            },
        },
    })
    .to_string();
    let transcript = Transcript::parse(&text).unwrap();
    assert_eq!(transcript.replies.len(), 1);
    let judge = transcript.judge.expect("the judge member");
    assert_eq!(
        (judge.requests, judge.script.model.as_str()),
        (0, "judge-9.9.9")
    );
    let answers = &judge.script.answers;
    assert_eq!(answers["done"], noul(1.0));
    assert_eq!(answers["kind"]["probabilities"]["docs"], 0.25);
    // A score answer keeps its own legend text.
    assert_eq!(answers["risk"], legend_score);
    // A score legend must name each level.
    let mut short = legend_score.clone();
    short["legend"] = json!({"0": "low"});
    let text = json!({"schema": 1, "replies": [{"text": "x"}],
        "judge": {"model": "m", "requests": 1, "answers": {"risk": short}}});
    assert!(Transcript::parse(&text.to_string()).is_err());
    // Without the member there is no judge script.
    let plain = Transcript::parse(&doc(json!([{"text": "x"}]))).unwrap();
    assert!(plain.judge.is_none());
}
