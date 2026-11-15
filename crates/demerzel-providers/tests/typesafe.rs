//! Conformance tests for the `typesafe` judge against the mock System
//! One route. Hermetic: no network, no model.
//!
//! The timeouts and waits stay short so the file runs in a few
//! seconds; each bound on elapsed time keeps a wide margin for a
//! loaded machine.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use demerzel_core::{Answer, Judge, JudgeError, JudgeRequest, JudgeUsage, Question};
use demerzel_mock::systemone::{self, JudgeScript};
use demerzel_mock::{MockResponse, MockServer};
use demerzel_providers::typesafe::{self, PROVIDER_ID, TypeSafeConfig, TypeSafeJudge};
use demerzel_providers::{KeyCheck, ProviderError, verify_instance_key};
use serde_json::{Value, json};

/// A clearly fake key. The mock echoes it into error bodies, so each
/// error test also checks the redaction.
const KEY: &str = "fake-judge-key-0000";

/// The budget of most requests: long enough for a few local round
/// trips on a loaded machine.
const BUDGET: Duration = Duration::from_secs(2);

fn fixture(name: &str) -> Value {
    let text = match name {
        "noul" => include_str!("fixtures/typesafe/noul.json"),
        "choice" => include_str!("fixtures/typesafe/choice.json"),
        "score" => include_str!("fixtures/typesafe/score.json"),
        "unknown_field" => include_str!("fixtures/typesafe/unknown_field.json"),
        "model_pinned" => include_str!("fixtures/typesafe/model_pinned.json"),
        "model_alias" => include_str!("fixtures/typesafe/model_alias.json"),
        other => panic!("no fixture {other}"),
    };
    serde_json::from_str(text).unwrap()
}

fn judge_with(mock: &MockServer, change: impl FnOnce(&mut TypeSafeConfig)) -> TypeSafeJudge {
    let mut config = TypeSafeConfig::new("judge", format!("{}/", mock.url()), "jev-1.14.0");
    config.base_delay = Duration::from_millis(1);
    change(&mut config);
    TypeSafeJudge::new(config, KEY.to_string())
}

fn judge(mock: &MockServer) -> TypeSafeJudge {
    judge_with(mock, |_| {})
}

fn ask_one(id: &str, question: Question, timeout: Duration) -> JudgeRequest {
    let mut request = JudgeRequest::new("the loop state".into(), timeout);
    request.questions.insert(id.into(), question);
    request
}

fn tests_pass() -> JudgeRequest {
    ask_one(
        "tests_pass",
        Question::yes_no("Did the tests pass?"),
        BUDGET,
    )
}

fn named(names: &[&str]) -> Vec<(String, Option<String>)> {
    names.iter().map(|name| (name.to_string(), None)).collect()
}

fn levels(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| name.to_string()).collect()
}

/// The error class and text of a failed ask, with the key check.
fn transport(error: JudgeError) -> ProviderError {
    assert!(!error.to_string().contains(KEY), "{error}");
    match error {
        JudgeError::Transport(error) => error,
        other => panic!("expected a transport error, got {other:?}"),
    }
}

#[tokio::test]
async fn each_answer_type_maps_from_its_fixture() {
    let mock = MockServer::start(
        vec![
            MockResponse::json(fixture("noul")),
            MockResponse::json(fixture("choice")),
            MockResponse::json(fixture("score")),
        ],
        Vec::new(),
    )
    .await;
    let judge: Arc<dyn Judge> = Arc::new(judge(&mock));
    assert_eq!(judge.id(), PROVIDER_ID);

    let reply = judge.ask(tests_pass()).await.unwrap();
    assert_eq!(reply.answers["tests_pass"], Answer::yes_no(0.9125));
    assert_eq!(reply.model, "jev-1.14.0");
    assert_eq!(reply.usage, JudgeUsage::new(214, 3));

    let options = named(&["edit_code", "run_tests", "stop"]);
    let choice = Question::choice("What comes next?", options).unwrap();
    let reply = judge
        .ask(ask_one("next_step", choice, BUDGET))
        .await
        .unwrap();
    let probabilities = BTreeMap::from([
        ("edit_code".to_string(), 0.07),
        ("run_tests".to_string(), 0.86),
        ("stop".to_string(), 0.07),
    ]);
    assert_eq!(
        reply.answers["next_step"],
        Answer::choice("run_tests".into(), probabilities, Some(0.82))
    );
    assert_eq!(reply.usage, JudgeUsage::new(388, 5));

    let scope = levels(&["One file", "One module", "Several modules"]);
    let score = Question::score("How wide is the change?", scope).unwrap();
    let reply = judge
        .ask(ask_one("change_scope", score, BUDGET))
        .await
        .unwrap();
    assert_eq!(
        reply.answers["change_scope"],
        Answer::score(1.3, vec![0.15, 0.4, 0.45], Some(0.71))
    );
}

#[tokio::test]
async fn unknown_fields_at_the_top_and_in_an_answer_are_ignored() {
    let mock = MockServer::start(
        vec![MockResponse::json(fixture("unknown_field"))],
        Vec::new(),
    )
    .await;
    let reply = judge(&mock).ask(tests_pass()).await.unwrap();
    assert_eq!(reply.answers["tests_pass"], Answer::yes_no(0.25));
    assert_eq!(reply.usage, JudgeUsage::new(96, 2));
}

#[tokio::test]
async fn the_reply_keeps_a_pinned_or_an_alias_model_id() {
    let mock = MockServer::start(
        vec![
            MockResponse::json(fixture("model_pinned")),
            MockResponse::json(fixture("model_alias")),
        ],
        Vec::new(),
    )
    .await;
    let judge = judge(&mock);
    let pinned = judge.ask(tests_pass()).await.unwrap();
    assert_eq!(pinned.model, "jev-1.14.0");
    assert!(!typesafe::is_alias(&pinned.model));
    let alias = judge.ask(tests_pass()).await.unwrap();
    assert_eq!(alias.model, "jev-latest");
    assert!(typesafe::is_alias(&alias.model));
    assert!(typesafe::is_alias(typesafe::DEFAULT_MODEL));
}

#[tokio::test]
async fn one_request_carries_every_question_under_its_id_with_its_wire_type() {
    let mock = MockServer::start(Vec::new(), Vec::new()).await;
    mock.set_judge(JudgeScript {
        model: "jev-1.14.0".into(),
        answers: BTreeMap::from([
            ("plain".to_string(), systemone::noul(0.5)),
            ("framed".to_string(), systemone::noul(0.125)),
            (
                "pick".to_string(),
                systemone::choice("b", &[("b", 0.75), ("a", 0.25)], 0.5),
            ),
            (
                "rate".to_string(),
                systemone::score(0.25, &[0.75, 0.25], 0.75),
            ),
        ]),
    });
    let mut request = JudgeRequest::new("state text".into(), BUDGET);
    request
        .questions
        .insert("plain".into(), Question::yes_no("Plain?"));
    request.questions.insert(
        "framed".into(),
        Question::yes_no_with("Framed?", "it is framed", "it is not"),
    );
    let options = vec![
        ("b".to_string(), Some("the second".to_string())),
        ("a".to_string(), None),
    ];
    request.questions.insert(
        "pick".into(),
        Question::choice("Which one?", options).unwrap(),
    );
    request.questions.insert(
        "rate".into(),
        Question::score("How much?", levels(&["little", "much"])).unwrap(),
    );

    let reply = judge(&mock).ask(request).await.unwrap();
    assert_eq!(reply.answers.len(), 4);
    assert_eq!(reply.answers["framed"], Answer::yes_no(0.125));
    assert_eq!(
        reply.answers["rate"],
        Answer::score(0.25, vec![0.75, 0.25], Some(0.75))
    );
    assert_eq!(mock.judged(), 1);

    let received = mock.received();
    assert_eq!(received.len(), 1);
    let sent = &received[0];
    assert_eq!(sent.path, "/v1/systemone");
    assert_eq!(
        sent.authorization.as_deref(),
        Some(format!("Bearer {KEY}").as_str())
    );
    assert_eq!(sent.content_type.as_deref(), Some("application/json"));
    let body = &sent.body;
    assert_eq!(body["model"], "jev-1.14.0");
    assert_eq!(body["state"], "state text");
    let questions = body["questions"].as_object().unwrap();
    assert_eq!(questions.len(), 4);
    assert_eq!(
        questions["plain"],
        json!({"type": "noul", "instructions": "Plain?"})
    );
    assert_eq!(
        questions["framed"],
        json!({"type": "noul", "instructions": "Framed?",
               "criteria": {"true": "it is framed", "false": "it is not"}})
    );
    assert_eq!(
        questions["pick"],
        json!({"type": "choice", "instructions": "Which one?",
               "criteria": {"b": "the second", "a": null}})
    );
    assert_eq!(
        questions["rate"],
        json!({"type": "score", "instructions": "How much?", "criteria": ["little", "much"]})
    );
}

#[tokio::test]
async fn a_request_error_fails_on_the_first_attempt() {
    for status in [400, 401, 408, 413, 415, 422] {
        let mock = MockServer::start(
            vec![
                MockResponse::error(status),
                MockResponse::json(fixture("noul")),
            ],
            Vec::new(),
        )
        .await;
        let error = transport(judge(&mock).ask(tests_pass()).await.unwrap_err());
        match status {
            401 => assert!(matches!(error, ProviderError::Auth(_)), "{error:?}"),
            _ => assert!(
                matches!(&error, ProviderError::InvalidRequest(text)
                    if text.contains(&status.to_string())),
                "{status}: {error:?}"
            ),
        }
        assert_eq!(mock.received().len(), 1, "{status}");
    }
}

#[tokio::test]
async fn a_429_honors_retry_after_and_then_answers() {
    let mock = MockServer::start(
        vec![
            MockResponse::error_with_retry_after(429, 0),
            MockResponse::json(fixture("noul")),
        ],
        Vec::new(),
    )
    .await;
    // The backoff alone would wait past the budget: only the server's
    // `retry-after: 0` lets the second attempt run.
    let judge = judge_with(&mock, |config| config.base_delay = Duration::from_secs(30));
    let reply = judge.ask(tests_pass()).await.unwrap();
    assert_eq!(reply.answers["tests_pass"], Answer::yes_no(0.9125));
    assert_eq!(mock.received().len(), 2);
}

#[tokio::test]
async fn an_overload_or_a_server_error_retries_and_then_answers() {
    for status in [529, 500, 503] {
        let mock = MockServer::start(
            vec![
                MockResponse::error(status),
                MockResponse::json(fixture("noul")),
            ],
            Vec::new(),
        )
        .await;
        let reply = judge(&mock).ask(tests_pass()).await.unwrap();
        assert_eq!(
            reply.answers["tests_pass"],
            Answer::yes_no(0.9125),
            "{status}"
        );
        assert_eq!(mock.received().len(), 2, "{status}");
    }
}

#[tokio::test]
async fn retries_stop_after_the_last_attempt() {
    let mock = MockServer::start(
        vec![
            MockResponse::error(529),
            MockResponse::error(529),
            MockResponse::json(fixture("noul")),
        ],
        Vec::new(),
    )
    .await;
    let judge = judge_with(&mock, |config| config.max_attempts = 2);
    let error = transport(judge.ask(tests_pass()).await.unwrap_err());
    assert!(matches!(error, ProviderError::Server(_)), "{error:?}");
    assert_eq!(mock.received().len(), 2);
}

#[tokio::test]
async fn a_wait_past_the_budget_fails_without_a_second_request() {
    // A wait of 60 s against a budget of 30 s: a client that waited for
    // either would take 30 s or more.
    let budget = Duration::from_secs(30);
    for (first, class) in [
        (MockResponse::error_with_retry_after(429, 60), "rate"),
        (MockResponse::error_with_retry_after(529, 60), "server"),
    ] {
        let mock =
            MockServer::start(vec![first, MockResponse::json(fixture("noul"))], Vec::new()).await;
        let started = Instant::now();
        let error = judge(&mock)
            .ask(ask_one("tests_pass", Question::yes_no("q"), budget))
            .await
            .unwrap_err();
        // It failed at once: far below the budget and the wait.
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        let error = transport(error);
        match class {
            "rate" => assert!(
                matches!(
                    error,
                    ProviderError::RateLimited { retry_after: Some(wait) }
                        if wait == Duration::from_secs(60)
                ),
                "{error:?}"
            ),
            _ => assert!(matches!(error, ProviderError::Server(_)), "{error:?}"),
        }
        assert_eq!(mock.received().len(), 1, "{class}");
    }
}

#[tokio::test]
async fn a_stalled_reply_times_out_inside_the_budget() {
    let budget = Duration::from_millis(300);
    let stalled = MockResponse {
        stall: true,
        ..MockResponse::json(fixture("noul"))
    };
    let mock = MockServer::start(vec![stalled], Vec::new()).await;
    let started = Instant::now();
    let error = judge(&mock)
        .ask(ask_one("tests_pass", Question::yes_no("q"), budget))
        .await
        .unwrap_err();
    let elapsed = started.elapsed();
    assert!(matches!(error, JudgeError::Timeout(300)), "{error:?}");
    assert!(elapsed >= budget, "{elapsed:?}");
    assert!(elapsed < budget + Duration::from_secs(1), "{elapsed:?}");
    assert_eq!(mock.received().len(), 1);
}

#[tokio::test]
async fn a_network_failure_retries_and_then_fails() {
    // A server that accepts each connection and closes it at once.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = std::sync::Arc::clone(&accepted);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            drop(stream);
        }
    });
    let mut config = TypeSafeConfig::new("judge", format!("http://127.0.0.1:{port}"), "m");
    config.base_delay = Duration::from_millis(1);
    config.max_attempts = 3;
    let judge = TypeSafeJudge::new(config, KEY.to_string());
    let error = transport(judge.ask(tests_pass()).await.unwrap_err());
    assert!(matches!(error, ProviderError::Network(_)), "{error:?}");
    // Every attempt reached the server: the failure retried.
    assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[tokio::test]
async fn a_reply_that_fails_the_check_is_an_error() {
    let mut wrong_type = fixture("noul");
    wrong_type["answers"]["tests_pass"] = systemone::choice("a", &[("a", 1.0)], 1.0);
    let mut missing = fixture("noul");
    missing["answers"] = json!({"other": systemone::noul(0.5)});
    let mut out_of_range = fixture("noul");
    out_of_range["answers"]["tests_pass"] = systemone::noul(1.5);
    let mut other_type = fixture("noul");
    other_type["answers"]["tests_pass"] = json!({"type": "box", "boxes": []});
    let mut no_usage = fixture("noul");
    no_usage.as_object_mut().unwrap().remove("usage");
    for reply in [wrong_type, missing, out_of_range, other_type, no_usage] {
        let mock = MockServer::start(vec![MockResponse::json(reply.clone())], Vec::new()).await;
        let error = judge(&mock).ask(tests_pass()).await.unwrap_err();
        assert!(matches!(error, JudgeError::Reply(_)), "{reply}: {error:?}");
        // A bad reply does not retry.
        assert_eq!(mock.received().len(), 1);
    }

    // A score answer without a level index.
    let mut gap = fixture("score");
    gap["answers"]["change_scope"]["probabilities"] = json!({"0": 0.5, "2": 0.5});
    let mock = MockServer::start(vec![MockResponse::json(gap)], Vec::new()).await;
    let scope = levels(&["One file", "One module", "Several modules"]);
    let score = Question::score("How wide?", scope).unwrap();
    let error = judge(&mock)
        .ask(ask_one("change_scope", score, BUDGET))
        .await
        .unwrap_err();
    assert!(matches!(error, JudgeError::Reply(_)), "{error:?}");
}

#[tokio::test]
async fn a_reply_that_is_not_json_is_a_protocol_error_with_its_start() {
    let page = format!("<html>{KEY} {}</html>", "x".repeat(4096));
    let mock = MockServer::start(
        vec![MockResponse::sse_body(page).with_content_type("text/html")],
        Vec::new(),
    )
    .await;
    let error = transport(judge(&mock).ask(tests_pass()).await.unwrap_err());
    let ProviderError::Protocol(text) = &error else {
        panic!("expected a protocol error, got {error:?}");
    };
    assert!(text.contains("<html>"), "{text}");
    assert!(text.len() < 1024, "{}", text.len());
}

#[tokio::test]
async fn the_key_check_accepts_only_a_2xx() {
    let models = json!({"models": [{"name": "jev-latest", "description": "d",
                                    "release_date": "2026-10-14"}]});
    let mock = MockServer::start(
        vec![
            MockResponse::json(models),
            MockResponse::error(401),
            MockResponse::error(404),
            MockResponse::error(500),
        ],
        Vec::new(),
    )
    .await;
    let http = reqwest::Client::new();
    let base_url = format!("{}/", mock.url());

    let check = typesafe::verify_key(&http, "judge", &base_url, KEY)
        .await
        .unwrap();
    assert_eq!(check, KeyCheck::Accepted);
    let error = typesafe::verify_key(&http, "judge", &base_url, KEY)
        .await
        .unwrap_err();
    assert!(matches!(error, ProviderError::Auth(_)), "{error:?}");
    // Every other status is an error too: the route exists on the
    // service, so nothing short of a 2xx confirms the key.
    for status in ["404", "500"] {
        let error = typesafe::verify_key(&http, "judge", &base_url, KEY)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(status), "{error}");
        assert!(!error.to_string().contains(KEY), "{error}");
    }

    let received = mock.received();
    assert_eq!(received.len(), 4);
    for request in &received {
        assert_eq!(request.path, "/v1/models");
        assert_eq!(
            request.authorization.as_deref(),
            Some(format!("Bearer {KEY}").as_str())
        );
    }

    // A closed port fails, and stores nothing.
    let closed = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let error = typesafe::verify_key(&http, "judge", &format!("http://127.0.0.1:{closed}"), KEY)
        .await
        .unwrap_err();
    assert!(matches!(error, ProviderError::Network(_)), "{error:?}");
}

#[tokio::test]
async fn the_instance_key_check_follows_the_kind() {
    let mock = MockServer::start(
        vec![
            MockResponse::json(json!({"models": []})),
            MockResponse::error(401),
        ],
        Vec::new(),
    )
    .await;
    let http = reqwest::Client::new();
    let url = mock.url();
    // The judge kind checks `{base_url}/v1/models`.
    let check = verify_instance_key(PROVIDER_ID, &http, "judge", &url, KEY)
        .await
        .unwrap();
    assert_eq!(check, KeyCheck::Accepted);
    // A compatible endpoint checks `{base_url}/models`, under its API
    // root.
    let error = verify_instance_key("openai-compat", &http, "local", &format!("{url}/v1"), KEY)
        .await
        .unwrap_err();
    assert!(matches!(error, ProviderError::Auth(_)), "{error:?}");
    let paths: Vec<String> = mock.received().into_iter().map(|r| r.path).collect();
    assert_eq!(paths, ["/v1/models", "/v1/models"]);
    // A kind without stored instance keys sends nothing.
    for kind in ["anthropic", "openai-codex", "not-a-kind"] {
        assert!(
            verify_instance_key(kind, &http, "x", &url, KEY)
                .await
                .is_err()
        );
    }
    assert_eq!(mock.received().len(), 2);
}
