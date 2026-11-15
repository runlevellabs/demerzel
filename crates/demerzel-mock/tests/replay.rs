//! The `demerzel-mock` binary: the stdout lines, replay order, the
//! stream a two-call reply serves, the tool-result check, the routes
//! that take a reply, the judge route of a `judge` member, and refusal
//! of a bad transcript.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_demerzel-mock");

/// How long a test waits for the mock before it fails.
const WAIT: Duration = Duration::from_secs(10);

/// Write `transcript` to a file unique to `name` and this process.
fn transcript(name: &str, transcript: &Value) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("replay-{name}-{}.json", std::process::id()));
    std::fs::write(&path, transcript.to_string()).unwrap();
    path
}

fn script(replies: Value) -> Value {
    json!({"schema": 1, "replies": replies})
}

/// The running binary. Dropping it kills the process.
struct Mock {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    url: String,
}

impl Mock {
    /// Start the binary on `path` and read the base URL it prints.
    fn start(path: &PathBuf) -> Self {
        let mut child = Command::new(BIN)
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let (send, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if send.send(line).is_err() {
                    break;
                }
            }
        });
        let url = lines.recv_timeout(WAIT).expect("the URL line");
        Self {
            child,
            stdin,
            lines,
            url,
        }
    }

    /// One HTTP/1.1 request; the status and the body.
    fn request(&self, method: &str, path: &str, body: &Value) -> (u16, String) {
        self.request_with(method, path, None, body)
    }

    /// One HTTP/1.1 request with an optional bearer key.
    fn request_with(
        &self,
        method: &str,
        path: &str,
        key: Option<&str>,
        body: &Value,
    ) -> (u16, String) {
        let addr = self.url.strip_prefix("http://").expect("an http base URL");
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.set_read_timeout(Some(WAIT)).unwrap();
        stream.set_write_timeout(Some(WAIT)).unwrap();
        let body = body.to_string();
        let auth = key.map_or(String::new(), |key| {
            format!("Authorization: Bearer {key}\r\n")
        });
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\n{auth}\
             Content-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let (head, body) = response.split_once("\r\n\r\n").expect("a header block");
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .expect("a status line");
        (status, body.to_string())
    }

    fn chat(&self, body: &Value) -> (u16, String) {
        self.request("POST", "/v1/chat/completions", body)
    }

    /// One judge request that asks the yes-or-no questions `ids`.
    fn judge(&self, key: Option<&str>, ids: &[&str]) -> (u16, Value) {
        let questions: serde_json::Map<String, Value> = ids
            .iter()
            .map(|id| (id.to_string(), json!({"type": "noul", "instructions": "q"})))
            .collect();
        let body = json!({"model": "judge-9.9.9", "state": "s", "questions": questions});
        let (status, body) = self.request_with("POST", JUDGE_ROUTE, key, &body);
        (status, serde_json::from_str(&body).unwrap_or(Value::Null))
    }

    /// Close stdin; the summary line and the exit status.
    fn finish(mut self) -> (String, ExitStatus) {
        drop(self.stdin.take());
        let line = self.lines.recv_timeout(WAIT).expect("the summary line");
        let deadline = Instant::now() + WAIT;
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "the mock did not exit");
            std::thread::sleep(Duration::from_millis(10));
        };
        (line, status)
    }
}

/// The judge route of the binary.
const JUDGE_ROUTE: &str = "/judge/v1/systemone";

/// A clearly fake bearer key.
const KEY: Option<&str> = Some("fake-judge-key");

/// The exact summary line for these counts.
fn summary(served: u64, unused: u64, refused: u64) -> String {
    format!(r#"{{"served":{served},"unused":{unused},"refused":{refused}}}"#)
}

impl Drop for Mock {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A request body with no history of calls.
fn first() -> Value {
    json!({"model": "m", "stream": true, "messages": [{"role": "user", "content": "go"}]})
}

/// A request body that answers the calls `ids` of the last reply.
fn answers(ids: &[&str]) -> Value {
    let calls: Vec<Value> = ids
        .iter()
        .map(|id| {
            json!({"id": id, "type": "function", "function": {"name": "read", "arguments": "{}"}})
        })
        .collect();
    let mut messages = vec![
        json!({"role": "user", "content": "go"}),
        json!({"role": "assistant", "content": null, "tool_calls": calls}),
    ];
    messages.extend(
        ids.iter()
            .map(|id| json!({"role": "tool", "tool_call_id": id, "content": "ok"})),
    );
    json!({"model": "m", "stream": true, "messages": messages})
}

/// The JSON chunks of an SSE body, without the `[DONE]` marker.
fn chunks(sse: &str) -> Vec<Value> {
    sse.split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

fn text_of(sse: &str) -> String {
    chunks(sse)
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect()
}

#[test]
fn replies_come_in_order_once_each_then_an_error() {
    let path = transcript(
        "order",
        &script(json!([
            {"text": "first"},
            {"tool_calls": [
                {"id": "call_a", "name": "read", "input": {"path": "a.txt"}},
                {"id": "call_b", "name": "grep", "input": {"pattern": "b", "glob": "*.rs"}},
            ]},
            {"text": "last"},
        ])),
    );
    let mock = Mock::start(&path);
    assert!(mock.url.starts_with("http://127.0.0.1:"), "{}", mock.url);

    let (status, body) = mock.chat(&first());
    assert_eq!(status, 200);
    assert_eq!(text_of(&body), "first");
    assert!(body.contains("data: [DONE]"));

    // The two-call reply: fragments at wire index 0 and 1 carry the ids
    // and names, and each call's argument text joins to its input. This
    // is the shape the openai-compat adapter keys calls on.
    let (status, body) = mock.chat(&first());
    assert_eq!(status, 200);
    let mut calls: Vec<(String, String, String)> = vec![Default::default(); 2];
    let mut finish = None;
    for chunk in chunks(&body) {
        let Some(choice) = chunk["choices"].get(0) else {
            continue;
        };
        if let Some(reason) = choice["finish_reason"].as_str() {
            finish = Some(reason.to_string());
        }
        for fragment in choice["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let index = fragment["index"]
                .as_u64()
                .expect("every fragment has an index");
            let call = &mut calls[usize::try_from(index).unwrap()];
            if let Some(id) = fragment["id"].as_str() {
                call.0 = id.to_string();
            }
            if let Some(name) = fragment["function"]["name"].as_str() {
                call.1 = name.to_string();
            }
            call.2
                .push_str(fragment["function"]["arguments"].as_str().unwrap_or(""));
        }
    }
    let parsed: Vec<(&str, &str, Value)> = calls
        .iter()
        .map(|(id, name, arguments)| {
            (
                id.as_str(),
                name.as_str(),
                serde_json::from_str(arguments).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        parsed,
        vec![
            ("call_a", "read", json!({"path": "a.txt"})),
            ("call_b", "grep", json!({"pattern": "b", "glob": "*.rs"})),
        ]
    );
    assert_eq!(finish.as_deref(), Some("tool_calls"));

    let (status, body) = mock.chat(&answers(&["call_a", "call_b"]));
    assert_eq!(status, 200);
    assert_eq!(text_of(&body), "last");

    // The script does not loop: the next request fails.
    let (status, _) = mock.chat(&first());
    assert!(status >= 400, "status {status}");
    let (status, _) = mock.chat(&first());
    assert!(status >= 400, "status {status}");
    let (line, status) = mock.finish();
    assert!(status.success(), "{status}");
    assert_eq!(line, summary(3, 0, 2));
}

#[test]
fn closing_stdin_reports_served_and_unused_replies() {
    let path = transcript(
        "counts",
        &script(json!([{"text": "one"}, {"text": "two"}, {"text": "three"}])),
    );
    for requests in 0..=3 {
        let mock = Mock::start(&path);
        for _ in 0..requests {
            assert_eq!(mock.chat(&first()).0, 200);
        }
        let (line, status) = mock.finish();
        assert!(status.success(), "{status}");
        assert_eq!(
            line,
            summary(requests, 3 - requests, 0),
            "after {requests} requests"
        );
    }
}

#[test]
fn a_request_without_a_tool_result_is_refused() {
    let path = transcript(
        "unanswered",
        &script(json!([
            {"tool_calls": [
                {"id": "call_a", "name": "read", "input": {"path": "a"}},
                {"id": "call_b", "name": "read", "input": {"path": "b"}},
            ]},
            {"text": "done"},
        ])),
    );
    for body in [
        first(),
        answers(&["call_a"]),
        answers(&["call_a", "call_c"]),
    ] {
        let mock = Mock::start(&path);
        assert_eq!(mock.chat(&first()).0, 200);
        let (status, _) = mock.chat(&body);
        assert!((400..500).contains(&status), "status {status} for {body}");
        // The refused request got no reply.
        let (line, _) = mock.finish();
        assert_eq!(line, summary(1, 1, 1), "{body}");
    }
}

#[test]
fn requests_that_answer_every_call_are_served() {
    let path = transcript(
        "answered",
        &script(json!([
            {"tool_calls": [{"id": "call_a", "name": "read", "input": {"path": "a"}}]},
            {"tool_calls": [
                {"id": "call_b", "name": "read", "input": {"path": "b"}},
                {"id": "call_c", "name": "grep", "input": {"pattern": "c"}},
            ]},
            {"text": "done"},
        ])),
    );
    let mock = Mock::start(&path);
    assert_eq!(mock.chat(&first()).0, 200);
    assert_eq!(mock.chat(&answers(&["call_a"])).0, 200);
    let (status, body) = mock.chat(&answers(&["call_b", "call_c"]));
    assert_eq!(status, 200);
    assert_eq!(text_of(&body), "done");
    let (line, status) = mock.finish();
    assert!(status.success());
    assert_eq!(line, summary(3, 0, 0));
}

#[test]
fn other_routes_take_no_reply_and_count_as_refused() {
    let path = transcript("routes", &script(json!([{"text": "one"}])));
    let mock = Mock::start(&path);
    let (status, _) = mock.request("GET", "/v1/models", &json!({}));
    assert!(status >= 400, "status {status}");
    let (status, _) = mock.request("POST", "/v1/messages", &first());
    assert!(status >= 400, "status {status}");
    let (status, _) = mock.request("GET", "/v1/chat/completions", &json!({}));
    assert!(status >= 400, "status {status}");
    let (status, body) = mock.chat(&first());
    assert_eq!(status, 200);
    assert_eq!(text_of(&body), "one");
    let (line, _) = mock.finish();
    assert_eq!(line, summary(1, 0, 3));
}

#[test]
fn a_bad_transcript_fails_before_any_output_on_stdout() {
    let path = transcript("bad", &script(json!([{"text": "ok"}, {"tool_calls": []}])));
    let output = Command::new(BIN).arg(&path).output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains(&*path.to_string_lossy()), "{stderr}");
}

#[test]
fn a_missing_file_fails() {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("replay-missing-{}.json", std::process::id()));
    let output = Command::new(BIN).arg(&path).output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
}

#[test]
fn arguments_other_than_one_path_are_refused() {
    for args in [&[][..], &["a.json", "b.json"][..], &["--port"][..]] {
        let output = Command::new(BIN).args(args).output().unwrap();
        assert!(!output.status.success(), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?}");
    }
    for flag in ["-h", "--help"] {
        let output = Command::new(BIN).arg(flag).output().unwrap();
        assert!(output.status.success(), "{flag}");
    }
}

fn judged_script(judge: Value) -> Value {
    json!({"schema": 1, "replies": [{"text": "one"}], "judge": judge})
}

fn judge_member() -> Value {
    json!({
        "model": "judge-9.9.9",
        "requests": 2,
        "answers": {
            "outcome": {"type": "noul", "noul": 0.25},
            "claim": {"type": "noul", "noul": 0.75},
        },
    })
}

#[test]
fn a_judge_member_answers_by_id_and_counts_judged_requests() {
    let path = transcript("judged", &judged_script(judge_member()));
    let mock = Mock::start(&path);
    let (status, reply) = mock.judge(KEY, &["outcome"]);
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["model"], "judge-9.9.9");
    assert_eq!(
        reply["answers"],
        json!({"outcome": {"type": "noul", "noul": 0.25}})
    );
    let (status, reply) = mock.judge(KEY, &["outcome", "claim"]);
    assert_eq!(status, 200, "{reply}");
    assert_eq!(
        reply["answers"]["claim"],
        json!({"type": "noul", "noul": 0.75})
    );
    // No bearer key, and a question without a scripted answer: both
    // refused, neither judged.
    assert_eq!(mock.judge(None, &["outcome"]).0, 401);
    assert_eq!(mock.judge(KEY, &["outcome", "progress"]).0, 422);
    // The model route is unchanged.
    assert_eq!(text_of(&mock.chat(&first()).1), "one");
    let (line, status) = mock.finish();
    assert!(status.success(), "{status}");
    assert_eq!(line, r#"{"served":1,"unused":0,"refused":2,"judged":2}"#);
}

#[test]
fn without_a_judge_member_the_judge_route_is_refused_and_the_line_keeps_three_keys() {
    let path = transcript("unjudged", &script(json!([{"text": "one"}])));
    let mock = Mock::start(&path);
    assert_eq!(mock.judge(KEY, &["outcome"]).0, 404);
    let (line, status) = mock.finish();
    assert!(status.success(), "{status}");
    assert_eq!(line, summary(0, 1, 1));
}

#[test]
fn a_bad_judge_member_fails_before_the_url_line() {
    let noul = json!({"type": "noul", "noul": 0.5});
    let mut cases = Vec::new();
    for (field, value) in [
        ("extra", json!(1)),
        ("answers", json!([noul])),
        ("answers", json!({})),
        ("answers", json!({"outcome": "yes"})),
        (
            "answers",
            json!({"outcome": {"type": "noul", "noul": 0.5, "extra": 1}}),
        ),
        (
            "answers",
            json!({"outcome": {"type": "yes_no", "yes": 0.5}}),
        ),
        ("requests", json!(-1)),
        ("requests", json!(1.5)),
        ("model", json!("")),
    ] {
        let mut judge = judge_member();
        judge[field] = value;
        cases.push(judge);
    }
    let mut no_model = judge_member();
    no_model.as_object_mut().unwrap().remove("model");
    cases.extend([no_model, Value::Null, json!([])]);
    for (index, judge) in cases.into_iter().enumerate() {
        let path = transcript(&format!("bad-judge-{index}"), &judged_script(judge.clone()));
        let output = Command::new(BIN).arg(&path).output().unwrap();
        assert!(!output.status.success(), "accepted {judge}");
        assert!(output.stdout.is_empty(), "{judge}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("judge"), "{judge}: {stderr}");
    }
}
