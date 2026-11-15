//! Hermetic judgments: the runtime host against the mock's scripted
//! judge. The main model is a named `openai-compat` instance on the
//! mock's Chat Completions route; the judge is a `typesafe` instance on
//! the same mock's System One route, with a stored key. No network.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use demerzel::config::{self, Check, CheckAction, Judgments, Layer};
use demerzel::{AuthPaths, Resume, RunConfig, Runtime, SessionSettings};
use demerzel_core::{
    ApprovalAnswer, ApprovalMode, Approver, BoxFuture, EngineEvent, EventSink, TurnOutcome,
};
use demerzel_mock::compat::{text_turn, tool_calls_turn, tool_turn};
use demerzel_mock::systemone::noul;
use demerzel_mock::{JudgeScript, MockResponse, MockServer, Received};
use demerzel_providers::auth::AuthStore;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const JUDGE_MODEL: &str = "jev-1.14.0";
const JUDGE_KEY: &str = "fake-judge-key";
/// The judge instance; a name no notice text holds by itself.
const JUDGE: &str = "referee";

/// The question ids of the built-in checks.
const DESTRUCTIVE: &str = "approval.destructive";
const IN_SCOPE: &str = "approval.in_scope";
const OUTCOME: &str = "outcome";
const PROGRESS: &str = "progress";
const CLAIM: &str = "claim";

/// A scripted judge that answers every built-in question; `destructive`
/// and `in_scope` set the approval answers.
fn judge_script(destructive: f64, in_scope: f64) -> JudgeScript {
    JudgeScript {
        model: JUDGE_MODEL.into(),
        answers: BTreeMap::from([
            (DESTRUCTIVE.to_string(), noul(destructive)),
            (IN_SCOPE.to_string(), noul(in_scope)),
            (OUTCOME.to_string(), noul(0.1)),
            (PROGRESS.to_string(), noul(0.2)),
            (CLAIM.to_string(), noul(0.3)),
        ]),
    }
}

struct Bed {
    mock: MockServer,
    project: tempfile::TempDir,
    data: tempfile::TempDir,
}

impl Bed {
    /// A mock with the model `script` and a judge that would allow
    /// every call.
    async fn start(script: Vec<MockResponse>) -> Self {
        let mock = MockServer::start(script, Vec::new()).await;
        mock.set_judge(judge_script(0.01, 0.99));
        Self {
            mock,
            project: tempfile::tempdir().unwrap(),
            data: tempfile::tempdir().unwrap(),
        }
    }

    fn auth_path(&self) -> PathBuf {
        self.data.path().join("auth.json")
    }

    /// A run without judgments: instance `local` of the compatible
    /// adapter at the mock's `/v1` root, in `mode`.
    fn plain(&self, mode: ApprovalMode) -> RunConfig {
        let mut config = RunConfig::new(self.project.path().to_path_buf());
        config.instance.name = "local".into();
        config.instance.kind = "openai-compat".into();
        config.model = "m1".into();
        config.instance.base_url = Some(format!("{}/v1", self.mock.url()));
        config.approval = mode;
        config.data_root = Some(self.data.path().join("store"));
        config.auth = Some(AuthPaths {
            store: self.auth_path(),
            codex_cli: None,
        });
        config
    }

    /// The same run with judge instance `judge` on `judge_url` and
    /// `judgments`. The judge's key is in the run's own store, bound
    /// to `judge_url`, when `key` is set.
    fn judged_at(
        &self,
        mode: ApprovalMode,
        judgments: Judgments,
        judge_url: &str,
        key: bool,
    ) -> RunConfig {
        let mut config = self.plain(mode);
        let text = format!(
            "[providers.{JUDGE}]\nkind = \"typesafe\"\nbase_url = \"{judge_url}\"\n\
             model = \"{JUDGE_MODEL}\"\n"
        );
        let user = Layer::parse(&text).unwrap().0;
        config.other_instances =
            config::run_instances(&config::Flags::default(), &|_| None, None, Some(&user))
                .into_iter()
                .filter(|resolved| resolved.instance.name != "local")
                .collect();
        config.judgments = Some(judgments);
        if key {
            AuthStore::set_instance_key(&self.auth_path(), JUDGE, JUDGE_KEY, judge_url).unwrap();
        }
        config
    }

    /// Judgments on the bed's own mock, every check at `log`.
    fn judged(&self, mode: ApprovalMode) -> RunConfig {
        self.judged_at(mode, Judgments::new(JUDGE), &self.mock.url(), true)
    }

    fn judge_requests(&self) -> Vec<Received> {
        judge_requests(&self.mock)
    }

    fn model_bodies(&self) -> Vec<Value> {
        self.mock
            .received()
            .into_iter()
            .filter(|request| request.path == "/v1/chat/completions")
            .map(|request| request.body)
            .collect()
    }
}

fn judge_requests(mock: &MockServer) -> Vec<Received> {
    mock.received()
        .into_iter()
        .filter(|request| request.path.ends_with("/v1/systemone"))
        .collect()
}

/// The question ids of each judge request, in request order.
fn asked(requests: &[Received]) -> Vec<BTreeSet<String>> {
    requests
        .iter()
        .map(|request| {
            request.body["questions"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect()
        })
        .collect()
}

fn ids(list: &[&str]) -> BTreeSet<String> {
    list.iter().map(|id| id.to_string()).collect()
}

/// The lowercase hex SHA-256 of `text`.
fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Every record of a session file.
fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn judgments_of(path: &Path) -> Vec<Value> {
    records(path)
        .into_iter()
        .filter(|record| record["type"] == "judgment")
        .collect()
}

/// The records without the `judgment` records and the clock fields.
fn without_judgments(path: &Path) -> Vec<Value> {
    records(path)
        .into_iter()
        .filter(|record| record["type"] != "judgment")
        .map(|mut record| {
            let object = record.as_object_mut().unwrap();
            // The clocks, and the project path of each copy.
            for clock in ["at", "duration_ms", "created", "cwd"] {
                object.remove(clock);
            }
            record
        })
        .collect()
}

/// Every engine event of a run, as its debug text.
#[derive(Default)]
struct Events(Mutex<Vec<String>>);

impl EventSink for Events {
    fn event(&self, event: EngineEvent<'_>) {
        self.0.lock().unwrap().push(format!("{event:?}"));
    }
}

/// An approver that gives scripted answers, then denies.
struct Answers(VecDeque<ApprovalAnswer>);

impl Answers {
    fn new(answers: &[ApprovalAnswer]) -> Self {
        Self(answers.iter().copied().collect())
    }
}

impl Approver for Answers {
    fn approve<'a>(&'a mut self, _: &'a str, _: &'a Value) -> BoxFuture<'a, ApprovalAnswer> {
        let answer = self.0.pop_front().unwrap_or(ApprovalAnswer::Deny);
        Box::pin(std::future::ready(answer))
    }
}

/// One finished session.
struct Run {
    outcomes: Vec<TurnOutcome>,
    events: Vec<String>,
    path: PathBuf,
}

/// Run `prompts` as the turns of one fresh session, then settle the
/// judgment requests in flight.
async fn run_session(runtime: &Runtime, prompts: &[&str], answers: &[ApprovalAnswer]) -> Run {
    let settings = SessionSettings::from_config(runtime.config());
    let mut opened = runtime.open(&Resume::Fresh, &settings, prompts[0]).unwrap();
    let events = Events::default();
    let mut approver = Answers::new(answers);
    let mut outcomes = Vec::new();
    for prompt in prompts {
        let outcome = opened
            .engine
            .user_turn(prompt.to_string(), &events, &mut approver)
            .await
            .unwrap();
        outcomes.push(outcome);
    }
    let path = opened.engine.session().path().to_path_buf();
    drop(opened);
    runtime.settle().await;
    Run {
        outcomes,
        events: events.0.into_inner().unwrap(),
        path,
    }
}

/// A turn of calls of every access class: a passing command, a failing
/// command, the failing command again, a file creation, a read, and a
/// final text.
fn mixed_script() -> Vec<MockResponse> {
    vec![
        tool_turn("c1", "bash", json!({"command": "echo one"})),
        tool_turn("c2", "bash", json!({"command": "exit 3"})),
        tool_turn("c3", "bash", json!({"command": "exit 3"})),
        tool_turn(
            "c4",
            "create",
            json!({"path": "notes.txt", "content": "one\n"}),
        ),
        tool_turn("c5", "read", json!({"path": "f.txt"})),
        text_turn("done"),
    ]
}

fn seed_project(project: &Path) {
    std::fs::write(project.join("f.txt"), "alpha\nbeta\n").unwrap();
}

#[tokio::test]
async fn one_request_carries_every_question_of_one_event() {
    // A failed command, then the same tool again: the second call's
    // end asks `outcome` and `progress` in one request.
    let bed = Bed::start(vec![
        tool_turn("c1", "bash", json!({"command": "exit 3"})),
        tool_turn("c2", "bash", json!({"command": "true"})),
        text_turn("done"),
    ])
    .await;
    let runtime = Runtime::new(bed.judged(ApprovalMode::Auto)).unwrap();
    let run = run_session(&runtime, &["go"], &[]).await;
    assert_eq!(run.outcomes, [TurnOutcome::Completed]);

    let requests = bed.judge_requests();
    let asked = asked(&requests);
    assert_eq!(asked.len(), 4, "{asked:?}");
    // Each command's approval sends both approval questions together.
    let approval = ids(&[DESTRUCTIVE, IN_SCOPE]);
    assert_eq!(asked.iter().filter(|set| **set == approval).count(), 2);
    assert_eq!(
        asked
            .iter()
            .filter(|set| **set == ids(&[OUTCOME, PROGRESS]))
            .count(),
        1,
        "{asked:?}"
    );
    assert_eq!(asked.iter().filter(|set| **set == ids(&[CLAIM])).count(), 1);
    // Every request names the model and carries the bearer key.
    for request in &requests {
        assert_eq!(request.path, "/v1/systemone");
        assert_eq!(request.body["model"], JUDGE_MODEL);
        assert_eq!(
            request.authorization.as_deref(),
            Some(format!("Bearer {JUDGE_KEY}").as_str())
        );
    }
    assert_eq!(bed.mock.judged(), 4);
    assert_eq!(bed.mock.refused(), 0);
    assert_eq!(judgments_of(&run.path).len(), 4);
    assert!(runtime.take_notices().is_empty());
}

/// Run the mixed script once without and once with judgments, each on
/// its own mock and its own copy of the project, and compare what the
/// loop did.
async fn compare_runs(mode: ApprovalMode, answers: &[ApprovalAnswer], destructive: f64) {
    let plain = Bed::start(mixed_script()).await;
    let judged = Bed::start(mixed_script()).await;
    judged
        .mock
        .set_judge(judge_script(destructive, 1.0 - destructive));
    seed_project(plain.project.path());
    seed_project(judged.project.path());

    let runtime = Runtime::new(plain.plain(mode)).unwrap();
    let without = run_session(&runtime, &["go"], answers).await;
    let runtime = Runtime::new(judged.judged(mode)).unwrap();
    let with = run_session(&runtime, &["go"], answers).await;

    assert!(!judged.judge_requests().is_empty(), "{mode}: no check ran");
    assert!(!judgments_of(&with.path).is_empty());
    // No byte of any request to the main model differs.
    assert_eq!(plain.model_bodies(), judged.model_bodies(), "{mode}");
    // No engine event and no outcome differs.
    assert_eq!(without.events, with.events, "{mode}");
    assert_eq!(without.outcomes, with.outcomes, "{mode}");
    // No session record differs, approval records included.
    assert_eq!(
        without_judgments(&without.path),
        without_judgments(&with.path),
        "{mode}"
    );
    assert!(judgments_of(&without.path).is_empty());
}

#[tokio::test]
async fn a_log_check_changes_nothing_in_auto_mode() {
    // The judge rates every call destructive: at `log` the policy
    // still allows each one.
    compare_runs(ApprovalMode::Auto, &[], 0.99).await;
}

#[tokio::test]
async fn a_log_check_changes_nothing_in_ask_mode() {
    // The judge would allow every call: at `log` the user still
    // decides. The user allows the commands and denies the creation.
    use ApprovalAnswer::{Allow, Deny};
    compare_runs(ApprovalMode::Ask, &[Allow, Allow, Allow, Deny], 0.01).await;
}

#[tokio::test]
async fn each_approval_judgment_pairs_with_its_approval_record() {
    use ApprovalAnswer::{Allow, Deny};
    let bed = Bed::start(mixed_script()).await;
    seed_project(bed.project.path());
    let runtime = Runtime::new(bed.judged(ApprovalMode::Ask)).unwrap();
    let run = run_session(&runtime, &["go"], &[Allow, Deny, Allow, Deny]).await;
    let all = records(&run.path);
    let before: Vec<&Value> = all
        .iter()
        .filter(|record| record["type"] == "judgment" && record["event"] == "tool_before")
        .collect();
    // Three commands and one creation reach the policy; the read does
    // not.
    assert_eq!(before.len(), 4);
    for judgment in before {
        let call_id = judgment["call_id"].as_str().unwrap();
        let approval = all
            .iter()
            .find(|record| record["type"] == "approval" && record["tool_use_id"] == call_id)
            .unwrap_or_else(|| panic!("no approval record for {call_id}"));
        // The shadow keeps the policy's question.
        assert_eq!(approval["decision"], "ask_user");
        assert_eq!(judgment["tool"], approval["tool"]);
        assert_eq!(judgment["checks"], json!({"approval": "log"}));
    }
}

#[tokio::test]
async fn no_check_runs_in_read_only_mode() {
    let bed = Bed::start(mixed_script()).await;
    seed_project(bed.project.path());
    let runtime = Runtime::new(bed.judged(ApprovalMode::ReadOnly)).unwrap();
    let run = run_session(&runtime, &["go"], &[]).await;
    let asked = asked(&bed.judge_requests());
    assert!(
        asked
            .iter()
            .all(|set| !set.contains(DESTRUCTIVE) && !set.contains(IN_SCOPE)),
        "{asked:?}"
    );
    assert!(
        judgments_of(&run.path)
            .iter()
            .all(|record| record["event"] != "tool_before")
    );
    // The policy refused the write tools, and nothing ran them.
    assert!(!bed.project.path().join("notes.txt").exists());
}

#[tokio::test]
async fn the_call_cap_stops_checks_with_one_notice() {
    let bed = Bed::start(vec![
        tool_turn("c1", "bash", json!({"command": "true"})),
        tool_turn("c2", "bash", json!({"command": "true"})),
        tool_turn("c3", "bash", json!({"command": "true"})),
        text_turn("done"),
        tool_turn("c4", "bash", json!({"command": "true"})),
        text_turn("done"),
        // The next session.
        tool_turn("c5", "bash", json!({"command": "true"})),
        text_turn("done"),
    ])
    .await;
    let mut judgments = Judgments::new(JUDGE);
    judgments.max_calls = 2;
    let config = bed.judged_at(ApprovalMode::Auto, judgments, &bed.mock.url(), true);
    let runtime = Runtime::new(config).unwrap();
    let run = run_session(&runtime, &["go", "again"], &[]).await;
    assert_eq!(run.outcomes, [TurnOutcome::Completed; 2]);
    // Ten events would ask; two requests went out.
    assert_eq!(bed.judge_requests().len(), 2);
    assert_eq!(bed.mock.judged(), 2);
    assert_eq!(judgments_of(&run.path).len(), 2);
    let notices = runtime.take_notices();
    assert_eq!(notices.len(), 1, "{notices:?}");
    // The notice gives the cap.
    assert!(notices[0].contains('2'), "{notices:?}");
    assert!(runtime.take_notices().is_empty());

    // The cap counts per opened session: the next session asks again.
    let next = run_session(&runtime, &["more"], &[]).await;
    assert_eq!(bed.judge_requests().len(), 4);
    assert_eq!(judgments_of(&next.path).len(), 2);
    assert_eq!(runtime.take_notices().len(), 1);
}

#[tokio::test]
async fn the_outcome_check_asks_only_after_an_execute_call_without_its_error_flag() {
    let bed = Bed::start(vec![
        tool_turn("c1", "bash", json!({"command": "exit 3"})),
        tool_turn("c2", "create", json!({"path": "a.txt", "content": "a"})),
        tool_turn("c3", "read", json!({"path": "a.txt"})),
        tool_turn("c4", "bash", json!({"command": "echo ok"})),
        text_turn("done"),
    ])
    .await;
    let mut judgments = Judgments::new(JUDGE);
    judgments.checks = config::Checks::uniform(CheckAction::Off);
    judgments.checks.set(Check::Outcome, CheckAction::Log);
    let config = bed.judged_at(ApprovalMode::Auto, judgments, &bed.mock.url(), true);
    let runtime = Runtime::new(config).unwrap();
    let run = run_session(&runtime, &["go"], &[]).await;
    assert_eq!(asked(&bed.judge_requests()), [ids(&[OUTCOME])]);
    let records = judgments_of(&run.path);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["event"], "tool_after");
    assert_eq!(records[0]["call_id"], "c4");
    assert_eq!(records[0]["checks"], json!({"outcome": "log"}));
}

#[tokio::test]
async fn the_progress_check_asks_only_after_an_earlier_failed_call_of_the_same_tool() {
    let bed = Bed::start(vec![
        // A failed read does not make a later command a repeat.
        tool_turn("c1", "read", json!({"path": "missing.txt"})),
        tool_turn("c2", "bash", json!({"command": "true"})),
        tool_turn("c3", "bash", json!({"command": "exit 3"})),
        tool_turn("c4", "glob", json!({"pattern": "*.txt"})),
        tool_turn("c5", "bash", json!({"command": "exit 4"})),
        text_turn("done"),
        // A new turn does not see the last turn's failures.
        tool_turn("c6", "bash", json!({"command": "true"})),
        text_turn("done"),
    ])
    .await;
    seed_project(bed.project.path());
    let mut judgments = Judgments::new(JUDGE);
    judgments.checks = config::Checks::uniform(CheckAction::Off);
    judgments.checks.set(Check::Progress, CheckAction::Log);
    let config = bed.judged_at(ApprovalMode::Auto, judgments, &bed.mock.url(), true);
    let runtime = Runtime::new(config).unwrap();
    let run = run_session(&runtime, &["go", "again"], &[]).await;
    assert_eq!(asked(&bed.judge_requests()), [ids(&[PROGRESS])]);
    let records = judgments_of(&run.path);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["call_id"], "c5");
    // The state carries the earlier failed input.
    let state = bed.judge_requests()[0].body["state"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        state.contains("exit 3") && state.contains("exit 4"),
        "{state}"
    );
}

#[tokio::test]
async fn the_claim_check_asks_only_after_a_completed_turn_with_a_call() {
    let bed = Bed::start(vec![
        // A text-only turn.
        text_turn("hello"),
        // A turn with a call that the provider ends with an error.
        tool_turn("c1", "read", json!({"path": "f.txt"})),
        MockResponse::error(400),
        // A completed turn with a call.
        tool_turn("c2", "read", json!({"path": "f.txt"})),
        text_turn("the file holds alpha and beta"),
    ])
    .await;
    seed_project(bed.project.path());
    let mut judgments = Judgments::new(JUDGE);
    judgments.checks = config::Checks::uniform(CheckAction::Off);
    judgments.checks.set(Check::Claim, CheckAction::Log);
    let config = bed.judged_at(ApprovalMode::Auto, judgments, &bed.mock.url(), true);
    let runtime = Runtime::new(config).unwrap();

    let settings = SessionSettings::from_config(runtime.config());
    let mut opened = runtime.open(&Resume::Fresh, &settings, "one").unwrap();
    let events = Events::default();
    let mut approver = Answers::new(&[]);
    for prompt in ["one", "two", "three"] {
        let _ = opened
            .engine
            .user_turn(prompt.into(), &events, &mut approver)
            .await;
    }
    let path = opened.engine.session().path().to_path_buf();
    drop(opened);
    runtime.settle().await;

    assert_eq!(asked(&bed.judge_requests()), [ids(&[CLAIM])]);
    let records = judgments_of(&path);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["event"], "turn_end");
    assert!(records[0].get("call_id").is_none());
    let state = bed.judge_requests()[0].body["state"]
        .as_str()
        .unwrap()
        .to_string();
    // The claim's state: the final message and the last call's output.
    assert!(state.contains("the file holds alpha and beta"), "{state}");
    assert!(state.contains("1:alpha"), "{state}");
    // The state holds the current turn only.
    assert!(
        state.contains("three") && !state.contains("hello"),
        "{state}"
    );
}

#[tokio::test]
async fn the_approval_check_asks_only_for_a_write_or_execute_call_at_the_policy() {
    use ApprovalAnswer::Allow;
    let bed = Bed::start(vec![
        tool_turn("c1", "read", json!({"path": "f.txt"})),
        tool_turn("c2", "create", json!({"path": "a.txt", "content": "a"})),
        tool_turn("c3", "glob", json!({"pattern": "*.txt"})),
        tool_turn("c4", "bash", json!({"command": "true"})),
        text_turn("done"),
    ])
    .await;
    seed_project(bed.project.path());
    let mut judgments = Judgments::new(JUDGE);
    judgments.checks = config::Checks::uniform(CheckAction::Off);
    judgments.checks.set(Check::Approval, CheckAction::Log);
    let config = bed.judged_at(ApprovalMode::Ask, judgments, &bed.mock.url(), true);
    let runtime = Runtime::new(config).unwrap();
    let run = run_session(&runtime, &["go"], &[Allow, Allow]).await;
    let approval = ids(&[DESTRUCTIVE, IN_SCOPE]);
    assert_eq!(asked(&bed.judge_requests()), [approval.clone(), approval]);
    let calls: Vec<Value> = judgments_of(&run.path)
        .iter()
        .map(|record| record["call_id"].clone())
        .collect();
    assert_eq!(calls.len(), 2);
    assert!(calls.contains(&json!("c2")) && calls.contains(&json!("c4")));
}

#[tokio::test]
async fn every_check_off_sends_no_request() {
    let bed = Bed::start(mixed_script()).await;
    seed_project(bed.project.path());
    let mut judgments = Judgments::new(JUDGE);
    judgments.checks = config::Checks::uniform(CheckAction::Off);
    let config = bed.judged_at(ApprovalMode::Auto, judgments, &bed.mock.url(), true);
    let runtime = Runtime::new(config).unwrap();
    let run = run_session(&runtime, &["go"], &[]).await;
    assert_eq!(run.outcomes, [TurnOutcome::Completed]);
    assert!(bed.judge_requests().is_empty());
    assert!(judgments_of(&run.path).is_empty());
}

#[tokio::test]
async fn a_judge_that_refuses_the_key_writes_a_record_with_the_error() {
    let judge = MockServer::start(
        (0..4).map(|_| MockResponse::error(401)).collect(),
        Vec::new(),
    )
    .await;
    let bed = Bed::start(vec![
        tool_turn("c1", "bash", json!({"command": "true"})),
        text_turn("done"),
    ])
    .await;
    let config = bed.judged_at(
        ApprovalMode::Auto,
        Judgments::new(JUDGE),
        &judge.url(),
        true,
    );
    let runtime = Runtime::new(config).unwrap();
    let run = run_session(&runtime, &["go"], &[]).await;
    assert_eq!(run.outcomes, [TurnOutcome::Completed]);
    // Approval, outcome, and claim: one attempt each, no retry.
    assert_eq!(judge_requests(&judge).len(), 3);
    let records = judgments_of(&run.path);
    assert_eq!(records.len(), 3);
    for record in &records {
        let error = record["error"].as_str().unwrap();
        assert!(error.contains("401"), "{error}");
        assert!(!error.contains(JUDGE_KEY), "{error}");
        assert!(record.get("answers").is_none());
        assert!(record.get("model").is_none());
    }
    assert!(runtime.take_notices().is_empty());
}

#[tokio::test]
async fn a_judge_without_a_key_sends_nothing_and_gives_one_notice() {
    let bed = Bed::start(vec![
        tool_turn("c1", "bash", json!({"command": "true"})),
        tool_turn("c2", "bash", json!({"command": "true"})),
        text_turn("done"),
        text_turn("still here"),
    ])
    .await;
    let config = bed.judged_at(
        ApprovalMode::Auto,
        Judgments::new(JUDGE),
        &bed.mock.url(),
        false,
    );
    let runtime = Runtime::new(config).unwrap();

    // The open builds no judge: no notice before the first check.
    let settings = SessionSettings::from_config(runtime.config());
    let mut opened = runtime.open(&Resume::Fresh, &settings, "go").unwrap();
    assert!(runtime.take_notices().is_empty());
    let events = Events::default();
    let mut approver = Answers::new(&[]);
    let outcome = opened
        .engine
        .user_turn("go".into(), &events, &mut approver)
        .await
        .unwrap();
    assert_eq!(outcome, TurnOutcome::Completed);
    let outcome = opened
        .engine
        .user_turn("again".into(), &events, &mut approver)
        .await
        .unwrap();
    assert_eq!(outcome, TurnOutcome::Completed);
    let path = opened.engine.session().path().to_path_buf();
    drop(opened);
    runtime.settle().await;

    assert!(bed.judge_requests().is_empty());
    assert!(judgments_of(&path).is_empty());
    let notices = runtime.take_notices();
    assert_eq!(notices.len(), 1, "{notices:?}");
    // The notice names the judge instance.
    assert!(notices[0].contains(JUDGE), "{notices:?}");
    // The turn is the same as a run without judgments.
    assert_eq!(bed.model_bodies().len(), 4);
}

#[tokio::test]
async fn a_judgment_record_describes_the_request_and_its_answers() {
    let bed = Bed::start(vec![
        tool_turn("c1", "bash", json!({"command": "echo one"})),
        text_turn("done"),
    ])
    .await;
    let runtime = Runtime::new(bed.judged(ApprovalMode::Auto)).unwrap();
    let run = run_session(&runtime, &["go"], &[]).await;
    let judgments = judgments_of(&run.path);
    let requests = bed.judge_requests();
    assert_eq!(judgments.len(), 3);
    assert_eq!(requests.len(), 3);
    for record in &judgments {
        // The record pairs with the request whose state hashes to it.
        let sha = record["state_sha256"].as_str().unwrap();
        let request = requests
            .iter()
            .find(|request| {
                let state = request.body["state"].as_str().unwrap();
                sha256_hex(state) == sha
            })
            .unwrap_or_else(|| panic!("no request with state {sha}"));
        let state = request.body["state"].as_str().unwrap();
        assert_eq!(record["state_bytes"], state.len());
        assert_eq!(record["instance"], JUDGE);
        assert_eq!(record["model"], JUDGE_MODEL);
        assert_eq!(
            record["usage"],
            json!({"input_tokens": state.len() / 4, "output_tokens": 0})
        );
        assert!(record.get("error").is_none());
        assert!(record["duration_ms"].is_u64());
        // An answer for each question, by id, as the judge gave it.
        let questions: BTreeSet<&String> = request.body["questions"]
            .as_object()
            .unwrap()
            .keys()
            .collect();
        let answers = record["answers"].as_object().unwrap();
        assert_eq!(answers.keys().collect::<BTreeSet<_>>(), questions);
        for (id, answer) in answers {
            let scripted = judge_script(0.01, 0.99).answers[id]["noul"].clone();
            assert_eq!(answer, &json!({"type": "yes_no", "yes": scripted}));
        }
        match record["event"].as_str().unwrap() {
            "tool_before" | "tool_after" => {
                assert_eq!(record["call_id"], "c1");
                assert_eq!(record["tool"], "bash");
            }
            "turn_end" => {
                assert!(record.get("call_id").is_none() && record.get("tool").is_none());
            }
            other => panic!("event {other}"),
        }
    }
}

#[tokio::test]
async fn a_record_that_completes_after_new_lands_in_the_session_that_asked() {
    let bed = Bed::start(vec![
        tool_turn("c1", "read", json!({"path": "f.txt"})),
        text_turn("done"),
        text_turn("fresh"),
    ])
    .await;
    seed_project(bed.project.path());
    let runtime = Runtime::new(bed.judged(ApprovalMode::Auto)).unwrap();
    let settings = SessionSettings::from_config(runtime.config());
    let mut first = runtime.open(&Resume::Fresh, &settings, "go").unwrap();
    first
        .engine
        .user_turn("go".into(), &Events::default(), &mut Answers::new(&[]))
        .await
        .unwrap();
    // The claim request is in flight: the turn returned at its spawn.
    let first_path = first.engine.session().path().to_path_buf();
    assert!(judgments_of(&first_path).is_empty());
    // `/new`: the engine drops and a fresh session opens.
    drop(first);
    let mut second = runtime.open(&Resume::Fresh, &settings, "next").unwrap();
    second
        .engine
        .user_turn("next".into(), &Events::default(), &mut Answers::new(&[]))
        .await
        .unwrap();
    let second_path = second.engine.session().path().to_path_buf();
    drop(second);
    runtime.settle().await;
    assert_ne!(first_path, second_path);
    let landed = judgments_of(&first_path);
    assert_eq!(landed.len(), 1);
    assert_eq!(landed[0]["event"], "turn_end");
    assert!(judgments_of(&second_path).is_empty());
}

#[tokio::test]
async fn the_state_holds_neither_the_system_prompt_nor_earlier_turns() {
    let bed = Bed::start(vec![
        text_turn("first answer"),
        tool_turn("c1", "bash", json!({"command": "echo two"})),
        text_turn("done"),
    ])
    .await;
    let base = bed.data.path().join("base.md");
    std::fs::write(&base, "BASE-PROMPT-MARKER: you are a test agent.").unwrap();
    let mut config = bed.judged(ApprovalMode::Auto);
    config.prompt_base_file = Some(base);
    let runtime = Runtime::new(config).unwrap();
    run_session(&runtime, &["FIRST-TURN-MARKER", "second turn"], &[]).await;
    // The system prompt went to the model.
    let system = bed.model_bodies()[0]["messages"][0].to_string();
    assert!(system.contains("BASE-PROMPT-MARKER"), "{system}");
    // The tool definitions the main model received.
    let descriptions: Vec<String> = bed.model_bodies()[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| {
            tool["function"]["description"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert!(descriptions.len() > 1);
    let requests = bed.judge_requests();
    assert_eq!(requests.len(), 3);
    for request in requests {
        let state = request.body["state"].as_str().unwrap();
        assert!(state.contains("second turn"), "{state}");
        for absent in ["BASE-PROMPT-MARKER", "FIRST-TURN-MARKER", "first answer"] {
            assert!(!state.contains(absent), "{absent} in {state}");
        }
        for description in &descriptions {
            assert!(
                !state.contains(description.as_str()),
                "{description} in {state}"
            );
        }
    }
}

#[tokio::test]
async fn a_burst_of_calls_gets_every_answer_through_the_permits() {
    let input = json!({"command": "true"});
    let ids: Vec<String> = (0..12).map(|i| format!("c{i}")).collect();
    let calls: Vec<(&str, &str, &Value)> =
        ids.iter().map(|id| (id.as_str(), "bash", &input)).collect();
    let bed = Bed::start(vec![tool_calls_turn(&calls), text_turn("done")]).await;
    let runtime = Runtime::new(bed.judged(ApprovalMode::Auto)).unwrap();
    let run = run_session(&runtime, &["go"], &[]).await;
    assert_eq!(run.outcomes, [TurnOutcome::Completed]);
    // Twelve approvals, twelve outcomes, and the claim: each answered.
    let judgments = judgments_of(&run.path);
    assert_eq!(judgments.len(), 25);
    assert!(judgments.iter().all(|record| {
        record.get("error").is_none()
            && record["answers"]
                .as_object()
                .is_some_and(|answers| !answers.is_empty())
    }));
    assert_eq!(bed.mock.judged(), 25);
}

#[tokio::test]
async fn a_judge_that_does_not_build_takes_none_of_the_call_cap() {
    // Two calls in one reply, a cap of one request, and no key.
    let input = json!({"command": "true"});
    let bed = Bed::start(vec![
        tool_calls_turn(&[("c1", "bash", &input), ("c2", "bash", &input)]),
        text_turn("done"),
    ])
    .await;
    let mut judgments = Judgments::new(JUDGE);
    judgments.max_calls = 1;
    let config = bed.judged_at(ApprovalMode::Auto, judgments, &bed.mock.url(), false);
    let runtime = Runtime::new(config).unwrap();
    run_session(&runtime, &["go"], &[]).await;
    assert!(bed.judge_requests().is_empty());
    // One notice, for the judge; nothing reached the cap.
    let notices = runtime.take_notices();
    assert_eq!(notices.len(), 1, "{notices:?}");
    assert!(notices[0].contains(JUDGE), "{notices:?}");
}

#[tokio::test]
async fn without_judgments_the_run_sends_no_judgment_request() {
    let bed = Bed::start(mixed_script()).await;
    seed_project(bed.project.path());
    let runtime = Runtime::new(bed.plain(ApprovalMode::Auto)).unwrap();
    let run = run_session(&runtime, &["go"], &[]).await;
    assert_eq!(run.outcomes, [TurnOutcome::Completed]);
    assert!(bed.judge_requests().is_empty());
    assert_eq!(bed.mock.judged(), 0);
    assert!(judgments_of(&run.path).is_empty());
    assert!(runtime.take_notices().is_empty());
}

#[tokio::test]
async fn the_judge_builds_at_the_first_check() {
    // A text-only turn sends nothing: the claim needs a call.
    let bed = Bed::start(vec![
        text_turn("hello"),
        tool_turn("c1", "read", json!({"path": "f.txt"})),
        text_turn("done"),
    ])
    .await;
    seed_project(bed.project.path());
    let config = bed.judged_at(
        ApprovalMode::Auto,
        Judgments::new(JUDGE),
        &bed.mock.url(),
        false,
    );
    let runtime = Runtime::new(config).unwrap();
    let settings = SessionSettings::from_config(runtime.config());
    let mut opened = runtime.open(&Resume::Fresh, &settings, "hi").unwrap();
    let events = Events::default();
    let mut approver = Answers::new(&[]);
    opened
        .engine
        .user_turn("hi".into(), &events, &mut approver)
        .await
        .unwrap();
    runtime.settle().await;
    // No check fired: the keyless judge did not try to build.
    assert!(runtime.take_notices().is_empty());
    assert!(bed.judge_requests().is_empty());

    // A login lands between two prompts; the first check builds the
    // judge with it.
    AuthStore::set_instance_key(&bed.auth_path(), JUDGE, JUDGE_KEY, &bed.mock.url()).unwrap();
    opened
        .engine
        .user_turn("read it".into(), &events, &mut approver)
        .await
        .unwrap();
    let path = opened.engine.session().path().to_path_buf();
    drop(opened);
    runtime.settle().await;
    assert!(runtime.take_notices().is_empty());
    assert_eq!(asked(&bed.judge_requests()), [ids(&[CLAIM])]);
    assert_eq!(judgments_of(&path).len(), 1);
}

/// A judge that accepts each connection, reads the request, and never
/// answers.
fn silent_judge() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            std::thread::spawn(move || {
                use std::io::Read as _;
                let mut buf = [0u8; 4096];
                while matches!(stream.read(&mut buf), Ok(n) if n > 0) {}
            });
        }
    });
    url
}

/// A one-shot run of the binary in `bed`'s project, with a judge at
/// `judge` that never answers. Its stdout and stderr lines arrive on
/// the channels, each with the instant the test read it.
struct OneShot {
    child: std::process::Child,
    stdout: std::sync::mpsc::Receiver<(std::time::Instant, String)>,
    stderr: std::sync::mpsc::Receiver<(std::time::Instant, String)>,
    _root: tempfile::TempDir,
}

impl OneShot {
    fn start(bed: &Bed, judge: &str) -> Self {
        use std::process::{Command, Stdio};

        let root = tempfile::tempdir().unwrap();
        let (home, config, data) = (
            root.path().join("home"),
            root.path().join("config"),
            root.path().join("data"),
        );
        for dir in [&home, &config.join("demerzel"), &data] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(
            config.join("demerzel").join("config.toml"),
            format!(
                "provider = \"local\"\napproval = \"auto\"\n\
                 [providers.local]\nkind = \"openai-compat\"\nbase_url = \"{}/v1\"\nmodel = \"m1\"\n\
                 [providers.{JUDGE}]\nkind = \"typesafe\"\nbase_url = \"{judge}\"\n\
                 model = \"{JUDGE_MODEL}\"\napi_key_env = \"DMZ_TEST_JUDGE_KEY\"\n\
                 [judgments]\ninstance = \"{JUDGE}\"\n",
                bed.mock.url()
            ),
        )
        .unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_demerzel"))
            .arg("go")
            .current_dir(bed.project.path())
            .env("HOME", &home)
            .env("XDG_CONFIG_HOME", &config)
            .env("XDG_DATA_HOME", &data)
            .env("DMZ_TEST_JUDGE_KEY", JUDGE_KEY)
            .env_remove("DEMERZEL_PROVIDER")
            .env_remove("DEMERZEL_MODEL")
            .env_remove("DEMERZEL_BASE_URL")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let lines = |stream: Box<dyn std::io::Read + Send>| {
            let (send, receive) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                use std::io::BufRead as _;
                for line in std::io::BufReader::new(stream)
                    .lines()
                    .map_while(Result::ok)
                {
                    let _ = send.send((std::time::Instant::now(), line));
                }
            });
            receive
        };
        let stdout = lines(Box::new(child.stdout.take().unwrap()));
        let stderr = lines(Box::new(child.stderr.take().unwrap()));
        Self {
            child,
            stdout,
            stderr,
            _root: root,
        }
    }

    /// Wait for a line on `channel` that holds `part`.
    fn line(
        channel: &std::sync::mpsc::Receiver<(std::time::Instant, String)>,
        part: &str,
    ) -> (std::time::Instant, String) {
        let mut seen = Vec::new();
        loop {
            let (at, line) = channel
                .recv_timeout(std::time::Duration::from_secs(20))
                .unwrap_or_else(|_| panic!("no line with {part:?} in {seen:?}"));
            if line.contains(part) {
                return (at, line);
            }
            seen.push(line);
        }
    }

    /// Send SIGINT, then wait for the exit: the status and the time it
    /// took after the signal.
    fn interrupt(&mut self) -> (std::process::ExitStatus, std::time::Duration) {
        let pid = i32::try_from(self.child.id()).unwrap();
        // SAFETY: `kill` takes a pid and a signal number; the pid is
        // the test's own child, which has not been reaped.
        assert_eq!(unsafe { libc::kill(pid, libc::SIGINT) }, 0);
        let sent = std::time::Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return (status, sent.elapsed());
            }
            assert!(
                sent.elapsed() < std::time::Duration::from_secs(10),
                "the run did not end"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// The stderr lines left after the exit.
    fn rest(&self) -> Vec<String> {
        self.stderr.try_iter().map(|(_, line)| line).collect()
    }
}

/// The exit waits at most two request timeouts and a grace, 4.25 s: a
/// run that ends within 1 s of the signal did not wait it out.
const QUICK: std::time::Duration = std::time::Duration::from_secs(1);

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_during_the_exit_wait_keeps_the_turns_exit_status() {
    let bed = Bed::start(vec![
        tool_turn("c1", "bash", json!({"command": "true"})),
        text_turn("done"),
    ])
    .await;
    let mut run = OneShot::start(&bed, &silent_judge());
    // The turn completes; the judge never answers, so the exit waits
    // and says so, in one exact line.
    let (_, waiting) = OneShot::line(&run.stderr, "judgment records");
    assert_eq!(waiting, format!("demerzel: {}", demerzel::WAIT_LINE));
    let (status, after) = run.interrupt();
    // The turn completed, so the run exits 0, not 130, and at once.
    assert_eq!(status.code(), Some(0), "{:?}", run.rest());
    assert!(after < QUICK, "{after:?}");
    assert!(run.stdout.try_iter().any(|(_, line)| line.contains("done")));
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_before_the_wait_line_ends_the_wait_too() {
    let bed = Bed::start(vec![
        tool_turn("c1", "bash", json!({"command": "true"})),
        text_turn("done"),
    ])
    .await;
    let mut run = OneShot::start(&bed, &silent_judge());
    // The reply's last line goes out as the turn ends, right before
    // the wait; the signal follows it at once.
    let (read, _) = OneShot::line(&run.stdout, "done");
    let late = read.elapsed();
    let (status, after) = run.interrupt();
    assert_eq!(status.code(), Some(0), "{:?}", run.rest());
    assert!(after < QUICK, "{after:?}");
    // A signal within the first 100 ms comes before the wait line.
    if late < std::time::Duration::from_millis(50) {
        let rest = run.rest();
        assert!(
            rest.iter().all(|line| !line.contains("judgment records")),
            "{rest:?}"
        );
    }
}

#[tokio::test]
async fn the_state_lists_the_last_ten_earlier_calls_of_the_turn() {
    let mut script: Vec<MockResponse> = (1..=12)
        .map(|i| {
            tool_turn(
                &format!("c{i}"),
                "bash",
                json!({"command": format!("echo {i}")}),
            )
        })
        .collect();
    script.push(text_turn("done"));
    // A new turn starts its list again.
    script.push(tool_turn("d1", "bash", json!({"command": "echo again"})));
    script.push(text_turn("done"));
    let bed = Bed::start(script).await;
    let mut judgments = Judgments::new(JUDGE);
    judgments.checks = config::Checks::uniform(CheckAction::Off);
    judgments.checks.set(Check::Outcome, CheckAction::Log);
    let runtime =
        Runtime::new(bed.judged_at(ApprovalMode::Auto, judgments, &bed.mock.url(), true)).unwrap();
    run_session(&runtime, &["go", "again"], &[]).await;
    let states: Vec<String> = bed
        .judge_requests()
        .iter()
        .map(|request| request.body["state"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(states.len(), 13);
    let calls = |state: &str| -> Option<Vec<String>> {
        let start = state.find("\n\ncalls:\n")? + "\n\ncalls:\n".len();
        Some(
            state[start..]
                .lines()
                .take_while(|line| !line.is_empty())
                .map(str::to_string)
                .collect(),
        )
    };
    let of = |command: &str| {
        states
            .iter()
            .find(|state| state.contains(&format!("\"command\":\"{command}\"")))
            .unwrap()
    };
    // The first call has no earlier call; the fifth lists four; the
    // twelfth lists the ten before it, never itself.
    assert_eq!(calls(of("echo 1")), None);
    assert_eq!(calls(of("echo 5")).unwrap().len(), 4);
    assert_eq!(
        calls(of("echo 12")).unwrap(),
        vec!["bash ok".to_string(); 10]
    );
    assert_eq!(calls(of("echo again")), None);
}
