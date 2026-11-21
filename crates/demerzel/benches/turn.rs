//! Layer-2 benchmark: per-turn latency against a local mock endpoint.
//!
//! The benches decompose the number:
//! - `transport_floor`: one bare HTTP POST to the mock, full body
//!   read. Everything below the harness: localhost round-trip, mock
//!   parse, SSE serialize.
//! - `turn_text`: one full engine turn with a text-only reply. The
//!   harness-added latency is `turn_text - transport_floor`.
//! - `turn_read_tool`: a turn with one read dispatch, two model calls.
//! - `turn_text_compat`: `turn_text` through a named `openai-compat`
//!   instance on the Chat Completions route. The run also defines a
//!   judge instance, with judgments off.
//! - `turn_text_responses`: `turn_text` through a named
//!   `openai-responses` instance on the Responses API route.
//! - `turn_text_judged`: `turn_text_compat` with judgments on: every
//!   check at `log` and a judge key in the store. A text turn sends no
//!   judge request, so the case measures the loop seam alone; the bench
//!   checks after the case that the mock judged and refused nothing.
//!
//! The bench drives `demerzel::Runtime`, the same wiring the
//! interfaces use. Session and engine construction, and session-file
//! cleanup, run outside the timed span. Session files land under the
//! cargo target directory, not tmpfs.
//!
//! The mock runs on its own single-worker runtime, so server work
//! never shares the bench thread. The bench runtime is
//! `current_thread`, the flavor the shipped binary uses. The auth
//! store keeps its default token URL; the script never answers 401,
//! so no refresh fires.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use demerzel::{AuthPaths, RunConfig, Runtime};
use demerzel_core::{ApprovalMode, EngineEvent, NoPrompt, TurnOutcome};
use demerzel_mock::codex::{text_turn, tool_turn};
use demerzel_mock::compat;
use demerzel_mock::responses;
use demerzel_mock::{MockResponse, MockServer};
use serde_json::json;

fn noop(_: EngineEvent<'_>) {}

/// The workspace target directory: a real filesystem, unlike `/tmp`.
fn bench_root() -> PathBuf {
    let target = match std::env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"),
    };
    let root = target.join("bench-data");
    std::fs::create_dir_all(&root).expect("create the bench data root");
    root
}

/// The long-lived pieces one bench shares across iterations. Field
/// order matters: the mock drops before its runtime.
struct Bed {
    mock: MockServer,
    _server_rt: tokio::runtime::Runtime,
    project: tempfile::TempDir,
    _data: tempfile::TempDir,
    runtime: Runtime,
}

impl Bed {
    fn start(script: Vec<MockResponse>) -> Self {
        Self::start_with(script, |_| {})
    }

    /// A bed whose run config `adjust` edits before the runtime
    /// builds.
    fn start_with(script: Vec<MockResponse>, adjust: impl FnOnce(&mut RunConfig)) -> Self {
        let root = bench_root();
        let server_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("build the mock-server runtime");
        let mock = server_rt.block_on(MockServer::start_looping(script));
        let project = tempfile::tempdir_in(&root).unwrap();
        let data = tempfile::tempdir_in(&root).unwrap();
        let auth_path = data.path().join("auth.json");
        std::fs::write(
            &auth_path,
            json!({
                "version": 3,
                "openai_codex": {"access_token": "bench-token", "refresh_token": "bench-refresh"},
            })
            .to_string(),
        )
        .unwrap();

        let mut config = RunConfig::new(project.path().to_path_buf());
        config.model = "bench-model".into();
        config.approval = ApprovalMode::Auto;
        config.data_root = Some(data.path().join("sessions"));
        config.instance.base_url = Some(mock.url());
        config.auth = Some(AuthPaths {
            store: auth_path,
            codex_cli: None,
        });
        adjust(&mut config);
        let runtime = Runtime::new(config).expect("build the harness runtime");

        Self {
            mock,
            _server_rt: server_rt,
            project,
            _data: data,
            runtime,
        }
    }

    /// One measured turn: build the engine and remove the session file
    /// outside the timed span; assert the workload really ran.
    async fn timed_turn(&self, expected_history: usize) -> Duration {
        let config = self.runtime.config();
        let settings = demerzel::SessionSettings::from_config(config);
        let mut engine = self
            .runtime
            .open(&config.resume, &settings, "bench")
            .unwrap()
            .engine;
        let start = Instant::now();
        let outcome = engine
            .user_turn("bench".into(), &noop, &mut NoPrompt)
            .await
            .unwrap();
        let elapsed = start.elapsed();
        assert_eq!(outcome, TurnOutcome::Completed);
        assert_eq!(engine.history().len(), expected_history);
        let path = engine.session().path().to_path_buf();
        drop(engine);
        std::fs::remove_file(path).unwrap();
        elapsed
    }
}

/// The judge model of the judged beds.
const JUDGE_MODEL: &str = "jev-1.14.0";

/// A named compatible head at the mock's `/v1` root, and a judge
/// instance at the mock's root. With `judged`, every check runs at
/// `log` and the judge's key is in the run's store.
fn compat_with_judge(config: &mut RunConfig, judged: bool) {
    let url = config.instance.base_url.take().expect("the mock URL");
    config.instance.name = "bench-compat".into();
    config.instance.kind = "openai-compat".into();
    config.instance.base_url = Some(format!("{url}/v1"));
    let user = demerzel::config::Layer::parse(&format!(
        "[providers.judge]\nkind = \"typesafe\"\nbase_url = \"{url}\"\nmodel = \"{JUDGE_MODEL}\"\n"
    ))
    .expect("parse the judge instance")
    .0;
    config.other_instances = demerzel::config::run_instances(
        &demerzel::config::Flags::default(),
        &|_| None,
        None,
        Some(&user),
    )
    .into_iter()
    .filter(|resolved| resolved.instance.name != "bench-compat")
    .collect();
    if judged {
        let store = &config.auth.as_ref().expect("the bench store").store;
        demerzel_providers::auth::AuthStore::set_instance_key(
            store,
            "judge",
            "fake-bench-judge-key",
            &url,
        )
        .expect("store the judge key");
        let mut judgments = demerzel::config::Judgments::new("judge");
        judgments.checks = demerzel::config::Checks::uniform(demerzel::config::CheckAction::Log);
        config.judgments = Some(judgments);
    }
}

/// Settle the bed's judgment requests, outside any timed span.
fn bed_rt_settle(runtime: &tokio::runtime::Runtime, bed: &Bed) {
    runtime.block_on(bed.runtime.settle());
}

fn bench_turns(c: &mut Criterion) {
    let bench_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build the bench runtime");

    // The transport-and-mock floor under every turn.
    let bed = Bed::start(vec![text_turn("done")]);
    let client = reqwest::Client::new();
    let url = format!("{}/responses", bed.mock.url());
    c.bench_function("transport_floor", |b| {
        b.to_async(&bench_rt).iter_custom(|iters| {
            let client = &client;
            let url = &url;
            async move {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    let body = client
                        .post(url)
                        .json(&json!({"model": "bench-model"}))
                        .send()
                        .await
                        .unwrap()
                        .text()
                        .await
                        .unwrap();
                    total += start.elapsed();
                    assert!(body.contains("response.completed"));
                }
                total
            }
        })
    });
    drop(bed);

    // One model call, text only: user and assistant in the history.
    let bed = Bed::start(vec![text_turn("done")]);
    c.bench_function("turn_text", |b| {
        b.to_async(&bench_rt).iter_custom(|iters| {
            let bed = &bed;
            async move {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += bed.timed_turn(2).await;
                }
                total
            }
        })
    });
    drop(bed);

    // Two model calls with one read dispatch between them: user,
    // tool-call, tool-result, and final assistant in the history.
    let bed = Bed::start(vec![
        tool_turn("call_1", "read", json!({"path": "f.txt"})),
        text_turn("done"),
    ]);
    std::fs::write(bed.project.path().join("f.txt"), "alpha\nbeta\n").unwrap();
    c.bench_function("turn_read_tool", |b| {
        b.to_async(&bench_rt).iter_custom(|iters| {
            let bed = &bed;
            async move {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += bed.timed_turn(4).await;
                }
                total
            }
        })
    });
    drop(bed);

    // `turn_text` through a named compatible instance: the same turn
    // on the Chat Completions route, with no credential. The run also
    // knows a judge instance, so the case differs from
    // `turn_text_judged` only by judgments.
    let bed = Bed::start_with(vec![compat::text_turn("done")], |config| {
        compat_with_judge(config, false);
    });
    c.bench_function("turn_text_compat", |b| {
        b.to_async(&bench_rt).iter_custom(|iters| {
            let bed = &bed;
            async move {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += bed.timed_turn(2).await;
                }
                total
            }
        })
    });
    drop(bed);

    // `turn_text_compat` with judgments: every check at `log`, and a
    // judge key in the store, so a check that fired would send a
    // request. A text turn has no call, so none fires.
    let bed = Bed::start_with(vec![compat::text_turn("done")], |config| {
        compat_with_judge(config, true);
    });
    bed.mock.set_judge(demerzel_mock::JudgeScript::new(
        JUDGE_MODEL,
        [
            "approval.destructive",
            "approval.in_scope",
            "outcome",
            "progress",
            "claim",
        ]
        .into_iter()
        .map(|id| (id.to_string(), demerzel_mock::systemone::noul(0.5)))
        .collect(),
    ));
    c.bench_function("turn_text_judged", |b| {
        b.to_async(&bench_rt).iter_custom(|iters| {
            let bed = &bed;
            async move {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += bed.timed_turn(2).await;
                }
                total
            }
        })
    });
    bed_rt_settle(&bench_rt, &bed);
    assert_eq!(bed.mock.judged(), 0, "a text turn sends no judge request");
    assert_eq!(bed.mock.refused(), 0, "the mock refused a request");
    assert!(bed.runtime.take_notices().is_empty());
    drop(bed);

    // `turn_text` through a named Responses instance: the same turn on
    // the Responses API route, with no credential.
    let bed = Bed::start_with(vec![responses::text_turn("done")], |config| {
        config.instance.name = "bench-responses".into();
        config.instance.kind = "openai-responses".into();
        config.instance.base_url = config
            .instance
            .base_url
            .take()
            .map(|url| format!("{url}/v1"));
    });
    c.bench_function("turn_text_responses", |b| {
        b.to_async(&bench_rt).iter_custom(|iters| {
            let bed = &bed;
            async move {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += bed.timed_turn(2).await;
                }
                total
            }
        })
    });
}

criterion_group!(benches, bench_turns);
criterion_main!(benches);
