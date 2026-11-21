//! Judgments: the run's judge beside the loop.
//!
//! A `Host` sits on each session's engine. At each loop event it
//! filters the built-in checks and matching rules, builds the state
//! of the event, sends one request for all questions of the event,
//! writes one `judgment` record, and applies configured actions
//! (`log`, `warn`, `note`, `act`, or rule `deny`).
//!
//! Checks at `log` and `tool_after` `warn` run asynchronously beside
//! the turn. Blocking checks (`tool_before` above `log`, `tool_after`
//! `note`, and `turn_end` `warn`) await the judge within its timeout
//! and leave the policy decision unchanged on timeout or failure.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Write as _;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use demerzel_core::{
    Answer, ApprovalDecision, ApprovalMode, BoxFuture, CallSeen, Finding, HostReply, Judge,
    JudgeError, JudgeReply, JudgeRequest, JudgmentRecord, JudgmentRuleRecord, LoopHost,
    ProviderError, Question, SessionWriter, ToolAccess, ToolAfter, ToolBefore, ToolGate, TurnEnd,
    TurnEndOutcome, TurnOutcome, TurnStart,
};
use demerzel_providers::typesafe::{self, TypeSafeConfig, TypeSafeJudge};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use crate::config::{self, Check, CheckAction, Checks, Rule, RuleAction, RuleEvent};
use crate::{AuthPaths, RunError};

/// The budget of one judge request, retries included.
pub(crate) const TIMEOUT: Duration = Duration::from_secs(2);

/// The judge requests one run sends at the same time, over all its
/// sessions. A request past it waits for a permit, at most the request
/// timeout from its event; then its record says `busy`.
pub(crate) const MAX_IN_FLIGHT: usize = 8;

/// What [`Judging::settle`] waits past the permit wait and the request:
/// the last record's write.
const SETTLE_GRACE: Duration = Duration::from_millis(250);

/// Cap on `busy` records written by one session when the judge is
/// stalled, so a long turn cannot flood the session file without
/// counting against `max_calls`.
const BUSY_RECORD_CAP: u32 = 32;

/// The `error` of a record whose request waited too long for a permit
/// and sent nothing.
pub(crate) const BUSY: &str = "busy";
/// The `error` of a record whose request had no answer in time.
pub(crate) const TIMED_OUT: &str = "timeout";

/// The built-in firing threshold for `outcome`, `progress`, and `claim`.
pub const CHECK_THRESHOLD: f64 = 0.8;
/// Maximum destructive probability for an `ask`-mode judge approval.
pub const APPROVAL_ASK_MAX_DESTRUCTIVE: f64 = 0.05;
/// Minimum in-scope probability for an `ask`-mode judge approval.
pub const APPROVAL_ASK_MIN_IN_SCOPE: f64 = 0.9;
/// Minimum destructive probability for an `auto`-mode judge escalation.
pub const APPROVAL_AUTO_MIN_DESTRUCTIVE: f64 = 0.5;

/// The question ids. They stay in code; the records key the answers
/// by them.
pub const APPROVAL_DESTRUCTIVE: &str = "approval.destructive";
pub const APPROVAL_IN_SCOPE: &str = "approval.in_scope";
pub const OUTCOME: &str = "outcome";
pub const PROGRESS: &str = "progress";
pub const CLAIM: &str = "claim";

/// Whether `ask`-mode `approval = "act"` removes the approval prompt.
pub fn approval_ask_allows(destructive: f64, in_scope: f64) -> bool {
    destructive <= APPROVAL_ASK_MAX_DESTRUCTIVE && in_scope >= APPROVAL_ASK_MIN_IN_SCOPE
}

/// Whether `auto`-mode `approval = "act"` escalates a call to an approval prompt.
pub fn approval_auto_asks(destructive: f64) -> bool {
    destructive >= APPROVAL_AUTO_MIN_DESTRUCTIVE
}

/// Whether a built-in `outcome`, `progress`, or `claim` check fires.
pub fn check_fires(yes: f64) -> bool {
    yes >= CHECK_THRESHOLD
}

/// Build the approval-check state text using the exact production state builder.
pub fn approval_state(prompt: &str, tool: &str, input: &Value, calls: &[(&str, bool)]) -> String {
    build_state(&StateParts {
        prompt,
        tool: Some(tool),
        input: Some(input),
        calls,
        ..StateParts::default()
    })
}

/// Build the `outcome`-check state text using the exact production state builder.
pub fn outcome_state(
    prompt: &str,
    tool: &str,
    input: &Value,
    output: &str,
    calls: &[(&str, bool)],
) -> String {
    build_state(&StateParts {
        prompt,
        tool: Some(tool),
        input: Some(input),
        output: Some(output),
        calls,
        ..StateParts::default()
    })
}

/// Build the `progress`-check state text using the exact production state builder.
pub fn progress_state(
    prompt: &str,
    tool: &str,
    input: &Value,
    earlier_failed_input: &Value,
    output: &str,
    calls: &[(&str, bool)],
) -> String {
    let earlier = JsonHead::new(earlier_failed_input, INPUT_CAP);
    build_state(&StateParts {
        prompt,
        tool: Some(tool),
        input: Some(input),
        earlier_failed_input: Some(&earlier),
        output: Some(output),
        calls,
        ..StateParts::default()
    })
}

/// Build the `claim`-check state text using the exact production state builder.
pub fn claim_state(
    prompt: &str,
    last_output: &str,
    calls: &[(&str, bool)],
    final_message: &str,
) -> String {
    build_state(&StateParts {
        prompt,
        output: Some(last_output),
        calls,
        final_message: Some(final_message),
        ..StateParts::default()
    })
}

/// The production approval-check questions.
pub fn approval_questions() -> BTreeMap<String, Question> {
    questions(&[Check::Approval], &[], &[])
}

/// The production questions for a set of built-in checks.
pub fn check_questions(checks: &[Check]) -> BTreeMap<String, Question> {
    questions(checks, &[], &[])
}

/// The question texts. The wording is not contract.
const APPROVAL_DESTRUCTIVE_TEXT: &str = "Is this tool call destructive or hard to undo? \
     It deletes or overwrites data, changes state outside the project, or cannot be reverted.";
const APPROVAL_IN_SCOPE_TEXT: &str =
    "Is this tool call inside what the user asked for in the prompt of this turn?";
const OUTCOME_TEXT: &str = "The command returned without its error flag. Did it fail, \
     or return something other than what it ran for?";
const PROGRESS_TEXT: &str = "An earlier call of the same tool failed in this turn. \
     Does this call repeat the failed action without a material change?";
const CLAIM_TEXT: &str =
    "Does the final message claim a result that the tool evidence of this turn does not show?";

/// The state caps, in bytes of UTF-8. Each cut marker counts inside
/// its cap.
const PROMPT_CAP: usize = 4 * 1024;
const TOOL_NAME_CAP: usize = 128;
const INPUT_CAP: usize = 4 * 1024;
const OUTPUT_CAP: usize = 8 * 1024;
const FINAL_CAP: usize = 4 * 1024;
/// The calls the state lists: the turn's last ones.
const CALLS_SHOWN: usize = 10;
/// The whole state of one request.
pub(crate) const STATE_CAP: usize = 16 * 1024;

/// The most bytes of an error text a record keeps.
const ERROR_CAP: usize = 1024;

/// The section labels, in their order in the state.
const PROMPT_LABEL: &str = "prompt";
const TOOL_LABEL: &str = "tool";
const INPUT_LABEL: &str = "input";
const EARLIER_FAILED_INPUT_LABEL: &str = "earlier failed input";
const OUTPUT_TAIL_LABEL: &str = "output tail";
const CALLS_LABEL: &str = "calls";
const FINAL_MESSAGE_LABEL: &str = "final message";

/// The order the builder cuts sections in when the state is over its
/// cap.
const CUT_ORDER: [&str; 5] = [
    OUTPUT_TAIL_LABEL,
    CALLS_LABEL,
    EARLIER_FAILED_INPUT_LABEL,
    INPUT_LABEL,
    PROMPT_LABEL,
];

/// The run's judgments: the judge slot, the checks, and what every
/// session's host shares. One per runtime.
pub(crate) struct Judging {
    shared: Arc<Shared>,
}

/// What each host and each request task share.
struct Shared {
    judge: JudgeSlot,
    checks: Checks,
    rules: Vec<Rule>,
    /// True when any configured check or rule acts above `log`.
    acts_above_log: bool,
    max_calls: u32,
    /// The request timeout; [`TIMEOUT`] outside tests.
    timeout: Duration,
    /// The run's request permits, [`MAX_IN_FLIGHT`] of them.
    permits: Arc<Semaphore>,
    flight: Arc<Flight>,
    notices: Mutex<Vec<String>>,
    /// The latest opened session's state, for `/judgments`.
    latest_session: Mutex<Option<Arc<SessionState>>>,
    /// The tasks the hosts spawned, for tests.
    #[cfg(test)]
    spawned: AtomicUsize,
}

/// The judge of the run's judge instance, built at the first check
/// that sends a request. A failed build is not kept: the next turn
/// tries again, so a login between two prompts takes effect.
struct JudgeSlot {
    instance: config::Instance,
    auth: Option<AuthPaths>,
    judge: OnceLock<Arc<dyn Judge>>,
}

/// The request tasks that have not ended: a count, and a wake-up when
/// it reaches 0.
#[derive(Default)]
struct Flight {
    count: AtomicUsize,
    idle: tokio::sync::Notify,
}

/// One task's hold on [`Flight`]. It lets go when the task ends or
/// drops.
struct FlightGuard(Arc<Flight>);

impl Flight {
    fn enter(self: &Arc<Self>) -> FlightGuard {
        self.count.fetch_add(1, Ordering::AcqRel);
        FlightGuard(Arc::clone(self))
    }

    /// Wait until no task is in flight.
    async fn idle(&self) {
        loop {
            let notified = self.idle.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.count.load(Ordering::Acquire) == 0 {
                return;
            }
            notified.await;
        }
    }
}

impl Drop for FlightGuard {
    fn drop(&mut self) {
        if self.0.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.idle.notify_waiters();
        }
    }
}

/// The judgment requests one opened session may send. A shared
/// handle, so a child session can count against its parent's cap. A
/// request counts when it goes out.
pub(crate) struct Budget {
    used: AtomicU32,
    cap: u32,
    /// Set once the session's cap notice went out.
    noticed: AtomicBool,
}

impl Budget {
    fn new(cap: u32) -> Self {
        Self {
            used: AtomicU32::new(0),
            cap,
            noticed: AtomicBool::new(false),
        }
    }

    /// Whether the cap is reached; takes nothing.
    fn spent(&self) -> bool {
        self.used.load(Ordering::Acquire) >= self.cap
    }

    /// Take one request from the budget; `false` at the cap.
    fn take(&self) -> bool {
        let mut used = self.used.load(Ordering::Acquire);
        while used < self.cap {
            match self.used.compare_exchange_weak(
                used,
                used + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(current) => used = current,
            }
        }
        false
    }

    /// The session's one cap notice, the first time.
    fn notice(&self, shared: &Shared) {
        if !self.noticed.swap(true, Ordering::AcqRel) {
            shared.notice(format!(
                "judgments: this session reached judgments.max_calls ({}); its checks send no \
                 more requests",
                self.cap
            ));
        }
    }
}

/// Live counters for one opened session, shown by `/judgments`.
#[derive(Default)]
struct SessionStats {
    fired_warn: AtomicU32,
    fired_note: AtomicU32,
    fired_allow: AtomicU32,
    fired_ask: AtomicU32,
    fired_deny: AtomicU32,
    errors: AtomicU32,
    timeouts: AtomicU32,
    busy: AtomicU32,
}

impl SessionStats {
    fn record_fired(&self, action: &str) {
        match action {
            "warn" => self.fired_warn.fetch_add(1, Ordering::Relaxed),
            "note" => self.fired_note.fetch_add(1, Ordering::Relaxed),
            "allow" => self.fired_allow.fetch_add(1, Ordering::Relaxed),
            "ask" => self.fired_ask.fetch_add(1, Ordering::Relaxed),
            "deny" => self.fired_deny.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
    }
}

/// What a host and its tasks share for one session.
struct SessionState {
    budget: Arc<Budget>,
    /// The number of the current turn, from 1.
    turn: AtomicU64,
    /// The last turn in which the judge did not build; 0 for none.
    /// Later events of that turn skip their check.
    build_failed: AtomicU64,
    /// Set once the session's notice for a judge that does not build
    /// went out.
    build_noticed: AtomicBool,
    /// Set once the session's notice for a transport error or timeout
    /// went out (only when a check or rule acts above `log`).
    transport_noticed: AtomicBool,
    /// Set once the session's notice for hitting the `busy` cap went out.
    busy_noticed: AtomicBool,
    /// Fast-path flag so `take_findings()` avoids locking on every
    /// streaming event when no async finding is queued.
    has_findings: AtomicBool,
    /// Async findings from `tool_after` `warn` tasks, drained by the engine.
    findings: Mutex<Vec<Finding>>,
    /// In-flight `tool_after` `warn` task handles for the current turn,
    /// awaited at `turn_end` if still running.
    pending_warns: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    stats: SessionStats,
}

impl SessionState {
    fn push_findings(&self, mut findings: Vec<Finding>) {
        if findings.is_empty() {
            return;
        }
        let mut queue = self.findings.lock().unwrap_or_else(PoisonError::into_inner);
        queue.append(&mut findings);
        self.has_findings.store(true, Ordering::Release);
    }

    fn take_findings(&self) -> Vec<Finding> {
        if !self.has_findings.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut queue = self.findings.lock().unwrap_or_else(PoisonError::into_inner);
        self.has_findings.store(false, Ordering::Release);
        std::mem::take(&mut *queue)
    }

    fn push_pending_warn(&self, handle: tokio::task::JoinHandle<()>) {
        let mut warns = self
            .pending_warns
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        warns.retain(|h| !h.is_finished());
        warns.push(handle);
    }

    fn take_pending_warns(&self) -> Vec<tokio::task::JoinHandle<()>> {
        std::mem::take(
            &mut *self
                .pending_warns
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        )
    }

    fn note_transport_failure(&self, shared: &Shared, error: &str) {
        if shared.acts_above_log && !self.transport_noticed.swap(true, Ordering::AcqRel) {
            shared.notice(format!(
                "judgments: judge `{}` failed ({error}); checks fall back to the approval \
                 policy (DEMERZEL_LOG=warn logs each failure)",
                shared.judge.instance.name
            ));
        }
    }
}

impl Judging {
    /// The judgments of a run over `instance`, the judge instance.
    /// `None` when every check is off and no rule is configured: no
    /// host, no request.
    pub(crate) fn new(
        judgments: &config::Judgments,
        instance: config::Instance,
        auth: Option<AuthPaths>,
    ) -> Option<Self> {
        let checks = judgments.checks;
        let all_off = Check::ALL
            .iter()
            .all(|&check| checks.get(check) == CheckAction::Off);
        if all_off && judgments.rules.is_empty() {
            return None;
        }
        let acts_above_log = Check::ALL
            .iter()
            .any(|&check| checks.get(check) > CheckAction::Log)
            || judgments
                .rules
                .iter()
                .any(|rule| rule.action != RuleAction::Log);
        Some(Self {
            shared: Arc::new(Shared {
                judge: JudgeSlot {
                    instance,
                    auth,
                    judge: OnceLock::new(),
                },
                checks,
                rules: judgments.rules.clone(),
                acts_above_log,
                max_calls: judgments.max_calls,
                timeout: TIMEOUT,
                permits: Arc::new(Semaphore::new(MAX_IN_FLIGHT)),
                flight: Arc::default(),
                notices: Mutex::new(Vec::new()),
                latest_session: Mutex::new(None),
                #[cfg(test)]
                spawned: AtomicUsize::new(0),
            }),
        })
    }

    /// Set the request timeout before any session opens. Tests use a
    /// short one.
    #[cfg(test)]
    pub(crate) fn set_timeout(&mut self, timeout: Duration) {
        Arc::get_mut(&mut self.shared)
            .expect("no session holds the judgments yet")
            .timeout = timeout;
    }

    /// The host of one opened session: a new call budget, and the
    /// session's own writer, so each record lands in the file of the
    /// session that asked.
    pub(crate) fn host(&self, writer: SessionWriter) -> Host {
        let session = Arc::new(SessionState {
            budget: Arc::new(Budget::new(self.shared.max_calls)),
            turn: AtomicU64::new(1),
            build_failed: AtomicU64::new(0),
            build_noticed: AtomicBool::new(false),
            transport_noticed: AtomicBool::new(false),
            busy_noticed: AtomicBool::new(false),
            has_findings: AtomicBool::new(false),
            findings: Mutex::new(Vec::new()),
            pending_warns: Mutex::new(Vec::new()),
            stats: SessionStats::default(),
        });
        *self
            .shared
            .latest_session
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(&session));
        Host {
            shared: Arc::clone(&self.shared),
            session,
            writer,
            calls: Mutex::new(TurnCalls::default()),
        }
    }

    /// Drop the active session stats handle (for `/new`).
    pub(crate) fn clear_session(&self) {
        *self
            .shared
            .latest_session
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
    }

    /// Human-readable status for `/judgments`.
    pub(crate) fn summary(&self) -> String {
        let shared = &self.shared;
        let instance = &shared.judge.instance;
        let model = instance
            .model
            .as_deref()
            .or_else(|| demerzel_providers::default_model(&instance.kind))
            .unwrap_or("unknown");
        let mut out = format!(
            "judgments: {} ({}, model {model})",
            instance.name, instance.kind
        );
        let check_parts: Vec<String> = Check::ALL
            .iter()
            .map(|&check| format!("{}={}", check.as_str(), shared.checks.get(check).as_str()))
            .collect();
        let _ = write!(out, "\n  checks: {}", check_parts.join(", "));
        if shared.rules.is_empty() {
            out.push_str("\n  rules: none");
        } else {
            let _ = write!(out, "\n  rules ({}):", shared.rules.len());
            for (index, rule) in shared.rules.iter().enumerate() {
                let tool = rule
                    .tool
                    .as_deref()
                    .map(|t| format!(" tool={t}"))
                    .unwrap_or_default();
                let _ = write!(
                    out,
                    "\n    rule.{index} [{}]{tool} {} (threshold {:.2}, {}): {}",
                    rule.event, rule.action, rule.threshold, rule.origin, rule.ask
                );
            }
        }
        let session = shared
            .latest_session
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        match session {
            Some(session) => {
                let used = session.budget.used.load(Ordering::Acquire);
                let s = &session.stats;
                let _ = write!(out, "\n  session: {used}/{} calls", shared.max_calls);
                let fired = [
                    ("allow", s.fired_allow.load(Ordering::Relaxed)),
                    ("ask", s.fired_ask.load(Ordering::Relaxed)),
                    ("deny", s.fired_deny.load(Ordering::Relaxed)),
                    ("warn", s.fired_warn.load(Ordering::Relaxed)),
                    ("note", s.fired_note.load(Ordering::Relaxed)),
                ];
                let active_fired: Vec<String> = fired
                    .into_iter()
                    .filter(|&(_, count)| count > 0)
                    .map(|(name, count)| format!("{name}={count}"))
                    .collect();
                if !active_fired.is_empty() {
                    let _ = write!(out, " · fired: {}", active_fired.join(", "));
                }
                let errors = s.errors.load(Ordering::Relaxed);
                let timeouts = s.timeouts.load(Ordering::Relaxed);
                let busy = s.busy.load(Ordering::Relaxed);
                if errors > 0 || timeouts > 0 || busy > 0 {
                    let _ = write!(
                        out,
                        " · failures: errors={errors}, timeouts={timeouts}, busy={busy}"
                    );
                }
            }
            None => {
                let _ = write!(out, "\n  session: none (cap {} calls)", shared.max_calls);
            }
        }
        out
    }

    /// The notices since the last call.
    pub(crate) fn take_notices(&self) -> Vec<String> {
        std::mem::take(
            &mut *self
                .shared
                .notices
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        )
    }

    /// The request tasks that have not ended.
    pub(crate) fn in_flight(&self) -> usize {
        self.shared.flight.count.load(Ordering::Acquire)
    }

    /// Wait for the request tasks in flight. A task lives at most the
    /// permit wait and the request, one timeout each; the bound adds a
    /// grace for the last write.
    pub(crate) async fn settle(&self) {
        let flight = &self.shared.flight;
        if flight.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let wait = self.shared.timeout * 2 + SETTLE_GRACE;
        if tokio::time::timeout(wait, flight.idle()).await.is_err() {
            log::warn!(
                "judgments: {} requests still in flight after {} ms; their records are lost",
                flight.count.load(Ordering::Acquire),
                wait.as_millis()
            );
        }
    }
}

impl Shared {
    fn notice(&self, notice: String) {
        self.notices
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(notice);
    }
}

impl JudgeSlot {
    fn get(&self) -> Result<Arc<dyn Judge>, RunError> {
        if let Some(judge) = self.judge.get() {
            return Ok(Arc::clone(judge));
        }
        let judge = build_judge(&self.instance, self.auth.as_ref())?;
        Ok(Arc::clone(self.judge.get_or_init(|| judge)))
    }
}

/// The endpoint a judge instance's requests and its stored key go to:
/// the instance's `base_url`, else its kind's default. `dmz login`
/// binds a key to the same endpoint.
fn judge_endpoint(instance: &config::Instance) -> Option<String> {
    instance
        .base_url
        .clone()
        .or_else(|| demerzel_providers::default_base_url(&instance.kind).map(str::to_string))
}

/// Build the judge of `instance`, with its key: the instance's own
/// key variable, else its stored key for its endpoint.
fn build_judge(
    instance: &config::Instance,
    auth: Option<&AuthPaths>,
) -> Result<Arc<dyn Judge>, RunError> {
    let name = &instance.name;
    let kind = instance.kind.as_str();
    if kind != typesafe::PROVIDER_ID {
        return Err(RunError::Config(format!(
            "provider `{name}` ({kind}) cannot serve as a judge"
        )));
    }
    let base_url = judge_endpoint(instance)
        .ok_or_else(|| RunError::Config(format!("provider `{name}` ({kind}) needs `base_url`")))?;
    let model = instance
        .model
        .clone()
        .or_else(|| demerzel_providers::default_model(kind).map(str::to_string))
        .ok_or_else(|| RunError::Config(format!("provider `{name}` ({kind}) needs `model`")))?;
    let key = match crate::named_key(instance)? {
        Some(key) => key,
        None => crate::stored_instance_key(name, auth, &base_url)?.ok_or_else(|| {
            RunError::Auth(ProviderError::Auth(format!(
                "provider `{name}` needs a key: run `dmz login {name}`, or set `api_key_env`"
            )))
        })?,
    };
    let config = TypeSafeConfig::new(name.clone(), base_url, model);
    Ok(Arc::new(TypeSafeJudge::new(config, key)))
}

/// The judgments host of one session.
pub(crate) struct Host {
    shared: Arc<Shared>,
    session: Arc<SessionState>,
    writer: SessionWriter,
    /// What the host keeps of the current turn's calls; emptied at
    /// `turn_start`.
    calls: Mutex<TurnCalls>,
}

/// The current turn's calls as the host keeps them, filled at each
/// `tool_after`: every event reads them in constant time, however long
/// the turn.
#[derive(Default)]
struct TurnCalls {
    /// The input of the turn's latest failed call of each tool, as the
    /// state keeps it: the `progress` filter and its state.
    failed: HashMap<String, JsonHead>,
    /// The turn's last calls that ran, oldest first: the name and the
    /// error flag, at most [`CALLS_SHOWN`].
    recent: VecDeque<(String, bool)>,
}

impl TurnCalls {
    /// The recent calls, as the state builder takes them.
    fn recent(&self) -> Vec<(&str, bool)> {
        self.recent
            .iter()
            .map(|(name, is_error)| (name.as_str(), *is_error))
            .collect()
    }
}

impl Host {
    /// Prepare an [`Ask`] when the session is eligible to call the
    /// judge (not build-failed this turn and budget not spent).
    fn prepare(
        &self,
        event: &'static str,
        call: Option<(&str, &str)>,
        asked: Vec<Check>,
        rule_ids: Vec<usize>,
        state: impl FnOnce() -> String,
    ) -> Option<Ask> {
        let session = &self.session;
        let turn = session.turn.load(Ordering::Acquire);
        if session.build_failed.load(Ordering::Acquire) == turn {
            return None;
        }
        if session.budget.spent() {
            session.budget.notice(&self.shared);
            return None;
        }
        Some(Ask {
            shared: Arc::clone(&self.shared),
            session: Arc::clone(session),
            writer: self.writer.clone(),
            event,
            turn,
            call_id: call.map(|(id, _)| id.to_string()),
            tool: call.map(|(_, tool)| tool.to_string()),
            asked,
            rule_ids,
            state: state(),
            spawned: Instant::now(),
            _guard: self.shared.flight.enter(),
        })
    }

    /// Spawn a background log-only ask task.
    fn spawn_log(
        &self,
        event: &'static str,
        call: Option<(&str, &str)>,
        asked: Vec<Check>,
        rule_ids: Vec<usize>,
        state: impl FnOnce() -> String,
    ) {
        let Some(task) = self.prepare(event, call, asked, rule_ids, state) else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            log::warn!("judgments: no async runtime at {event}; the checks send no request");
            return;
        };
        #[cfg(test)]
        self.shared.spawned.fetch_add(1, Ordering::Relaxed);
        runtime.spawn(async move {
            let _ = task.execute(BTreeMap::new()).await;
        });
    }

    fn calls(&self) -> std::sync::MutexGuard<'_, TurnCalls> {
        self.calls.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl LoopHost for Host {
    fn turn_start<'a>(&'a self, _event: &'a TurnStart<'a>) -> BoxFuture<'a, HostReply> {
        self.session.turn.fetch_add(1, Ordering::AcqRel);
        *self.calls() = TurnCalls::default();
        self.session.take_findings();
        // Abort any leftover warn tasks from a previously cancelled turn
        // so they do not keep occupying judge permits.
        for handle in self.session.take_pending_warns() {
            handle.abort();
        }
        Box::pin(std::future::ready(HostReply::default()))
    }

    fn tool_before<'a>(&'a self, event: &'a ToolBefore<'a>) -> BoxFuture<'a, ToolGate> {
        // In `read-only` mode (or any policy refusal), the policy
        // refuses the call before any judgment.
        if event.decision == ApprovalDecision::Deny {
            return Box::pin(std::future::ready(ToolGate::keep(event.decision)));
        }
        let checks = &self.shared.checks;
        let run_approval = checks.approval != CheckAction::Off
            && approval_applies(event.access, event.mode, event.decision);
        let rule_ids = matching_rules(&self.shared.rules, RuleEvent::ToolBefore, Some(event.tool));
        if !run_approval && rule_ids.is_empty() {
            return Box::pin(std::future::ready(ToolGate::keep(event.decision)));
        }
        let mut asked = Vec::with_capacity(1);
        if run_approval {
            asked.push(Check::Approval);
        }
        let blocking = (run_approval && checks.approval > CheckAction::Log)
            || rule_ids
                .iter()
                .any(|&i| self.shared.rules[i].action != RuleAction::Log);
        let build = || {
            let turn = self.calls();
            build_state(&StateParts {
                prompt: event.turn.prompt(),
                tool: Some(event.tool),
                input: Some(event.input),
                calls: &turn.recent(),
                ..StateParts::default()
            })
        };
        if !blocking {
            self.spawn_log(
                "tool_before",
                Some((event.call_id, event.tool)),
                asked,
                rule_ids,
                build,
            );
            return Box::pin(std::future::ready(ToolGate::keep(event.decision)));
        }
        let Some(task) = self.prepare(
            "tool_before",
            Some((event.call_id, event.tool)),
            asked,
            rule_ids,
            build,
        ) else {
            return Box::pin(std::future::ready(ToolGate::keep(event.decision)));
        };
        let mode = event.mode;
        let decision = event.decision;
        Box::pin(async move { task.run_tool_before(mode, decision).await })
    }

    fn tool_after(&self, event: &ToolAfter<'_>) -> BoxFuture<'static, HostReply> {
        let checks = &self.shared.checks;
        let outcome = checks.outcome != CheckAction::Off
            && outcome_applies(event.access, event.output.is_error);
        let progress_enabled = checks.progress != CheckAction::Off;
        let rule_ids = matching_rules(&self.shared.rules, RuleEvent::ToolAfter, Some(event.tool));

        // Capture the earlier failed input and recent calls while
        // updating `self.calls`, then drop the lock before calling
        // `self.prepare` (which may lock `shared.notices`).
        let task_opt = {
            let mut turn = self.calls();
            let has_earlier = progress_enabled && turn.failed.contains_key(event.tool);
            let mut asked = Vec::new();
            if outcome {
                asked.push(Check::Outcome);
            }
            if has_earlier {
                asked.push(Check::Progress);
            }
            let eligible = (!asked.is_empty() || !rule_ids.is_empty()) && {
                let t = self.session.turn.load(Ordering::Acquire);
                self.session.build_failed.load(Ordering::Acquire) != t
                    && !self.session.budget.spent()
            };
            let state = eligible.then(|| {
                let earlier = if progress_enabled {
                    turn.failed.get(event.tool)
                } else {
                    None
                };
                build_state(&StateParts {
                    prompt: event.turn.prompt(),
                    tool: Some(event.tool),
                    input: Some(event.input),
                    earlier_failed_input: earlier,
                    output: Some(&event.output.content),
                    calls: &turn.recent(),
                    ..StateParts::default()
                })
            });
            // This call is an earlier failed call for the next ones.
            if progress_enabled && event.output.is_error {
                turn.failed.insert(
                    event.tool.to_string(),
                    JsonHead::new(event.input, INPUT_CAP),
                );
            }
            // This call is an earlier call for the next events.
            if turn.recent.len() == CALLS_SHOWN {
                turn.recent.pop_front();
            }
            turn.recent
                .push_back((event.tool.to_string(), event.output.is_error));
            drop(turn);

            if !asked.is_empty() || !rule_ids.is_empty() {
                self.prepare(
                    "tool_after",
                    Some((event.call_id, event.tool)),
                    asked,
                    rule_ids,
                    || state.unwrap_or_default(),
                )
            } else {
                None
            }
        };

        let Some(task) = task_opt else {
            return Box::pin(std::future::ready(HostReply::default()));
        };
        let needs_note = task
            .asked
            .iter()
            .any(|&check| checks.get(check) == CheckAction::Note)
            || task
                .rule_ids
                .iter()
                .any(|&i| self.shared.rules[i].action == RuleAction::Note);
        if needs_note {
            return Box::pin(async move { task.run_tool_after().await });
        }
        let needs_warn = task
            .asked
            .iter()
            .any(|&check| checks.get(check) >= CheckAction::Warn)
            || task
                .rule_ids
                .iter()
                .any(|&i| self.shared.rules[i].action != RuleAction::Log);
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            log::warn!("judgments: no async runtime at tool_after; the checks send no request");
            return Box::pin(std::future::ready(HostReply::default()));
        };
        #[cfg(test)]
        self.shared.spawned.fetch_add(1, Ordering::Relaxed);
        if needs_warn {
            let session = Arc::clone(&self.session);
            let task_turn = task.turn;
            let handle = runtime.spawn(async move {
                let reply = task.run_tool_after().await;
                if session.turn.load(Ordering::Acquire) == task_turn {
                    session.push_findings(reply.findings);
                }
            });
            self.session.push_pending_warn(handle);
        } else {
            runtime.spawn(async move {
                let _ = task.execute(BTreeMap::new()).await;
            });
        }
        Box::pin(std::future::ready(HostReply::default()))
    }

    fn turn_end(&self, event: &TurnEnd<'_>) -> BoxFuture<'static, HostReply> {
        let completed = event.outcome == TurnEndOutcome::Outcome(TurnOutcome::Completed);
        let claim_enabled = completed && self.shared.checks.claim != CheckAction::Off;
        let rule_ids = if completed {
            matching_rules(&self.shared.rules, RuleEvent::TurnEnd, None)
        } else {
            Vec::new()
        };
        let seen: Vec<CallSeen<'_>> = if claim_enabled || !rule_ids.is_empty() {
            event.turn.calls().collect()
        } else {
            Vec::new()
        };
        let run_claim = claim_enabled && claim_applies(event.outcome, seen.len());
        let mut asked = Vec::new();
        if run_claim {
            asked.push(Check::Claim);
        }
        let task_opt = if !asked.is_empty() || !rule_ids.is_empty() {
            self.prepare("turn_end", None, asked, rule_ids, || {
                let calls = call_list(seen.iter().copied());
                let final_message = event.turn.final_text();
                build_state(&StateParts {
                    prompt: event.turn.prompt(),
                    output: seen.last().map(|call| call.output),
                    calls: &calls,
                    final_message: final_message.as_deref(),
                    ..StateParts::default()
                })
            })
        } else {
            None
        };
        let pending_warns = self.session.take_pending_warns();
        if !completed {
            // On abort or provider error, return immediately without
            // stalling on background warn tasks.
            for handle in pending_warns {
                handle.abort();
            }
            let mut reply = HostReply::default();
            reply.findings = self.session.take_findings();
            return Box::pin(std::future::ready(reply));
        }
        let needs_warn = task_opt.as_ref().is_some_and(|task| {
            task.asked
                .iter()
                .any(|&check| self.shared.checks.get(check) >= CheckAction::Warn)
                || task
                    .rule_ids
                    .iter()
                    .any(|&i| self.shared.rules[i].action != RuleAction::Log)
        });
        if !needs_warn {
            if let Some(task) = task_opt {
                if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                    #[cfg(test)]
                    self.shared.spawned.fetch_add(1, Ordering::Relaxed);
                    runtime.spawn(async move {
                        let _ = task.execute(BTreeMap::new()).await;
                    });
                } else {
                    log::warn!(
                        "judgments: no async runtime at turn_end; the checks send no request"
                    );
                }
            }
            if pending_warns.is_empty() {
                let mut reply = HostReply::default();
                reply.findings = self.session.take_findings();
                return Box::pin(std::future::ready(reply));
            }
            let session = Arc::clone(&self.session);
            return Box::pin(async move {
                for handle in pending_warns {
                    let _ = handle.await;
                }
                let mut reply = HostReply::default();
                reply.findings = session.take_findings();
                reply
            });
        }
        let session = Arc::clone(&self.session);
        Box::pin(async move {
            let mut findings = Vec::new();
            if let Some(task) = task_opt {
                findings.extend(task.run_turn_end().await);
            }
            for handle in pending_warns {
                let _ = handle.await;
            }
            let mut all = session.take_findings();
            all.extend(findings);
            let mut reply = HostReply::default();
            reply.findings = all;
            reply
        })
    }

    fn take_findings(&self) -> Vec<Finding> {
        self.session.take_findings()
    }
}

/// Indices of rules matching `event` and optional `tool`.
fn matching_rules(rules: &[Rule], event: RuleEvent, tool: Option<&str>) -> Vec<usize> {
    rules
        .iter()
        .enumerate()
        .filter(|(_, rule)| {
            rule.event == event
                && match (&rule.tool, tool) {
                    (None, _) => true,
                    (Some(expected), Some(actual)) => expected == actual,
                    (Some(_), None) => false,
                }
        })
        .map(|(index, _)| index)
        .collect()
}

/// Question id for rule `index`.
pub(crate) fn rule_question_id(index: usize) -> String {
    format!("rule.{index}")
}

/// Extract the probability of yes for `id` from `answers`.
fn yes_prob(answers: &BTreeMap<String, Answer>, id: &str) -> Option<f64> {
    match answers.get(id) {
        Some(Answer::YesNo { yes, .. }) => Some(*yes),
        _ => None,
    }
}

fn approval_concern(destructive: f64, in_scope: f64) -> String {
    format!("approval concern: destructive {destructive:.2}, in scope {in_scope:.2}")
}

fn approval_destructive_concern(destructive: f64) -> String {
    format!("destructive call (probability {destructive:.2})")
}

/// Whether the approval check runs: a write or execute call that the
/// policy asks about in `ask` mode, or allows in `auto` mode. In
/// `read-only` mode the policy refuses such a call first.
fn approval_applies(access: ToolAccess, mode: ApprovalMode, decision: ApprovalDecision) -> bool {
    matches!(access, ToolAccess::Write | ToolAccess::Execute)
        && matches!(
            (mode, decision),
            (ApprovalMode::Ask, ApprovalDecision::AskUser)
                | (ApprovalMode::Auto, ApprovalDecision::Allow)
        )
}

/// Whether the outcome check runs: an execute call that returned
/// without its error flag.
fn outcome_applies(access: Option<ToolAccess>, is_error: bool) -> bool {
    access == Some(ToolAccess::Execute) && !is_error
}

/// Whether the claim check runs: a completed turn with at least one
/// call.
fn claim_applies(outcome: TurnEndOutcome, calls: usize) -> bool {
    outcome == TurnEndOutcome::Outcome(TurnOutcome::Completed) && calls > 0
}

/// The name and error flag of each call, for the state.
fn call_list<'a>(calls: impl Iterator<Item = CallSeen<'a>>) -> Vec<(&'a str, bool)> {
    calls.map(|call| (call.name, call.is_error)).collect()
}

/// The questions of the checks in `asked` and rules in `rule_ids`, by id.
fn questions(asked: &[Check], rule_ids: &[usize], rules: &[Rule]) -> BTreeMap<String, Question> {
    let mut questions = BTreeMap::new();
    for check in asked {
        let mut add = |id: &str, text: &str| {
            questions.insert(id.to_string(), Question::yes_no(text));
        };
        match check {
            Check::Approval => {
                add(APPROVAL_DESTRUCTIVE, APPROVAL_DESTRUCTIVE_TEXT);
                add(APPROVAL_IN_SCOPE, APPROVAL_IN_SCOPE_TEXT);
            }
            Check::Outcome => add(OUTCOME, OUTCOME_TEXT),
            Check::Progress => add(PROGRESS, PROGRESS_TEXT),
            Check::Claim => add(CLAIM, CLAIM_TEXT),
        }
    }
    for &index in rule_ids {
        if let Some(rule) = rules.get(index) {
            questions.insert(rule_question_id(index), Question::yes_no(&rule.ask));
        }
    }
    questions
}

/// One request, as its task runs it.
struct Ask {
    shared: Arc<Shared>,
    session: Arc<SessionState>,
    writer: SessionWriter,
    event: &'static str,
    /// The session's turn at the event.
    turn: u64,
    call_id: Option<String>,
    tool: Option<String>,
    asked: Vec<Check>,
    rule_ids: Vec<usize>,
    state: String,
    /// When the event spawned the task: the permit wait counts from
    /// here.
    spawned: Instant,
    _guard: FlightGuard,
}

/// The state's length and its SHA-256, as the record names them.
struct StateFacts {
    bytes: u64,
    sha256: String,
}

/// A completed transport attempt before writing its `judgment` record.
struct Attempt {
    facts: StateFacts,
    at: SystemTime,
    round_trip: Duration,
    outcome: Result<JudgeReply, String>,
}

impl Ask {
    /// Run the judge request, returning `Some(Attempt)` if a record
    /// should be written (success, timeout, error, or busy within cap),
    /// or `None` if skipped (build failure, budget cap, or busy past cap).
    async fn send(&mut self) -> Option<Attempt> {
        if self.session.build_failed.load(Ordering::Acquire) == self.turn {
            return None;
        }
        let shared = Arc::clone(&self.shared);
        let name = shared.judge.instance.name.as_str();
        let judge = match shared.judge.get() {
            Ok(judge) => judge,
            Err(error) => {
                let session = &self.session;
                if session.build_failed.swap(self.turn, Ordering::AcqRel) != self.turn {
                    log::warn!("judgments: judge `{name}` does not build: {error}");
                }
                if !session.build_noticed.swap(true, Ordering::AcqRel) {
                    shared.notice(format!(
                        "judgments: judge `{name}` cannot run: {error}. The checks send no \
                         request until it can."
                    ));
                }
                return None;
            }
        };
        let deadline = tokio::time::Instant::from_std(self.spawned + shared.timeout);
        let permit = tokio::time::timeout_at(deadline, Arc::clone(&shared.permits).acquire_owned())
            .await
            .ok()
            .and_then(Result::ok);
        let state = std::mem::take(&mut self.state);
        let Some(_permit) = permit else {
            if !self.session.budget.spent() {
                let count = self.session.stats.busy.fetch_add(1, Ordering::Relaxed) + 1;
                let cap = BUSY_RECORD_CAP.min(shared.max_calls.max(1));
                if count <= cap {
                    let facts = StateFacts {
                        bytes: state.len() as u64,
                        sha256: hex(&Sha256::digest(state.as_bytes())),
                    };
                    return Some(Attempt {
                        facts,
                        at: SystemTime::now(),
                        round_trip: Duration::ZERO,
                        outcome: Err(BUSY.to_string()),
                    });
                } else if !self.session.busy_noticed.swap(true, Ordering::AcqRel) {
                    shared.notice(format!(
                        "judgments: judge `{name}` is busy; suppressing further busy records \
                         in this session"
                    ));
                }
            }
            return None;
        };
        if !self.session.budget.take() {
            self.session.budget.notice(&shared);
            return None;
        }
        let facts = StateFacts {
            bytes: state.len() as u64,
            sha256: hex(&Sha256::digest(state.as_bytes())),
        };
        let at = SystemTime::now();
        let sent = Instant::now();
        let mut request = JudgeRequest::new(state, shared.timeout);
        request.questions = questions(&self.asked, &self.rule_ids, &shared.rules);
        let outcome = match tokio::time::timeout(shared.timeout, judge.ask(request)).await {
            Ok(Ok(reply)) => Ok(reply),
            Ok(Err(JudgeError::Timeout(_))) | Err(_) => {
                self.session.stats.timeouts.fetch_add(1, Ordering::Relaxed);
                self.session.note_transport_failure(&shared, TIMED_OUT);
                Err(TIMED_OUT.to_string())
            }
            Ok(Err(error)) => {
                self.session.stats.errors.fetch_add(1, Ordering::Relaxed);
                let text = capped(&error.to_string(), ERROR_CAP).to_string();
                log::warn!("judgments: judge `{name}` failed: {text}");
                self.session.note_transport_failure(&shared, &text);
                Err(text)
            }
        };
        Some(Attempt {
            facts,
            at,
            round_trip: sent.elapsed(),
            outcome,
        })
    }

    /// Log-only execution: send and write the record with `fired`.
    async fn execute(mut self, fired: BTreeMap<String, String>) {
        if let Some(attempt) = self.send().await {
            self.write(attempt, fired);
        }
    }

    /// Blocking `tool_before` evaluation: rules first, then approval check.
    async fn run_tool_before(mut self, mode: ApprovalMode, policy: ApprovalDecision) -> ToolGate {
        let Some(attempt) = self.send().await else {
            return ToolGate::keep(policy);
        };
        let mut fired: BTreeMap<String, String> = BTreeMap::new();
        let mut gate = ToolGate::keep(policy);
        if let Ok(reply) = &attempt.outcome {
            let shared = &self.shared;
            let mut denied = false;
            // Stage 1: plain-language rules.
            for &index in &self.rule_ids {
                let rule = &shared.rules[index];
                let qid = rule_question_id(index);
                if let Some(yes) = yes_prob(&reply.answers, &qid)
                    && yes >= rule.threshold
                {
                    match rule.action {
                        RuleAction::Deny => {
                            fired.insert(qid.clone(), "deny".to_string());
                            self.session.stats.record_fired("deny");
                            gate.findings.push(Finding::new(
                                self.call_id.clone(),
                                qid,
                                "deny",
                                &rule.reason,
                            ));
                            if !denied {
                                denied = true;
                                gate.decision = ApprovalDecision::Deny;
                                gate.reason = Some(rule.reason.clone());
                                gate.answer = Some("judge".to_string());
                            }
                        }
                        RuleAction::Warn => {
                            fired.insert(qid.clone(), "warn".to_string());
                            self.session.stats.record_fired("warn");
                            gate.findings.push(Finding::new(
                                self.call_id.clone(),
                                qid,
                                "warn",
                                &rule.reason,
                            ));
                        }
                        RuleAction::Log | RuleAction::Note => {}
                    }
                }
            }
            // Stage 2: built-in approval check (never overrides a rule denial).
            if !denied
                && self.asked.contains(&Check::Approval)
                && let (Some(p_d), Some(p_s)) = (
                    yes_prob(&reply.answers, APPROVAL_DESTRUCTIVE),
                    yes_prob(&reply.answers, APPROVAL_IN_SCOPE),
                )
            {
                let action = shared.checks.approval;
                match (mode, policy, action) {
                    (ApprovalMode::Ask, ApprovalDecision::AskUser, CheckAction::Act) => {
                        if approval_ask_allows(p_d, p_s) {
                            fired.insert("approval".to_string(), "allow".to_string());
                            self.session.stats.record_fired("allow");
                            gate.decision = ApprovalDecision::Allow;
                            gate.answer = Some("judge".to_string());
                            gate.findings.push(Finding::new(
                                self.call_id.clone(),
                                "approval",
                                "allow",
                                format!(
                                    "approved by judge (destructive {p_d:.2}, in scope {p_s:.2})"
                                ),
                            ));
                        } else {
                            fired.insert("approval".to_string(), "warn".to_string());
                            self.session.stats.record_fired("warn");
                            gate.findings.push(Finding::new(
                                self.call_id.clone(),
                                "approval",
                                "warn",
                                approval_concern(p_d, p_s),
                            ));
                        }
                    }
                    (ApprovalMode::Ask, ApprovalDecision::AskUser, CheckAction::Warn)
                        if !approval_ask_allows(p_d, p_s) =>
                    {
                        fired.insert("approval".to_string(), "warn".to_string());
                        self.session.stats.record_fired("warn");
                        gate.findings.push(Finding::new(
                            self.call_id.clone(),
                            "approval",
                            "warn",
                            approval_concern(p_d, p_s),
                        ));
                    }
                    (ApprovalMode::Auto, ApprovalDecision::Allow, CheckAction::Act)
                        if approval_auto_asks(p_d) =>
                    {
                        let concern = approval_destructive_concern(p_d);
                        fired.insert("approval".to_string(), "ask".to_string());
                        self.session.stats.record_fired("ask");
                        gate.decision = ApprovalDecision::AskUser;
                        gate.reason = Some(concern.clone());
                        gate.findings.push(Finding::new(
                            self.call_id.clone(),
                            "approval",
                            "ask",
                            concern,
                        ));
                    }
                    (ApprovalMode::Auto, ApprovalDecision::Allow, CheckAction::Warn)
                        if approval_auto_asks(p_d) =>
                    {
                        fired.insert("approval".to_string(), "warn".to_string());
                        self.session.stats.record_fired("warn");
                        gate.findings.push(Finding::new(
                            self.call_id.clone(),
                            "approval",
                            "warn",
                            approval_destructive_concern(p_d),
                        ));
                    }
                    _ => {}
                }
            }
        }
        self.write(attempt, fired);
        gate
    }

    /// Evaluate `tool_after` checks (`outcome`, `progress`) and rules.
    async fn run_tool_after(mut self) -> HostReply {
        let Some(attempt) = self.send().await else {
            return HostReply::default();
        };
        let mut fired: BTreeMap<String, String> = BTreeMap::new();
        let mut reply_out = HostReply::default();
        let mut notes: Vec<String> = Vec::new();
        if let Ok(reply) = &attempt.outcome {
            let shared = &self.shared;
            for &check in &self.asked {
                let (qid, label, text_fn): (&str, &str, fn(f64) -> String) = match check {
                    Check::Outcome => (OUTCOME, "outcome", |p| {
                        format!("command may have failed despite exit 0 (probability {p:.2})")
                    }),
                    Check::Progress => (PROGRESS, "progress", |p| {
                        format!(
                            "call may repeat an earlier failed action without material change \
                             (probability {p:.2})"
                        )
                    }),
                    _ => continue,
                };
                let action = shared.checks.get(check);
                if action <= CheckAction::Log {
                    continue;
                }
                if let Some(yes) = yes_prob(&reply.answers, qid)
                    && check_fires(yes)
                {
                    let text = text_fn(yes);
                    let action_str = action.as_str();
                    fired.insert(label.to_string(), action_str.to_string());
                    self.session.stats.record_fired(action_str);
                    reply_out.findings.push(Finding::new(
                        self.call_id.clone(),
                        label,
                        action_str,
                        &text,
                    ));
                    if action == CheckAction::Note {
                        notes.push(format!("[judgment: {label}] {text}"));
                    }
                }
            }
            for &index in &self.rule_ids {
                let rule = &shared.rules[index];
                if rule.action == RuleAction::Log {
                    continue;
                }
                let qid = rule_question_id(index);
                if let Some(yes) = yes_prob(&reply.answers, &qid)
                    && yes >= rule.threshold
                {
                    let action_str = rule.action.as_str();
                    fired.insert(qid.clone(), action_str.to_string());
                    self.session.stats.record_fired(action_str);
                    reply_out.findings.push(Finding::new(
                        self.call_id.clone(),
                        qid.clone(),
                        action_str,
                        &rule.reason,
                    ));
                    if rule.action == RuleAction::Note {
                        notes.push(format!("[judgment: {qid}] {}", rule.reason));
                    }
                }
            }
        }
        if !notes.is_empty() {
            reply_out.note = Some(notes.join("\n"));
        }
        self.write(attempt, fired);
        reply_out
    }

    /// Evaluate `turn_end` checks (`claim`) and rules.
    async fn run_turn_end(mut self) -> Vec<Finding> {
        let Some(attempt) = self.send().await else {
            return Vec::new();
        };
        let mut fired: BTreeMap<String, String> = BTreeMap::new();
        let mut findings = Vec::new();
        if let Ok(reply) = &attempt.outcome {
            let shared = &self.shared;
            if self.asked.contains(&Check::Claim)
                && shared.checks.claim >= CheckAction::Warn
                && let Some(yes) = yes_prob(&reply.answers, CLAIM)
                && check_fires(yes)
            {
                let text = format!(
                    "final message may claim a result not shown by tool evidence \
                     (probability {yes:.2})"
                );
                fired.insert("claim".to_string(), "warn".to_string());
                self.session.stats.record_fired("warn");
                findings.push(Finding::new(None, "claim", "warn", text));
            }
            for &index in &self.rule_ids {
                let rule = &shared.rules[index];
                if rule.action == RuleAction::Log {
                    continue;
                }
                let qid = rule_question_id(index);
                if let Some(yes) = yes_prob(&reply.answers, &qid)
                    && yes >= rule.threshold
                {
                    let action_str = rule.action.as_str();
                    fired.insert(qid.clone(), action_str.to_string());
                    self.session.stats.record_fired(action_str);
                    findings.push(Finding::new(None, qid, action_str, &rule.reason));
                }
            }
        }
        self.write(attempt, fired);
        findings
    }

    /// Write the record of this request: sent at `at` and answered
    /// after `round_trip`, or failed with an error. A request that
    /// never went out has a round trip of 0.
    fn write(&self, attempt: Attempt, fired: BTreeMap<String, String>) {
        let shared = &self.shared;
        let mut record = JudgmentRecord::new(self.event, shared.judge.instance.name.as_str());
        record.call_id.clone_from(&self.call_id);
        record.tool.clone_from(&self.tool);
        record.checks = self
            .asked
            .iter()
            .map(|&check| {
                let action = shared.checks.get(check);
                (check.as_str().to_string(), action.as_str().to_string())
            })
            .collect();
        record.rules = self
            .rule_ids
            .iter()
            .filter_map(|&index| {
                let rule = shared.rules.get(index)?;
                Some((
                    rule_question_id(index),
                    JudgmentRuleRecord::new(
                        rule.action.as_str(),
                        rule.source_name(),
                        rule.threshold,
                    ),
                ))
            })
            .collect();
        record.fired = fired;
        record.at = humantime::format_rfc3339_millis(attempt.at).to_string();
        record.duration_ms = u64::try_from(attempt.round_trip.as_millis()).unwrap_or(u64::MAX);
        record.state_bytes = attempt.facts.bytes;
        record.state_sha256.clone_from(&attempt.facts.sha256);
        match attempt.outcome {
            Ok(reply) => {
                record.answers = reply
                    .answers
                    .into_iter()
                    .map(|(id, answer)| (id, serde_json::to_value(answer).unwrap_or(Value::Null)))
                    .collect();
                record.model = Some(reply.model);
                record.usage = Some(reply.usage);
            }
            Err(error) => record.error = Some(error),
        }
        log::info!(
            "judgment {} call_id={} duration_ms={} state_bytes={} answers={} fired={} error={}",
            record.event,
            record.call_id.as_deref().unwrap_or("-"),
            record.duration_ms,
            record.state_bytes,
            record.answers.len(),
            record.fired.len(),
            record.error.as_deref().unwrap_or("-")
        );
        if let Err(error) = self.writer.append_judgment(&record) {
            log::warn!("judgments: the session file took no judgment record: {error}");
        }
    }
}

/// Lowercase hex of `bytes`.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// `text` cut at `max` bytes, at a char boundary.
fn capped(text: &str, max: usize) -> &str {
    &text[..text.floor_char_boundary(max)]
}

/// What one request's state holds. The builder caps each part and the
/// whole; see [`build_state`].
#[derive(Default)]
struct StateParts<'a> {
    /// The user prompt of the current turn.
    prompt: &'a str,
    tool: Option<&'a str>,
    input: Option<&'a Value>,
    /// For `progress`: the input of the latest earlier failed call of
    /// the same tool, as the host keeps it.
    earlier_failed_input: Option<&'a JsonHead>,
    /// This call's output; for `claim`, the turn's last call's output.
    output: Option<&'a str>,
    /// The turn's answered calls, oldest first: the name and the error
    /// flag. The builder keeps the last ten.
    calls: &'a [(&'a str, bool)],
    /// For `claim`: the turn's final message.
    final_message: Option<&'a str>,
}

/// The state text: labeled sections in a fixed order, each `label:`
/// on its own line, then the content, with a blank line between
/// sections. Caps: the prompt 4 KiB and the inputs 4 KiB each, head
/// kept; the output 8 KiB, tail kept; the last ten calls as name and
/// outcome; the final message 4 KiB, head kept; at most 16 KiB in
/// total. Over the total, the builder cuts the output tail first, then
/// the calls (oldest first), the earlier failed input, the input, and
/// the prompt. Each cut leaves a `[cut: N bytes]` marker inside its
/// cap and respects UTF-8 boundaries.
fn build_state(parts: &StateParts<'_>) -> String {
    let mut sections = Vec::with_capacity(7);
    sections.push(Section::new(
        PROMPT_LABEL,
        Body::head(parts.prompt),
        PROMPT_CAP,
    ));
    if let Some(tool) = parts.tool {
        sections.push(Section::new(TOOL_LABEL, Body::head(tool), TOOL_NAME_CAP));
    }
    let input = parts.input.map(|input| JsonHead::new(input, INPUT_CAP));
    if let Some(input) = &input {
        sections.push(Section::new(INPUT_LABEL, input.body(), INPUT_CAP));
    }
    if let Some(input) = parts.earlier_failed_input {
        sections.push(Section::new(
            EARLIER_FAILED_INPUT_LABEL,
            input.body(),
            INPUT_CAP,
        ));
    }
    if let Some(output) = parts.output {
        sections.push(Section::new(
            OUTPUT_TAIL_LABEL,
            Body::Tail(output),
            OUTPUT_CAP,
        ));
    }
    if !parts.calls.is_empty() {
        let shown = &parts.calls[parts.calls.len().saturating_sub(CALLS_SHOWN)..];
        let lines = shown
            .iter()
            .map(|(name, is_error)| {
                let outcome = if *is_error { "error" } else { "ok" };
                let name = cut_head_with(name, Length::Exact(name.len()), TOOL_NAME_CAP, " ");
                format!("{name} {outcome}")
            })
            .collect();
        sections.push(Section::new(CALLS_LABEL, Body::Lines(lines), usize::MAX));
    }
    if let Some(text) = parts.final_message {
        sections.push(Section::new(
            FINAL_MESSAGE_LABEL,
            Body::head(text),
            FINAL_CAP,
        ));
    }

    for label in CUT_ORDER {
        let size = state_size(&sections);
        if size <= STATE_CAP {
            break;
        }
        if let Some(section) = sections.iter_mut().find(|section| section.label == label) {
            let cap = section.text.len().saturating_sub(size - STATE_CAP);
            section.text = section.body.render(cap);
        }
    }

    let mut state = String::with_capacity(state_size(&sections));
    for (index, section) in sections.iter().enumerate() {
        if index > 0 {
            state.push('\n');
        }
        state.push_str(section.label);
        state.push_str(":\n");
        state.push_str(&section.text);
        state.push('\n');
    }
    state
}

/// The rendered size of `sections`, as [`build_state`] joins them.
fn state_size(sections: &[Section<'_>]) -> usize {
    let own: usize = sections
        .iter()
        .map(|section| section.label.len() + 2 + section.text.len() + 1)
        .sum();
    own + sections.len().saturating_sub(1)
}

/// One state section: its label, its source, and its text as cut.
struct Section<'a> {
    label: &'static str,
    body: Body<'a>,
    text: String,
}

impl<'a> Section<'a> {
    fn new(label: &'static str, body: Body<'a>, cap: usize) -> Self {
        let text = body.render(cap);
        Self { label, body, text }
    }
}

/// The length of a text the state keeps the head of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Length {
    Exact(usize),
    /// Longer than this: the writer stopped before the end.
    MoreThan(usize),
}

/// The source of a section's text.
enum Body<'a> {
    /// Kept from the head. `text` is the whole content, or its first
    /// bytes when the content was never whole in memory.
    Head { text: Cow<'a, str>, length: Length },
    /// Kept from the tail.
    Tail(&'a str),
    /// One line per call, oldest first; a cut drops the oldest lines.
    Lines(Vec<String>),
}

impl<'a> Body<'a> {
    fn head(text: &'a str) -> Self {
        Body::Head {
            text: Cow::Borrowed(text),
            length: Length::Exact(text.len()),
        }
    }

    fn render(&self, cap: usize) -> String {
        match self {
            Body::Head { text, length } => cut_head(text, *length, cap).into_owned(),
            Body::Tail(text) => cut_tail(text, cap).into_owned(),
            Body::Lines(lines) => cut_lines(lines, cap),
        }
    }
}

/// The head of a JSON value's text: the first bytes the state can
/// keep, and the length.
pub(crate) struct JsonHead {
    text: String,
    length: Length,
}

impl JsonHead {
    /// The JSON text of `value`, of which at most `keep` bytes and one
    /// more stay. The writer stops past them, so a large value costs
    /// no more than its head.
    fn new(value: &Value, keep: usize) -> Self {
        let mut head = HeadBuf {
            bytes: Vec::new(),
            limit: keep + 1,
        };
        let whole = serde_json::to_writer(&mut head, value).is_ok();
        let text = match String::from_utf8(head.bytes) {
            Ok(text) => text,
            // The stop can split the last char.
            Err(error) => {
                let valid = error.utf8_error().valid_up_to();
                let mut bytes = error.into_bytes();
                bytes.truncate(valid);
                String::from_utf8(bytes).unwrap_or_default()
            }
        };
        let length = if whole {
            Length::Exact(text.len())
        } else {
            Length::MoreThan(text.len())
        };
        Self { text, length }
    }

    fn body(&self) -> Body<'_> {
        Body::Head {
            text: Cow::Borrowed(&self.text),
            length: self.length,
        }
    }
}

/// A writer that keeps the first `limit` bytes, then stops the writer
/// with an error.
struct HeadBuf {
    bytes: Vec<u8>,
    limit: usize,
}

impl io::Write for HeadBuf {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let room = self.limit.saturating_sub(self.bytes.len());
        if data.len() > room {
            self.bytes.extend_from_slice(&data[..room]);
            return Err(io::Error::other("the head is full"));
        }
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The marker for a cut of `bytes` bytes.
fn marker(bytes: usize) -> String {
    format!("[cut: {bytes} bytes]")
}

/// The marker for a cut of `bytes` bytes or more, when the end of the
/// text is unknown.
fn marker_at_least(bytes: usize) -> String {
    format!("[cut: at least {bytes} bytes]")
}

/// The head of a text of `length`, of which `text` holds at least the
/// first `cap` bytes, in at most `cap` bytes with the marker on its
/// own line. When not even the marker fits, the marker alone: the
/// whole text is cut, and the builder's total cut goes on.
fn cut_head(text: &str, length: Length, cap: usize) -> Cow<'_, str> {
    cut_head_with(text, length, cap, "\n")
}

/// [`cut_head`] with `separator` between the head and the marker.
fn cut_head_with<'t>(text: &'t str, length: Length, cap: usize, separator: &str) -> Cow<'t, str> {
    let (total, mark): (usize, fn(usize) -> String) = match length {
        Length::Exact(total) if total <= cap => return Cow::Borrowed(text),
        Length::Exact(total) => (total, marker),
        Length::MoreThan(seen) => (seen, marker_at_least),
    };
    // The marker for the largest possible cut bounds every marker.
    let room = cap.saturating_sub(mark(total).len() + separator.len());
    if room == 0 {
        return Cow::Owned(mark(total));
    }
    let keep = text.floor_char_boundary(room.min(text.len()));
    Cow::Owned(format!(
        "{}{separator}{}",
        &text[..keep],
        mark(total - keep)
    ))
}

/// The tail of `text` in at most `cap` bytes with the marker; the
/// marker alone when not even the marker fits.
fn cut_tail(text: &str, cap: usize) -> Cow<'_, str> {
    if text.len() <= cap {
        return Cow::Borrowed(text);
    }
    let room = cap.saturating_sub(marker(text.len()).len() + 1);
    if room == 0 {
        return Cow::Owned(marker(text.len()));
    }
    let start = text.ceil_char_boundary(text.len() - room);
    Cow::Owned(format!("{}\n{}", marker(start), &text[start..]))
}

/// The newest `lines` in at most `cap` bytes, joined by newlines, with
/// a marker line for the oldest lines cut; the marker alone when not
/// even the marker fits.
fn cut_lines(lines: &[String], cap: usize) -> String {
    let total = lines.iter().map(String::len).sum::<usize>() + lines.len().saturating_sub(1);
    if total <= cap {
        return lines.join("\n");
    }
    let room = cap.saturating_sub(marker(total).len() + 1);
    let mut kept = 0;
    let mut size = 0;
    for line in lines.iter().rev() {
        let next = size + line.len() + usize::from(kept > 0);
        if next > room {
            break;
        }
        size = next;
        kept += 1;
    }
    let cut = total - size - usize::from(kept > 0);
    let mut out = marker(cut);
    for line in &lines[lines.len() - kept..] {
        out.push('\n');
        out.push_str(line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Resume;
    use serde_json::json;

    /// The labels a state can hold, in their order.
    const LABELS: [&str; 7] = [
        PROMPT_LABEL,
        TOOL_LABEL,
        INPUT_LABEL,
        EARLIER_FAILED_INPUT_LABEL,
        OUTPUT_TAIL_LABEL,
        CALLS_LABEL,
        FINAL_MESSAGE_LABEL,
    ];

    /// The content of section `label` in `state`; `None` when absent.
    fn section<'s>(state: &'s str, label: &str) -> Option<&'s str> {
        let head = format!("{label}:\n");
        let start = if state.starts_with(&head) {
            head.len()
        } else {
            state.find(&format!("\n\n{head}"))? + 2 + head.len()
        };
        let rest = &state[start..];
        let end = LABELS
            .iter()
            .filter_map(|next| rest.find(&format!("\n\n{next}:\n")))
            .min()
            .unwrap_or(rest.len() - 1);
        Some(&rest[..end])
    }

    /// The byte count of the cut marker in `text`; `None` when uncut.
    fn cut_bytes(text: &str) -> Option<usize> {
        let mut start = text.find("[cut: ")? + "[cut: ".len();
        if text[start..].starts_with("at least ") {
            start += "at least ".len();
        }
        let end = start + text[start..].find(" bytes]")?;
        text[start..end].parse().ok()
    }

    /// `n` bytes of text whose chars are 1, 2, 3, and 4 bytes wide, so
    /// any cut can fall inside a char.
    fn wide(n: usize) -> String {
        let mut text = String::new();
        for c in ['a', 'é', '€', '😀'].iter().cycle() {
            if text.len() + c.len_utf8() > n {
                break;
            }
            text.push(*c);
        }
        text
    }

    #[test]
    fn a_small_state_holds_every_part_whole_in_order() {
        let input = json!({"command": "ls"});
        let earlier = JsonHead::new(&json!({"command": "lss"}), INPUT_CAP);
        let calls = [("bash", true), ("read", false)];
        let state = build_state(&StateParts {
            prompt: "list the files",
            tool: Some("bash"),
            input: Some(&input),
            earlier_failed_input: Some(&earlier),
            output: Some("a.txt\nb.txt"),
            calls: &calls,
            final_message: Some("two files"),
        });
        assert_eq!(section(&state, PROMPT_LABEL), Some("list the files"));
        assert_eq!(section(&state, TOOL_LABEL), Some("bash"));
        assert_eq!(section(&state, INPUT_LABEL), Some(r#"{"command":"ls"}"#));
        assert_eq!(
            section(&state, EARLIER_FAILED_INPUT_LABEL),
            Some(r#"{"command":"lss"}"#)
        );
        assert_eq!(section(&state, OUTPUT_TAIL_LABEL), Some("a.txt\nb.txt"));
        assert_eq!(section(&state, CALLS_LABEL), Some("bash error\nread ok"));
        assert_eq!(section(&state, FINAL_MESSAGE_LABEL), Some("two files"));
        let positions: Vec<usize> = LABELS
            .iter()
            .map(|label| state.find(&format!("{label}:\n")).unwrap())
            .collect();
        assert!(positions.is_sorted(), "{state}");
        assert!(!state.contains("[cut:"));
        // A part the event does not have has no section.
        let state = build_state(&StateParts {
            prompt: "hi",
            ..StateParts::default()
        });
        assert_eq!(state, "prompt:\nhi\n");
    }

    #[test]
    fn each_part_keeps_its_cap_at_a_char_boundary() {
        let prompt = wide(PROMPT_CAP + 1001);
        let input = json!({"content": wide(INPUT_CAP * 2)});
        let input_head = JsonHead::new(&input, INPUT_CAP);
        let output = wide(OUTPUT_CAP + 999);
        let final_message = wide(FINAL_CAP + 7);
        let tool = wide(TOOL_NAME_CAP + 50);
        let names: Vec<String> = (0..15).map(|i| format!("tool{i}")).collect();
        let calls: Vec<(&str, bool)> = names.iter().map(|name| (name.as_str(), false)).collect();
        for (parts, label, original, head, cap) in [
            (
                StateParts {
                    prompt: &prompt,
                    ..StateParts::default()
                },
                PROMPT_LABEL,
                prompt.clone(),
                true,
                PROMPT_CAP,
            ),
            (
                StateParts {
                    input: Some(&input),
                    ..StateParts::default()
                },
                INPUT_LABEL,
                input.to_string(),
                true,
                INPUT_CAP,
            ),
            (
                StateParts {
                    earlier_failed_input: Some(&input_head),
                    ..StateParts::default()
                },
                EARLIER_FAILED_INPUT_LABEL,
                input.to_string(),
                true,
                INPUT_CAP,
            ),
            (
                StateParts {
                    output: Some(&output),
                    ..StateParts::default()
                },
                OUTPUT_TAIL_LABEL,
                output.clone(),
                false,
                OUTPUT_CAP,
            ),
            (
                StateParts {
                    final_message: Some(&final_message),
                    ..StateParts::default()
                },
                FINAL_MESSAGE_LABEL,
                final_message.clone(),
                true,
                FINAL_CAP,
            ),
            (
                StateParts {
                    tool: Some(&tool),
                    ..StateParts::default()
                },
                TOOL_LABEL,
                tool.clone(),
                true,
                TOOL_NAME_CAP,
            ),
        ] {
            let state = build_state(&parts);
            let text = section(&state, label).unwrap();
            assert!(text.len() <= cap, "{label}: {} > {cap}", text.len());
            let cut = cut_bytes(text).unwrap_or_else(|| panic!("{label}: no marker"));
            let kept = if head {
                let kept = &text[..text.rfind("\n[cut:").unwrap()];
                assert!(original.starts_with(kept), "{label}");
                kept
            } else {
                let kept = &text[text.find("]\n").unwrap() + 2..];
                assert!(original.ends_with(kept), "{label}");
                kept
            };
            // A large input stops its writer: the marker gives a lower
            // bound. Every other part knows its length.
            if text.contains("[cut: at least ") {
                assert!(matches!(label, INPUT_LABEL | EARLIER_FAILED_INPUT_LABEL));
                assert!(kept.len() + cut <= original.len(), "{label}");
            } else {
                assert_eq!(kept.len() + cut, original.len(), "{label}");
            }
            // The kept text fills the cap, short of a char and the
            // marker's spare digits.
            assert!(cap - text.len() < 8, "{label}: {} of {cap}", text.len());
        }
        // The last ten calls, oldest first.
        let state = build_state(&StateParts {
            calls: &calls,
            ..StateParts::default()
        });
        let lines: Vec<&str> = section(&state, CALLS_LABEL).unwrap().lines().collect();
        assert_eq!(lines.len(), CALLS_SHOWN);
        assert_eq!(lines[0], "tool5 ok");
        assert_eq!(lines[9], "tool14 ok");
    }

    #[test]
    fn over_the_total_the_output_tail_goes_first() {
        let prompt = wide(PROMPT_CAP);
        let input = json!({"content": wide(INPUT_CAP - 20)});
        let earlier_value = json!({"content": wide(INPUT_CAP - 20)});
        let earlier = JsonHead::new(&earlier_value, INPUT_CAP);
        let output = wide(OUTPUT_CAP * 3);
        let names: Vec<String> = (0..12).map(|i| format!("{i}{}", "n".repeat(200))).collect();
        let calls: Vec<(&str, bool)> = names.iter().map(|name| (name.as_str(), true)).collect();
        let parts = StateParts {
            prompt: &prompt,
            tool: Some("bash"),
            input: Some(&input),
            earlier_failed_input: Some(&earlier),
            output: Some(&output),
            calls: &calls,
            final_message: None,
        };
        let state = build_state(&parts);
        assert!(state.len() <= STATE_CAP, "{}", state.len());
        // Only the output tail is cut past its own cap.
        assert_eq!(section(&state, PROMPT_LABEL), Some(prompt.as_str()));
        assert_eq!(
            section(&state, INPUT_LABEL),
            Some(input.to_string().as_str())
        );
        assert_eq!(
            section(&state, EARLIER_FAILED_INPUT_LABEL),
            Some(earlier_value.to_string().as_str())
        );
        assert_eq!(
            section(&state, CALLS_LABEL).unwrap().lines().count(),
            CALLS_SHOWN
        );
        let tail = section(&state, OUTPUT_TAIL_LABEL).unwrap();
        assert!(tail.len() < OUTPUT_CAP / 2, "{}", tail.len());
        assert!(output.ends_with(&tail[tail.find("]\n").unwrap() + 2..]));
        // Each long name keeps its own cap on its line.
        assert!(
            section(&state, CALLS_LABEL)
                .unwrap()
                .lines()
                .all(|line| line.len() <= TOOL_NAME_CAP + " error".len())
        );

        // With a final message too, the calls go next, oldest first.
        let final_message = wide(FINAL_CAP);
        let small = json!({"content": wide(INPUT_CAP - 700)});
        let small_head = JsonHead::new(&small, INPUT_CAP);
        let state = build_state(&StateParts {
            input: Some(&small),
            earlier_failed_input: Some(&small_head),
            final_message: Some(&final_message),
            ..parts
        });
        assert!(state.len() <= STATE_CAP, "{}", state.len());
        let tail = section(&state, OUTPUT_TAIL_LABEL).unwrap();
        assert_eq!(cut_bytes(tail), Some(output.len()), "{tail}");
        let listed = section(&state, CALLS_LABEL).unwrap();
        assert!(listed.starts_with("[cut: "), "{listed}");
        let lines = listed.lines().count();
        assert!(lines > 1 && lines <= CALLS_SHOWN, "{listed}");
        let newest = cut_head_with(
            &names[11],
            Length::Exact(names[11].len()),
            TOOL_NAME_CAP,
            " ",
        );
        assert!(listed.ends_with(&format!("{newest} error")), "{listed}");
        for (label, whole) in [
            (EARLIER_FAILED_INPUT_LABEL, small.to_string()),
            (INPUT_LABEL, small.to_string()),
            (PROMPT_LABEL, prompt.clone()),
            (FINAL_MESSAGE_LABEL, final_message.clone()),
        ] {
            assert_eq!(section(&state, label), Some(whole.as_str()), "{label}");
        }

        // Then the earlier failed input; the input and the prompt stay.
        let state = build_state(&StateParts {
            final_message: Some(&final_message),
            ..parts
        });
        assert!(state.len() <= STATE_CAP, "{}", state.len());
        let listed = section(&state, CALLS_LABEL).unwrap();
        // Every call line is cut: the marker alone stays.
        assert_eq!(listed, marker(cut_bytes(listed).unwrap()));
        let earlier_text = section(&state, EARLIER_FAILED_INPUT_LABEL).unwrap();
        assert!(cut_bytes(earlier_text).is_some(), "{earlier_text}");
        for (label, whole) in [
            (INPUT_LABEL, input.to_string()),
            (PROMPT_LABEL, prompt.clone()),
            (FINAL_MESSAGE_LABEL, final_message.clone()),
        ] {
            assert_eq!(section(&state, label), Some(whole.as_str()), "{label}");
        }
    }

    #[test]
    fn the_total_never_exceeds_its_cap() {
        let sizes = [0, 1, 100, 4095, 4096, 4097, 9000, 40_000];
        for (i, &a) in sizes.iter().enumerate() {
            for &b in &sizes[i..] {
                let prompt = wide(a);
                let input = json!({"x": wide(b)});
                let input_head = JsonHead::new(&input, INPUT_CAP);
                let output = wide(a + b);
                let final_message = wide(b);
                let names: Vec<String> = (0..a % 23).map(|i| "x".repeat(i * 13)).collect();
                let calls: Vec<(&str, bool)> = names.iter().map(|n| (n.as_str(), false)).collect();
                let state = build_state(&StateParts {
                    prompt: &prompt,
                    tool: Some(&prompt),
                    input: Some(&input),
                    earlier_failed_input: Some(&input_head),
                    output: Some(&output),
                    calls: &calls,
                    final_message: Some(&final_message),
                });
                assert!(state.len() <= STATE_CAP, "{a} {b}: {}", state.len());
            }
        }
    }

    #[test]
    fn the_cuts_handle_tiny_caps() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        // Below the marker's width, the marker alone says all is cut.
        let marker_only = marker(text.len());
        let length = Length::Exact(text.len());
        assert_eq!(cut_head(text, length, 3), marker_only);
        assert_eq!(cut_tail(text, 3), marker_only);
        let lines = ["a".repeat(20), "b".repeat(5)];
        assert_eq!(cut_lines(&lines, 3), marker(26));
        assert_eq!(cut_head(text, length, marker_only.len()), marker_only);
        assert_eq!(cut_tail(text, marker_only.len()), marker_only);
        assert_eq!(
            cut_head_with(text, length, marker(26).len() + 3, " "),
            format!("ab {}", marker(24))
        );
        // A text whose end the writer did not reach says so.
        assert_eq!(
            cut_head(
                text,
                Length::MoreThan(text.len()),
                marker_at_least(26).len() + 3
            ),
            format!("ab\n{}", marker_at_least(24))
        );
        assert_eq!(cut_lines(&lines, 100), format!("{}\nbbbbb", lines[0]));
        assert_eq!(
            cut_lines(&lines, marker(26).len() + 6),
            format!("{}\nbbbbb", marker(20))
        );
    }

    #[test]
    fn the_filters_follow_the_checks_table() {
        use ApprovalDecision::{Allow, AskUser, Deny};
        use ApprovalMode::{Ask, Auto, ReadOnly};
        use ToolAccess::{Execute, ReadOnly as Read, Write};
        for access in [Write, Execute] {
            assert!(approval_applies(access, Ask, AskUser));
            assert!(approval_applies(access, Auto, Allow));
            // A session grant in `ask` mode, a refusal, and `read-only`.
            assert!(!approval_applies(access, Ask, Allow));
            assert!(!approval_applies(access, Ask, Deny));
            assert!(!approval_applies(access, Auto, Deny));
            assert!(!approval_applies(access, ReadOnly, Deny));
        }
        assert!(!approval_applies(Read, Auto, Allow));
        assert!(!approval_applies(Read, Ask, Allow));

        assert!(outcome_applies(Some(Execute), false));
        assert!(!outcome_applies(Some(Execute), true));
        assert!(!outcome_applies(Some(Write), false));
        assert!(!outcome_applies(Some(Read), false));
        assert!(!outcome_applies(None, false));

        let completed = TurnEndOutcome::Outcome(TurnOutcome::Completed);
        assert!(claim_applies(completed, 1));
        assert!(!claim_applies(completed, 0));
        for outcome in [
            TurnEndOutcome::ProviderError,
            TurnEndOutcome::Outcome(TurnOutcome::Aborted),
            TurnEndOutcome::Outcome(TurnOutcome::CallLimit),
            TurnEndOutcome::Outcome(TurnOutcome::MaxTokens),
            TurnEndOutcome::Outcome(TurnOutcome::Refused),
        ] {
            assert!(!claim_applies(outcome, 3), "{outcome:?}");
        }
    }

    #[test]
    fn a_large_input_stops_its_writer_at_the_head() {
        let small = JsonHead::new(&json!({"a": 1}), 100);
        assert_eq!(small.text, r#"{"a":1}"#);
        assert_eq!(small.length, Length::Exact(7));
        let huge = json!({"content": "é".repeat(1_000_000)});
        let head = JsonHead::new(&huge, 100);
        assert!(head.text.len() <= 101, "{}", head.text.len());
        assert!(huge.to_string().starts_with(&head.text));
        assert_eq!(head.length, Length::MoreThan(head.text.len()));
    }

    #[test]
    fn one_request_asks_the_questions_of_its_checks() {
        let ids = |asked: &[Check]| questions(asked, &[], &[]).into_keys().collect::<Vec<_>>();
        assert_eq!(
            ids(&[Check::Approval]),
            [APPROVAL_DESTRUCTIVE, APPROVAL_IN_SCOPE]
        );
        assert_eq!(ids(&[Check::Outcome, Check::Progress]), [OUTCOME, PROGRESS]);
        assert_eq!(ids(&[Check::Claim]), [CLAIM]);
    }

    #[test]
    fn the_budget_stops_at_its_cap() {
        let budget = Budget::new(2);
        assert!(!budget.spent());
        assert!(budget.take());
        assert!(!budget.spent());
        assert!(budget.take());
        assert!(budget.spent());
        assert!(!budget.take());
        let empty = Budget::new(0);
        assert!(empty.spent() && !empty.take());
    }

    /// A runtime whose head is a compatible instance on `model` and
    /// whose judge instance `judge` sits at `judge_url`, every check at
    /// `log`, with the request timeout `timeout`. The judge's key is in
    /// the run's store when `key` is set.
    struct Bed {
        runtime: crate::Runtime,
        _dirs: (tempfile::TempDir, tempfile::TempDir),
    }

    impl Bed {
        fn new(model: &str, judge_url: &str, key: bool, timeout: Duration) -> Self {
            let project = tempfile::tempdir().unwrap();
            let data = tempfile::tempdir().unwrap();
            let auth = data.path().join("auth.json");
            let mut config = crate::RunConfig::new(project.path().to_path_buf());
            config.instance = config::Instance::new("local", "openai-compat");
            config.instance.base_url = Some(format!("{model}/v1"));
            config.model = "m1".into();
            config.approval = ApprovalMode::Auto;
            config.data_root = Some(data.path().join("store"));
            config.auth = Some(AuthPaths {
                store: auth.clone(),
                codex_cli: None,
            });
            let user = config::Layer::parse(&format!(
                "[providers.judge]\nkind = \"typesafe\"\nbase_url = \"{judge_url}\"\n"
            ))
            .unwrap()
            .0;
            config.other_instances =
                config::run_instances(&config::Flags::default(), &|_| None, None, Some(&user))
                    .into_iter()
                    .filter(|resolved| resolved.instance.name != "local")
                    .collect();
            config.judgments = Some(config::Judgments::new("judge"));
            if key {
                demerzel_providers::auth::AuthStore::set_instance_key(
                    &auth,
                    "judge",
                    "fake-judge-key",
                    judge_url,
                )
                .unwrap();
            }
            let mut runtime = crate::Runtime::new(config).unwrap();
            runtime.judging.as_mut().unwrap().set_timeout(timeout);
            Self {
                runtime,
                _dirs: (project, data),
            }
        }

        fn judging(&self) -> &Judging {
            self.runtime.judging.as_ref().unwrap()
        }

        /// Run `prompts` in one fresh session; the session file.
        async fn run(&self, prompts: &[&str]) -> std::path::PathBuf {
            let settings = crate::SessionSettings::from_config(self.runtime.config());
            let mut opened = self
                .runtime
                .open(&Resume::Fresh, &settings, prompts[0])
                .unwrap();
            for prompt in prompts {
                let outcome = opened
                    .engine
                    .user_turn(
                        prompt.to_string(),
                        &|_: demerzel_core::EngineEvent<'_>| {},
                        &mut demerzel_core::NoPrompt,
                    )
                    .await
                    .unwrap();
                assert_eq!(outcome, TurnOutcome::Completed);
            }
            opened.engine.session().path().to_path_buf()
        }
    }

    fn records(path: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    /// A judge that accepts each connection, reads the request, and
    /// never answers. It counts the connections it accepted.
    fn silent_judge() -> (String, Arc<AtomicUsize>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let accepted = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&accepted);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                counter.fetch_add(1, Ordering::SeqCst);
                std::thread::spawn(move || {
                    use std::io::Read as _;
                    let mut buf = [0u8; 4096];
                    while matches!(stream.read(&mut buf), Ok(n) if n > 0) {}
                });
            }
        });
        (url, accepted)
    }

    #[tokio::test]
    async fn a_stalled_judge_never_holds_the_turn_and_settle_waits_for_its_record() {
        use demerzel_mock::MockServer;
        use demerzel_mock::compat::{text_turn, tool_turn};

        let (judge, _) = silent_judge();
        let model = MockServer::start(
            vec![
                tool_turn("c1", "bash", json!({"command": "true"})),
                text_turn("done"),
            ],
            Vec::new(),
        )
        .await;
        let bed = Bed::new(&model.url(), &judge, true, Duration::from_millis(200));
        let path = bed.run(&["go"]).await;
        let flight = &bed.judging().shared.flight;
        // The turn ended with the claim request in flight: its record
        // is not in the file yet.
        assert!(flight.count.load(Ordering::Acquire) >= 1);
        assert!(!records(&path).iter().any(|r| r["event"] == "turn_end"));

        bed.runtime.settle().await;
        assert_eq!(flight.count.load(Ordering::Acquire), 0);
        let all = records(&path);
        let judgments: Vec<&Value> = all.iter().filter(|r| r["type"] == "judgment").collect();
        assert_eq!(judgments.len(), 3, "{all:?}");
        for record in &judgments {
            assert_eq!(record["error"], TIMED_OUT, "{record}");
            assert!(record.get("answers").is_none());
            // The round trip of a request that went out.
            assert!(record["duration_ms"].as_u64().unwrap() >= 150, "{record}");
        }
        // The turn record went out before the claim's record.
        let position = |pick: &dyn Fn(&Value) -> bool| all.iter().position(pick).unwrap();
        let turn = position(&|r| r["type"] == "turn");
        let claim = position(&|r| r["event"] == "turn_end");
        assert!(turn < claim, "{all:?}");
        assert!(bed.runtime.take_notices().is_empty());
    }

    #[tokio::test]
    async fn the_run_sends_at_most_its_permits_and_the_rest_record_busy() {
        use demerzel_mock::MockServer;
        use demerzel_mock::compat::{text_turn, tool_calls_turn};

        let (judge, accepted) = silent_judge();
        let input = json!({"command": "true"});
        let calls: Vec<(String, &str, &Value)> =
            (0..12).map(|i| (format!("c{i}"), "bash", &input)).collect();
        let calls: Vec<(&str, &str, &Value)> = calls
            .iter()
            .map(|(id, name, input)| (id.as_str(), *name, *input))
            .collect();
        let model =
            MockServer::start(vec![tool_calls_turn(&calls), text_turn("done")], Vec::new()).await;
        let timeout = Duration::from_millis(600);
        let bed = Bed::new(&model.url(), &judge, true, timeout);
        let started = Instant::now();
        let path = bed.run(&["go"]).await;
        // The turn returned with every request still in flight: 12
        // approvals, 12 outcomes, and the claim. The checks of this
        // test hold when the turn ends before the first requests time
        // out; a machine too slow for that skips them.
        let quick = started.elapsed() < timeout;
        if quick {
            assert_eq!(bed.judging().in_flight(), 25);
        }
        // While the first requests stall, only the permits went out.
        while accepted.load(Ordering::SeqCst) < MAX_IN_FLIGHT && started.elapsed() < timeout {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        if started.elapsed() < timeout {
            assert_eq!(accepted.load(Ordering::SeqCst), MAX_IN_FLIGHT);
        }

        bed.runtime.settle().await;
        assert_eq!(bed.judging().in_flight(), 0);
        let judgments: Vec<Value> = records(&path)
            .into_iter()
            .filter(|r| r["type"] == "judgment")
            .collect();
        assert_eq!(judgments.len(), 25);
        let busy: Vec<&Value> = judgments.iter().filter(|r| r["error"] == BUSY).collect();
        let sent = judgments.iter().filter(|r| r["error"] == TIMED_OUT).count();
        if quick {
            assert!(!busy.is_empty());
        }
        assert_eq!(busy.len() + sent, 25);
        // Each request that went out reached the judge; a busy one
        // sent nothing and took no time.
        assert_eq!(accepted.load(Ordering::SeqCst), sent);
        for record in busy {
            assert_eq!(record["duration_ms"], 0, "{record}");
            assert!(record.get("answers").is_none() && record.get("model").is_none());
        }
    }

    #[tokio::test]
    async fn a_judge_that_does_not_build_is_tried_once_per_turn() {
        use demerzel_mock::MockServer;
        use demerzel_mock::compat::{text_turn, tool_turn};

        let script = || {
            vec![
                tool_turn("c1", "bash", json!({"command": "true"})),
                tool_turn("c2", "bash", json!({"command": "true"})),
                tool_turn("c3", "bash", json!({"command": "true"})),
                text_turn("done"),
            ]
        };
        let model = MockServer::start([script(), script()].concat(), Vec::new()).await;
        let bed = Bed::new(&model.url(), "http://127.0.0.1:9", false, TIMEOUT);
        let path = bed.run(&["one"]).await;
        bed.runtime.settle().await;
        // Seven events would ask; after the first failed build, the
        // turn's later events built no state and spawned nothing.
        let spawned = || bed.judging().shared.spawned.load(Ordering::Relaxed);
        assert_eq!(spawned(), 1);
        let notices = bed.runtime.take_notices();
        assert_eq!(notices.len(), 1, "{notices:?}");
        assert!(notices[0].contains("`judge`"), "{notices:?}");
        assert!(records(&path).iter().all(|r| r["type"] != "judgment"));
        // The next turn tries again, once.
        let _ = bed.run(&["two"]).await;
        bed.runtime.settle().await;
        assert_eq!(spawned(), 2);
        // A new session gives its own notice.
        assert_eq!(bed.runtime.take_notices().len(), 1);
    }
}
