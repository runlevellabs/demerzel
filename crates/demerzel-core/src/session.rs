//! JSONL session store: one append-only file per session.
//!
//! The file is the source of truth for resume and replay, and it is
//! the fixture format for the mock server. It is plaintext and
//! self-contained. It can hold user content and tool output, so it
//! must not be treated as secret-free.
//! Provider credentials belong in the auth store, never in session fields.
//!
//! Layout: `<data root>/projects/<project-id>/sessions/<id>.jsonl`.
//! The first record is metadata. A later meta record marks a resume
//! or a model switch; the last meta record describes the current
//! run. Every other record is one event: a message, an approval
//! decision, a usage report, a tool call, a turn end, or a judge's
//! answers. Readers skip record types they do not know, so old
//! binaries read new files. A torn final line — a crash mid-append —
//! is dropped on replay and truncated away on resume.
//!
//! The file is also the measurement trace for one session. Meta
//! records carry the project, provider, and model. Messages carry tool
//! steps. Usage, tool, and turn records carry counts, timings, sizes,
//! and outcomes. A turn without a closing turn record was cut off.
//!
//! The clock: every usage, tool, and turn record carries `at`, the
//! event's start on the system clock, and `duration_ms`, its wall
//! time on the monotonic clock. The engine measures both at the
//! boundary — the request send, the tool dispatch, the turn — so the
//! file says when each call ran and how long it took. A file from
//! before the clock fields replays the same; the fields are absent.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::approval::{ApprovalDecision, ApprovalMode};
use crate::judge::JudgeUsage;
use crate::message::{Message, Usage};
use crate::paths;
use crate::provider::Effort;
use crate::route::Target;

/// Version of the session-file schema, written into every meta record.
pub const SCHEMA_VERSION: u32 = 1;

/// The `type` of a `judgment` record.
const JUDGMENT_TAG: &str = "judgment";

/// The clock format of the `usage`, `tool`, and `turn` records: RFC
/// 3339 in UTC at millisecond precision. `meta.created` keeps the
/// second precision it has had since schema 1; the session id carries
/// the creation millisecond.
fn timestamp(at: SystemTime) -> String {
    humantime::format_rfc3339_millis(at).to_string()
}

/// When an event started and how long it took: the clock of one
/// usage, tool, or turn record. The start comes from the system
/// clock and the duration from the monotonic clock, so a clock step
/// mid-event moves `at` and not `duration`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timing {
    pub at: SystemTime,
    pub duration: Duration,
}

impl Timing {
    pub fn new(at: SystemTime, duration: Duration) -> Self {
        Self { at, duration }
    }

    /// The duration in whole milliseconds; a duration past the
    /// 64-bit range saturates.
    pub fn duration_ms(self) -> u64 {
        u64::try_from(self.duration.as_millis()).unwrap_or(u64::MAX)
    }
}

/// The metadata record that opens every session file. A resume, a
/// model switch, an effort switch, and an approval switch append a
/// fresh one, so the last meta record describes the current run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SessionMeta {
    pub schema: u32,
    /// Project directory the session ran in.
    pub cwd: String,
    /// The provider instance name, e.g. "local" for a
    /// `[providers.local]` table. A shipped adapter's implicit
    /// instance carries the adapter id, e.g. "openai-codex".
    pub provider: String,
    /// The adapter id behind the instance, e.g. "openai-compat";
    /// absent in a file from before the field, and readers keep such
    /// a file. There, `provider` is the adapter id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub model: String,
    /// The session's reasoning effort as the run's requests carry it,
    /// by its level name; absent when the provider's default applies.
    /// The adapter maps the level to its wire: it can clamp it to its
    /// range, or omit it, as an instance with thinking off does. A
    /// name, not the enum, so a reader keeps the file when a later
    /// build writes a level it does not know.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// The approval mode in force, by its name; absent in a file from
    /// before the field, and readers keep such a file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval: Option<String>,
    /// RFC 3339 creation time.
    pub created: String,
}

impl SessionMeta {
    /// Metadata for a run starting now. The stored `cwd` is canonical,
    /// so one project has one spelling across resumes.
    pub fn now(open: &SessionOpen<'_>) -> Self {
        let canonical = open
            .project
            .canonicalize()
            .unwrap_or_else(|_| open.project.to_path_buf());
        Self {
            schema: SCHEMA_VERSION,
            cwd: canonical.display().to_string(),
            provider: open.provider.to_string(),
            kind: Some(open.kind.to_string()),
            model: open.model.to_string(),
            effort: open.effort.map(|effort| effort.as_str().to_string()),
            approval: Some(open.approval.as_str().to_string()),
            created: humantime::format_rfc3339_seconds(SystemTime::now()).to_string(),
        }
    }
}

/// What a meta record names: the project, the provider instance and
/// its adapter, the model, the effort, and the approval mode. One
/// struct for [`SessionMeta::now`] and [`SessionStore::create`], so a
/// later field changes no signature. Non-exhaustive: build one with
/// [`SessionOpen::new`] and set the optional fields.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SessionOpen<'a> {
    /// The project directory the session is scoped to.
    pub project: &'a Path,
    /// The provider instance name.
    pub provider: &'a str,
    /// The adapter id behind the instance.
    pub kind: &'a str,
    pub model: &'a str,
    /// The reasoning effort the requests carry; `None` when the
    /// provider's default applies. The adapter maps it to its wire;
    /// see [`SessionMeta::effort`].
    pub effort: Option<Effort>,
    pub approval: ApprovalMode,
}

impl<'a> SessionOpen<'a> {
    /// A record for instance `provider` of adapter `kind`, with no
    /// effort and the `ask` mode.
    pub fn new(project: &'a Path, provider: &'a str, kind: &'a str, model: &'a str) -> Self {
        Self {
            project,
            provider,
            kind,
            model,
            effort: None,
            approval: ApprovalMode::Ask,
        }
    }
}

/// One line of a session file. The serialized shape is part of the
/// session-file format; changing it is a schema change. Non-exhaustive:
/// a later record type is additive, and readers skip what they do not
/// know.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SessionRecord {
    Meta(SessionMeta),
    Message(Message),
    /// One approval decision, recorded as a non-prompt event so the
    /// record stays auditable and the prompt prefix stays stable.
    Approval {
        tool: String,
        mode: ApprovalMode,
        decision: ApprovalDecision,
        /// Id of the tool call the decision was for.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_use_id: Option<String>,
        /// Outcome when the user was asked (`AskUser`) or when a host
        /// overrode the policy decision (`Allow` or `Deny`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        approved: Option<bool>,
        /// The literal answer for the audit trail: `allow`,
        /// `allow_for_session`, `deny`, `abort`, or `judge`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        answer: Option<String>,
    },
    /// The token counts of one model call, with the target that
    /// answered it and the call's clock. `provider` names the
    /// instance and `model` the model: with a fallback chain, the
    /// target can differ from the meta record's head. `at` is when
    /// the request went out, and `duration_ms` is the wall time from
    /// the send to the stream's terminal event, the failed attempts
    /// before the answer included. Each field is absent in a file
    /// from before it, and readers keep such a file.
    Usage {
        #[serde(flatten)]
        usage: Usage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        at: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    /// One tool call the engine answered, appended after the user
    /// message that carries the call's result: one record per
    /// `tool_use` block, in the batch's order. `at` is when the run
    /// started and `duration_ms` is its wall time; the wait at an
    /// approval prompt is not in it, and a call the policy, the user,
    /// or an abort declined records the time to build its error
    /// result. `output_bytes` counts the result text the model
    /// receives, and `truncated` says whether a size limit cut it.
    Tool {
        call_id: String,
        name: String,
        at: String,
        duration_ms: u64,
        is_error: bool,
        output_bytes: u64,
        truncated: bool,
    },
    /// How one user turn ended. `outcome` is one of `completed`,
    /// `max_tokens`, `refused`, `call_limit`, `aborted`, or `error`;
    /// the first five are the engine's turn outcomes by their
    /// `as_str` names, and `error` marks a provider failure, with the
    /// message in `error`, cut at 1 KiB. `at` is when the turn
    /// started and `duration_ms` is the whole user turn's wall time;
    /// both are absent in a file from before the fields.
    Turn {
        outcome: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        at: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    /// One judge request and its answers. Replay and resume skip it;
    /// see [`JudgmentRecord`].
    Judgment(JudgmentRecord),
    /// A record type from a newer schema. Replay skips it, so old
    /// binaries read new files. The harness never writes this variant.
    #[serde(other)]
    Unknown,
}

/// Metadata for one configured rule that asked in a `judgment` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JudgmentRuleRecord {
    /// The rule's configured action: `log`, `warn`, `note`, or `deny`.
    pub action: String,
    /// Where the rule came from: `user` or `project`.
    pub source: String,
    /// The rule's firing threshold.
    pub threshold: f64,
}

impl JudgmentRuleRecord {
    pub fn new(action: impl Into<String>, source: impl Into<String>, threshold: f64) -> Self {
        Self {
            action: action.into(),
            source: source.into(),
            threshold,
        }
    }
}

/// One judge request with its answers, or its failure. The record
/// completes off the turn's path, so it can land after records that
/// follow its event; `at` places it. A `tool_before` record pairs with
/// the `approval` record whose `tool_use_id` equals its `call_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JudgmentRecord {
    /// "tool_before", "tool_after", or "turn_end".
    pub event: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    /// Each built-in check that asked in the request, with the action
    /// config sets for it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub checks: BTreeMap<String, String>,
    /// Each configured rule that asked in the request, keyed by its
    /// question id, with its configured action, source, and threshold.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub rules: BTreeMap<String, JudgmentRuleRecord>,
    /// Each check or rule whose answer crossed its threshold, mapped to
    /// the action that fired (for example `warn`, `note`, `allow`,
    /// `ask`, `deny`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fired: BTreeMap<String, String>,
    /// The judge instance name.
    pub instance: String,
    /// The model id from the reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Each answer by question id, as [`crate::Answer`] serializes
    /// it. JSON values, so an answer type from a later build does not
    /// stop this build from reading the file.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub answers: BTreeMap<String, serde_json::Value>,
    /// The state's UTF-8 length.
    pub state_bytes: u64,
    /// Lowercase hex SHA-256 of the state's bytes.
    pub state_sha256: String,
    /// When the request started, in the clock format of the other
    /// records.
    pub at: String,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<JudgeUsage>,
    /// "timeout", "busy", or the error text, cut at 1 KiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl JudgmentRecord {
    /// A record of `event` from judge instance `instance`, with every
    /// other field empty.
    pub fn new(event: impl Into<String>, instance: impl Into<String>) -> Self {
        Self {
            event: event.into(),
            call_id: None,
            tool: None,
            checks: BTreeMap::new(),
            rules: BTreeMap::new(),
            fired: BTreeMap::new(),
            instance: instance.into(),
            model: None,
            answers: BTreeMap::new(),
            state_bytes: 0,
            state_sha256: String::new(),
            at: String::new(),
            duration_ms: 0,
            usage: None,
            error: None,
        }
    }
}

/// Borrowed mirror of [`SessionRecord`], so appends serialize without
/// cloning the payload. The serialized shape is identical.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RecordRef<'a> {
    Meta(&'a SessionMeta),
    Message(&'a Message),
    Approval {
        tool: &'a str,
        mode: ApprovalMode,
        decision: ApprovalDecision,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_use_id: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        approved: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        answer: Option<&'a str>,
    },
    Usage {
        #[serde(flatten)]
        usage: &'a Usage,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        at: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    Tool {
        call_id: &'a str,
        name: &'a str,
        at: &'a str,
        duration_ms: u64,
        is_error: bool,
        output_bytes: u64,
        truncated: bool,
    },
    Turn {
        outcome: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        at: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    Judgment(&'a JudgmentRecord),
}

/// The record type alone. Only the replay error path uses it, to name
/// the record type in the message; the happy path parses each line
/// once.
#[derive(Deserialize)]
struct RecordTag<'a> {
    #[serde(borrow, rename = "type")]
    kind: Option<std::borrow::Cow<'a, str>>,
}

/// History rebuilt from a session file: the messages the model
/// receives again, and the records beside them that an interface
/// shows again — the usage lines and the clock of each tool call.
/// Readers get one from [`replay`] or [`Session::resume`]; the
/// struct grows with the records a reader shows.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Replay {
    pub meta: Option<SessionMeta>,
    pub messages: Vec<Message>,
    /// The usage records, in file order. Each counts the messages
    /// before it, so a reader places the line where the call ended.
    pub usage: Vec<ReplayUsage>,
    /// The tool records, in file order: the clock of each answered
    /// call, by call id.
    pub tools: Vec<ReplayToolClock>,
}

/// One usage record on the replay.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayUsage {
    /// The count of messages before the record in the file.
    pub messages_before: usize,
    pub usage: Usage,
    /// The call's clock, when the record carries one.
    pub timing: Option<Timing>,
}

/// The clock of one tool record on the replay.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayToolClock {
    pub call_id: String,
    /// The call's wall time, as the record kept it.
    pub duration: Duration,
}

/// The session directory tree under one data root.
#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    /// Open the store at the default data root (see
    /// [`crate::paths::data_dir`]).
    pub fn open_default() -> io::Result<Self> {
        Ok(Self::at(paths::data_dir()?))
    }

    /// Open the store at an explicit data root. Tests use a temp dir.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The sessions directory for one project.
    pub fn sessions_dir(&self, project: &Path) -> PathBuf {
        self.root
            .join("projects")
            .join(project_id(project))
            .join("sessions")
    }

    /// Create a new session file for `open.project` and write its meta
    /// record.
    pub fn create(&self, open: &SessionOpen<'_>) -> io::Result<Session> {
        let project = open.project;
        let dir = self.sessions_dir(project);
        std::fs::create_dir_all(&dir)?;
        let mut attempt = 0;
        let (id, path, file) = loop {
            let id = new_session_id();
            let path = dir.join(format!("{id}.jsonl"));
            match OpenOptions::new().create_new(true).append(true).open(&path) {
                Ok(file) => break (id, path, file),
                // A cross-process id collision: mint a fresh id once.
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists && attempt == 0 => {
                    attempt += 1;
                }
                Err(error) => return Err(error),
            }
        };
        let mut session = Session {
            id,
            path,
            file: Arc::new(Mutex::new(Appender::new(file))),
            cwd: project.to_path_buf(),
            meta: None,
            buf: Vec::new(),
        };
        session.append_meta(&SessionMeta::now(open))?;
        log::info!(
            "session created id={} path={}",
            session.id,
            session.path.display()
        );
        Ok(session)
    }

    /// The session file with this id in the project, or `None` when
    /// no file has the id. An id has the shape `create` mints: 26
    /// digits and capital letters. Anything else — a path, a
    /// lowercase spelling, an empty string — names no file, so a
    /// `--session` argument cannot reach outside the project's
    /// directory, and a case-insensitive file system answers the
    /// same as a case-sensitive one.
    pub fn find(&self, project: &Path, id: &str) -> io::Result<Option<PathBuf>> {
        let shaped = id.len() == SESSION_ID_LEN
            && id
                .bytes()
                .all(|byte| byte.is_ascii_digit() || byte.is_ascii_uppercase());
        if !shaped {
            return Ok(None);
        }
        let path = self.sessions_dir(project).join(format!("{id}.jsonl"));
        match std::fs::metadata(&path) {
            Ok(meta) if meta.is_file() => Ok(Some(path)),
            Ok(_) => Ok(None),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// The newest session file for the project, by id. Session ids
    /// start with a millisecond timestamp, so the name order is the
    /// creation order.
    pub fn latest(&self, project: &Path) -> io::Result<Option<PathBuf>> {
        let dir = self.sessions_dir(project);
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut newest: Option<PathBuf> = None;
        for entry in entries {
            let path = entry?.path();
            if path.extension().is_none_or(|ext| ext != "jsonl") {
                continue;
            }
            if newest.as_ref().is_none_or(|best| path > *best) {
                newest = Some(path);
            }
        }
        Ok(newest)
    }
}

/// One open session: an append handle on its file.
#[derive(Debug)]
pub struct Session {
    id: String,
    path: PathBuf,
    /// The append handle, shared with every [`SessionWriter`]: each
    /// line goes out in one write under the lock, and the file closes
    /// when the last handle drops.
    file: Arc<Mutex<Appender<File>>>,
    /// The project directory the session is scoped to, kept so a
    /// mid-run meta record — a model switch — names the same project
    /// as the one that opened the file.
    cwd: PathBuf,
    /// The last meta record this handle wrote or resumed from. A
    /// switch record starts from it, so a field that one record
    /// carries stays on the next.
    meta: Option<SessionMeta>,
    /// Reused serialization buffer, so appends allocate nothing steady
    /// state.
    buf: Vec<u8>,
}

impl Session {
    /// The last meta record this handle wrote or resumed from.
    pub fn meta(&self) -> Option<&SessionMeta> {
        self.meta.as_ref()
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// The project directory the session is scoped to.
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one record as one line. The line is written with a
    /// single call, so a reader never sees a torn record boundary.
    pub fn append(&mut self, record: &SessionRecord) -> io::Result<()> {
        let record = match record {
            // Through `append_meta`, so the session adopts the
            // meta's cwd on every path a meta record can enter.
            SessionRecord::Meta(meta) => return self.append_meta(meta),
            SessionRecord::Message(message) => RecordRef::Message(message),
            SessionRecord::Approval {
                tool,
                mode,
                decision,
                tool_use_id,
                approved,
                answer,
            } => RecordRef::Approval {
                tool,
                mode: *mode,
                decision: *decision,
                tool_use_id: tool_use_id.as_deref(),
                approved: *approved,
                answer: answer.as_deref(),
            },
            SessionRecord::Usage {
                usage,
                provider,
                model,
                at,
                duration_ms,
            } => RecordRef::Usage {
                usage,
                provider: provider.as_deref(),
                model: model.as_deref(),
                at: at.as_deref(),
                duration_ms: *duration_ms,
            },
            SessionRecord::Tool {
                call_id,
                name,
                at,
                duration_ms,
                is_error,
                output_bytes,
                truncated,
            } => RecordRef::Tool {
                call_id,
                name,
                at,
                duration_ms: *duration_ms,
                is_error: *is_error,
                output_bytes: *output_bytes,
                truncated: *truncated,
            },
            SessionRecord::Turn {
                outcome,
                error,
                at,
                duration_ms,
            } => RecordRef::Turn {
                outcome,
                error: error.as_deref(),
                at: at.as_deref(),
                duration_ms: *duration_ms,
            },
            SessionRecord::Judgment(record) => RecordRef::Judgment(record),
            // Replay-only marker; there is nothing to write.
            SessionRecord::Unknown => return Ok(()),
        };
        self.append_ref(&record)
    }

    pub fn append_meta(&mut self, meta: &SessionMeta) -> io::Result<()> {
        self.append_ref(&RecordRef::Meta(meta))?;
        // The session adopts the meta's canonical cwd, so a later
        // meta append — a model switch — names the same project even
        // when the file the session resumed had no meta record.
        self.cwd = PathBuf::from(&meta.cwd);
        self.meta = Some(meta.clone());
        Ok(())
    }

    pub fn append_message(&mut self, message: &Message) -> io::Result<()> {
        self.append_ref(&RecordRef::Message(message))
    }

    pub fn append_approval(
        &mut self,
        tool: &str,
        mode: ApprovalMode,
        decision: ApprovalDecision,
        tool_use_id: &str,
        approved: Option<bool>,
        answer: Option<&str>,
    ) -> io::Result<()> {
        self.append_ref(&RecordRef::Approval {
            tool,
            mode,
            decision,
            tool_use_id: Some(tool_use_id),
            approved,
            answer,
        })
    }

    /// Record one model call's token counts, the target that
    /// answered it, and its clock. See [`SessionRecord::Usage`].
    pub fn append_usage(
        &mut self,
        usage: &Usage,
        target: &Target,
        timing: Timing,
    ) -> io::Result<()> {
        let at = timestamp(timing.at);
        self.append_ref(&RecordRef::Usage {
            usage,
            provider: Some(&target.instance),
            model: Some(&target.model),
            at: Some(&at),
            duration_ms: Some(timing.duration_ms()),
        })
    }

    /// Record one answered tool call. See [`SessionRecord::Tool`].
    pub fn append_tool(
        &mut self,
        call_id: &str,
        name: &str,
        timing: Timing,
        is_error: bool,
        output_bytes: u64,
        truncated: bool,
    ) -> io::Result<()> {
        let at = timestamp(timing.at);
        self.append_ref(&RecordRef::Tool {
            call_id,
            name,
            at: &at,
            duration_ms: timing.duration_ms(),
            is_error,
            output_bytes,
            truncated,
        })
    }

    /// Close one user turn with its outcome and its clock. See
    /// [`SessionRecord::Turn`] for the outcome vocabulary.
    pub fn append_turn(
        &mut self,
        outcome: &str,
        error: Option<&str>,
        timing: Timing,
    ) -> io::Result<()> {
        let at = timestamp(timing.at);
        self.append_ref(&RecordRef::Turn {
            outcome,
            error,
            at: Some(&at),
            duration_ms: Some(timing.duration_ms()),
        })
    }

    fn append_ref(&mut self, record: &RecordRef<'_>) -> io::Result<()> {
        self.buf.clear();
        serde_json::to_writer(&mut self.buf, record).map_err(io::Error::other)?;
        self.buf.push(b'\n');
        write_line(&self.file, &self.buf)
    }

    /// An append handle on this session's file, for records that
    /// complete off the turn's path. The handle keeps the file open
    /// after the session drops.
    pub fn writer(&self) -> SessionWriter {
        SessionWriter {
            file: Arc::clone(&self.file),
        }
    }

    /// Reopen an existing session file for appending and rebuild its
    /// history. A torn final line is truncated away first, so the next
    /// append starts on a clean record boundary.
    pub fn resume(path: &Path) -> io::Result<(Self, Replay)> {
        truncate_torn_tail(path)?;
        let replay = replay(path)?;
        let id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_string();
        let file = OpenOptions::new().append(true).open(path)?;
        // The stored cwd is already canonical, so a later meta append
        // names the same spelling. A file without a meta record is
        // outside the format; an empty cwd keeps the append total.
        let cwd = replay
            .meta
            .as_ref()
            .map(|meta| PathBuf::from(&meta.cwd))
            .unwrap_or_default();
        log::info!(
            "session resumed id={id} path={} messages={}",
            path.display(),
            replay.messages.len()
        );
        Ok((
            Self {
                id,
                path: path.to_path_buf(),
                file: Arc::new(Mutex::new(Appender::new(file))),
                cwd,
                meta: replay.meta.clone(),
                buf: Vec::new(),
            },
            replay,
        ))
    }
}

/// A cloneable append handle on one session's file, for records that
/// complete off the turn's path. Every line goes out in one write
/// under the file's lock, so lines from the session and from each
/// clone never interleave. The file closes when the last handle drops.
#[derive(Debug, Clone)]
pub struct SessionWriter {
    file: Arc<Mutex<Appender<File>>>,
}

impl SessionWriter {
    /// Append one `judgment` record as one line.
    pub fn append_judgment(&self, record: &JudgmentRecord) -> io::Result<()> {
        let mut line =
            serde_json::to_vec(&RecordRef::Judgment(record)).map_err(io::Error::other)?;
        line.push(b'\n');
        write_line(&self.file, &line)
    }
}

/// Write one complete line under the file's lock. The lock guards
/// state that every append keeps whole, so a poisoned lock still
/// writes.
fn write_line(file: &Mutex<Appender<File>>, line: &[u8]) -> io::Result<()> {
    file.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .append(line)
}

/// A file a line goes to, as [`Appender`] needs it.
trait LineFile {
    /// One write call: the bytes it took.
    fn write_some(&mut self, bytes: &[u8]) -> io::Result<usize>;
    fn length(&mut self) -> io::Result<u64>;
    fn cut_to(&mut self, length: u64) -> io::Result<()>;
}

impl LineFile for File {
    fn write_some(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.write(bytes)
    }

    fn length(&mut self) -> io::Result<u64> {
        Ok(self.metadata()?.len())
    }

    fn cut_to(&mut self, length: u64) -> io::Result<()> {
        self.set_len(length)
    }
}

/// The append side of a session file, shared by the session and its
/// writers.
#[derive(Debug)]
struct Appender<F> {
    file: F,
    /// Set when a failed append could not be cut back: the file ends in
    /// a fragment, and a later line would join it.
    torn: bool,
}

impl<F: LineFile> Appender<F> {
    fn new(file: F) -> Self {
        Self { file, torn: false }
    }

    /// Append one whole line or nothing: a write that fails part way,
    /// as on a full disk, is cut back by the bytes it wrote, so no
    /// fragment joins the next line and the file stays readable. A
    /// write that succeeds makes no other system call. When the cut
    /// fails too, every later append fails: a resume cuts the fragment
    /// away.
    fn append(&mut self, line: &[u8]) -> io::Result<()> {
        if self.torn {
            return Err(io::Error::other(
                "the session file ends in a partial line from a failed write; resume the \
                 session to repair it",
            ));
        }
        // `write_all`, with a count of the bytes written.
        let mut written = 0;
        let error = loop {
            if written == line.len() {
                return Ok(());
            }
            match self.file.write_some(&line[written..]) {
                Ok(0) => break io::Error::from(io::ErrorKind::WriteZero),
                Ok(n) => written += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => break error,
            }
        };
        if written > 0 {
            let cut = self
                .file
                .length()
                .and_then(|length| self.file.cut_to(length.saturating_sub(written as u64)));
            if let Err(cut) = cut {
                log::error!("session file: a failed append left a partial line: {cut}");
                self.torn = true;
            }
        }
        Err(error)
    }
}

/// Cut an unterminated final line, the leftover of a crash mid-append.
/// Reads backward in blocks from the file end, so a long session never
/// loads whole into memory.
fn truncate_torn_tail(path: &Path) -> io::Result<()> {
    use std::io::{Read as _, Seek as _, SeekFrom};
    const BLOCK: u64 = 8192;
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(());
    }
    let mut byte = [0u8; 1];
    file.seek(SeekFrom::Start(len - 1))?;
    file.read_exact(&mut byte)?;
    if byte[0] == b'\n' {
        return Ok(());
    }
    let mut end = len - 1;
    let mut buf = vec![0u8; BLOCK as usize];
    loop {
        let start = end.saturating_sub(BLOCK);
        let read_len = (end - start) as usize;
        if read_len == 0 {
            return file.set_len(0);
        }
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut buf[..read_len])?;
        if let Some(position) = buf[..read_len].iter().rposition(|b| *b == b'\n') {
            return file.set_len(start + position as u64 + 1);
        }
        end = start;
    }
}

/// Rebuild history from a session file. Unknown record types are
/// skipped; a malformed known record or a record without a type is an
/// error; an unterminated final line is ignored.
pub fn replay(path: &Path) -> io::Result<Replay> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut out = Replay::default();
    let mut line = String::new();
    let mut number = 0usize;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        number += 1;
        if !line.ends_with('\n') {
            // Torn tail from a crash mid-append; resume truncates it.
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let bad = |what: &str, detail: String| {
            io::Error::other(format!("{}:{number}: {what}: {detail}", path.display()))
        };
        let record: SessionRecord = match serde_json::from_str(trimmed) {
            Ok(record) => record,
            Err(error) => {
                // Name the record type in the error when the line can
                // still identify itself.
                let kind = serde_json::from_str::<RecordTag<'_>>(trimmed)
                    .ok()
                    .and_then(|tag| tag.kind);
                // Replay does not read judgment records, so their shape
                // can change in a later build: one that does not parse
                // is skipped by its tag.
                if kind.as_deref() == Some(JUDGMENT_TAG) {
                    continue;
                }
                return Err(match kind {
                    Some(kind) => bad(&format!("bad {kind} record"), error.to_string()),
                    None => bad("bad session record", error.to_string()),
                });
            }
        };
        match record {
            SessionRecord::Meta(meta) => {
                if meta.schema > SCHEMA_VERSION {
                    return Err(bad(
                        "unsupported session schema",
                        format!(
                            "file has schema {}, this build reads {SCHEMA_VERSION}",
                            meta.schema
                        ),
                    ));
                }
                out.meta = Some(meta);
            }
            SessionRecord::Message(message) => out.messages.push(message),
            SessionRecord::Usage {
                usage,
                at,
                duration_ms,
                ..
            } => {
                // A clock needs both halves; a file from before the
                // fields has neither, and a record with one alone
                // has no clock a reader can place.
                let timing = at
                    .as_deref()
                    .and_then(|at| humantime::parse_rfc3339(at).ok())
                    .zip(duration_ms)
                    .map(|(at, ms)| Timing::new(at, Duration::from_millis(ms)));
                out.usage.push(ReplayUsage {
                    messages_before: out.messages.len(),
                    usage,
                    timing,
                });
            }
            SessionRecord::Tool {
                call_id,
                duration_ms,
                ..
            } => out.tools.push(ReplayToolClock {
                call_id,
                duration: Duration::from_millis(duration_ms),
            }),
            SessionRecord::Approval { .. }
            | SessionRecord::Turn { .. }
            | SessionRecord::Judgment(_) => {}
            SessionRecord::Unknown => {}
        }
    }
    Ok(out)
}

/// Directory name for one project: the canonical path folded to safe
/// characters, bounded in length, plus a short hash of the unfolded
/// path so distinct projects never share a directory.
fn project_id(project: &Path) -> String {
    const MAX_STEM: usize = 96;
    let canonical = project
        .canonicalize()
        .unwrap_or_else(|_| project.to_path_buf());
    let display = canonical.display().to_string();
    let mut stem: String = display
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    stem.truncate(MAX_STEM);
    let hash = fnv1a64(display.as_bytes());
    format!("{stem}-{:08x}", (hash >> 32) as u32 ^ hash as u32)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Crockford base32, the ULID alphabet.
const ULID_ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Characters in a session id: 128 bits in base32.
const SESSION_ID_LEN: usize = 26;

const ENTROPY_MASK: u128 = (1 << 80) - 1;

/// The last issued id as `(millis, entropy)`, so ids stay strictly
/// increasing inside one process even when the clock steps backward.
static LAST_ID: Mutex<(u128, u128)> = Mutex::new((0, 0));

/// A ULID-format session id: 48-bit millisecond timestamp plus 80 bits
/// of local entropy, 26 characters of Crockford base32. Ids sort by
/// creation time; inside one millisecond the entropy increments, so
/// the order stays strict. The entropy seeds from the clock and the
/// process id; it is unique per machine, not cryptographic.
fn new_session_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let raw_millis =
        u128::from(u64::try_from(now.as_millis()).unwrap_or(u64::MAX)) & ((1 << 48) - 1);
    let seed = now.as_nanos() as u64 ^ (u64::from(std::process::id()) << 32);
    let (millis, entropy) = {
        let mut last = LAST_ID.lock().unwrap();
        // Clamp a backward clock step, so a new id never sorts before
        // an issued one and `latest` keeps returning the newest.
        let millis = raw_millis.max(last.0);
        let entropy = if last.0 == millis {
            (last.1 + 1) & ENTROPY_MASK
        } else {
            (u128::from(splitmix64(seed)) << 16 | u128::from(splitmix64(seed ^ 0xDEAD) >> 48))
                & ENTROPY_MASK
        };
        *last = (millis, entropy);
        (millis, entropy)
    };
    let value = (millis << 80) | entropy;
    let mut id = String::with_capacity(SESSION_ID_LEN);
    for shift in (0..SESSION_ID_LEN).rev() {
        let index = ((value >> (shift * 5)) & 0x1F) as usize;
        id.push(ULID_ALPHABET[index] as char);
    }
    id
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, Role};

    fn text_message(role: Role, text: &str) -> Message {
        Message {
            role,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[test]
    fn create_append_and_replay_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let project = dir.path().join("project");
        std::fs::create_dir(&project).unwrap();

        let mut session = store
            .create(&SessionOpen::new(
                &project,
                "openai-codex",
                "openai-codex",
                "gpt-test",
            ))
            .unwrap();
        session
            .append_message(&text_message(Role::User, "hi"))
            .unwrap();
        session
            .append_message(&text_message(Role::Assistant, "hello"))
            .unwrap();
        session
            .append_usage(
                &Usage {
                    input_tokens: 10,
                    output_tokens: 2,
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                },
                &Target::new("openai-codex", "gpt-test"),
                Timing::new(SystemTime::now(), Duration::from_millis(5)),
            )
            .unwrap();

        let replay = replay(session.path()).unwrap();
        let meta = replay.meta.unwrap();
        assert_eq!(meta.schema, SCHEMA_VERSION);
        assert_eq!(meta.provider, "openai-codex");
        assert_eq!(meta.kind.as_deref(), Some("openai-codex"));
        assert_eq!(meta.model, "gpt-test");
        assert_eq!(meta.approval.as_deref(), Some("ask"));
        assert_eq!(replay.messages.len(), 2);
        assert_eq!(replay.messages[0], text_message(Role::User, "hi"));
        // The usage record follows both messages, with its clock.
        assert_eq!(replay.usage.len(), 1);
        assert_eq!(replay.usage[0].messages_before, 2);
        assert_eq!(replay.usage[0].usage.input_tokens, 10);
        assert_eq!(
            replay.usage[0].timing.map(|timing| timing.duration),
            Some(Duration::from_millis(5))
        );
        assert!(replay.tools.is_empty());
    }

    #[test]
    fn a_meta_record_names_the_instance_and_its_adapter() {
        let dir = tempfile::tempdir().unwrap();
        let mut open = SessionOpen::new(dir.path(), "local", "openai-compat", "qwen3");
        open.effort = Some(Effort::Low);
        let meta = SessionMeta::now(&open);
        assert_eq!(meta.provider, "local");
        assert_eq!(meta.kind.as_deref(), Some("openai-compat"));
        assert_eq!(meta.model, "qwen3");
        assert_eq!(meta.effort.as_deref(), Some("low"));
        assert_eq!(meta.approval.as_deref(), Some("ask"));

        // A record from before the field reads without a kind.
        let old: SessionMeta = serde_json::from_str(
            "{\"schema\":1,\"cwd\":\"/x\",\"provider\":\"anthropic\",\"model\":\"m\",\
             \"created\":\"2026-08-07T00:00:00Z\"}",
        )
        .unwrap();
        assert_eq!(old.provider, "anthropic");
        assert_eq!(old.kind, None);
    }

    #[test]
    fn the_replay_keeps_the_tool_clocks_and_places_each_usage_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let project = dir.path().join("project");
        std::fs::create_dir(&project).unwrap();
        let mut open = SessionOpen::new(&project, "openai-codex", "openai-codex", "gpt-test");
        open.approval = ApprovalMode::Auto;
        let mut session = store.create(&open).unwrap();
        let usage = Usage {
            input_tokens: 1,
            output_tokens: 1,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        };
        let clock = Timing::new(SystemTime::UNIX_EPOCH, Duration::from_millis(40));
        session
            .append_message(&text_message(Role::User, "go"))
            .unwrap();
        session
            .append_message(&text_message(Role::Assistant, "calling"))
            .unwrap();
        session
            .append_usage(&usage, &Target::new("p", "m"), clock)
            .unwrap();
        session
            .append_message(&text_message(Role::User, "result"))
            .unwrap();
        session
            .append_tool("call_1", "read", clock, false, 12, false)
            .unwrap();
        session
            .append_message(&text_message(Role::Assistant, "done"))
            .unwrap();
        session
            .append_usage(&usage, &Target::new("p", "m"), clock)
            .unwrap();

        let replay = replay(session.path()).unwrap();
        assert_eq!(replay.messages.len(), 4);
        let placed: Vec<usize> = replay
            .usage
            .iter()
            .map(|record| record.messages_before)
            .collect();
        assert_eq!(placed, vec![2, 4]);
        assert_eq!(replay.usage[0].timing, Some(clock));
        assert_eq!(
            replay.tools,
            vec![ReplayToolClock {
                call_id: "call_1".into(),
                duration: Duration::from_millis(40),
            }]
        );
    }

    #[test]
    fn record_shapes_are_stable_and_the_borrowed_mirror_matches() {
        // The session-file format depends on these exact shapes.
        let record = SessionRecord::Message(text_message(Role::User, "hi"));
        let value = serde_json::to_value(&record).unwrap();
        assert_eq!(value["type"], "message");
        assert_eq!(value["role"], "user");
        assert_eq!(value["content"][0]["type"], "text");

        let approval = SessionRecord::Approval {
            tool: "bash".into(),
            mode: ApprovalMode::Ask,
            decision: ApprovalDecision::AskUser,
            tool_use_id: Some("call_9".into()),
            approved: Some(true),
            answer: Some("allow".into()),
        };
        let value = serde_json::to_value(&approval).unwrap();
        assert_eq!(value["type"], "approval");
        assert_eq!(value["mode"], "ask");
        assert_eq!(value["decision"], "ask_user");
        assert_eq!(value["tool_use_id"], "call_9");
        assert_eq!(value["approved"], true);

        let usage = SessionRecord::Usage {
            usage: Usage::default(),
            provider: Some("local".into()),
            model: Some("m1".into()),
            at: Some("2026-09-01T10:00:00.123Z".into()),
            duration_ms: Some(1500),
        };
        let value = serde_json::to_value(&usage).unwrap();
        assert_eq!(value["type"], "usage");
        // The token counts sit flat on the record, beside the target
        // and the clock.
        assert_eq!(value["input_tokens"], 0);
        assert_eq!(value["provider"], "local");
        assert_eq!(value["model"], "m1");
        assert_eq!(value["at"], "2026-09-01T10:00:00.123Z");
        assert_eq!(value["duration_ms"], 1500);

        let tool = SessionRecord::Tool {
            call_id: "call_9".into(),
            name: "bash".into(),
            at: "2026-09-01T10:00:01.000Z".into(),
            duration_ms: 42,
            is_error: false,
            output_bytes: 17,
            truncated: false,
        };
        let value = serde_json::to_value(&tool).unwrap();
        assert_eq!(value["type"], "tool");
        assert_eq!(value["call_id"], "call_9");
        assert_eq!(value["name"], "bash");
        assert_eq!(value["at"], "2026-09-01T10:00:01.000Z");
        assert_eq!(value["duration_ms"], 42);
        assert_eq!(value["is_error"], false);
        assert_eq!(value["output_bytes"], 17);
        assert_eq!(value["truncated"], false);

        let turn = SessionRecord::Turn {
            outcome: "completed".into(),
            error: None,
            at: Some("2026-09-01T10:00:00.000Z".into()),
            duration_ms: Some(2000),
        };
        let value = serde_json::to_value(&turn).unwrap();
        assert_eq!(value["type"], "turn");
        assert_eq!(value["outcome"], "completed");
        assert!(value.get("error").is_none());
        assert_eq!(value["at"], "2026-09-01T10:00:00.000Z");
        assert_eq!(value["duration_ms"], 2000);
        let failed = SessionRecord::Turn {
            outcome: "error".into(),
            error: Some("rate limited".into()),
            at: None,
            duration_ms: None,
        };
        let value = serde_json::to_value(&failed).unwrap();
        assert_eq!(value["error"], "rate limited");
        assert!(value.get("at").is_none());
        assert!(value.get("duration_ms").is_none());

        // The borrowed append path serializes the identical shape.
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let mut session = store
            .create(&SessionOpen::new(dir.path(), "p", "p", "m"))
            .unwrap();
        session
            .append_approval(
                "bash",
                ApprovalMode::Ask,
                ApprovalDecision::AskUser,
                "call_9",
                Some(true),
                Some("allow"),
            )
            .unwrap();
        let text = std::fs::read_to_string(session.path()).unwrap();
        let line = text.lines().last().unwrap();
        assert_eq!(line, serde_json::to_string(&approval).unwrap());
    }

    #[test]
    fn resume_appends_to_the_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let project = dir.path().to_path_buf();
        let mut session = store
            .create(&SessionOpen::new(&project, "p", "p", "m"))
            .unwrap();
        session
            .append_message(&text_message(Role::User, "one"))
            .unwrap();
        let path = session.path().to_path_buf();
        drop(session);

        let (mut resumed, replayed) = Session::resume(&path).unwrap();
        assert_eq!(replayed.messages.len(), 1);
        resumed
            .append_message(&text_message(Role::Assistant, "two"))
            .unwrap();
        let replayed = replay(&path).unwrap();
        assert_eq!(replayed.messages.len(), 2);
    }

    #[test]
    fn a_torn_final_line_is_dropped_and_truncated_on_resume() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let mut session = store
            .create(&SessionOpen::new(dir.path(), "p", "p", "m"))
            .unwrap();
        session
            .append_message(&text_message(Role::User, "kept"))
            .unwrap();
        let path = session.path().to_path_buf();
        drop(session);
        // A crash mid-append leaves a record without its newline.
        use std::io::Write as _;
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        write!(file, "{{\"type\":\"message\",\"TORN_FRAGMENT").unwrap();
        drop(file);

        // Replay ignores the torn tail.
        let replayed = replay(&path).unwrap();
        assert_eq!(replayed.messages.len(), 1);

        // Resume truncates it, so the next append starts clean.
        let (mut resumed, _) = Session::resume(&path).unwrap();
        resumed
            .append_message(&text_message(Role::Assistant, "next"))
            .unwrap();
        let replayed = replay(&path).unwrap();
        assert_eq!(replayed.messages.len(), 2);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("TORN_FRAGMENT"));
    }

    #[test]
    fn a_torn_tail_longer_than_one_scan_block_is_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let mut session = store
            .create(&SessionOpen::new(dir.path(), "p", "p", "m"))
            .unwrap();
        session
            .append_message(&text_message(Role::User, "kept"))
            .unwrap();
        let path = session.path().to_path_buf();
        drop(session);
        use std::io::Write as _;
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        // A torn record larger than the 8 KiB backward-scan block.
        write!(
            file,
            "{{\"type\":\"message\",\"x\":\"{}",
            "y".repeat(40_000)
        )
        .unwrap();
        drop(file);

        let (_, replayed) = Session::resume(&path).unwrap();
        assert_eq!(replayed.messages.len(), 1);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with('\n'));
        assert!(!text.contains("yyyy"));
    }

    #[test]
    fn append_matches_the_owned_record_serialization_for_every_variant() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let mut session = store
            .create(&SessionOpen::new(dir.path(), "p", "p", "m"))
            .unwrap();
        let records = [
            SessionRecord::Meta(SessionMeta::now(&SessionOpen::new(
                dir.path(),
                "p2",
                "k2",
                "m2",
            ))),
            SessionRecord::Message(text_message(Role::Assistant, "hi")),
            SessionRecord::Approval {
                tool: "bash".into(),
                mode: ApprovalMode::Auto,
                decision: ApprovalDecision::Allow,
                tool_use_id: Some("call_3".into()),
                approved: None,
                answer: None,
            },
            SessionRecord::Usage {
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 2,
                    cache_read_tokens: 3,
                    cache_creation_tokens: 4,
                },
                provider: None,
                model: None,
                at: Some("2026-09-01T10:00:00.123Z".into()),
                duration_ms: Some(7),
            },
            SessionRecord::Tool {
                call_id: "call_3".into(),
                name: "read".into(),
                at: "2026-09-01T10:00:00.130Z".into(),
                duration_ms: 1,
                is_error: true,
                output_bytes: 9,
                truncated: true,
            },
            SessionRecord::Turn {
                outcome: "completed".into(),
                error: None,
                at: Some("2026-09-01T10:00:00.000Z".into()),
                duration_ms: Some(140),
            },
            SessionRecord::Turn {
                outcome: "error".into(),
                error: Some("server error: 503".into()),
                at: Some("2026-09-01T10:00:01.000Z".into()),
                duration_ms: Some(3),
            },
            // The clock fields absent: the mirror must elide them the
            // same way, or an old-shape record would not round-trip.
            SessionRecord::Usage {
                usage: Usage::default(),
                provider: None,
                model: None,
                at: None,
                duration_ms: None,
            },
            SessionRecord::Turn {
                outcome: "aborted".into(),
                error: None,
                at: None,
                duration_ms: None,
            },
            SessionRecord::Judgment(full_judgment()),
            // The optional fields absent.
            SessionRecord::Judgment(JudgmentRecord::new("turn_end", "judge")),
        ];
        for record in &records {
            session.append(record).unwrap();
        }
        let text = std::fs::read_to_string(session.path()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        // Line 0 is the create-time meta; the appended records follow.
        for (line, record) in lines[1..].iter().zip(&records) {
            assert_eq!(*line, serde_json::to_string(record).unwrap());
        }
    }

    /// A `judgment` record with every field set.
    fn full_judgment() -> JudgmentRecord {
        let mut record = JudgmentRecord::new("tool_after", "judge");
        record.call_id = Some("call_3".into());
        record.tool = Some("bash".into());
        record.checks.insert("outcome".into(), "log".into());
        record.checks.insert("progress".into(), "log".into());
        record.rules.insert(
            "rule.0".into(),
            JudgmentRuleRecord::new("warn", "user", 0.8),
        );
        record.fired.insert("rule.0".into(), "warn".into());
        record.model = Some("judge-1.0.0".into());
        record.answers.insert(
            "outcome".into(),
            serde_json::to_value(crate::judge::Answer::yes_no(0.875)).unwrap(),
        );
        record.answers.insert(
            "progress".into(),
            serde_json::json!({"type": "yes_no", "yes": 0.5}),
        );
        record.state_bytes = 1234;
        record.state_sha256 = "ab".repeat(32);
        record.at = "2026-09-01T10:00:00.123Z".into();
        record.duration_ms = 210;
        record.usage = Some(JudgeUsage::new(400, 9));
        record.error = Some("timeout".into());
        record
    }

    #[test]
    fn a_judgment_record_has_its_field_names_and_round_trips() {
        let record = SessionRecord::Judgment(full_judgment());
        let value = serde_json::to_value(&record).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "type": "judgment",
                "event": "tool_after",
                "call_id": "call_3",
                "tool": "bash",
                "checks": {"outcome": "log", "progress": "log"},
                "rules": {"rule.0": {"action": "warn", "source": "user", "threshold": 0.8}},
                "fired": {"rule.0": "warn"},
                "instance": "judge",
                "model": "judge-1.0.0",
                "answers": {
                    "outcome": {"type": "yes_no", "yes": 0.875},
                    "progress": {"type": "yes_no", "yes": 0.5}
                },
                "state_bytes": 1234,
                "state_sha256": "ab".repeat(32),
                "at": "2026-09-01T10:00:00.123Z",
                "duration_ms": 210,
                "usage": {"input_tokens": 400, "output_tokens": 9},
                "error": "timeout"
            })
        );
        assert_eq!(
            serde_json::from_value::<SessionRecord>(value).unwrap(),
            record
        );

        // The optional fields stay out when empty, and come back empty.
        let sparse = SessionRecord::Judgment(JudgmentRecord::new("turn_end", "judge"));
        let value = serde_json::to_value(&sparse).unwrap();
        for absent in [
            "call_id", "tool", "checks", "rules", "fired", "model", "answers", "usage", "error",
        ] {
            assert!(value.get(absent).is_none(), "{absent}: {value}");
        }
        assert_eq!(
            serde_json::from_value::<SessionRecord>(value).unwrap(),
            sparse
        );

        // Pre-8b records without `rules` or `fired` still deserialize cleanly.
        let pre_8b = serde_json::json!({
            "type": "judgment",
            "event": "turn_end",
            "checks": {"claim": "log"},
            "instance": "judge",
            "state_bytes": 10,
            "state_sha256": "00".repeat(32),
            "at": "2026-11-15T10:00:00.000Z",
            "duration_ms": 50
        });
        let SessionRecord::Judgment(parsed) = serde_json::from_value(pre_8b).unwrap() else {
            panic!("expected judgment");
        };
        assert!(parsed.rules.is_empty());
        assert!(parsed.fired.is_empty());
    }

    #[test]
    fn replay_and_resume_skip_judgment_records() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let mut session = store
            .create(&SessionOpen::new(dir.path(), "p", "p", "m"))
            .unwrap();
        session
            .append_message(&text_message(Role::User, "hi"))
            .unwrap();
        session.writer().append_judgment(&full_judgment()).unwrap();
        session
            .append_message(&text_message(Role::Assistant, "hello"))
            .unwrap();
        let path = session.path().to_path_buf();
        let replayed = replay(&path).unwrap();
        assert_eq!(replayed.messages.len(), 2);
        assert!(replayed.usage.is_empty());
        assert!(replayed.tools.is_empty());
        drop(session);

        let (mut resumed, replayed) = Session::resume(&path).unwrap();
        assert_eq!(replayed.messages.len(), 2);
        resumed.writer().append_judgment(&full_judgment()).unwrap();
        resumed
            .append_message(&text_message(Role::User, "more"))
            .unwrap();
        assert_eq!(replay(&path).unwrap().messages.len(), 3);
    }

    #[test]
    fn a_writer_clone_appends_after_the_session_drops() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let mut session = store
            .create(&SessionOpen::new(dir.path(), "p", "p", "m"))
            .unwrap();
        let path = session.path().to_path_buf();
        let writer = session.writer().clone();
        session
            .append_message(&text_message(Role::User, "hi"))
            .unwrap();
        drop(session);
        writer.append_judgment(&full_judgment()).unwrap();
        drop(writer);

        let text = std::fs::read_to_string(&path).unwrap();
        let last: SessionRecord = serde_json::from_str(text.lines().last().unwrap()).unwrap();
        assert_eq!(last, SessionRecord::Judgment(full_judgment()));
        assert_eq!(replay(&path).unwrap().messages.len(), 1);
    }

    #[test]
    fn concurrent_appends_from_writer_clones_and_the_session_stay_whole_lines() {
        const EACH: usize = 100;
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let mut session = store
            .create(&SessionOpen::new(dir.path(), "p", "p", "m"))
            .unwrap();
        // Long lines, so an interleaved write would tear one.
        let mut record = full_judgment();
        record.error = Some("x".repeat(32 * 1024));
        let writers = [session.writer(), session.writer().clone()];
        std::thread::scope(|scope| {
            for (index, writer) in writers.iter().enumerate() {
                let mut record = record.clone();
                record.instance = format!("judge-{index}");
                scope.spawn(move || {
                    for _ in 0..EACH {
                        writer.append_judgment(&record).unwrap();
                    }
                });
            }
            let text = "y".repeat(16 * 1024);
            for _ in 0..EACH {
                session
                    .append_message(&text_message(Role::User, &text))
                    .unwrap();
            }
        });

        let text = std::fs::read_to_string(session.path()).unwrap();
        let mut judgments = [0usize; 2];
        let mut messages = 0;
        for line in text.lines() {
            match serde_json::from_str::<SessionRecord>(line).unwrap() {
                SessionRecord::Judgment(judgment) => {
                    let index = usize::from(judgment.instance == "judge-1");
                    judgments[index] += 1;
                }
                SessionRecord::Message(_) => messages += 1,
                _ => {}
            }
        }
        assert_eq!(judgments, [EACH, EACH]);
        assert_eq!(messages, EACH);
        assert_eq!(replay(session.path()).unwrap().messages.len(), EACH);
    }

    #[test]
    fn a_judgment_record_that_does_not_parse_still_replays_and_resumes() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let project = dir.path().to_path_buf();
        let mut session = store
            .create(&SessionOpen::new(&project, "p", "p", "m"))
            .unwrap();
        session
            .append_message(&text_message(Role::User, "hi"))
            .unwrap();
        let full = serde_json::to_value(RecordRef::Judgment(&full_judgment())).unwrap();
        // One record without a required field, one with a field of
        // another type.
        let mut missing = full.clone();
        missing.as_object_mut().unwrap().remove("state_sha256");
        let mut retyped = full;
        retyped["state_bytes"] = serde_json::json!("many");
        use std::io::Write as _;
        let mut file = OpenOptions::new()
            .append(true)
            .open(session.path())
            .unwrap();
        writeln!(file, "{missing}").unwrap();
        writeln!(file, "{retyped}").unwrap();
        drop(file);
        session
            .append_message(&text_message(Role::Assistant, "hello"))
            .unwrap();
        let path = session.path().to_path_buf();
        drop(session);

        assert_eq!(replay(&path).unwrap().messages.len(), 2);
        let (mut resumed, replayed) = Session::resume(&path).unwrap();
        assert_eq!(replayed.messages.len(), 2);
        resumed
            .append_message(&text_message(Role::User, "again"))
            .unwrap();
        assert_eq!(replay(resumed.path()).unwrap().messages.len(), 3);

        // Any other record that does not parse still stops the replay.
        let mut file = OpenOptions::new()
            .append(true)
            .open(resumed.path())
            .unwrap();
        writeln!(file, r#"{{"type":"message","role":7}}"#).unwrap();
        drop(file);
        assert!(replay(resumed.path()).is_err());
    }

    /// A file that writes part of a line, then fails, as a full disk
    /// does.
    struct FullDisk {
        bytes: Vec<u8>,
        room: usize,
        /// Whether a cut fails too.
        stuck: bool,
    }

    impl LineFile for FullDisk {
        fn length(&mut self) -> io::Result<u64> {
            Ok(self.bytes.len() as u64)
        }

        fn write_some(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.room == 0 {
                return Err(io::Error::new(io::ErrorKind::StorageFull, "no space"));
            }
            let fits = bytes.len().min(self.room);
            self.bytes.extend_from_slice(&bytes[..fits]);
            self.room -= fits;
            Ok(fits)
        }

        fn cut_to(&mut self, length: u64) -> io::Result<()> {
            if self.stuck {
                return Err(io::Error::other("read-only"));
            }
            self.bytes.truncate(length as usize);
            Ok(())
        }
    }

    #[test]
    fn a_failed_append_leaves_no_partial_line() {
        let line = b"{\"type\":\"turn\"}\n";
        let mut file = Appender::new(FullDisk {
            bytes: Vec::new(),
            room: 30,
            stuck: false,
        });
        file.append(line).unwrap();
        let error = file
            .append(b"{\"type\":\"judgment\",\"x\":1}\n")
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::StorageFull);
        // The file holds the first line whole and nothing of the second.
        assert_eq!(file.file.bytes, line);
        file.file.room = 100;
        file.append(line).unwrap();
        assert_eq!(file.file.bytes, [&line[..], &line[..]].concat());
    }

    /// A file whose writes take a few bytes each and are interrupted
    /// once.
    struct Choppy {
        bytes: Vec<u8>,
        interrupted: bool,
    }

    impl LineFile for Choppy {
        fn write_some(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(io::ErrorKind::Interrupted.into());
            }
            let fits = bytes.len().min(3);
            self.bytes.extend_from_slice(&bytes[..fits]);
            Ok(fits)
        }

        fn length(&mut self) -> io::Result<u64> {
            unreachable!("a write that succeeds reads no length")
        }

        fn cut_to(&mut self, _: u64) -> io::Result<()> {
            unreachable!("a write that succeeds cuts nothing")
        }
    }

    #[test]
    fn a_short_or_interrupted_write_goes_on_to_the_whole_line() {
        let line = b"{\"type\":\"turn\"}\n";
        let mut file = Appender::new(Choppy {
            bytes: Vec::new(),
            interrupted: false,
        });
        file.append(line).unwrap();
        assert_eq!(file.file.bytes, line);
    }

    #[test]
    fn a_fragment_that_cannot_be_cut_stops_every_later_append() {
        let line = b"{\"type\":\"turn\"}\n";
        let mut file = Appender::new(FullDisk {
            bytes: Vec::new(),
            room: 20,
            stuck: true,
        });
        file.append(line).unwrap();
        assert!(file.append(line).is_err());
        // The fragment stays, and no later line joins it.
        let torn = file.file.bytes.clone();
        assert_eq!(torn.len(), 20);
        file.file.room = 100;
        assert!(file.append(line).is_err());
        assert_eq!(file.file.bytes, torn);
    }

    #[test]
    fn unknown_record_types_are_skipped_on_replay() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let project = dir.path().to_path_buf();
        let mut session = store
            .create(&SessionOpen::new(&project, "p", "p", "m"))
            .unwrap();
        session
            .append_message(&text_message(Role::User, "hi"))
            .unwrap();
        // A record type from a future schema version.
        use std::io::Write as _;
        let mut file = OpenOptions::new()
            .append(true)
            .open(session.path())
            .unwrap();
        writeln!(file, r#"{{"type":"hologram","data":42}}"#).unwrap();
        drop(file);
        session.writer().append_judgment(&full_judgment()).unwrap();

        let replayed = replay(session.path()).unwrap();
        assert_eq!(replayed.messages.len(), 1);

        // A reader from before the `judgment` record skips it through
        // its unknown arm, nested members and all.
        #[derive(Deserialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum OldRecord {
            Meta(SessionMeta),
            Message(Message),
            #[serde(other)]
            Unknown,
        }
        let text = std::fs::read_to_string(session.path()).unwrap();
        let old: Vec<OldRecord> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(matches!(&old[0], OldRecord::Meta(meta) if meta.provider == "p"));
        assert!(matches!(&old[1], OldRecord::Message(message) if message.role == Role::User));
        assert!(matches!(old[2], OldRecord::Unknown));
        assert!(matches!(old[3], OldRecord::Unknown));
        assert!(
            text.lines()
                .last()
                .unwrap()
                .contains("\"type\":\"judgment\"")
        );
    }

    #[test]
    fn records_from_before_the_clock_fields_still_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.jsonl");
        // The shapes a build before the clock fields wrote: usage
        // with the counts alone, and a turn with its outcome alone.
        std::fs::write(
            &path,
            concat!(
                "{\"type\":\"meta\",\"schema\":1,\"cwd\":\"/x\",\"provider\":\"p\",\"model\":\"m\",\"created\":\"2026-08-07T00:00:00Z\"}\n",
                "{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}\n",
                "{\"type\":\"usage\",\"input_tokens\":10,\"output_tokens\":2,\"cache_read_tokens\":1,\"cache_creation_tokens\":0}\n",
                "{\"type\":\"turn\",\"outcome\":\"completed\"}\n",
            ),
        )
        .unwrap();
        let replayed = replay(&path).unwrap();
        assert_eq!(replayed.messages.len(), 1);
        // The usage record replays without a clock.
        assert_eq!(replayed.usage.len(), 1);
        assert_eq!(replayed.usage[0].messages_before, 1);
        assert_eq!(replayed.usage[0].timing, None);

        // The old records parse into the current variants with the
        // clock fields absent, so a reader of the typed records sees
        // one shape.
        let usage: SessionRecord = serde_json::from_str(
            "{\"type\":\"usage\",\"input_tokens\":10,\"output_tokens\":2,\"cache_read_tokens\":1,\"cache_creation_tokens\":0}",
        )
        .unwrap();
        assert_eq!(
            usage,
            SessionRecord::Usage {
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 2,
                    cache_read_tokens: 1,
                    cache_creation_tokens: 0,
                },
                provider: None,
                model: None,
                at: None,
                duration_ms: None,
            }
        );
        let turn: SessionRecord =
            serde_json::from_str("{\"type\":\"turn\",\"outcome\":\"completed\"}").unwrap();
        assert_eq!(
            turn,
            SessionRecord::Turn {
                outcome: "completed".into(),
                error: None,
                at: None,
                duration_ms: None,
            }
        );
        // A usage record from before the cache-write count still
        // reads: the counts default, the clock stays absent.
        let sparse: SessionRecord =
            serde_json::from_str("{\"type\":\"usage\",\"input_tokens\":3}").unwrap();
        assert!(matches!(
            sparse,
            SessionRecord::Usage {
                usage: Usage {
                    input_tokens: 3,
                    cache_creation_tokens: 0,
                    ..
                },
                provider: None,
                model: None,
                at: None,
                duration_ms: None,
            }
        ));
    }

    #[test]
    fn the_clock_fields_round_trip_and_the_timestamp_is_rfc3339_millis() {
        let at = UNIX_EPOCH + Duration::from_millis(1_788_256_800_123);
        assert_eq!(timestamp(at), "2026-09-01T10:00:00.123Z");
        let timing = Timing::new(at, Duration::from_micros(1_500_400));
        assert_eq!(timing.duration_ms(), 1500);

        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let mut session = store
            .create(&SessionOpen::new(dir.path(), "p", "p", "m"))
            .unwrap();
        session
            .append_usage(&Usage::default(), &Target::new("p", "m"), timing)
            .unwrap();
        session
            .append_tool("call_1", "read", timing, false, 12, false)
            .unwrap();
        session.append_turn("completed", None, timing).unwrap();
        let text = std::fs::read_to_string(session.path()).unwrap();
        let records: Vec<SessionRecord> = text
            .lines()
            .skip(1)
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            records,
            [
                SessionRecord::Usage {
                    usage: Usage::default(),
                    provider: Some("p".into()),
                    model: Some("m".into()),
                    at: Some("2026-09-01T10:00:00.123Z".into()),
                    duration_ms: Some(1500),
                },
                SessionRecord::Tool {
                    call_id: "call_1".into(),
                    name: "read".into(),
                    at: "2026-09-01T10:00:00.123Z".into(),
                    duration_ms: 1500,
                    is_error: false,
                    output_bytes: 12,
                    truncated: false,
                },
                SessionRecord::Turn {
                    outcome: "completed".into(),
                    error: None,
                    at: Some("2026-09-01T10:00:00.123Z".into()),
                    duration_ms: Some(1500),
                },
            ]
        );
    }

    #[test]
    fn malformed_and_untyped_records_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        std::fs::write(&path, "{\"type\":\"message\",\"role\":\"nobody\"}\n").unwrap();
        let error = replay(&path).unwrap_err();
        assert!(error.to_string().contains("bad message record"));

        std::fs::write(&path, "{\"role\":\"user\"}\n").unwrap();
        let error = replay(&path).unwrap_err();
        assert!(error.to_string().contains("bad session record"));
    }

    #[test]
    fn a_newer_schema_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.jsonl");
        std::fs::write(
            &path,
            format!(
                "{{\"type\":\"meta\",\"schema\":{},\"cwd\":\"/x\",\"provider\":\"p\",\"model\":\"m\",\"created\":\"2026-08-07T00:00:00Z\"}}\n",
                SCHEMA_VERSION + 1
            ),
        )
        .unwrap();
        let error = replay(&path).unwrap_err();
        assert!(error.to_string().contains("unsupported session schema"));
    }

    #[test]
    fn latest_returns_the_newest_session() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let project = dir.path().to_path_buf();
        assert!(store.latest(&project).unwrap().is_none());

        let first = store
            .create(&SessionOpen::new(&project, "p", "p", "m"))
            .unwrap();
        let second = store
            .create(&SessionOpen::new(&project, "p", "p", "m"))
            .unwrap();
        let latest = store.latest(&project).unwrap().unwrap();
        assert_eq!(latest, second.path());
        assert_ne!(first.path(), second.path());
    }

    #[test]
    fn find_names_one_session_by_id_and_refuses_every_other_shape() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let project = dir.path().to_path_buf();
        assert!(
            store
                .find(&project, "01ARZ3NDEKTSV4RRFFQ69G5FAV")
                .unwrap()
                .is_none()
        );

        let session = store
            .create(&SessionOpen::new(&project, "p", "p", "m"))
            .unwrap();
        let id = session.id().to_string();
        let found = store.find(&project, &id).unwrap().unwrap();
        assert_eq!(found, session.path());
        // Not a session id: no file, whatever the path would name.
        // The planted file sits three levels above the sessions
        // directory, where a path-shaped id would reach.
        std::fs::write(dir.path().join("outside.jsonl"), "").unwrap();
        let lowercase = id.to_ascii_lowercase();
        let short = &id[..SESSION_ID_LEN - 1];
        let long = format!("{id}0");
        for other in [
            "",
            "../../../outside",
            "../outside",
            ".",
            "a/b",
            "01ARZ3.jsonl",
            lowercase.as_str(),
            short,
            long.as_str(),
        ] {
            assert!(store.find(&project, other).unwrap().is_none(), "{other:?}");
        }
    }

    #[test]
    fn delete_means_nothing_remains() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::at(dir.path());
        let project = dir.path().to_path_buf();
        let session = store
            .create(&SessionOpen::new(&project, "p", "p", "m"))
            .unwrap();
        let path = session.path().to_path_buf();
        drop(session);

        std::fs::remove_file(&path).unwrap();
        assert!(store.latest(&project).unwrap().is_none());
        // The only artifact of a session is its one file.
        let dir_entries: Vec<_> = std::fs::read_dir(store.sessions_dir(&project))
            .unwrap()
            .collect();
        assert!(dir_entries.is_empty());
    }

    #[test]
    fn session_ids_are_unique_and_strictly_increasing() {
        let ids: Vec<String> = (0..64).map(|_| new_session_id()).collect();
        for pair in ids.windows(2) {
            assert!(pair[0] < pair[1], "{} !< {}", pair[0], pair[1]);
        }
        for id in &ids {
            assert_eq!(id.len(), 26);
            assert!(id.bytes().all(|b| ULID_ALPHABET.contains(&b)));
        }
    }

    #[test]
    fn project_ids_are_readable_safe_and_collision_free() {
        let id = project_id(Path::new("/does/not/exist/my project"));
        assert!(id.starts_with("-does-not-exist-my-project-"));
        assert!(!id.contains('/'));

        // Paths that fold to the same stem stay distinct.
        let a = project_id(Path::new("/does/not/exist/a-b/x"));
        let b = project_id(Path::new("/does/not/exist/a/b/x"));
        assert_ne!(a, b);

        // A deep path stays inside one filename component.
        let deep = "/does/not/exist/".to_string() + &"d/".repeat(200);
        let id = project_id(Path::new(&deep));
        assert!(id.len() <= 96 + 1 + 8);
    }
}
