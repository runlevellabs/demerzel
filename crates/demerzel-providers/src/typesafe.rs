//! The `typesafe` judge kind: the System One API, which answers typed
//! questions about one state value with probabilities.
//!
//! One request carries the state and every question of one loop
//! event, keyed by the caller's ids. A yes-or-no question goes on the
//! wire as the type `noul`. The reply holds one answer per id, the
//! model id, and the token counts. The parser ignores unknown fields
//! at every level, so additive server changes never break a check.
//!
//! The whole request, its retries included, stays inside the
//! request's timeout. Status 429, 529, another 5xx, and a network
//! failure retry with backoff, or after the server's `retry-after`;
//! a wait past the time left fails at once. Every other status fails
//! on the first attempt. A kind of this module serves only as a
//! judge: no session sends model turns to it.

use std::collections::BTreeMap;
use std::time::Duration;

use demerzel_core::{
    Answer, Judge, JudgeError, JudgeReply, JudgeRequest, JudgeUsage, Question, check_reply,
};
use serde::ser::{Error as _, SerializeMap};
use serde::{Deserialize, Serialize, Serializer};
use tokio::time::Instant;

use crate::retry::{backoff, redact, request_id, retry_after, retry_delay};
use crate::{BoxFuture, KeyCheck, ProviderError};

/// Stable backend id: the `kind` of a judge instance.
pub const PROVIDER_ID: &str = "typesafe";

/// Default API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";

/// Default model: an alias, which can move to a new version.
pub const DEFAULT_MODEL: &str = "jev-latest";

/// The request route under the base URL.
const ROUTE: &str = "/v1/systemone";

/// The key-check route under the base URL.
const MODELS_ROUTE: &str = "/v1/models";

/// The most reply bytes the judge reads. The replies to the harness's
/// questions are a few hundred bytes; a wrong endpoint can answer
/// with a page of any size.
const MAX_REPLY_BYTES: usize = 1 << 20;

/// The most bytes of a server's text that an error carries.
const ERROR_TEXT_BYTES: usize = 1024;

/// The most bytes of a reply that is not JSON that an error carries.
const REPLY_HEAD_BYTES: usize = 512;

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TypeSafeConfig {
    /// The instance name, for the error text.
    pub instance: String,
    /// The API root, without the route.
    pub base_url: String,
    /// The model id or alias every request names.
    pub model: String,
    /// Total attempts for retryable failures (429, 5xx, network).
    pub max_attempts: u32,
    pub base_delay: Duration,
}

impl TypeSafeConfig {
    /// The config for instance `instance` at `base_url` with `model`,
    /// with the default retry policy. A trailing `/` on the URL is
    /// dropped.
    pub fn new(
        instance: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let mut base_url = base_url.into();
        while base_url.ends_with('/') {
            base_url.pop();
        }
        Self {
            instance: instance.into(),
            base_url,
            model: model.into(),
            max_attempts: 4,
            base_delay: Duration::from_millis(250),
        }
    }
}

/// A judge on one System One endpoint with one key.
pub struct TypeSafeJudge {
    config: TypeSafeConfig,
    http: reqwest::Client,
    /// The full request URL.
    url: String,
    api_key: String,
}

/// Idle connections a judge keeps to its endpoint: as many as the
/// harness lets run at once (8), so a steady burst reuses them.
const IDLE_CONNECTIONS: usize = 8;
/// How long an idle connection stays open.
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);

impl TypeSafeJudge {
    pub fn new(config: TypeSafeConfig, api_key: String) -> Self {
        let url = format!("{}{ROUTE}", config.base_url.trim_end_matches('/'));
        // A builder with only pool settings does not fail; the default
        // client is the fallback all the same.
        let http = reqwest::Client::builder()
            .pool_max_idle_per_host(IDLE_CONNECTIONS)
            .pool_idle_timeout(IDLE_TIMEOUT)
            .build()
            .unwrap_or_default();
        Self {
            config,
            http,
            url,
            api_key,
        }
    }

    /// POST `body` until an answer, a failure that no retry mends, or
    /// the end of the budget. Returns the reply bytes of a 2xx status.
    async fn post(&self, body: Vec<u8>, timeout: Duration) -> Result<Vec<u8>, JudgeError> {
        let deadline = Instant::now() + timeout;
        let budget = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        let instance = self.config.instance.as_str();
        let secret = self.api_key.as_str();
        let max_attempts = self.config.max_attempts.max(1);
        log::debug!("request POST {ROUTE} body_bytes={}", body.len());
        log::trace!("request body {}", String::from_utf8_lossy(&body));
        // A URL that does not parse, or a key that is no header value,
        // fails here, once: no retry mends it.
        let request = self
            .http
            .post(&self.url)
            .bearer_auth(secret)
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .body(body)
            .build()
            .map_err(|error| {
                ProviderError::InvalidRequest(redact(
                    &format!(
                        "the request of instance `{instance}` to `{}` does not build: {error}; \
                         check its base_url and key",
                        self.url
                    ),
                    secret,
                ))
            })?;
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(JudgeError::Timeout(budget));
            }
            let attempt_request = request.try_clone().expect("a byte body clones");
            let outcome =
                match tokio::time::timeout(left, self.attempt(attempt_request, attempt)).await {
                    Ok(outcome) => outcome,
                    Err(_) => {
                        log::warn!(
                            "judge request timed out after {budget}ms (attempt {attempt}) \
                             instance={instance}"
                        );
                        return Err(JudgeError::Timeout(budget));
                    }
                };
            let (error, delay) = match outcome {
                Attempt::Answered(bytes) => return Ok(bytes),
                Attempt::Failed(error) => return Err(error.into()),
                Attempt::Retry { error, delay } => (error, delay),
            };
            if attempt >= max_attempts {
                log::error!(
                    "judge request failed after {attempt} of {max_attempts} attempts: {error} \
                     instance={instance}"
                );
                return Err(error.into());
            }
            // A wait that ends at or past the deadline leaves no time
            // for the next attempt: fail now with what the last one
            // found.
            if delay >= deadline.saturating_duration_since(Instant::now()) {
                log::warn!(
                    "judge request failed: the {}ms wait after {error} is past the budget \
                     instance={instance}",
                    delay.as_millis()
                );
                return Err(error.into());
            }
            log::info!(
                "retry in {}ms after {error} (attempt {attempt} of {max_attempts}) \
                 instance={instance}",
                delay.as_millis()
            );
            tokio::time::sleep(delay).await;
        }
    }

    /// One attempt: send, then read the reply or the error text.
    async fn attempt(&self, request: reqwest::Request, attempt: u32) -> Attempt {
        let instance = self.config.instance.as_str();
        let secret = self.api_key.as_str();
        let base_delay = self.config.base_delay;
        let response = match self.http.execute(request).await {
            Ok(response) => response,
            Err(error) => {
                return Attempt::Retry {
                    error: ProviderError::Network(redact(&error.to_string(), secret)),
                    delay: backoff(base_delay, attempt),
                };
            }
        };
        let status = response.status();
        log::info!(
            "POST {ROUTE} attempt {attempt}: status {} request-id={} instance={instance}",
            status.as_u16(),
            request_id(&response)
        );
        if status.is_success() {
            return match read_body(response, MAX_REPLY_BYTES).await {
                Ok((bytes, false)) => Attempt::Answered(bytes),
                Ok((_, true)) => Attempt::Failed(ProviderError::Protocol(format!(
                    "the endpoint of instance `{instance}` answered with more than \
                     {MAX_REPLY_BYTES} bytes; check its base_url"
                ))),
                // The reply broke off: the same as a failed send.
                Err(error) => Attempt::Retry {
                    error: ProviderError::Network(redact(&error.to_string(), secret)),
                    delay: backoff(base_delay, attempt),
                },
            };
        }
        match status.as_u16() {
            // A key has no refresh flow: a 401 fails at once and
            // points at the key.
            401 => Attempt::Failed(ProviderError::Auth(format!(
                "the endpoint of instance `{instance}` rejected the key (status 401): check its \
                 `api_key_env`, or run `dmz login {instance}`"
            ))),
            429 => Attempt::Retry {
                error: ProviderError::RateLimited {
                    retry_after: retry_after(&response),
                },
                delay: retry_delay(&response, base_delay, attempt),
            },
            // 529 is a 5xx: the server is overloaded.
            _ if status.is_server_error() => Attempt::Retry {
                error: ProviderError::Server(format!("status {status}")),
                delay: retry_delay(&response, base_delay, attempt),
            },
            // 400, 408, 413, 415, 422, and any other status: the same
            // request fails the same way again.
            _ => {
                let text = server_text(response, ERROR_TEXT_BYTES, secret).await;
                Attempt::Failed(ProviderError::InvalidRequest(format!(
                    "status {status}: {text}"
                )))
            }
        }
    }
}

impl Judge for TypeSafeJudge {
    fn id(&self) -> &'static str {
        PROVIDER_ID
    }

    fn ask(&self, request: JudgeRequest) -> BoxFuture<'_, Result<JudgeReply, JudgeError>> {
        // Serialize once, before the round trip; each attempt shares
        // the bytes.
        let body = serde_json::to_vec(&WireRequest {
            model: &self.config.model,
            request: &request,
        });
        Box::pin(async move {
            let body = body.map_err(|error| {
                ProviderError::InvalidRequest(format!("the judge request does not build: {error}"))
            })?;
            let bytes = self.post(body, request.timeout).await?;
            let reply = read_reply(&bytes, &self.api_key)?;
            check_reply(&request, &reply).map_err(JudgeError::Reply)?;
            Ok(reply)
        })
    }
}

/// What one attempt found.
enum Attempt {
    /// A 2xx reply, read whole.
    Answered(Vec<u8>),
    /// A failure that the next attempt can mend, after `delay`.
    Retry {
        error: ProviderError,
        delay: Duration,
    },
    /// A failure that no retry mends.
    Failed(ProviderError),
}

/// Read at most `cap` body bytes. The flag says the body holds more.
async fn read_body(
    mut response: reqwest::Response,
    cap: usize,
) -> Result<(Vec<u8>, bool), reqwest::Error> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len() + chunk.len() > cap {
            body.extend_from_slice(&chunk[..cap - body.len()]);
            return Ok((body, true));
        }
        body.extend_from_slice(&chunk);
    }
    Ok((body, false))
}

/// `bytes` as text for an error: the secret redacted, then cut at
/// `cap` bytes on a character boundary. The redaction runs first, so
/// the cut can never leave part of the secret.
fn error_text(bytes: &[u8], cap: usize, secret: &str) -> String {
    let text = redact(&String::from_utf8_lossy(bytes), secret);
    if text.len() <= cap {
        return text;
    }
    let mut text = text;
    text.truncate(text.floor_char_boundary(cap));
    text
}

/// The start of a server's error text, at most `cap` bytes, redacted.
/// Reads `cap` plus the secret's length, so a secret that straddles
/// the cut is still whole when the redaction runs.
async fn server_text(response: reqwest::Response, cap: usize, secret: &str) -> String {
    match read_body(response, cap + secret.len()).await {
        Ok((bytes, _)) => error_text(&bytes, cap, secret),
        Err(_) => String::new(),
    }
}

/// The request body, serialized from the borrowed request: the
/// options of a choice keep the caller's order.
struct WireRequest<'a> {
    model: &'a str,
    request: &'a JudgeRequest,
}

impl Serialize for WireRequest<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("model", self.model)?;
        map.serialize_entry("state", &self.request.state)?;
        map.serialize_entry("questions", &WireQuestions(&self.request.questions))?;
        map.end()
    }
}

struct WireQuestions<'a>(&'a BTreeMap<String, Question>);

impl Serialize for WireQuestions<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (id, question) in self.0 {
            map.serialize_entry(id, &WireQuestion(question))?;
        }
        map.end()
    }
}

/// One question in the wire shape of its type.
struct WireQuestion<'a>(&'a Question);

impl Serialize for WireQuestion<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Question::YesNo {
                instructions,
                yes,
                no,
                ..
            } => {
                let criteria = yes.is_some() || no.is_some();
                let mut map = serializer.serialize_map(Some(2 + usize::from(criteria)))?;
                map.serialize_entry("type", "noul")?;
                map.serialize_entry("instructions", instructions)?;
                if criteria {
                    map.serialize_entry(
                        "criteria",
                        &YesNoCriteria {
                            yes: yes.as_deref(),
                            no: no.as_deref(),
                        },
                    )?;
                }
                map.end()
            }
            Question::Choice {
                instructions,
                options,
                ..
            } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "choice")?;
                map.serialize_entry("instructions", instructions)?;
                map.serialize_entry("criteria", &ChoiceCriteria(options))?;
                map.end()
            }
            Question::Score {
                instructions,
                levels,
                ..
            } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "score")?;
                map.serialize_entry("instructions", instructions)?;
                map.serialize_entry("criteria", levels)?;
                map.end()
            }
            _ => Err(S::Error::custom(
                "the request holds a question type this judge does not take",
            )),
        }
    }
}

/// The criteria of a yes-or-no question: an unset side stays off the
/// wire.
#[derive(Serialize)]
struct YesNoCriteria<'a> {
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    yes: Option<&'a str>,
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    no: Option<&'a str>,
}

/// The options of a choice, in order: name to description, or `null`.
struct ChoiceCriteria<'a>(&'a [(String, Option<String>)]);

impl Serialize for ChoiceCriteria<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (name, description) in self.0 {
            map.serialize_entry(name, description)?;
        }
        map.end()
    }
}

/// The reply shape. Serde ignores the fields it does not name.
#[derive(Deserialize)]
struct WireReply {
    model: String,
    answers: BTreeMap<String, WireAnswer>,
    usage: WireUsage,
}

#[derive(Deserialize)]
struct WireUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireAnswer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    /// `legend` repeats the request's levels; the judge does not read
    /// it.
    Score {
        score: f64,
        /// Keyed by the level index as text, from `"0"`.
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    /// An answer type this judge does not read.
    #[serde(other)]
    Other,
}

/// Map the reply bytes to the core reply. Bytes that are not JSON are
/// a protocol error with their start; JSON that does not fit the
/// answer shape is a reply error.
fn read_reply(bytes: &[u8], secret: &str) -> Result<JudgeReply, JudgeError> {
    let wire: WireReply = serde_json::from_slice(bytes).map_err(|error| {
        if error.is_data() {
            JudgeError::Reply(redact(
                &format!("the reply does not fit the answer shape: {error}"),
                secret,
            ))
        } else {
            JudgeError::Transport(ProviderError::Protocol(format!(
                "the reply is not JSON ({error}); check the base_url: {}",
                error_text(bytes, REPLY_HEAD_BYTES, secret)
            )))
        }
    })?;
    let usage = JudgeUsage::new(wire.usage.input_tokens, wire.usage.output_tokens);
    let mut reply = JudgeReply::new(wire.model, usage);
    for (id, answer) in wire.answers {
        let answer = match answer {
            WireAnswer::Noul { noul } => Answer::yes_no(noul),
            WireAnswer::Choice {
                choice,
                probabilities,
                confidence,
            } => Answer::choice(choice, probabilities, Some(confidence)),
            WireAnswer::Score {
                score,
                probabilities,
                confidence,
            } => Answer::score(
                score,
                level_probabilities(&id, &probabilities)?,
                Some(confidence),
            ),
            WireAnswer::Other => {
                return Err(JudgeError::Reply(format!(
                    "the answer to `{id}` has a type this judge does not read"
                )));
            }
        };
        reply.answers.insert(id, answer);
    }
    Ok(reply)
}

/// The probabilities of a score by level index: one for each index
/// from 0 to the count less one, or an error that names the gap.
fn level_probabilities(id: &str, map: &BTreeMap<String, f64>) -> Result<Vec<f64>, JudgeError> {
    (0..map.len())
        .map(|level| {
            map.get(level.to_string().as_str()).copied().ok_or_else(|| {
                JudgeError::Reply(format!(
                    "the answer to `{id}` has no probability for level {level}"
                ))
            })
        })
        .collect()
}

/// Whether `model` is an alias: an id without a `-N.N.N` version
/// suffix. An alias can move to a new version, and its answers can
/// change with it.
pub fn is_alias(model: &str) -> bool {
    let Some((name, version)) = model.rsplit_once('-') else {
        return true;
    };
    let mut parts = 0;
    let numeric = version.split('.').all(|part| {
        parts += 1;
        !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())
    });
    name.is_empty() || !numeric || parts != 3
}

/// Check a key with `GET {base_url}/v1/models`. Only a 2xx reply
/// accepts it. A 401, any other status, a request that does not
/// build, and a request that fails are each an error that names
/// what happened: the service serves this route, so nothing short of
/// a 2xx confirms the key. `instance` names the instance in the text.
pub async fn verify_key(
    http: &reqwest::Client,
    instance: &str,
    base_url: &str,
    key: &str,
) -> Result<KeyCheck, ProviderError> {
    let url = format!("{}{MODELS_ROUTE}", base_url.trim_end_matches('/'));
    let request = http.get(&url).bearer_auth(key).build().map_err(|error| {
        ProviderError::InvalidRequest(redact(
            &format!(
                "the key check of instance `{instance}` at `{url}` does not build: {error}; \
                 check its base_url and key"
            ),
            key,
        ))
    })?;
    let response = http.execute(request).await.map_err(|error| {
        ProviderError::Network(redact(
            &format!("the key check of instance `{instance}`: GET {url} failed: {error}"),
            key,
        ))
    })?;
    let status = response.status();
    if status.is_success() {
        return Ok(KeyCheck::Accepted);
    }
    let what = format!("status {status} from GET {url}");
    Err(match status.as_u16() {
        401 => ProviderError::Auth(format!(
            "the endpoint of instance `{instance}` rejected the key ({what})"
        )),
        429 => ProviderError::RateLimited {
            retry_after: retry_after(&response),
        },
        _ if status.is_server_error() => ProviderError::Server(format!(
            "the key check of instance `{instance}` did not complete ({what})"
        )),
        _ => ProviderError::InvalidRequest(format!(
            "the endpoint of instance `{instance}` refused the key check ({what}); check its \
             base_url"
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_id_without_a_version_suffix_is_an_alias() {
        for alias in [
            DEFAULT_MODEL,
            "jev-preview",
            "jev",
            "1.2.3",
            "-1.2.3",
            "jev-1.2",
            "jev-1.2.3.4",
            "jev-1..3",
            "jev-1.2.x",
            "",
        ] {
            assert!(is_alias(alias), "{alias}");
        }
        for pinned in ["jev-1.14.0", "jev-1.13.0", "other-model-0.0.1"] {
            assert!(!is_alias(pinned), "{pinned}");
        }
    }

    #[test]
    fn the_config_drops_a_trailing_slash() {
        let config = TypeSafeConfig::new("judge", "http://127.0.0.1:1/judge//", "m");
        assert_eq!(config.base_url, "http://127.0.0.1:1/judge");
        assert_eq!(config.max_attempts, 4);
    }

    #[test]
    fn an_error_text_never_holds_part_of_the_secret() {
        let secret = "fake-secret-0123";
        let text = format!("{}{secret} tail", "x".repeat(10));
        // The cut falls inside the secret: the redaction ran first.
        let cut = error_text(text.as_bytes(), 14, secret);
        assert!(!cut.contains("fake-se"), "{cut}");
        // A cut on a multibyte character keeps a whole character.
        let wide = error_text("ééé".as_bytes(), 3, "");
        assert_eq!(wide, "é");
    }

    #[test]
    fn score_probabilities_need_every_level_index() {
        let full = BTreeMap::from([("1".to_string(), 0.75), ("0".to_string(), 0.25)]);
        assert_eq!(level_probabilities("q", &full).unwrap(), vec![0.25, 0.75]);
        let gap = BTreeMap::from([("0".to_string(), 0.25), ("2".to_string(), 0.75)]);
        assert!(level_probabilities("q", &gap).is_err());
        let named = BTreeMap::from([("low".to_string(), 1.0)]);
        assert!(level_probabilities("q", &named).is_err());
    }
}
