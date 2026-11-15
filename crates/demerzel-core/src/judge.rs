//! The judge contract: a fast model that answers typed questions about
//! the loop's state with probabilities.
//!
//! A judge writes no text and calls no tools. One request holds one
//! state value and every question of one loop event, keyed by an id
//! that stays in code. The reply holds one answer per id, the model id
//! that answered, and the usage. Code applies the thresholds and picks
//! the action; the judge only answers.
//!
//! The trait follows the [`crate::Provider`] style: it returns a boxed
//! future, so it is object-safe and needs no async runtime. Every type
//! is non-exhaustive and built through its constructors, so a later
//! field or question type stays additive.

use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::provider::ProviderError;
use crate::tool::BoxFuture;

/// The contract every judge backend implements.
pub trait Judge: Send + Sync {
    /// Stable backend id: the kind a config instance names.
    fn id(&self) -> &'static str;

    /// Every question of one loop event, in one request. The backend
    /// keeps inside `request.timeout`, its retries included.
    fn ask(&self, request: JudgeRequest) -> BoxFuture<'_, Result<JudgeReply, JudgeError>>;
}

/// One judge request: the state and the questions about it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct JudgeRequest {
    /// The loop state every question refers to. Caps count its UTF-8
    /// bytes.
    pub state: String,
    /// The questions by id: stable order, unique ids.
    pub questions: BTreeMap<String, Question>,
    /// The budget for the whole request, retries included.
    pub timeout: Duration,
}

impl JudgeRequest {
    /// A request over `state` with no questions yet.
    pub fn new(state: String, timeout: Duration) -> Self {
        Self {
            state,
            questions: BTreeMap::new(),
            timeout,
        }
    }
}

/// Most options a `Choice` question takes.
const MAX_OPTIONS: usize = 255;
/// Most levels a `Score` question takes.
const MAX_LEVELS: usize = 10;
/// Fewest options or levels a question takes: one alone asks nothing.
const MIN_ANSWERS: usize = 2;

/// One typed question. The instructions carry the question's full
/// meaning; the id beside it stays in code.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Question {
    /// A yes-or-no question, with optional criteria for each side.
    #[non_exhaustive]
    YesNo {
        instructions: String,
        yes: Option<String>,
        no: Option<String>,
    },
    /// One of named options, each with an optional description.
    #[non_exhaustive]
    Choice {
        instructions: String,
        options: Vec<(String, Option<String>)>,
    },
    /// A position on ordered levels, low to high.
    #[non_exhaustive]
    Score {
        instructions: String,
        levels: Vec<String>,
    },
}

impl Question {
    /// A yes-or-no question without criteria.
    pub fn yes_no(instructions: impl Into<String>) -> Self {
        Question::YesNo {
            instructions: instructions.into(),
            yes: None,
            no: None,
        }
    }

    /// A yes-or-no question with the criteria for each side.
    pub fn yes_no_with(
        instructions: impl Into<String>,
        yes: impl Into<String>,
        no: impl Into<String>,
    ) -> Self {
        Question::YesNo {
            instructions: instructions.into(),
            yes: Some(yes.into()),
            no: Some(no.into()),
        }
    }

    /// A choice among 2..=255 options with unique, non-empty names.
    pub fn choice(
        instructions: impl Into<String>,
        options: Vec<(String, Option<String>)>,
    ) -> Result<Self, String> {
        if !(MIN_ANSWERS..=MAX_OPTIONS).contains(&options.len()) {
            return Err(format!(
                "a choice takes {MIN_ANSWERS} to {MAX_OPTIONS} options, not {}",
                options.len()
            ));
        }
        let mut names = HashSet::with_capacity(options.len());
        for (name, _) in &options {
            if name.is_empty() {
                return Err("a choice option needs a name".to_string());
            }
            if !names.insert(name.as_str()) {
                return Err(format!("the choice option `{name}` repeats"));
            }
        }
        Ok(Question::Choice {
            instructions: instructions.into(),
            options,
        })
    }

    /// A score on 2..=10 levels, low to high.
    pub fn score(instructions: impl Into<String>, levels: Vec<String>) -> Result<Self, String> {
        if !(MIN_ANSWERS..=MAX_LEVELS).contains(&levels.len()) {
            return Err(format!(
                "a score takes {MIN_ANSWERS} to {MAX_LEVELS} levels, not {}",
                levels.len()
            ));
        }
        Ok(Question::Score {
            instructions: instructions.into(),
            levels,
        })
    }
}

/// One answer, of its question's type. The serialized shape is the
/// `answers` member of the session file's `judgment` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Answer {
    /// The probability of yes.
    #[non_exhaustive]
    YesNo { yes: f64 },
    /// The chosen option and a probability for each option, by name.
    #[non_exhaustive]
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
    },
    /// The probability-weighted position on the levels, from 0.
    #[non_exhaustive]
    Score {
        position: f64,
        /// One probability per level, by level index.
        probabilities: Vec<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
    },
}

impl Answer {
    pub fn yes_no(yes: f64) -> Self {
        Answer::YesNo { yes }
    }

    pub fn choice(
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: Option<f64>,
    ) -> Self {
        Answer::Choice {
            choice,
            probabilities,
            confidence,
        }
    }

    pub fn score(position: f64, probabilities: Vec<f64>, confidence: Option<f64>) -> Self {
        Answer::Score {
            position,
            probabilities,
            confidence,
        }
    }
}

/// One judge reply: an answer per question id.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct JudgeReply {
    pub answers: BTreeMap<String, Answer>,
    /// The model id the reply names.
    pub model: String,
    pub usage: JudgeUsage,
}

impl JudgeReply {
    /// A reply with no answers yet.
    pub fn new(model: impl Into<String>, usage: JudgeUsage) -> Self {
        Self {
            answers: BTreeMap::new(),
            model: model.into(),
            usage,
        }
    }
}

/// The token counts of one judge request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JudgeUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl JudgeUsage {
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }
}

/// Judge failure classes.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JudgeError {
    /// The transport failed: auth, rate limit, server, network, an
    /// invalid request, or a reply the backend could not read.
    #[error(transparent)]
    Transport(#[from] ProviderError),
    /// No answer within the request's budget, in milliseconds.
    #[error("no answer within {0} ms")]
    Timeout(u64),
    /// A reply that does not answer each question with its type.
    #[error("reply: {0}")]
    Reply(String),
}

/// One rule for every backend: each question id has an answer of its
/// question's type, and the reply has no other answer; each probability
/// and confidence is finite and in 0..=1; a `Choice` answer names one
/// of the options and gives probabilities for options only; a `Score`
/// answer has one probability per level and a position in
/// 0..=levels-1.
pub fn check_reply(request: &JudgeRequest, reply: &JudgeReply) -> Result<(), String> {
    if let Some(extra) = reply
        .answers
        .keys()
        .find(|id| !request.questions.contains_key(*id))
    {
        return Err(format!("the answer `{extra}` has no question"));
    }
    for (id, question) in &request.questions {
        let Some(answer) = reply.answers.get(id) else {
            return Err(format!("the question `{id}` has no answer"));
        };
        match (question, answer) {
            (Question::YesNo { .. }, Answer::YesNo { yes }) => unit(id, "yes", *yes)?,
            (
                Question::Choice { options, .. },
                Answer::Choice {
                    choice,
                    probabilities,
                    confidence,
                },
            ) => {
                let known = |name: &str| options.iter().any(|(option, _)| option == name);
                if !known(choice) {
                    return Err(format!("`{id}` chose `{choice}`, which is not an option"));
                }
                for (name, probability) in probabilities {
                    if !known(name) {
                        return Err(format!(
                            "`{id}` gives a probability for `{name}`, which is not an option"
                        ));
                    }
                    unit(id, name, *probability)?;
                }
                if let Some(confidence) = confidence {
                    unit(id, "confidence", *confidence)?;
                }
            }
            (
                Question::Score { levels, .. },
                Answer::Score {
                    position,
                    probabilities,
                    confidence,
                },
            ) => {
                if probabilities.len() != levels.len() {
                    return Err(format!(
                        "`{id}` gives {} probabilities for {} levels",
                        probabilities.len(),
                        levels.len()
                    ));
                }
                for (level, probability) in probabilities.iter().enumerate() {
                    unit(id, &format!("level {level}"), *probability)?;
                }
                let top = (levels.len() - 1) as f64;
                if !position.is_finite() || !(0.0..=top).contains(position) {
                    return Err(format!(
                        "`{id}` gives the position {position}, outside 0 to {top}"
                    ));
                }
                if let Some(confidence) = confidence {
                    unit(id, "confidence", *confidence)?;
                }
            }
            _ => {
                return Err(format!(
                    "the answer to `{id}` is not of its question's type"
                ));
            }
        }
    }
    Ok(())
}

/// `value` is a finite probability in 0..=1.
fn unit(id: &str, what: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(format!("`{id}` gives {what} {value}, outside 0 to 1"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        match std::pin::pin!(future).poll(&mut cx) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("the test judge must resolve at once"),
        }
    }

    fn options(names: &[&str]) -> Vec<(String, Option<String>)> {
        names.iter().map(|name| (name.to_string(), None)).collect()
    }

    fn levels(count: usize) -> Vec<String> {
        (0..count).map(|level| format!("level {level}")).collect()
    }

    /// Answers every question with the first option at full weight.
    struct FirstOption;

    impl Judge for FirstOption {
        fn id(&self) -> &'static str {
            "first-option"
        }

        fn ask(&self, request: JudgeRequest) -> BoxFuture<'_, Result<JudgeReply, JudgeError>> {
            let mut reply = JudgeReply::new("test-judge-1", JudgeUsage::new(12, 3));
            for (id, question) in &request.questions {
                let answer = match question {
                    Question::YesNo { .. } => Answer::yes_no(1.0),
                    Question::Choice { options, .. } => {
                        let first = options[0].0.clone();
                        Answer::choice(first.clone(), BTreeMap::from([(first, 1.0)]), None)
                    }
                    Question::Score { levels, .. } => {
                        let mut probabilities = vec![0.0; levels.len()];
                        probabilities[0] = 1.0;
                        Answer::score(0.0, probabilities, Some(0.9))
                    }
                };
                reply.answers.insert(id.clone(), answer);
            }
            Box::pin(std::future::ready(Ok(reply)))
        }
    }

    fn request() -> JudgeRequest {
        let mut request = JudgeRequest::new("state".into(), Duration::from_secs(2));
        request
            .questions
            .insert("done".into(), Question::yes_no("Is the task done?"));
        request.questions.insert(
            "kind".into(),
            Question::choice("Which kind?", options(&["fix", "feature", "docs"])).unwrap(),
        );
        request.questions.insert(
            "risk".into(),
            Question::score("How risky?", levels(3)).unwrap(),
        );
        request
    }

    #[test]
    fn a_judge_is_object_safe_and_its_reply_passes_the_check() {
        let judge: Arc<dyn Judge> = Arc::new(FirstOption);
        assert_eq!(judge.id(), "first-option");
        let request = request();
        let reply = block_on(judge.ask(request.clone())).unwrap();
        assert_eq!(reply.answers.len(), 3);
        assert_eq!(reply.model, "test-judge-1");
        assert_eq!(reply.usage, JudgeUsage::new(12, 3));
        check_reply(&request, &reply).unwrap();
    }

    #[test]
    fn a_choice_takes_2_to_255_unique_named_options() {
        assert!(Question::choice("q", options(&["a"])).is_err());
        assert!(Question::choice("q", Vec::new()).is_err());
        assert!(Question::choice("q", options(&["a", "b"])).is_ok());
        let most: Vec<(String, Option<String>)> =
            (0..255).map(|index| (format!("o{index}"), None)).collect();
        assert!(Question::choice("q", most.clone()).is_ok());
        let mut too_many = most;
        too_many.push(("o255".into(), None));
        assert!(Question::choice("q", too_many).is_err());
        assert!(Question::choice("q", options(&["a", "a"])).is_err());
        assert!(Question::choice("q", options(&["a", ""])).is_err());
        // A description is optional and does not count toward the name.
        let described = vec![
            ("a".to_string(), Some("first".to_string())),
            ("b".into(), None),
        ];
        assert!(Question::choice("q", described).is_ok());
    }

    #[test]
    fn a_score_takes_2_to_10_levels() {
        assert!(Question::score("q", levels(0)).is_err());
        assert!(Question::score("q", levels(1)).is_err());
        assert!(Question::score("q", levels(2)).is_ok());
        assert!(Question::score("q", levels(10)).is_ok());
        assert!(Question::score("q", levels(11)).is_err());
    }

    #[test]
    fn a_yes_no_question_carries_its_optional_criteria() {
        assert_eq!(
            Question::yes_no("q"),
            Question::YesNo {
                instructions: "q".into(),
                yes: None,
                no: None
            }
        );
        assert_eq!(
            Question::yes_no_with("q", "it ran", "it failed"),
            Question::YesNo {
                instructions: "q".into(),
                yes: Some("it ran".into()),
                no: Some("it failed".into())
            }
        );
    }

    fn good_reply() -> JudgeReply {
        block_on(FirstOption.ask(request())).unwrap()
    }

    /// `check_reply` refuses `reply` after `change`.
    fn refused(change: impl FnOnce(&mut JudgeReply)) {
        let mut reply = good_reply();
        change(&mut reply);
        assert!(check_reply(&request(), &reply).is_err(), "{reply:?}");
    }

    #[test]
    fn check_reply_refuses_a_missing_or_an_extra_answer() {
        refused(|reply| {
            reply.answers.remove("done");
        });
        refused(|reply| {
            reply.answers.insert("other".into(), Answer::yes_no(0.5));
        });
    }

    #[test]
    fn check_reply_refuses_an_answer_of_another_type() {
        refused(|reply| {
            reply
                .answers
                .insert("done".into(), Answer::score(0.0, vec![1.0, 0.0], None));
        });
        refused(|reply| {
            reply.answers.insert("risk".into(), Answer::yes_no(0.5));
        });
        refused(|reply| {
            reply
                .answers
                .insert("kind".into(), Answer::score(0.0, vec![1.0, 0.0, 0.0], None));
        });
    }

    #[test]
    fn check_reply_refuses_a_probability_outside_0_to_1_or_not_finite() {
        for bad in [f64::NAN, f64::INFINITY, -0.1, 1.5] {
            refused(|reply| {
                reply.answers.insert("done".into(), Answer::yes_no(bad));
            });
            refused(|reply| {
                reply.answers.insert(
                    "kind".into(),
                    Answer::choice("fix".into(), BTreeMap::from([("fix".into(), bad)]), None),
                );
            });
            refused(|reply| {
                reply.answers.insert(
                    "kind".into(),
                    Answer::choice("fix".into(), BTreeMap::new(), Some(bad)),
                );
            });
            refused(|reply| {
                reply
                    .answers
                    .insert("risk".into(), Answer::score(0.0, vec![bad, 0.0, 0.0], None));
            });
            refused(|reply| {
                reply.answers.insert(
                    "risk".into(),
                    Answer::score(0.0, vec![1.0, 0.0, 0.0], Some(bad)),
                );
            });
        }
        // The ends of the range are probabilities.
        let mut reply = good_reply();
        reply.answers.insert("done".into(), Answer::yes_no(0.0));
        check_reply(&request(), &reply).unwrap();
    }

    #[test]
    fn check_reply_refuses_a_choice_outside_the_options() {
        refused(|reply| {
            reply.answers.insert(
                "kind".into(),
                Answer::choice("chore".into(), BTreeMap::new(), None),
            );
        });
        refused(|reply| {
            reply.answers.insert(
                "kind".into(),
                Answer::choice(
                    "fix".into(),
                    BTreeMap::from([("fix".into(), 0.5), ("chore".into(), 0.5)]),
                    None,
                ),
            );
        });
    }

    #[test]
    fn check_reply_refuses_a_score_off_its_levels() {
        // One probability short.
        refused(|reply| {
            reply
                .answers
                .insert("risk".into(), Answer::score(0.0, vec![1.0, 0.0], None));
        });
        // A position past the top level, below the first, or not finite.
        for position in [2.5, -0.5, f64::NAN] {
            refused(|reply| {
                reply.answers.insert(
                    "risk".into(),
                    Answer::score(position, vec![0.0, 0.0, 1.0], None),
                );
            });
        }
        let mut reply = good_reply();
        reply
            .answers
            .insert("risk".into(), Answer::score(2.0, vec![0.0, 0.0, 1.0], None));
        check_reply(&request(), &reply).unwrap();
    }

    #[test]
    fn answers_serialize_with_their_type_and_round_trip() {
        let answers = [
            Answer::yes_no(0.25),
            Answer::choice(
                "fix".into(),
                BTreeMap::from([("fix".into(), 0.75), ("docs".into(), 0.25)]),
                Some(0.5),
            ),
            Answer::score(1.5, vec![0.0, 0.5, 0.5], None),
        ];
        let values: Vec<serde_json::Value> = answers
            .iter()
            .map(|answer| serde_json::to_value(answer).unwrap())
            .collect();
        assert_eq!(
            values[0],
            serde_json::json!({"type": "yes_no", "yes": 0.25})
        );
        assert_eq!(
            values[1],
            serde_json::json!({
                "type": "choice",
                "choice": "fix",
                "probabilities": {"docs": 0.25, "fix": 0.75},
                "confidence": 0.5
            })
        );
        assert_eq!(
            values[2],
            serde_json::json!({"type": "score", "position": 1.5, "probabilities": [0.0, 0.5, 0.5]})
        );
        for (answer, value) in answers.iter().zip(values) {
            assert_eq!(&serde_json::from_value::<Answer>(value).unwrap(), answer);
        }
    }

    #[test]
    fn a_transport_error_keeps_its_class() {
        let error = JudgeError::from(ProviderError::RateLimited { retry_after: None });
        assert!(matches!(
            error,
            JudgeError::Transport(ProviderError::RateLimited { .. })
        ));
    }
}
