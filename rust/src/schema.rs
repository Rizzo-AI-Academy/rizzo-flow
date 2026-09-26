//! Strict public request contract — Rust port of `schema.py` (Rizzo Flow, Apache-2.0).
//!
//! Model outputs never supply JSON or field names: the contract is validated *before* any
//! computation, and unknown fields are rejected (parity with the Python `extra="forbid"`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Every candidate (options + special ones) is one uppercase answer letter.
pub const MAX_SLOTS: usize = 26;

/// Validation error — parity with the Python `ValidationError` paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError(pub String);

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ValidationError {}

pub type Result<T> = std::result::Result<T, ValidationError>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(ValidationError(msg.into()))
}

/// `Text`: stripped, 1..=8000 characters.
pub fn text(value: &str, field: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 8000 {
        return err(format!("{field}: text must have 1..=8000 nonblank characters"));
    }
    Ok(trimmed.to_string())
}

fn finite(value: f64, field: &str) -> Result<f64> {
    if !value.is_finite() || value.abs() > 1e100 {
        return err(format!("{field}: number must be finite and within ±1e100"));
    }
    Ok(value)
}

/// Option id pattern: `^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$` (also rejects reserved `__…__` ids).
pub fn valid_option_id(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    let rest: Vec<char> = chars.collect();
    rest.len() <= 63 && rest.iter().all(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    #[serde(default = "default_true")]
    pub allow_abstain: bool,
    #[serde(default = "default_max_unavailable")]
    pub max_unavailable_probability: f64,
    #[serde(default)]
    pub min_top_probability: f64,
}

fn default_true() -> bool {
    true
}
fn default_max_unavailable() -> f64 {
    0.5
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            allow_abstain: true,
            max_unavailable_probability: 0.5,
            min_top_probability: 0.0,
        }
    }
}

impl Policy {
    fn validate(&self) -> Result<()> {
        if !(self.max_unavailable_probability > 0.0 && self.max_unavailable_probability <= 1.0) {
            return err("policy.max_unavailable_probability must be in (0, 1]");
        }
        if !(0.0..=1.0).contains(&self.min_top_probability) {
            return err("policy.min_top_probability must be in [0, 1]");
        }
        Ok(())
    }
}

fn require_slots(entries: usize, reserved: usize, policy: &Policy) -> Result<()> {
    let reserved = reserved + usize::from(policy.allow_abstain);
    if entries + reserved > MAX_SLOTS {
        return err(format!(
            "At most {} entries fit here: {MAX_SLOTS} answer letters, {reserved} reserved for abstention/out-of-range",
            MAX_SLOTS - reserved
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ChoiceOption {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Anchor {
    pub value: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BooleanQuestion {
    pub instructions: String,
    #[serde(default)]
    pub policy: Policy,
    #[serde(default = "default_true_description")]
    pub true_description: String,
    #[serde(default = "default_false_description")]
    pub false_description: String,
}

fn default_true_description() -> String {
    "Yes. The evidence supports an affirmative answer to the question.".to_string()
}
fn default_false_description() -> String {
    "No. The evidence supports a negative answer to the question.".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChoiceQuestion {
    pub instructions: String,
    #[serde(default)]
    pub policy: Policy,
    pub options: Vec<ChoiceOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreQuestion {
    pub instructions: String,
    #[serde(default)]
    pub policy: Policy,
    pub levels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericQuestion {
    pub instructions: String,
    #[serde(default)]
    pub policy: Policy,
    pub unit: String,
    pub anchors: Vec<Anchor>,
}

/// Discriminated question union (`type`: boolean | choice | score | numeric).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Boolean(BooleanQuestion),
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
    Numeric(NumericQuestion),
}

impl Question {
    pub fn policy(&self) -> &Policy {
        match self {
            Question::Boolean(q) => &q.policy,
            Question::Choice(q) => &q.policy,
            Question::Score(q) => &q.policy,
            Question::Numeric(q) => &q.policy,
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Question::Boolean(q) => &q.instructions,
            Question::Choice(q) => &q.instructions,
            Question::Score(q) => &q.instructions,
            Question::Numeric(q) => &q.instructions,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Question::Boolean(_) => "boolean",
            Question::Choice(_) => "choice",
            Question::Score(_) => "score",
            Question::Numeric(_) => "numeric",
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.policy().validate()?;
        text(self.instructions(), "instructions")?;
        match self {
            Question::Boolean(q) => {
                text(&q.true_description, "true_description")?;
                text(&q.false_description, "false_description")?;
                require_slots(0, 0, &q.policy)
            }
            Question::Choice(q) => {
                if q.options.len() < 2 || q.options.len() > MAX_SLOTS {
                    return err("choice.options must have 2..=26 entries");
                }
                for option in &q.options {
                    if !valid_option_id(&option.id) {
                        return err(format!(
                            "choice option id {:?} does not match ^[a-zA-Z0-9][a-zA-Z0-9_-]{{0,63}}$",
                            option.id
                        ));
                    }
                    text(&option.description, "option description")?;
                }
                let unique: std::collections::HashSet<&str> =
                    q.options.iter().map(|o| o.id.as_str()).collect();
                if unique.len() != q.options.len() {
                    return err("Option IDs must be unique");
                }
                require_slots(q.options.len(), 0, &q.policy)
            }
            Question::Score(q) => {
                if q.levels.len() < 2 || q.levels.len() > MAX_SLOTS {
                    return err("score.levels must have 2..=26 entries");
                }
                for level in &q.levels {
                    text(level, "level")?;
                }
                let unique: std::collections::HashSet<&str> =
                    q.levels.iter().map(|s| s.as_str()).collect();
                if unique.len() != q.levels.len() {
                    return err("Levels must have distinct descriptions");
                }
                require_slots(q.levels.len(), 0, &q.policy)
            }
            Question::Numeric(q) => {
                if q.anchors.len() < 2 || q.anchors.len() > MAX_SLOTS {
                    return err("numeric.anchors must have 2..=26 entries");
                }
                text(&q.unit, "unit")?;
                if q.unit.trim().chars().count() > 64 {
                    return err("numeric.unit must have at most 64 characters");
                }
                for anchor in &q.anchors {
                    finite(anchor.value, "anchor value")?;
                    text(&anchor.description, "anchor description")?;
                }
                if q.anchors.windows(2).any(|w| w[1].value <= w[0].value) {
                    return err("Numeric anchors must be strictly increasing");
                }
                require_slots(q.anchors.len(), 2, &q.policy)
            }
        }
    }
}

/// Question key → allowed field names (strict check, since `deny_unknown_fields` cannot be
/// combined with internally tagged enums in serde).
fn allowed_question_keys(kind: &str) -> Option<&'static [&'static str]> {
    match kind {
        "boolean" => Some(&[
            "type",
            "instructions",
            "policy",
            "true_description",
            "false_description",
        ]),
        "choice" => Some(&["type", "instructions", "policy", "options"]),
        "score" => Some(&["type", "instructions", "policy", "levels"]),
        "numeric" => Some(&["type", "instructions", "policy", "unit", "anchors"]),
        _ => None,
    }
}

const POLICY_KEYS: &[&str] = &[
    "allow_abstain",
    "max_unavailable_probability",
    "min_top_probability",
];

fn check_keys(value: &Value, allowed: &[&str], what: &str) -> Result<()> {
    if let Value::Object(map) = value {
        for key in map.keys() {
            if !allowed.contains(&key.as_str()) {
                return err(format!("{what}: unknown field {key:?}"));
            }
        }
        Ok(())
    } else {
        err(format!("{what}: expected an object"))
    }
}

fn check_question_strict(value: &Value) -> Result<()> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| ValidationError("question: missing `type`".into()))?;
    let allowed = allowed_question_keys(kind)
        .ok_or_else(|| ValidationError(format!("question: unknown type {kind:?}")))?;
    check_keys(value, allowed, "question")?;
    if let Some(policy) = value.get("policy") {
        check_keys(policy, POLICY_KEYS, "policy")?;
    }
    if let Some(options) = value.get("options").and_then(Value::as_array) {
        for option in options {
            check_keys(option, &["id", "description"], "option")?;
        }
    }
    if let Some(anchors) = value.get("anchors").and_then(Value::as_array) {
        for anchor in anchors {
            check_keys(anchor, &["value", "description"], "anchor")?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Shared,
    Direct,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub state: Value,
    pub questions: std::collections::BTreeMap<String, Question>,
    #[serde(default)]
    pub mode: Mode,
}

impl Request {
    /// Parse + strict validation (unknown fields, text constraints, slot limits, state size).
    pub fn from_json_str(input: &str) -> Result<Request> {
        let raw: Value =
            serde_json::from_str(input).map_err(|e| ValidationError(format!("invalid JSON: {e}")))?;
        Self::from_value(raw)
    }

    pub fn from_value(raw: Value) -> Result<Request> {
        check_keys(&raw, &["state", "questions", "mode"], "request")?;
        let questions = raw
            .get("questions")
            .and_then(Value::as_object)
            .ok_or_else(|| ValidationError("questions: expected an object".into()))?;
        if questions.is_empty() || questions.len() > 64 {
            return err("questions must have 1..=64 entries");
        }
        for key in questions.keys() {
            if key.trim().is_empty() || key.chars().count() > 128 {
                return err("Question IDs must have 1..=128 nonblank characters");
            }
        }
        for question in questions.values() {
            check_question_strict(question)?;
        }
        let request: Request = serde_json::from_value(raw)
            .map_err(|e| ValidationError(format!("contract violation: {e}")))?;
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<()> {
        match &self.state {
            Value::String(s) if s.trim().is_empty() => {
                return err("State must not be empty");
            }
            Value::Null => return err("State must not be empty"),
            _ => {}
        }
        let rendered = serde_json::to_string(&self.state)
            .map_err(|e| ValidationError(format!("state: {e}")))?;
        if rendered.len() > 256_000 {
            return err("State exceeds 256 KB; no silent truncation");
        }
        for question in self.questions.values() {
            question.validate()?;
        }
        Ok(())
    }
}
