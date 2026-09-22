//! TypeSafe-compatible wire format (`POST /v1/systemone`) translated to native questions.
//! Port of `rizzo_flow/compat.py` (Rizzo Flow, Apache-2.0).
//!
//! Only the interface matches the public TypeSafe docs. Answers come from the local
//! checkpoint: the response `model` field always reports the local model, never a Jev version.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::calibration::canonical;
use crate::schema::{Request, ValidationError, MAX_SLOTS};

pub const LOCAL_ALIAS: &str = "rizzo-latest";
/// Accepted so that clients written for the hosted API work unchanged against localhost.
pub const FOREIGN_PREFIX: &str = "jev-";
/// Abstention is disabled on this wire, so every answer letter is an option.
pub const MAX_OPTIONS: usize = MAX_SLOTS;
pub const MAX_LEVELS: usize = 10;
pub const MODEL_RELEASE_DATE: &str = "2026-09-21";

pub type Result<T> = std::result::Result<T, ValidationError>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(ValidationError(msg.into()))
}

/// `Structured = str | dict | list` (the wire allows free text or structured values).
fn is_structured(value: &Value) -> bool {
    value.is_string() || value.is_object() || value.is_array()
}

/// Free text is stripped; structured values fall back to canonical JSON (parity with `text`).
pub fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.trim().to_string(),
        other => canonical(other),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WireQuestion {
    Noul {
        instructions: Value,
        criteria: Option<NoulCriteria>,
    },
    Choice {
        instructions: Value,
        /// Option keys in wire order — this order defines the native slots.
        criteria: Map<String, Value>,
    },
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NoulCriteria {
    pub true_description: Option<Value>,
    pub false_description: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneRequest {
    pub state: Value,
    pub model: String,
    pub questions: BTreeMap<String, WireQuestion>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRequest {
    state: Value,
    model: String,
    questions: BTreeMap<String, RawQuestion>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawQuestion {
    #[serde(rename = "type")]
    kind: String,
    instructions: Option<Value>,
    criteria: Option<Value>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawNoulCriteria {
    #[serde(rename = "true")]
    true_description: Option<Value>,
    #[serde(rename = "false")]
    false_description: Option<Value>,
}

impl SystemOneRequest {
    /// Strict parse + validation (parity with the pydantic `Wire` model: extra="forbid").
    pub fn parse(value: Value) -> Result<SystemOneRequest> {
        let raw: RawRequest = serde_json::from_value(value)
            .map_err(|e| ValidationError(format!("invalid systemone request: {e}")))?;
        if !is_structured(&raw.state) {
            return err("state must be a string, an object or an array");
        }
        if raw.model.is_empty() || raw.model.chars().count() > 128 {
            return err("model must be between 1 and 128 characters");
        }
        if raw.questions.is_empty() || raw.questions.len() > 64 {
            return err("questions must contain between 1 and 64 entries");
        }
        let mut questions = BTreeMap::new();
        for (key, question) in raw.questions {
            let instructions = match &question.instructions {
                Some(v) if is_structured(v) => v.clone(),
                Some(_) => return err(format!("{key}: instructions must be structured")),
                None => return err(format!("{key}: instructions is required")),
            };
            let parsed = match question.kind.as_str() {
                "noul" => {
                    let criteria = match &question.criteria {
                        None | Some(Value::Null) => None,
                        Some(Value::Object(_)) => {
                            let c: RawNoulCriteria = serde_json::from_value(
                                question.criteria.clone().unwrap_or(Value::Null),
                            )
                            .map_err(|e| ValidationError(format!("{key}: criteria: {e}")))?;
                            if let Some(v) = c.true_description.as_ref() {
                                if !is_structured(v) {
                                    return err(format!("{key}: criteria.true must be structured"));
                                }
                            }
                            if let Some(v) = c.false_description.as_ref() {
                                if !is_structured(v) {
                                    return err(format!("{key}: criteria.false must be structured"));
                                }
                            }
                            Some(NoulCriteria {
                                true_description: c.true_description,
                                false_description: c.false_description,
                            })
                        }
                        Some(_) => return err(format!("{key}: criteria must be an object")),
                    };
                    WireQuestion::Noul {
                        instructions,
                        criteria,
                    }
                }
                "choice" => {
                    let criteria = match &question.criteria {
                        Some(Value::Object(map)) => map.clone(),
                        Some(_) => return err(format!("{key}: criteria must be an object")),
                        None => return err(format!("{key}: criteria is required")),
                    };
                    if criteria.len() < 2 || criteria.len() > MAX_OPTIONS {
                        return err(format!(
                            "{key}: criteria must contain between 2 and {MAX_OPTIONS} options"
                        ));
                    }
                    if criteria.keys().any(|k| k.trim().is_empty()) {
                        return err("Choice option keys must not be blank");
                    }
                    for (name, detail) in &criteria {
                        if !detail.is_null() && !is_structured(detail) {
                            return err(format!("{key}: criteria.{name} must be structured"));
                        }
                    }
                    WireQuestion::Choice {
                        instructions,
                        criteria,
                    }
                }
                "score" => {
                    let criteria = match &question.criteria {
                        Some(Value::Array(items)) => items.clone(),
                        Some(_) => return err(format!("{key}: criteria must be an array")),
                        None => return err(format!("{key}: criteria is required")),
                    };
                    if criteria.len() < 2 || criteria.len() > MAX_LEVELS {
                        return err(format!(
                            "{key}: criteria must contain between 2 and {MAX_LEVELS} levels"
                        ));
                    }
                    if criteria.iter().any(|v| !is_structured(v)) {
                        return err(format!("{key}: criteria entries must be structured"));
                    }
                    WireQuestion::Score {
                        instructions,
                        criteria,
                    }
                }
                other => {
                    return err(format!(
                        "{key}: unknown question type {other:?} (noul, choice or score)"
                    ))
                }
            };
            questions.insert(key, parsed);
        }
        Ok(SystemOneRequest {
            state: raw.state,
            model: raw.model,
            questions,
        })
    }
}

/// Served model name derived from the backend metadata (`rizzo-<checkpoint>-<precision>`).
pub fn model_name(metadata: &Value) -> String {
    let source = metadata
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("spark");
    let file = source.rsplit('/').next().unwrap_or(source).to_lowercase();
    let stem = match file.strip_suffix(".gguf") {
        Some(stem) => stem.to_string(),
        None => file,
    };
    let precision = metadata
        .get("precision")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    // A GGUF file name usually carries its own quant tag (…-Q4_K_M.gguf): drop it
    // from the checkpoint so the name does not repeat the precision.
    let mut checkpoint = stem;
    if precision != "unknown" {
        let suffix = precision.to_lowercase();
        let lower = checkpoint.to_lowercase();
        for sep in ['.', '-', '_'] {
            let candidate = format!("{sep}{suffix}");
            if lower.ends_with(&candidate) {
                checkpoint.truncate(checkpoint.len() - candidate.len());
                break;
            }
        }
    }
    format!("rizzo-{checkpoint}-{precision}")
}

/// Accept the local alias, the served name, or any `jev-*` alias; reject the rest.
pub fn resolve_model(requested: &str, metadata: &Value) -> Result<String> {
    let served = model_name(metadata);
    if requested == LOCAL_ALIAS || requested == served || requested.starts_with(FOREIGN_PREFIX) {
        return Ok(served);
    }
    err(format!(
        "Unknown model {requested:?}. Use {LOCAL_ALIAS:?}, {served:?} or a {FOREIGN_PREFIX}* alias."
    ))
}

pub fn list_models(metadata: &Value) -> Value {
    let served = model_name(metadata);
    let source = metadata
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("Spark");
    let local = format!("Local {source} scored with typed option logits.");
    json!({
        "models": [
            {"name": LOCAL_ALIAS, "description": local, "release_date": MODEL_RELEASE_DATE},
            {"name": served, "description": local, "release_date": MODEL_RELEASE_DATE},
            {
                "name": "jev-latest",
                "description": format!("Compatibility alias: answered by {served}, not by TypeSafe Jev."),
                "release_date": MODEL_RELEASE_DATE,
            },
        ]
    })
}

/// Translate the wire request to a native request, plus (per choice question) the
/// option keys in slot order.
pub fn to_native(
    request: &SystemOneRequest,
) -> Result<(Request, BTreeMap<String, Vec<String>>)> {
    let mut questions = Map::new();
    let mut options: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (key, question) in &request.questions {
        let (kind, instructions, extra) = match question {
            WireQuestion::Noul {
                instructions,
                criteria,
            } => {
                let mut native = Map::new();
                if let Some(c) = criteria {
                    if let Some(true_desc) = &c.true_description {
                        native.insert(
                            "true_description".into(),
                            Value::String(format!("Yes. {}", text(true_desc))),
                        );
                    }
                    if let Some(false_desc) = &c.false_description {
                        native.insert(
                            "false_description".into(),
                            Value::String(format!("No. {}", text(false_desc))),
                        );
                    }
                }
                ("boolean", instructions, native)
            }
            WireQuestion::Choice {
                instructions,
                criteria,
            } => {
                // Option keys are free-form strings, so native IDs are positional.
                let keys: Vec<String> = criteria.keys().cloned().collect();
                let native_options: Vec<Value> = keys
                    .iter()
                    .enumerate()
                    .map(|(index, name)| {
                        let detail = criteria.get(name).cloned().unwrap_or(Value::Null);
                        let description = if detail.is_null() {
                            name.clone()
                        } else {
                            format!("{name}: {}", text(&detail))
                        };
                        json!({"id": format!("o{index}"), "description": description})
                    })
                    .collect();
                options.insert(key.clone(), keys);
                let mut native = Map::new();
                native.insert("options".into(), Value::Array(native_options));
                ("choice", instructions, native)
            }
            WireQuestion::Score {
                instructions,
                criteria,
            } => {
                let mut native = Map::new();
                native.insert(
                    "levels".into(),
                    Value::Array(criteria.iter().map(|l| Value::String(text(l))).collect()),
                );
                ("score", instructions, native)
            }
        };
        let mut native = Map::new();
        native.insert("type".into(), Value::String(kind.into()));
        native.insert("instructions".into(), Value::String(text(instructions)));
        // the wire format has no abstention outcome
        native.insert("policy".into(), json!({"allow_abstain": false}));
        for (k, v) in extra {
            native.insert(k, v);
        }
        questions.insert(key.clone(), Value::Object(native));
    }
    let native = json!({"state": request.state, "questions": Value::Object(questions)});
    Ok((Request::from_value(native)?, options))
}

/// Peak-over-uniform statistic from the public Confidence page; not a calibrated accuracy.
pub fn confidence(probabilities: &[f64]) -> f64 {
    let count = probabilities.len();
    if count < 2 {
        return 0.0;
    }
    let peak = probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ((count as f64 * peak - 1.0) / (count as f64 - 1.0)).clamp(0.0, 1.0)
}

/// Translate a native response back to the wire format.
pub fn from_native(
    request: &SystemOneRequest,
    response: &Value,
    options: &BTreeMap<String, Vec<String>>,
    served: &str,
) -> Result<Value> {
    let answers_in = response
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| ValidationError("native response without answers".into()))?;
    let mut answers = Map::new();
    for (key, question) in &request.questions {
        let native = answers_in
            .get(key)
            .ok_or_else(|| ValidationError(format!("native response missing answer {key:?}")))?;
        let probabilities = native
            .get("probabilities")
            .and_then(Value::as_object)
            .ok_or_else(|| ValidationError(format!("{key}: answer without probabilities")))?;
        let answer = match question {
            WireQuestion::Noul { .. } => json!({
                "type": "noul",
                "noul": probabilities.get("true").cloned().unwrap_or(Value::Null),
            }),
            WireQuestion::Choice { .. } => {
                let keys = options.get(key).cloned().unwrap_or_default();
                let mut named = Map::new();
                for (index, name) in keys.iter().enumerate() {
                    let p = probabilities
                        .get(&format!("o{index}"))
                        .cloned()
                        .unwrap_or(json!(0.0));
                    named.insert(name.clone(), p);
                }
                // max(named, key=named.get): first maximum wins on ties
                let choice = named
                    .iter()
                    .max_by(|a, b| {
                        let av = a.1.as_f64().unwrap_or(0.0);
                        let bv = b.1.as_f64().unwrap_or(0.0);
                        av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(k, _)| k.clone());
                let values: Vec<f64> = named
                    .values()
                    .map(|v| v.as_f64().unwrap_or(0.0))
                    .collect();
                json!({
                    "type": "choice",
                    "choice": choice,
                    "probabilities": named,
                    "confidence": confidence(&values),
                })
            }
            WireQuestion::Score { criteria, .. } => {
                let values: Vec<f64> = probabilities
                    .values()
                    .map(|v| v.as_f64().unwrap_or(0.0))
                    .collect();
                let legend: Map<String, Value> = criteria
                    .iter()
                    .enumerate()
                    .map(|(i, level)| (i.to_string(), Value::String(text(level))))
                    .collect();
                json!({
                    "type": "score",
                    "score": native.get("score").cloned().unwrap_or(Value::Null),
                    "legend": legend,
                    "probabilities": probabilities.clone(),
                    "confidence": confidence(&values),
                })
            }
        };
        answers.insert(key.clone(), answer);
    }

    let timing = response.get("timing").cloned().unwrap_or(json!({}));
    let shared = timing
        .get("shared_prefix_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let n = answers_in.len() as u64;
    let total_input: u64 = answers_in
        .values()
        .map(|a| a.get("input_tokens").and_then(Value::as_u64).unwrap_or(0))
        .sum();
    // The shared state is evaluated once; nothing is ever generated.
    let input_tokens = total_input.saturating_sub(shared.saturating_mul(n.saturating_sub(1)));
    let mut statuses: Vec<String> = answers_in
        .values()
        .filter_map(|a| a.get("probability_status").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    statuses.sort();
    statuses.dedup();

    Ok(json!({
        "model": served,
        "answers": Value::Object(answers),
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        // Extension outside the TypeSafe contract; their SDKs ignore unknown fields.
        "x_rizzo": {
            "timing": timing,
            "probability_status": statuses,
            "fingerprint": response.get("model").and_then(|m| m.get("fingerprint")).cloned().unwrap_or(Value::Null),
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> Value {
        json!({"source": "qwen2.5-7b-instruct-q4_k_m.gguf", "precision": "q4_k_m"})
    }

    #[test]
    fn model_name_derives_from_metadata() {
        assert_eq!(model_name(&metadata()), "rizzo-qwen2.5-7b-instruct-q4_k_m");
        assert_eq!(model_name(&json!({})), "rizzo-spark-unknown");
    }

    #[test]
    fn resolve_model_accepts_aliases_and_rejects_unknown() {
        let md = metadata();
        assert_eq!(resolve_model("rizzo-latest", &md).unwrap(), model_name(&md));
        assert_eq!(resolve_model("jev-anything", &md).unwrap(), model_name(&md));
        assert_eq!(
            resolve_model("rizzo-qwen2.5-7b-instruct-q4_k_m", &md).unwrap(),
            model_name(&md)
        );
        assert!(resolve_model("gpt-4o", &md).is_err());
    }

    #[test]
    fn confidence_is_peak_over_uniform() {
        assert_eq!(confidence(&[1.0, 0.0]), 1.0);
        assert_eq!(confidence(&[0.5, 0.5]), 0.0);
        // 3 candidates, peak 0.5 → (3*0.5 - 1) / 2 = 0.25
        assert!((confidence(&[0.5, 0.25, 0.25]) - 0.25).abs() < 1e-12);
    }
}
