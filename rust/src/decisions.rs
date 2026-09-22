//! Pure, deterministic decoding of categorical logits into typed results.
//! Rust port of `decisions.py` (Rizzo Flow, Apache-2.0) — no I/O, no side effects.

use crate::schema::{Question, ValidationError};
use serde_json::{json, Map, Value};

pub const UNKNOWN: &str = "__insufficient__";
pub const BELOW: &str = "__below_range__";
pub const ABOVE: &str = "__above_range__";

pub type Result<T> = std::result::Result<T, ValidationError>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(ValidationError(msg.into()))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub description: String,
    pub value: Option<f64>,
}

/// Build the candidate list for a question (options + special abstention/range slots).
pub fn candidates(question: &Question) -> Vec<Candidate> {
    let mut result: Vec<Candidate> = match question {
        Question::Boolean(q) => vec![
            Candidate {
                id: "false".into(),
                description: q.false_description.clone(),
                value: Some(0.0),
            },
            Candidate {
                id: "true".into(),
                description: q.true_description.clone(),
                value: Some(1.0),
            },
        ],
        Question::Choice(q) => q
            .options
            .iter()
            .map(|o| Candidate {
                id: o.id.clone(),
                description: o.description.clone(),
                value: None,
            })
            .collect(),
        Question::Score(q) => q
            .levels
            .iter()
            .enumerate()
            .map(|(i, level)| Candidate {
                id: i.to_string(),
                description: level.clone(),
                value: Some(i as f64),
            })
            .collect(),
        Question::Numeric(q) => {
            let mut list: Vec<Candidate> = q
                .anchors
                .iter()
                .enumerate()
                .map(|(i, a)| Candidate {
                    id: i.to_string(),
                    description: format!("Approximately {} {}: {}", a.value, q.unit, a.description),
                    value: Some(a.value),
                })
                .collect();
            let first = q.anchors.first().map(|a| a.value).unwrap_or(0.0);
            let last = q.anchors.last().map(|a| a.value).unwrap_or(0.0);
            list.push(Candidate {
                id: BELOW.into(),
                description: format!("The value is below {first} {}.", q.unit),
                value: None,
            });
            list.push(Candidate {
                id: ABOVE.into(),
                description: format!("The value is above {last} {}.", q.unit),
                value: None,
            });
            list
        }
    };
    if question.policy().allow_abstain {
        result.push(Candidate {
            id: UNKNOWN.into(),
            description: "Cannot determine the answer: the required information is not provided \
                          or is contradictory. A known value outside the numeric range is not \
                          missing information."
                .into(),
            value: None,
        });
    }
    result
}

/// Numerically stable softmax with optional temperature.
pub fn softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return err("Temperature must be finite and positive");
    }
    if logits.len() < 2 || !logits.iter().all(|x| x.is_finite()) {
        return err("At least two finite logits are required");
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|x| ((x - maximum) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    Ok(weights.iter().map(|w| w / total).collect())
}

/// Probability-weighted summary for score/numeric answers.
pub fn summarize(values: &[f64], probabilities: &[f64]) -> Value {
    let mean: f64 = values
        .iter()
        .zip(probabilities)
        .map(|(v, p)| v * p)
        .sum();
    let variance: f64 = probabilities
        .iter()
        .zip(values)
        .map(|(p, v)| p * (v - mean).powi(2))
        .sum();
    let quantile = |q: f64| -> f64 {
        let mut cumulative = 0.0;
        for (v, p) in values.iter().zip(probabilities) {
            cumulative += p;
            if cumulative >= q {
                return *v;
            }
        }
        values.last().copied().unwrap_or(0.0)
    };
    json!({
        "mean": mean,
        "stddev": variance.sqrt(),
        "median": quantile(0.5),
        "anchor_quantiles": {"p10": quantile(0.1), "p90": quantile(0.9)},
    })
}

/// Decode logits (one per candidate) into a typed result — parity with `decisions.decode`.
pub fn decode(question: &Question, logits: &[f64], temperature: f64) -> Result<Value> {
    let choices = candidates(question);
    if logits.len() != choices.len() {
        return err("Logit count does not match the declared candidates");
    }
    let ps = softmax(logits, temperature)?;

    let distribution: Map<String, Value> = choices
        .iter()
        .zip(&ps)
        .map(|(c, p)| (c.id.clone(), json!(p)))
        .collect();
    let unavailable_ids = [UNKNOWN, BELOW, ABOVE];
    let mut valid: Vec<(&Candidate, f64)> = Vec::new();
    let mut unavailable = 0.0_f64;
    for (c, p) in choices.iter().zip(&ps) {
        if unavailable_ids.contains(&c.id.as_str()) {
            unavailable += p;
        } else {
            valid.push((c, *p));
        }
    }
    let available: f64 = valid.iter().map(|(_, p)| p).sum();

    let (winner_idx, top) = ps
        .iter()
        .enumerate()
        .fold((0_usize, f64::NEG_INFINITY), |(bi, bt), (i, p)| {
            if *p > bt {
                (i, *p)
            } else {
                (bi, bt)
            }
        });
    let winner = choices[winner_idx].id.as_str();
    let entropy: f64 = ps.iter().filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum();

    let policy = question.policy();
    let mut status = "ok";
    if unavailable_ids.contains(&winner) || unavailable >= policy.max_unavailable_probability {
        let get = |id: &str| distribution.get(id).and_then(Value::as_f64).unwrap_or(0.0);
        status = if get(BELOW) + get(ABOVE) > get(UNKNOWN) {
            "out_of_range"
        } else {
            "insufficient_evidence"
        };
    } else if top < policy.min_top_probability {
        status = "uncertain";
    }
    let concentration = (1.0 - entropy / (ps.len() as f64).ln()).clamp(0.0, 1.0);
    let probability_status = if temperature == 1.0 {
        "uncalibrated_conditional_option_scores"
    } else {
        "temperature_scaled_requires_held_out_validation"
    };

    let mut result = json!({
        "type": question.type_name(),
        "status": status,
        "probabilities": distribution,
        "option_logits": choices.iter().zip(logits)
            .map(|(c, x)| (c.id.clone(), json!(x)))
            .collect::<Map<String, Value>>(),
        "legend": choices.iter()
            .map(|c| (c.id.clone(), json!(c.description)))
            .collect::<Map<String, Value>>(),
        "uncertainty": {
            "top_probability": top,
            "entropy_nats": entropy,
            "concentration": concentration,
            "unavailable_probability": unavailable,
        },
        "probability_status": probability_status,
        "temperature": temperature,
    });

    let conditional: Option<Vec<f64>> = if available > 0.0 {
        Some(valid.iter().map(|(_, p)| p / available).collect())
    } else {
        None
    };

    match question {
        Question::Choice(_) => {
            result["choice"] = if status == "ok" { json!(winner) } else { Value::Null };
        }
        Question::Boolean(_) => {
            result["value"] = if status == "ok" {
                json!(winner == "true")
            } else {
                Value::Null
            };
            result["probability_true_given_available"] = conditional
                .as_ref()
                .and_then(|c| c.get(1))
                .map(|p| json!(p))
                .unwrap_or(Value::Null);
        }
        _ => {
            let values: Vec<f64> = valid.iter().filter_map(|(c, _)| c.value).collect();
            let stats = match (&conditional, status) {
                (Some(cond), "ok") => Some(summarize(&values, cond)),
                _ => None,
            };
            let mean = stats
                .as_ref()
                .and_then(|s| s.get("mean"))
                .and_then(Value::as_f64);
            result["statistics_given_available"] = stats.clone().unwrap_or(Value::Null);
            result["values"] = Value::Object(
                valid
                    .iter()
                    .map(|(c, _)| (c.id.clone(), json!(c.value)))
                    .collect::<Map<String, Value>>(),
            );
            result["support"] = json!([values.first().copied(), values.last().copied()]);
            match question {
                Question::Numeric(q) => {
                    result["value"] = mean.map(|m| json!(m)).unwrap_or(Value::Null);
                    result["unit"] = json!(q.unit);
                    result["range_probabilities"] = json!({
                        "below": distribution.get(BELOW).cloned().unwrap_or(json!(0.0)),
                        "above": distribution.get(ABOVE).cloned().unwrap_or(json!(0.0)),
                    });
                }
                Question::Score(_) => {
                    result["score"] = mean.map(|m| json!(m)).unwrap_or(Value::Null);
                    result["normalized_score"] = mean
                        .and_then(|m| values.last().map(|last| json!(m / last)))
                        .unwrap_or(Value::Null);
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(result)
}
