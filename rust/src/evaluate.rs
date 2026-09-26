//! Reproducible benchmark reports with coverage, proper scoring rules, and raw evidence.
//! Port of `rizzo_flow/evaluation.py` (Rizzo Flow, Apache-2.0).

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::calibration::{canonical, sha256_canonical};
use crate::schema::ValidationError;

pub type Result<T> = std::result::Result<T, ValidationError>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(ValidationError(msg.into()))
}

/// Anything that can turn a native request into a native response.
/// Implemented by the llama.cpp `Engine`; a fake implementation keeps the harness testable.
pub trait Decider {
    fn decide(&self, request: Value) -> Result<Value>;
}

fn total_seconds(response: &Value) -> f64 {
    response
        .get("timing")
        .and_then(|t| t.get("total_seconds"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

/// `statistics.median`: average of the two middle values for an even count.
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[middle])
    } else {
        Some((sorted[middle - 1] + sorted[middle]) / 2.0)
    }
}

fn opt(value: Option<f64>) -> Value {
    value.map(|v| json!(v)).unwrap_or(Value::Null)
}

/// Argmax over a probability map: the first maximum wins on ties (`max(ps, key=ps.get)`).
fn argmax(probabilities: &Map<String, Value>) -> Option<String> {
    let mut best: Option<(String, f64)> = None;
    for (key, value) in probabilities {
        let p = value.as_f64().unwrap_or(0.0);
        match &best {
            Some((_, current)) if *current >= p => {}
            _ => best = Some((key.clone(), p)),
        }
    }
    best.map(|(key, _)| key)
}

#[derive(Debug, Clone, Copy)]
struct Categorical {
    correct: bool,
    accepted: bool,
    nll: f64,
    brier: f64,
    confidence: f64,
}

/// Run the fixtures through the engine and return the report (parity with `evaluate`).
pub fn evaluate(
    engine: &dyn Decider,
    fixtures: &[Value],
    repeats: usize,
    compare_modes: bool,
) -> Result<Value> {
    if fixtures.is_empty() || repeats < 1 {
        return err("Provide fixtures and at least one repetition");
    }
    let request_of = |fixture: &Value| -> Result<Value> {
        fixture
            .get("request")
            .cloned()
            .ok_or_else(|| ValidationError("fixture without a request".into()))
    };

    // Explicit warmup is excluded from reported timings.
    engine.decide(request_of(&fixtures[0])?)?;

    let mut rows: Vec<Value> = Vec::new();
    let mut latencies: Vec<f64> = Vec::new();
    let mut categorical: Vec<Categorical> = Vec::new();
    let mut statuses: Vec<bool> = Vec::new();
    let mut accepted: Vec<bool> = Vec::new();
    let mut numeric: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
    let mut mode_times: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    mode_times.insert("shared".into(), Vec::new());
    mode_times.insert("direct".into(), Vec::new());
    let mut decision_count = 0usize;
    let mut mode_deltas: Vec<f64> = Vec::new();
    let mut changed = 0usize;

    for fixture in fixtures {
        let request = request_of(fixture)?;
        let mut responses = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            responses.push(engine.decide(request.clone())?);
        }
        for response in &responses {
            latencies.push(total_seconds(response));
            let mode = response
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("shared")
                .to_string();
            mode_times
                .entry(mode)
                .or_default()
                .push(total_seconds(response));
            decision_count += response
                .get("answers")
                .and_then(Value::as_object)
                .map(|answers| answers.len())
                .unwrap_or(0);
        }
        let response = responses[0].clone();
        let mut evidence = json!({
            "id": fixture.get("id").cloned().unwrap_or(Value::Null),
            "expected": fixture.get("expected").cloned().unwrap_or(json!({})),
            "response": response,
            "repeat_timings": responses
                .iter()
                .map(|r| r.get("timing").cloned().unwrap_or(json!({})))
                .collect::<Vec<_>>(),
        });

        if let Some(expected) = fixture.get("expected").and_then(Value::as_object) {
            for (key, expected) in expected {
                let answer = response
                    .get("answers")
                    .and_then(|a| a.get(key))
                    .ok_or_else(|| {
                        ValidationError(format!("response without answer {key:?}"))
                    })?;
                let is_accepted = answer.get("status") == Some(&json!("ok"));
                accepted.push(is_accepted);
                if let Some(status) = expected.get("status") {
                    statuses.push(answer.get("status") == Some(status));
                }
                if let Some(label) = expected.get("label") {
                    let label = label
                        .as_str()
                        .ok_or_else(|| ValidationError("expected label must be a string".into()))?;
                    let probabilities = answer
                        .get("probabilities")
                        .and_then(Value::as_object)
                        .ok_or_else(|| ValidationError("answer without probabilities".into()))?;
                    if !probabilities.contains_key(label) {
                        return err(format!(
                            "Unknown expected label {label} in fixture {}",
                            fixture.get("id").and_then(Value::as_str).unwrap_or("?")
                        ));
                    }
                    let predicted = argmax(probabilities);
                    let p_label = probabilities
                        .get(label)
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    let brier: f64 = probabilities
                        .iter()
                        .map(|(k, p)| {
                            let p = p.as_f64().unwrap_or(0.0);
                            let target = if k == label { 1.0 } else { 0.0 };
                            (p - target) * (p - target)
                        })
                        .sum();
                    let confidence = probabilities
                        .values()
                        .filter_map(Value::as_f64)
                        .fold(f64::NEG_INFINITY, f64::max);
                    categorical.push(Categorical {
                        correct: predicted.as_deref() == Some(label),
                        accepted: is_accepted,
                        nll: -p_label.max(1e-300).ln(),
                        brier,
                        confidence,
                    });
                }
                if let Some(target) = expected.get("value") {
                    let target = target
                        .as_f64()
                        .ok_or_else(|| ValidationError("expected value must be a number".into()))?;
                    if !target.is_finite() {
                        return err("Expected numeric targets must be finite");
                    }
                    let value = answer.get("value").or_else(|| answer.get("score"));
                    let group = canonical(&json!({
                        "type": answer.get("type").cloned().unwrap_or(Value::Null),
                        "unit": answer.get("unit").cloned().unwrap_or(Value::Null),
                        "support": answer.get("support").cloned().unwrap_or(Value::Null),
                    }));
                    numeric.entry(group).or_default().push(
                        value
                            .and_then(Value::as_f64)
                            .map(|v| v - target),
                    );
                }
            }
        }

        if compare_modes {
            let mut alternate = request.clone();
            let current = response
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("shared");
            alternate["mode"] = json!(if current == "shared" { "direct" } else { "shared" });
            let other = engine.decide(alternate)?;
            let other_mode = other
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("shared")
                .to_string();
            mode_times
                .entry(other_mode)
                .or_default()
                .push(total_seconds(&other));
            evidence["alternate_mode_response"] = other.clone();
            if let Some(answers) = response.get("answers").and_then(Value::as_object) {
                for (key, answer) in answers {
                    let a = answer
                        .get("probabilities")
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default();
                    let b = other
                        .get("answers")
                        .and_then(|answers| answers.get(key))
                        .and_then(|answer| answer.get("probabilities"))
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default();
                    let mut delta = 0.0_f64;
                    for (k, av) in &a {
                        let av = av.as_f64().unwrap_or(0.0);
                        let bv = b.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                        delta = delta.max((av - bv).abs());
                    }
                    mode_deltas.push(delta);
                    if argmax(&a) != argmax(&b) {
                        changed += 1;
                    }
                }
            }
        }
        rows.push(evidence);
    }

    // Reliability: ten confidence bins, expected calibration error.
    let mut bins: Vec<Value> = Vec::new();
    let mut ece = 0.0_f64;
    if !categorical.is_empty() {
        for i in 0..10_i64 {
            let subset: Vec<&Categorical> = categorical
                .iter()
                .filter(|row| ((row.confidence * 10.0) as i64).min(9) == i)
                .collect();
            if subset.is_empty() {
                continue;
            }
            let accuracy = mean(
                &subset
                    .iter()
                    .map(|row| if row.correct { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            )
            .unwrap_or(0.0);
            let confidence = mean(
                &subset.iter().map(|row| row.confidence).collect::<Vec<_>>(),
            )
            .unwrap_or(0.0);
            ece += subset.len() as f64 / categorical.len() as f64 * (accuracy - confidence).abs();
            bins.push(json!({
                "lower": i as f64 / 10.0,
                "count": subset.len(),
                "accuracy": accuracy,
                "mean_top_probability": confidence,
            }));
        }
    }

    let mut numeric_groups = Map::new();
    for (group, observations) in &numeric {
        let errors: Vec<f64> = observations.iter().flatten().copied().collect();
        let absolute: Vec<f64> = errors.iter().map(|x| x.abs()).collect();
        let squared: Vec<f64> = errors.iter().map(|x| x * x).collect();
        numeric_groups.insert(
            group.clone(),
            json!({
                "rows": observations.len(),
                "answered": errors.len(),
                "mae_on_answered": opt(mean(&absolute)),
                "rmse_on_answered": match mean(&squared) {
                    Some(value) => json!(value.sqrt()),
                    None => Value::Null,
                },
            }),
        );
    }

    let mut sorted_times = latencies.clone();
    sorted_times.sort_by(f64::total_cmp);
    let p95_index = (0.95 * sorted_times.len() as f64).ceil() as usize;
    let total_latency: f64 = latencies.iter().sum();
    let categorical_correct: Vec<f64> = categorical
        .iter()
        .map(|row| if row.correct { 1.0 } else { 0.0 })
        .collect();
    let accepted_only: Vec<f64> = categorical
        .iter()
        .filter(|row| row.accepted)
        .map(|row| if row.correct { 1.0 } else { 0.0 })
        .collect();

    let summary = json!({
        "requests": fixtures.len(),
        "repeats": repeats,
        "warmup_excluded": true,
        "decisions_per_second": if total_latency > 0.0 {
            json!(decision_count as f64 / total_latency)
        } else {
            Value::Null
        },
        "latency_seconds": {
            "median": opt(median(&latencies)),
            "p95": sorted_times.get(p95_index.saturating_sub(1)).copied(),
        },
        "labeled_decision_coverage": opt(mean(
            &accepted.iter().map(|a| if *a { 1.0 } else { 0.0 }).collect::<Vec<_>>()
        )),
        "status_accuracy": opt(mean(
            &statuses.iter().map(|s| if *s { 1.0 } else { 0.0 }).collect::<Vec<_>>()
        )),
        "categorical": {
            "rows": categorical.len(),
            "accuracy": opt(mean(&categorical_correct)),
            "accepted_accuracy": opt(mean(&accepted_only)),
            "nll": opt(mean(&categorical.iter().map(|r| r.nll).collect::<Vec<_>>())),
            "brier": opt(mean(&categorical.iter().map(|r| r.brier).collect::<Vec<_>>())),
            "ece_10_bins": if categorical.is_empty() { Value::Null } else { json!(ece) },
            "reliability_bins": bins,
        },
        "numeric": {
            "rows": numeric.values().map(|v| v.len()).sum::<usize>(),
            "by_type_unit_and_support": Value::Object(numeric_groups),
        },
        "mode_comparison": if compare_modes {
            json!({
                "decisions": mode_deltas.len(),
                "changed_argmaxes": changed,
                "max_probability_delta": mode_deltas.iter().copied().fold(0.0_f64, f64::max),
                "median_seconds": mode_times
                    .iter()
                    .map(|(k, v)| (k.clone(), opt(median(v))))
                    .collect::<Map<String, Value>>(),
            })
        } else {
            Value::Null
        },
    });

    Ok(json!({
        "dataset_sha256": sha256_canonical(&Value::Array(fixtures.to_vec())),
        "summary": summary,
        "rows": rows,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fake engine: shared mode is confident on "yes", direct flips it.
    struct Fake {
        warmups: std::cell::Cell<usize>,
    }

    impl Decider for Fake {
        fn decide(&self, request: Value) -> Result<Value> {
            self.warmups.set(self.warmups.get() + 1);
            let mode = request
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("shared");
            let probabilities = if mode == "direct" {
                json!({"yes": 0.4, "no": 0.6})
            } else {
                json!({"yes": 0.8, "no": 0.2})
            };
            Ok(json!({
                "mode": mode,
                "answers": {
                    "q": {
                        "type": "boolean",
                        "status": "ok",
                        "probabilities": probabilities,
                    },
                    "n": {
                        "type": "numeric",
                        "status": "ok",
                        "unit": "kg",
                        "support": [1.0, 5.0],
                        "value": 3.0,
                        "probabilities": {"0": 0.7, "1": 0.3},
                    },
                },
                "timing": {"total_seconds": 0.5},
            }))
        }
    }

    fn fixtures() -> Vec<Value> {
        vec![json!({
            "id": "f1",
            "request": {"state": "s", "questions": {}},
            "expected": {
                "q": {"label": "yes", "status": "ok"},
                "n": {"value": 2.5}
            }
        })]
    }

    #[test]
    fn evaluate_reports_coverage_scoring_and_numeric_error() {
        let engine = Fake {
            warmups: std::cell::Cell::new(0),
        };
        let report = evaluate(&engine, &fixtures(), 1, false).unwrap();
        let summary = &report["summary"];
        assert_eq!(summary["requests"], 1);
        assert_eq!(summary["warmup_excluded"], true);
        // warmup + 1 repetition
        assert_eq!(engine.warmups.get(), 2);
        // both expected answers are "ok"
        assert_eq!(summary["labeled_decision_coverage"], 1.0);
        assert_eq!(summary["status_accuracy"], 1.0);
        assert_eq!(summary["categorical"]["rows"], 1);
        assert_eq!(summary["categorical"]["accuracy"], 1.0);
        // nll of 0.8 and brier of the yes/no pair
        let nll = summary["categorical"]["nll"].as_f64().unwrap();
        assert!((nll + 0.8_f64.ln()).abs() < 1e-12);
        let brier = summary["categorical"]["brier"].as_f64().unwrap();
        assert!((brier - (0.2 * 0.2 + 0.2 * 0.2)).abs() < 1e-12);
        // numeric: |3.0 - 2.5| = 0.5
        let groups = summary["numeric"]["by_type_unit_and_support"]
            .as_object()
            .unwrap();
        assert_eq!(groups.len(), 1);
        let group = groups.values().next().unwrap();
        assert_eq!(group["rows"], 1);
        assert_eq!(group["answered"], 1);
        assert!((group["mae_on_answered"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(summary["mode_comparison"], Value::Null);
        assert_eq!(report["dataset_sha256"].as_str().unwrap().len(), 64);
        assert_eq!(report["rows"][0]["id"], "f1");
    }

    #[test]
    fn evaluate_compares_modes_and_counts_changed_argmaxes() {
        let engine = Fake {
            warmups: std::cell::Cell::new(0),
        };
        let report = evaluate(&engine, &fixtures(), 2, true).unwrap();
        let comparison = &report["summary"]["mode_comparison"];
        assert_eq!(comparison["decisions"], 2, "one decision per answer key");
        assert_eq!(comparison["changed_argmaxes"], 1, "yes -> no in direct mode");
        let delta = comparison["max_probability_delta"].as_f64().unwrap();
        assert!((delta - 0.4).abs() < 1e-12);
        // shared repeats (2) + the alternate direct run, per fixture
        assert!(report["rows"][0]["alternate_mode_response"]["mode"] == "direct");
        assert_eq!(report["summary"]["repeats"], 2);
    }

    #[test]
    fn evaluate_rejects_empty_fixtures_and_missing_labels() {
        let engine = Fake {
            warmups: std::cell::Cell::new(0),
        };
        assert!(evaluate(&engine, &[], 1, false).is_err());
        assert!(evaluate(&engine, &fixtures(), 0, false).is_err());

        let mut bad = fixtures();
        bad[0]["expected"]["q"]["label"] = json!("maybe");
        assert!(evaluate(&engine, &bad, 1, false).is_err());
    }

    #[test]
    fn median_matches_statistics_median() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }
}
