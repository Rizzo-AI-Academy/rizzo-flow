//! Calibration (temperature fit) — parity with `rizzo_flow.calibration`.
//!
//! A calibration file binds temperatures per question type to one model/runtime
//! fingerprint. Loading validates positivity/finiteness; the fingerprint must
//! match the engine's own fingerprint or the file is rejected (same contract as
//! the Python `Engine`).

use serde_json::Value;
use std::collections::BTreeMap;

use crate::schema::ValidationError;

pub type Result<T> = std::result::Result<T, ValidationError>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    #[serde(default = "default_version")]
    pub version: u8,
    pub fingerprint: String,
    pub dataset_sha256: String,
    pub temperatures: BTreeMap<String, f64>,
    pub fit_metrics: Value,
    #[serde(default = "default_status")]
    pub status: String,
}

fn default_version() -> u8 {
    1
}

fn default_status() -> String {
    "fitted_requires_held_out_validation".to_string()
}

/// One labeled logit row used to fit a temperature (`LabeledLogits` in calibration.py).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabeledLogits {
    #[serde(rename = "type")]
    pub kind: String,
    pub logits: Vec<f64>,
    pub label_index: usize,
}

pub const TYPES: [&str; 4] = ["boolean", "choice", "score", "numeric"];

/// Fit one temperature per question type on a separate labeled set.
///
/// Parity with `calibration.fit_temperature`: grid search (241 log-spaced points
/// from T=0.05 to T=20, plus T=1) minimising the average NLL, first minimum wins.
/// The returned status never claims validation — fitting loss is not evidence.
pub fn fit_temperature(rows: &[Value], fingerprint: &str) -> Result<Calibration> {
    if rows.is_empty() {
        return Err(ValidationError("Calibration requires labeled rows".into()));
    }
    let mut groups: BTreeMap<String, Vec<LabeledLogits>> = BTreeMap::new();
    for raw in rows {
        let row: LabeledLogits = serde_json::from_value(raw.clone())
            .map_err(|e| ValidationError(format!("invalid labeled row: {e}")))?;
        if !TYPES.contains(&row.kind.as_str()) {
            return Err(ValidationError(format!(
                "labeled row type must be one of: {}",
                TYPES.join(", ")
            )));
        }
        if row.logits.len() < 2 || row.logits.len() > 26 {
            return Err(ValidationError(
                "labeled row needs between 2 and 26 logits".into(),
            ));
        }
        // the same finiteness/positivity checks the Python softmax applies
        crate::decisions::softmax(&row.logits, 1.0)?;
        if row.label_index >= row.logits.len() {
            return Err(ValidationError(
                "Label index is outside the candidate list".into(),
            ));
        }
        groups.entry(row.kind.clone()).or_default().push(row);
    }

    let mut temperatures = BTreeMap::new();
    let mut metrics = serde_json::Map::new();
    for (kind, group) in &groups {
        if group.len() < 10 {
            return Err(ValidationError(format!(
                "Provide at least 10 calibration examples for {kind}"
            )));
        }
        let loss = |log_temperature: f64| -> f64 {
            let temperature = log_temperature.exp();
            let mut total = 0.0;
            for row in group {
                let values: Vec<f64> = row.logits.iter().map(|x| x / temperature).collect();
                let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let log_sum_exp = top + values.iter().map(|v| (v - top).exp()).sum::<f64>().ln();
                total += log_sum_exp - values[row.label_index];
            }
            total / group.len() as f64
        };
        // Positive one-dimensional search including T=1; bounded to avoid singular solutions.
        let mut grid: Vec<f64> = (0..=240)
            .map(|i| 0.05_f64.ln() + i as f64 * 400_f64.ln() / 240.0)
            .collect();
        grid.push(0.0);
        let mut best = grid[0];
        let mut best_loss = loss(grid[0]);
        for candidate in &grid[1..] {
            let value = loss(*candidate);
            if value < best_loss {
                best_loss = value;
                best = *candidate;
            }
        }
        temperatures.insert(kind.clone(), best.exp());
        metrics.insert(
            kind.clone(),
            serde_json::json!({
                "rows": group.len(),
                "fit_nll_before": loss(0.0),
                "fit_nll_after": best_loss,
            }),
        );
    }

    Ok(Calibration {
        version: 1,
        fingerprint: fingerprint.to_string(),
        dataset_sha256: sha256_canonical(&Value::Array(rows.to_vec())),
        temperatures,
        fit_metrics: Value::Object(metrics),
        status: default_status(),
    })
}

impl Calibration {
    /// Load and validate a calibration file (JSON). All temperatures must be
    /// finite and positive; missing types fall back to temperature 1.0.
    pub fn from_file(path: &str) -> Result<Calibration> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ValidationError(format!("calibration read: {e}")))?;
        let cal: Calibration = serde_json::from_str(&text)
            .map_err(|e| ValidationError(format!("invalid calibration: {e}")))?;
        cal.validate()?;
        Ok(cal)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != 1 {
            return Err(ValidationError("calibration version must be 1".into()));
        }
        if self.fingerprint.trim().is_empty() {
            return Err(ValidationError("calibration fingerprint is empty".into()));
        }
        for (kind, t) in &self.temperatures {
            if !TYPES.contains(&kind.as_str()) {
                return Err(ValidationError(format!(
                    "calibration temperature for unknown type {kind:?}"
                )));
            }
            if !t.is_finite() || *t <= 0.0 {
                return Err(ValidationError(format!(
                    "calibration temperature for {kind} must be finite and positive"
                )));
            }
        }
        Ok(())
    }

    /// Per-type temperature, default 1.0 (parity with `temperatures.get(type, 1.0)`).
    pub fn temperature(&self, type_name: &str) -> f64 {
        self.temperatures.get(type_name).copied().unwrap_or(1.0)
    }

    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// Canonical serialization: compact JSON with sorted object keys (parity with
/// the Python `prompts.canonical`). Used for framework/data fingerprints.
pub fn canonical(value: &Value) -> String {
    fn sorted(v: &Value) -> Value {
        match v {
            Value::Object(map) => {
                let mut sorted_map = serde_json::Map::new();
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                for k in keys {
                    sorted_map.insert(k.clone(), sorted(&map[k]));
                }
                Value::Object(sorted_map)
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&sorted(value)).unwrap_or_default()
}

/// SHA-256 hex digest of canonical JSON (parity with Python hashing).
pub fn sha256_canonical(value: &Value) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(canonical(value).as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_sorts_keys_recursively() {
        let v: Value = serde_json::json!({"b": 2, "a": {"d": 4, "c": 3}});
        assert_eq!(canonical(&v), r#"{"a":{"c":3,"d":4},"b":2}"#);
    }

    #[test]
    fn sha256_matches_python_canonical() {
        // Parity check against the Python reference:
        //   hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True,
        //                              separators=(",", ":")).encode()).hexdigest()
        // for v = "abc" the canonical form is the quoted string `"abc"`.
        assert_eq!(
            sha256_canonical(&Value::String("abc".into())),
            "6cc43f858fbb763301637b5af970e2a46b46f461f27e5a0f41e009c59b827b25"
        );
    }

    #[test]
    fn calibration_rejects_bad_temperature() {
        let raw = serde_json::json!({
            "version": 1,
            "fingerprint": "abc",
            "dataset_sha256": "def",
            "temperatures": {"choice": 0.0},
            "fit_metrics": {},
            "status": "fitted_requires_held_out_validation"
        });
        let cal: Calibration = serde_json::from_value(raw).unwrap();
        assert!(cal.validate().is_err());
    }

    fn overconfident_rows(kind: &str, count: usize, scale: f64) -> Vec<Value> {
        // logits scaled by `scale` with the label always at index 1: a temperature
        // near 1/scale should undo the overconfidence.
        (0..count)
            .map(|i| {
                serde_json::json!({
                    "type": kind,
                    "logits": [0.0 * scale, 4.0 * scale, -2.0 * scale, 0.5 * scale + i as f64 * 1e-3],
                    "label_index": 1
                })
            })
            .collect()
    }

    #[test]
    fn fit_temperature_requires_ten_rows_per_type() {
        let rows = overconfident_rows("choice", 9, 2.0);
        let err = fit_temperature(&rows, "fp").unwrap_err();
        assert!(err.0.contains("at least 10"), "{}", err.0);
    }

    #[test]
    fn fit_temperature_finds_a_smaller_temperature_for_overconfident_logits() {
        let rows = overconfident_rows("boolean", 40, 3.0);
        let cal = fit_temperature(&rows, "fp-abc").unwrap();
        let t = cal.temperature("boolean");
        assert!(t > 0.0 && t.is_finite());
        // overconfident logits (scaled 3x) need sharpening: T well below 1
        assert!(t < 0.8, "expected T < 0.8, got {t}");
        // fitting loss must improve (or stay equal) after fitting
        let before = cal.fit_metrics["boolean"]["fit_nll_before"].as_f64().unwrap();
        let after = cal.fit_metrics["boolean"]["fit_nll_after"].as_f64().unwrap();
        assert!(after <= before);
        assert_eq!(cal.fit_metrics["boolean"]["rows"], 40);
        // the fit never claims validation
        assert_eq!(cal.status, "fitted_requires_held_out_validation");
        assert_eq!(cal.version, 1);
        assert_eq!(cal.fingerprint, "fp-abc");
        assert_eq!(cal.dataset_sha256.len(), 64);
        assert_eq!(cal.temperature("score"), 1.0, "unfitted type defaults to 1.0");
    }

    #[test]
    fn fit_temperature_rejects_bad_rows() {
        let mut rows = overconfident_rows("choice", 12, 2.0);
        rows.push(serde_json::json!({"type": "choice", "logits": [1.0], "label_index": 0}));
        assert!(fit_temperature(&rows, "fp").is_err(), "single logit is invalid");

        let mut rows = overconfident_rows("choice", 12, 2.0);
        rows[0]["label_index"] = serde_json::json!(9);
        assert!(fit_temperature(&rows, "fp").is_err(), "label index out of range");

        let mut rows = overconfident_rows("choice", 12, 2.0);
        rows[0]["type"] = serde_json::json!("noul");
        assert!(fit_temperature(&rows, "fp").is_err(), "unknown type");

        assert!(fit_temperature(&[], "fp").is_err(), "empty set");
    }

    #[test]
    fn calibration_serializes_in_the_python_field_order() {
        let cal = fit_temperature(&overconfident_rows("numeric", 10, 2.0), "fp").unwrap();
        let text = serde_json::to_string(&cal).unwrap();
        let order: Vec<&str> = ["version", "fingerprint", "dataset_sha256", "temperatures", "fit_metrics", "status"]
            .into_iter()
            .collect();
        let mut last = 0;
        for key in order {
            let at = text.find(&format!("\"{key}\"")).unwrap_or_else(|| panic!("{key} missing"));
            assert!(at >= last, "{key} out of order");
            last = at;
        }
    }
}