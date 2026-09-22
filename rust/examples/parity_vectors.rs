//! Rust side of the decision-core parity check.
//!
//! Reads the golden vectors and prints the decoded answers in the same shape the
//! Python reference emits, so the two outputs can be diffed field by field.
//!
//!     cargo run --release --example parity_vectors -- parity/core_vectors.json

use serde_json::{json, Map, Value};

use rizzo_flow_rs::decisions::decode;
use rizzo_flow_rs::schema::Question;

fn main() {
    let path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("usage: parity_vectors <core_vectors.json>");
            std::process::exit(2);
        }
    };
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let vectors: Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("invalid vectors: {e}");
        std::process::exit(1);
    });
    let mut out = Map::new();
    for case in vectors["cases"].as_array().expect("cases must be an array") {
        let id = case["id"].as_str().expect("case without id").to_string();
        let question: Question = serde_json::from_value(case["question"].clone())
            .unwrap_or_else(|e| panic!("{id}: invalid question: {e}"));
        let logits: Vec<f64> = case["logits"]
            .as_array()
            .expect("logits must be an array")
            .iter()
            .map(|value| value.as_f64().expect("logits must be numbers"))
            .collect();
        let temperature = case["temperature"].as_f64().unwrap_or(1.0);
        let result = match decode(&question, &logits, temperature) {
            Ok(answer) => json!({ "ok": answer }),
            Err(error) => json!({ "error": error.0 }),
        };
        out.insert(id, result);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Object(out)).expect("serialize")
    );
}
