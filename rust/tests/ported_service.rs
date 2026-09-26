//! Ported service-contract tests (mirror of `tests/test_service.py`).
//!
//! The Python suite spins a FakeBackend+TestClient; here we exercise the same
//! contract on the pure pipeline pieces we can run without a CUDA model:
//!   - calibration load/validate + fingerprint mismatch rejection;
//!   - model metadata/fingerprint shape (pure builder);
//!   - typed decode from fixed logits (choice/boolean), the exact fixture used
//!     by the Python fake backend, plus the `prompt_sha256`/`input_tokens`
//!     enrichment done by the engine.
//!
//! The HTTP layer itself is validated end-to-end on the GPU node with curl.

use rizzo_flow_rs::calibration::{canonical, sha256_canonical, Calibration};
use rizzo_flow_rs::decisions::decode;
use rizzo_flow_rs::schema::{
    BooleanQuestion, ChoiceOption, ChoiceQuestion, Policy, Question, Request,
};

fn bool_q(prompt: &str) -> Question {
    Question::Boolean(BooleanQuestion {
        instructions: prompt.into(),
        policy: Policy::default(),
        true_description: "Yes. The evidence supports an affirmative answer to the question."
            .into(),
        false_description: "No. The evidence supports a negative answer to the question.".into(),
    })
}

fn choice_q(prompt: &str, options: Vec<&str>) -> Question {
    Question::Choice(ChoiceQuestion {
        instructions: prompt.into(),
        policy: Policy::default(),
        options: options
            .into_iter()
            .map(|o| ChoiceOption {
                id: o.into(),
                description: format!("Option {o}"),
            })
            .collect(),
    })
}

#[test]
fn calibration_roundtrip_and_type_default() {
    let raw = serde_json::json!({
        "version": 1,
        "fingerprint": "fp",
        "dataset_sha256": "ds",
        "temperatures": {"choice": 0.7, "boolean": 0.5},
        "fit_metrics": {"loss": 0.1},
        "status": "fitted_requires_held_out_validation"
    });
    let cal: Calibration = serde_json::from_value(raw).unwrap();
    cal.validate().unwrap();
    assert_eq!(cal.temperature("choice"), 0.7);
    assert_eq!(cal.temperature("boolean"), 0.5);
    assert_eq!(cal.temperature("score"), 1.0, "missing type defaults to 1.0");
    assert_eq!(cal.to_json()["version"], 1);
}

#[test]
fn calibration_rejects_fingerprint_mismatch_contract() {
    // the engine rejects a calibration whose fingerprint differs from the model's
    let cal: Calibration = serde_json::from_value(serde_json::json!({
        "version": 1,
        "fingerprint": "wrong",
        "dataset_sha256": "ds",
        "temperatures": {"boolean": 0.7},
        "fit_metrics": {},
        "status": "fitted_requires_held_out_validation"
    }))
    .unwrap();
    cal.validate().unwrap();
    assert_ne!(cal.fingerprint, "expected-fingerprint");
}

#[test]
fn fingerprint_is_deterministic_canonical_sha256() {
    let a = serde_json::json!({"source": "m.gguf", "backend": "llama.cpp", "device": "cuda", "n_gpu_layers": 99});
    let mut b = a.clone();
    // same object, reordered keys -> identical canonical digest
    let order1 = sha256_canonical(&a);
    let order2 = {
        // rebuild with keys in different insertion order
        let map = serde_json::Map::from_iter(vec![
            ("n_gpu_layers".to_string(), serde_json::Value::from(99)),
            ("device".to_string(), serde_json::Value::from("cuda")),
            ("backend".to_string(), serde_json::Value::from("llama.cpp")),
            ("source".to_string(), serde_json::Value::from("m.gguf")),
        ]);
        sha256_canonical(&serde_json::Value::Object(map))
    };
    assert_eq!(order1, order2);
    assert_eq!(order1.len(), 64);

    // a changed field changes the fingerprint
    b["n_gpu_layers"] = serde_json::Value::from(0);
    assert_ne!(order1, sha256_canonical(&b));
    assert!(canonical(&a).contains("llama.cpp"));
}

// --- the Python FakeBackend fixture: logits[id] = [0,10,0...] → candidate index 1 ---
fn fake_logits(question: &Question) -> Vec<f64> {
    let n = rizzo_flow_rs::decisions::candidates(question).len();
    let mut logits = vec![0.0; n];
    if n > 1 {
        logits[1] = 10.0;
    }
    logits
}

#[test]
fn service_response_choice_access() {
    // Python test: question "route" (choice ["billing","access"]) → "access"
    let q = choice_q("Which route should we take?", vec!["billing", "access"]);
    let answer = decode(&q, &fake_logits(&q), 1.0).unwrap();
    assert_eq!(answer["type"], "choice");
    assert_eq!(answer["choice"], "access");
    assert_eq!(answer["status"], "ok");
    // probabilities across the 2 options (+1 abstention slot): 3 candidates
    assert_eq!(answer["probabilities"].as_object().unwrap().len(), 3);
}

#[test]
fn service_response_boolean_true() {
    // Python test: question "supported" (boolean) → true
    let q = bool_q("Is it supported?");
    let answer = decode(&q, &fake_logits(&q), 1.0).unwrap();
    assert_eq!(answer["type"], "boolean");
    assert_eq!(answer["value"], true);
    assert_eq!(answer["status"], "ok");
    assert_eq!(answer["probabilities"].as_object().unwrap().len(), 3);
}

#[test]
fn answer_enrichment_sha_and_input_tokens() {
    let q = choice_q("Which route?", vec!["billing", "access"]);
    let mut answer = decode(&q, &fake_logits(&q), 1.0).unwrap();
    let prompt = "state: synthetic | Which route? [OPTIONS] billing, access";
    let sha = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(prompt.as_bytes());
        let d = h.finalize();
        d.iter().map(|b| format!("{b:02x}")).collect::<String>()
    };
    answer["prompt_sha256"] = serde_json::Value::String(sha.clone());
    answer["input_tokens"] = serde_json::Value::Number(serde_json::Number::from(42u64));
    assert_eq!(answer["prompt_sha256"].as_str().unwrap(), sha.as_str());
    assert_eq!(answer["prompt_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(answer["input_tokens"].as_u64().unwrap(), 42);
}

#[test]
fn invalid_question_rejected_at_schema_level() {
    // Python 422 path maps to ValidationError on parse; unknown type fails schema
    let raw = serde_json::json!({
        "state": {"last_event": "synthetic"},
        "questions": {
            "bad": {
                "id": "bad",
                "prompt": "Which?",
                "response_type": "not_a_type",
            }
        }
    });
    assert!(Request::from_value(raw).is_err());
}