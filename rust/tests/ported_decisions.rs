//! Ported tests — faithful translation of `tests/test_decisions.py` (Rizzo Flow, Apache-2.0).
//! Same scenarios, same expected numbers: this is the parity baseline for the Rust port.

use rizzo_flow_rs::decisions::{decode, softmax, ABOVE, BELOW, UNKNOWN};
use rizzo_flow_rs::schema::{NumericQuestion, Question, Request, ValidationError};

/// Helper mirroring the Python `question()` fixture: build a single-question request and
/// return the validated question. `extra` is a raw JSON fragment appended to the question.
fn question(kind: &str, extra: &str) -> Question {
    let payload = format!(
        r#"{{"state":"Example","questions":{{"q":{{"type":"{kind}","instructions":"Evaluate the evidence"{extra}}}}}}}"#
    );
    Request::from_json_str(&payload)
        .expect("request should validate")
        .questions
        .into_values()
        .next()
        .expect("one question")
}

fn approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps * b.abs().max(1.0)
}

#[test]
fn score_expected_value_and_polarization() {
    let q = question(
        "score",
        r#","levels":["low","medium","high"],"policy":{"allow_abstain":false}"#,
    );
    let result = decode(&q, &[0.05_f64.ln(), 0.06_f64.ln(), 0.89_f64.ln()], 1.0).unwrap();
    let score = result["score"].as_f64().unwrap();
    assert!(approx(score, 1.84, 1e-9), "score {score} != 1.84");

    let polar = decode(&q, &[0.0, -1000.0, 0.0], 1.0).unwrap();
    let center = decode(&q, &[-1000.0, 0.0, -1000.0], 1.0).unwrap();
    let p_score = polar["score"].as_f64().unwrap();
    let c_score = center["score"].as_f64().unwrap();
    assert!(approx(p_score, 1.0, 1e-9) && approx(c_score, 1.0, 1e-9));
    assert!(approx(
        polar["statistics_given_available"]["stddev"].as_f64().unwrap(),
        1.0,
        1e-9
    ));
    assert!(approx(
        center["statistics_given_available"]["stddev"].as_f64().unwrap(),
        0.0,
        1e-12
    ));
}

#[test]
fn numeric_real_anchors_and_nonuniform_spacing() {
    let q = question(
        "numeric",
        r#","unit":"EUR","anchors":[{"value":180000,"description":"180000"},{"value":225000,"description":"225000"},{"value":275000,"description":"275000"},{"value":325000,"description":"325000"},{"value":375000,"description":"375000"}],"policy":{"allow_abstain":false}"#,
    );
    let ps = [0.02_f64, 0.10, 0.65, 0.21, 0.02];
    let mut logits: Vec<f64> = ps.iter().map(|p| p.ln()).collect();
    logits.extend([-1000.0, -1000.0]);
    let result = decode(&q, &logits, 1.0).unwrap();
    let value = result["value"].as_f64().unwrap();
    assert!(approx(value, 280600.0, 1e-9), "value {value} != 280600");
    assert_eq!(result["unit"], "EUR");
    let total: f64 = result["probabilities"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_f64().unwrap())
        .sum();
    assert!(approx(total, 1.0, 1e-9));
}

#[test]
fn no_nonfinite_numbers() {
    // softmax rejects non-finite logits (three cases, like the parametrized Python test)
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(softmax(&[0.0, bad], 1.0).is_err());
    }
    // JSON cannot carry NaN/Infinity: parsing must fail, never clamp silently
    assert!(Request::from_json_str(
        r#"{"state":"x","questions":{"q":{"type":"numeric","unit":"x","anchors":[{"value":0,"description":"x"},{"value":NaN,"description":"y"}]}}}"#
    )
    .is_err());
    assert!(Request::from_json_str(
        r#"{"state":{"x":1e999},"questions":{"q":{"type":"boolean","instructions":"x"}}}"#
    )
    .is_err());
    // defense in depth: programmatic non-finite values are rejected by validation
    let mut q = NumericQuestion {
        instructions: "x".into(),
        policy: Default::default(),
        unit: "x".into(),
        anchors: vec![
            rizzo_flow_rs::schema::Anchor {
                value: 0.0,
                description: "x".into(),
            },
            rizzo_flow_rs::schema::Anchor {
                value: f64::INFINITY,
                description: "y".into(),
            },
        ],
    };
    assert!(Question::Numeric(q.clone()).validate().is_err());
    q.anchors[1].value = 1.0;
    assert!(Question::Numeric(q).validate().is_ok());
}

#[test]
fn numeric_out_of_range_and_insufficient_are_not_clamped() {
    let q = question(
        "numeric",
        r#","unit":"kg","anchors":[{"value":0,"description":"empty"},{"value":10,"description":"full"}]"#,
    );
    let cs = rizzo_flow_rs::candidates(&q);
    for (candidate, status) in [
        (ABOVE, "out_of_range"),
        (BELOW, "out_of_range"),
        (UNKNOWN, "insufficient_evidence"),
    ] {
        let logits: Vec<f64> = cs
            .iter()
            .map(|c| if c.id == candidate { 20.0 } else { 0.0 })
            .collect();
        let answer = decode(&q, &logits, 1.0).unwrap();
        assert_eq!(answer["status"], status, "candidate {candidate}");
        assert_eq!(answer["value"], serde_json::Value::Null);
        assert_eq!(
            answer["statistics_given_available"],
            serde_json::Value::Null
        );
    }
}

#[test]
fn abstention_uses_combined_unavailable_mass() {
    let q = question(
        "numeric",
        r#","unit":"x","anchors":[{"value":1,"description":"one"},{"value":2,"description":"two"}]"#,
    );
    let logits: Vec<f64> = [0.30_f64, 0.10, 0.20, 0.20, 0.20]
        .iter()
        .map(|p| p.ln())
        .collect();
    let a = decode(&q, &logits, 1.0).unwrap();
    assert_eq!(a["value"], serde_json::Value::Null);
}

#[test]
fn boolean_unknown_is_not_false() {
    let q = question("boolean", "");
    let a = decode(&q, &[0.0, 0.0, 20.0], 1.0).unwrap();
    assert_eq!(a["value"], serde_json::Value::Null);
    assert_eq!(a["status"], "insufficient_evidence");
}

#[test]
fn choice_rejects_duplicate_or_reserved_ids() {
    for ids in [("same", "same"), ("__insufficient__", "valid")] {
        let payload = format!(
            r#"{{"state":"Example","questions":{{"q":{{"type":"choice","instructions":"x","options":[{{"id":"{}","description":"a"}},{{"id":"{}","description":"b"}}]}}}}}}"#,
            ids.0, ids.1
        );
        let outcome = Request::from_json_str(&payload);
        assert!(
            matches!(outcome, Err(ValidationError(_))),
            "ids {ids:?} should be rejected"
        );
    }
}

#[test]
fn low_probability_policy() {
    let q = question(
        "boolean",
        r#","policy":{"allow_abstain":false,"min_top_probability":0.9}"#,
    );
    assert_eq!(decode(&q, &[0.0, 0.0], 1.0).unwrap()["status"], "uncertain");
}

#[test]
fn invalid_shape_and_temperature() {
    let q = question("boolean", "");
    assert!(decode(&q, &[0.0, 1.0], 1.0).is_err());
    for t in [0.0, -1.0, f64::NAN] {
        assert!(softmax(&[1.0, 2.0], t).is_err());
    }
}

#[test]
fn strict_request_and_numeric_order() {
    assert!(Request::from_json_str(
        r#"{"state":"Example","questions":{"q":{"type":"numeric","instructions":"x","unit":"x","anchors":[{"value":2,"description":"a"},{"value":1,"description":"b"}]}}}"#
    )
    .is_err());
    assert!(Request::from_json_str(
        r#"{"state":"Example","questions":{"q":{"type":"boolean","instructions":"x","unsupported":true}}}"#
    )
    .is_err());
}
