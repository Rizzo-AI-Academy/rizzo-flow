//! Ported compat tests — the TypeSafe wire format (`compat.py`).
//!
//! These exercise the pure translation layer end-to-end with a synthetic native
//! response, so they run without a model. The HTTP surface is verified on the GPU
//! node with curl.

use serde_json::{json, Value};

use rizzo_flow_rs::compat::{
    confidence, from_native, list_models, model_name, resolve_model, to_native, SystemOneRequest,
    FOREIGN_PREFIX, LOCAL_ALIAS,
};

fn metadata() -> Value {
    json!({
        "source": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "precision": "q4_k_m",
        "fingerprint": "fp-1234",
        "load_seconds": 1.5
    })
}

#[test]
fn systemone_parse_rejects_unknown_fields_and_shapes() {
    // extra field at top level (pydantic extra="forbid")
    let bad = json!({
        "state": {}, "model": "rizzo-latest",
        "questions": {"q": {"type": "noul", "instructions": "ok"}},
        "surprise": true
    });
    assert!(SystemOneRequest::parse(bad).is_err());

    // unknown question type
    let bad = json!({
        "state": {}, "model": "rizzo-latest",
        "questions": {"q": {"type": "numeric", "instructions": "ok"}}
    });
    assert!(SystemOneRequest::parse(bad).is_err());

    // choice with one option (min_length=2)
    let bad = json!({
        "state": {}, "model": "rizzo-latest",
        "questions": {"q": {"type": "choice", "instructions": "ok", "criteria": {"a": null}}}
    });
    assert!(SystemOneRequest::parse(bad).is_err());

    // score with 11 levels (max 10)
    let levels: Vec<String> = (0..11).map(|i| i.to_string()).collect();
    let bad = json!({
        "state": {}, "model": "rizzo-latest",
        "questions": {"q": {"type": "score", "instructions": "ok", "criteria": levels}}
    });
    assert!(SystemOneRequest::parse(bad).is_err());

    // model too long (>128)
    let bad = json!({
        "state": {}, "model": "x".repeat(129),
        "questions": {"q": {"type": "noul", "instructions": "ok"}}
    });
    assert!(SystemOneRequest::parse(bad).is_err());
}

#[test]
fn systemone_parse_accepts_the_three_question_kinds() {
    let ok = json!({
        "state": {"last_event": "synthetic"},
        "model": LOCAL_ALIAS,
        "questions": {
            "supported": {"type": "noul", "instructions": "Is it supported?",
                          "criteria": {"true": "yes it is", "false": "no it is not"}},
            "route": {"type": "choice", "instructions": "Which route?",
                      "criteria": {"billing": "the billing path", "access": null}},
            "urgency": {"type": "score", "instructions": "How urgent?",
                        "criteria": ["low", "medium", "high"]}
        }
    });
    let parsed = SystemOneRequest::parse(ok).unwrap();
    assert_eq!(parsed.questions.len(), 3);
}

#[test]
fn to_native_maps_noul_choice_score_and_keeps_option_order() {
    let wire = json!({
        "state": "some state",
        "model": "rizzo-latest",
        "questions": {
            "supported": {"type": "noul", "instructions": "Is it supported?",
                          "criteria": {"true": "yes it is", "false": "no it is not"}},
            "route": {"type": "choice", "instructions": "Which route?",
                      "criteria": {"zeta": "last first", "alpha": "first second", "mu": null}},
            "urgency": {"type": "score", "instructions": "How urgent?",
                        "criteria": ["low", "medium", "high"]}
        }
    });
    let wire = SystemOneRequest::parse(wire).unwrap();
    let (native, options) = to_native(&wire).unwrap();
    let native = serde_json::to_value(&native).unwrap();

    // noul -> boolean, descriptions prefixed
    assert_eq!(native["questions"]["supported"]["type"], "boolean");
    assert_eq!(
        native["questions"]["supported"]["true_description"],
        "Yes. yes it is"
    );
    assert_eq!(
        native["questions"]["supported"]["false_description"],
        "No. no it is not"
    );
    // the wire format has no abstention outcome
    assert_eq!(
        native["questions"]["supported"]["policy"]["allow_abstain"],
        false
    );

    // choice -> positional ids, wire order preserved (zeta, alpha, mu — NOT alphabetical)
    assert_eq!(
        options["route"],
        vec!["zeta".to_string(), "alpha".to_string(), "mu".to_string()]
    );
    assert_eq!(native["questions"]["route"]["options"][0]["id"], "o0");
    assert_eq!(
        native["questions"]["route"]["options"][0]["description"],
        "zeta: last first"
    );
    assert_eq!(native["questions"]["route"]["options"][1]["id"], "o1");
    assert_eq!(
        native["questions"]["route"]["options"][2]["description"],
        "mu"
    );

    // score -> levels as text
    assert_eq!(native["questions"]["urgency"]["type"], "score");
    assert_eq!(native["questions"]["urgency"]["levels"][1], "medium");
}

#[test]
fn from_native_translates_answers_back_to_the_wire() {
    let wire = json!({
        "state": "s",
        "model": "rizzo-latest",
        "questions": {
            "supported": {"type": "noul", "instructions": "ok"},
            "route": {"type": "choice", "instructions": "ok",
                      "criteria": {"billing": null, "access": null}}
        }
    });
    let wire = SystemOneRequest::parse(wire).unwrap();
    let (_native, options) = to_native(&wire).unwrap();

    // synthetic native response, as produced by the engine
    let response = json!({
        "model": {"fingerprint": "fp-1234"},
        "answers": {
            "supported": {
                "type": "boolean", "status": "ok",
                "probabilities": {"false": 0.05, "true": 0.95},
                "probability_status": "uncalibrated_conditional_option_scores",
                "input_tokens": 300
            },
            "route": {
                "type": "choice", "status": "ok",
                "probabilities": {"o0": 0.2, "o1": 0.8, "__insufficient__": 0.0},
                "probability_status": "uncalibrated_conditional_option_scores",
                "input_tokens": 280
            }
        },
        "timing": {"shared_prefix_tokens": 250}
    });

    let out = from_native(&wire, &response, &options, "rizzo-served").unwrap();
    assert_eq!(out["model"], "rizzo-served");
    assert_eq!(out["answers"]["supported"]["type"], "noul");
    assert_eq!(out["answers"]["supported"]["noul"], 0.95);
    assert_eq!(out["answers"]["route"]["choice"], "access");
    assert_eq!(out["answers"]["route"]["probabilities"]["billing"], 0.2);
    assert_eq!(out["answers"]["route"]["probabilities"]["access"], 0.8);
    // confidence: 2 candidates, peak 0.8 → (2*0.8 - 1) / 1 = 0.6
    let c = out["answers"]["route"]["confidence"].as_f64().unwrap();
    assert!((c - 0.6).abs() < 1e-12);
    // usage: (300 + 280) - 250 * (2 - 1) = 330; nothing is generated
    assert_eq!(out["usage"]["input_tokens"], 330);
    assert_eq!(out["usage"]["output_tokens"], 0);
    assert_eq!(out["x_rizzo"]["fingerprint"], "fp-1234");
    assert_eq!(
        out["x_rizzo"]["probability_status"][0],
        "uncalibrated_conditional_option_scores"
    );
}

#[test]
fn list_models_and_resolution_follow_the_hosted_contract() {
    let md = metadata();
    let catalogue = list_models(&md);
    let names: Vec<&str> = catalogue["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec![LOCAL_ALIAS, &model_name(&md), "jev-latest"]);
    assert!(names[2].starts_with(FOREIGN_PREFIX));
    // every entry carries the contract fields
    for entry in catalogue["models"].as_array().unwrap() {
        assert!(entry["description"].is_string());
        assert!(entry["release_date"].is_string());
    }
    assert!(resolve_model("jev-anything", &md).is_ok());
    assert!(resolve_model("some-other-model", &md).is_err());
}

#[test]
fn confidence_matches_the_public_formula() {
    assert_eq!(confidence(&[1.0, 0.0]), 1.0);
    assert_eq!(confidence(&[0.0, 1.0]), 1.0);
    assert_eq!(confidence(&[0.5, 0.5]), 0.0);
    assert!((confidence(&[0.8, 0.1, 0.1]) - 0.7).abs() < 1e-12);
    // clamp at both ends
    assert_eq!(confidence(&[1.0, 1.0]), 1.0);
}
