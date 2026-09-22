import copy
import json

import pytest
from fakes import CharacterTokenizer, FakeBackend
from fastapi.testclient import TestClient

from rizzo_flow.api import create_app
from rizzo_flow.calibration import fit_temperature
from rizzo_flow.engine import Engine
from rizzo_flow.evaluation import evaluate
from rizzo_flow.prompts import compile_request
from rizzo_flow.schema import Request


@pytest.fixture
def payload():
    return {
        "state": {"ticket": "Cannot log in"},
        "questions": {
            "supported": {"type": "boolean", "instructions": "Does the user need login help?"},
            "route": {
                "type": "choice",
                "instructions": "Choose a queue",
                "options": [
                    {"id": "billing", "description": "Payment problem"},
                    {"id": "access", "description": "Login problem"},
                ],
            },
        },
    }


def test_shared_prefix_and_state_mutation(payload):
    request = Request.model_validate(payload)
    prefix, jobs = compile_request(CharacterTokenizer(), request, 8192)
    assert all(job.tokens[: len(prefix)] == prefix for job in jobs)
    payload["state"]["ticket"] = "Changed"
    other, _ = compile_request(CharacterTokenizer(), Request.model_validate(payload), 8192)
    assert other != prefix


def test_limits_reject_without_truncation(payload):
    with pytest.raises(ValueError, match="no truncation"):
        Engine(FakeBackend(), ctx=10).decide(payload)


def test_api_and_all_input_validation(payload):
    with TestClient(create_app(Engine(FakeBackend()))) as client:
        assert client.get("/health").json()["status"] == "ready"
        response = client.post("/v1/decisions", json=payload)
        assert response.status_code == 200
        assert response.json()["answers"]["route"]["choice"] == "access"
        assert response.json()["answers"]["supported"]["value"] is True
        payload["questions"]["route"]["options"][1]["id"] = "billing"
        assert client.post("/v1/decisions", json=payload).status_code == 422


def test_temperature_fit_and_model_binding():
    rows = [{"type": "choice", "logits": [0, 8], "label_index": int(i % 2 == 0)} for i in range(20)]
    calibration = fit_temperature(rows, FakeBackend().metadata.fingerprint)
    assert calibration.temperatures["choice"] > 1
    metric = calibration.fit_metrics["choice"]
    assert metric.fit_nll_after < metric.fit_nll_before
    Engine(FakeBackend(), calibration=calibration)
    calibration.fingerprint = "different"
    with pytest.raises(ValueError, match="different"):
        Engine(FakeBackend(), calibration=calibration)


def test_evaluation_coverage_raw_evidence(payload):
    # A score question as well, so the numeric side of the report is exercised: the fake
    # favours the second of three levels, so the expected value is exactly 1.
    payload["questions"]["severity"] = {
        "type": "score",
        "instructions": "How bad is it?",
        "levels": ["low", "medium", "high"],
        "policy": {"allow_abstain": False},
    }
    fixtures = [
        {
            "id": "sample",
            "request": copy.deepcopy(payload),
            "expected": {
                "route": {"label": "access", "status": "ok"},
                "supported": {"label": "true"},
                "severity": {"value": 1},
            },
        }
    ]
    report = evaluate(Engine(FakeBackend()), fixtures, compare_modes=True)
    assert report.summary.categorical.accuracy == 1
    assert report.summary.mode_comparison.changed_argmaxes == 0
    assert report.rows[0].response.timing.generated_tokens == 0
    numeric = report.summary.numeric
    assert numeric.rows == 1
    group = '{"support":[0.0,2.0],"type":"score","unit":null}'
    assert list(numeric.by_type_unit_and_support) == [group]
    assert numeric.by_type_unit_and_support[group].answered == 1
    assert numeric.by_type_unit_and_support[group].mae_on_answered == pytest.approx(0, abs=1e-4)
    # The report is also a file format: it must survive the round trip to JSON.
    dumped = json.loads(json.dumps(report.model_dump()))
    assert dumped["summary"]["requests"] == 1
    assert dumped["rows"][0]["expected"]["severity"] == {"value": 1}
