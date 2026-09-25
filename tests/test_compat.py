import copy

import pytest
from fastapi.testclient import TestClient
from test_service import FakeBackend

from rizzo_flow.api import create_app
from rizzo_flow.compat import SystemOneRequest, confidence, model_name, to_native
from rizzo_flow.engine import Engine


@pytest.fixture
def body():
    return {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this convey urgency?",
                "criteria": {"true": "Explicitly time-sensitive"},
            },
            "department": {
                "type": "choice",
                "instructions": {"question": "Which team should handle `ticket`?", "ticket": 7},
                "criteria": {"Billing team": "Payments, invoicing", "technical": None, "sales": []},
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    }


def client(api_key=""):
    return TestClient(create_app(Engine(FakeBackend()), api_key=api_key))


def test_systemone_wire_shape(body):
    with client() as http:
        response = http.post("/v1/systemone", json=body)
        assert response.status_code == 200
        result = response.json()
    assert result["model"].startswith("rizzo-spark-x2.5-4b")
    assert set(result) == {"model", "answers", "usage", "x_rizzo"}
    answers = result["answers"]
    assert set(answers["is_urgent"]) == {"type", "noul"}
    assert answers["is_urgent"]["noul"] > 0.99
    # FakeBackend favors the second candidate; free-form option keys survive the round trip.
    assert answers["department"]["choice"] == "technical"
    assert list(answers["department"]["probabilities"]) == ["Billing team", "technical", "sales"]
    assert sum(answers["department"]["probabilities"].values()) == pytest.approx(1)
    assert set(answers["frustration"]) == {"type", "score", "legend", "probabilities", "confidence"}
    assert answers["frustration"]["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert answers["frustration"]["score"] == pytest.approx(1, abs=1e-3)
    assert 0.99 < answers["department"]["confidence"] <= 1
    assert result["usage"]["output_tokens"] == 0 and result["usage"]["input_tokens"] > 0


def test_no_abstention_and_structured_text(body):
    native, options = to_native(SystemOneRequest.model_validate(body))
    assert all(not q.policy.allow_abstain for q in native.questions.values())
    assert options["department"] == ["Billing team", "technical", "sales"]
    department = native.questions["department"]
    assert department.instructions == '{"question":"Which team should handle `ticket`?","ticket":7}'
    assert [o.description for o in department.options] == [
        "Billing team: Payments, invoicing",
        "technical",
        "sales: []",
    ]
    assert native.questions["is_urgent"].true_description == "Yes. Explicitly time-sensitive"


def test_confidence_statistic():
    assert confidence([1, 0, 0]) == 1
    assert confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0)
    assert confidence([0.9, 0.06, 0.04]) == pytest.approx(0.85)


def test_validation_models_and_auth(body):
    with client() as http:
        names = [m["name"] for m in http.get("/v1/models").json()["models"]]
        assert "rizzo-latest" in names and "jev-latest" in names
        unknown = http.post("/v1/systemone", json={**body, "model": "gpt-unknown"})
        assert unknown.status_code == 400  # the hosted API's shape, not a validation error
        assert unknown.json()["detail"]["error_type"] == "api_usage_error"
        missing = {k: v for k, v in body.items() if k != "model"}
        assert http.post("/v1/systemone", json=missing).status_code == 422
        # The hosted API does not forbid unknown top-level fields; the SDK forwards them.
        extra = http.post("/v1/systemone", json={**body, "x_trace_id": "abc", "seed": 7})
        assert extra.status_code == 200 and "x_trace_id" not in extra.json()
        unknown_question_field = copy.deepcopy(body)
        unknown_question_field["questions"]["frustration"]["temperature"] = 0.5
        assert http.post("/v1/systemone", json=unknown_question_field).status_code == 422
        body["questions"]["frustration"]["criteria"] = ["only one"]
        assert http.post("/v1/systemone", json=body).status_code == 422
        assert "Rizzo Flow" in http.get("/playground").text
        assert "/v1/decisions" in http.get("/snake").text
        assert http.get("/playground/logo.png").headers["content-type"] == "image/png"
    with client(api_key="secret") as http:
        assert http.get("/v1/models").status_code == 401
        assert http.post("/v1/systemone", json=body).status_code == 401
        assert http.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_twenty_six_answer_letters(body):
    def choice(count):
        body["questions"]["department"]["criteria"] = {f"option {i}": None for i in range(count)}
        return body

    def native(count, abstain):
        options = [{"id": f"o{i}", "description": f"Option {i}"} for i in range(count)]
        question = {"type": "choice", "instructions": "Pick", "options": options}
        question["policy"] = {"allow_abstain": abstain}
        return {"state": "evidence", "questions": {"q": question}}

    with client() as http:
        full = http.post("/v1/systemone", json=choice(26))
        assert full.status_code == 200
        assert len(full.json()["answers"]["department"]["probabilities"]) == 26
        assert http.post("/v1/systemone", json=choice(27)).status_code == 422
        # Natively, abstention reserves one of the 26 letters.
        assert http.post("/v1/decisions", json=native(26, False)).status_code == 200
        assert http.post("/v1/decisions", json=native(25, True)).status_code == 200
        assert http.post("/v1/decisions", json=native(26, True)).status_code == 422


def test_fine_tuned_weights_get_their_own_model_name():
    base = {"source": "XHToken/Spark-X2.5-4B", "precision": "q8_0"}
    assert model_name(base) == "rizzo-spark-x2.5-4b-q8_0"
    assert model_name({**base, "weights": "flow"}) == "rizzo-flow-4b-q8_0"
    small = {"source": "XHToken/Spark-X2.5-1.7B", "precision": "bf16", "weights": "flow"}
    assert model_name(small) == "rizzo-flow-1.7b-bf16"
