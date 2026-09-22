import copy

import pytest
from fastapi.testclient import TestClient

from rizzo_flow.api import create_app
from rizzo_flow.calibration import fit_temperature
from rizzo_flow.config import MODEL_ID
from rizzo_flow.engine import Engine
from rizzo_flow.evaluation import evaluate
from rizzo_flow.metadata import LlamaMetadata
from rizzo_flow.prompts import PROMPT_VERSION, compile_request
from rizzo_flow.responses import Timing
from rizzo_flow.schema import Request


class CharacterTokenizer:
    """Test tokenizer; deliberately distinct from real Spark tokenizer integration tests."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"


class FakeBackend:
    tokenizer = CharacterTokenizer()

    def __init__(self):
        self.metadata = LlamaMetadata(
            source=MODEL_ID,
            requested_revision="test",
            gguf_source=None,
            gguf_revision=None,
            source_files={"fake.gguf": "0" * 64},
            precision="q8_0",
            device="cpu",
            backend="cpu",
            runtime="llama.cpp",
            llama_cpp_release="test",
            llama_cpp_commit="test",
            prompt_version=PROMPT_VERSION,
        )

    def score(self, prefix, jobs, mode):
        logits = {j.id: [0, 10] + [0] * (len(j.slots) - 2) for j in jobs}
        return logits, Timing(
            inference_seconds=0.0,
            prefill_seconds=0.0,
            shared_prefix_tokens=len(prefix),
            evaluated_tokens_including_padding=sum(len(j.tokens) for j in jobs),
            logical_input_tokens=sum(len(j.tokens) for j in jobs),
            batches=1,
            generated_tokens=0,
        )


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
    assert metric["fit_nll_after"] < metric["fit_nll_before"]
    Engine(FakeBackend(), calibration=calibration)
    calibration.fingerprint = "different"
    with pytest.raises(ValueError, match="different"):
        Engine(FakeBackend(), calibration=calibration)


def test_evaluation_coverage_raw_evidence(payload):
    fixtures = [
        {
            "id": "sample",
            "request": copy.deepcopy(payload),
            "expected": {
                "route": {"label": "access", "status": "ok"},
                "supported": {"label": "true"},
            },
        }
    ]
    report = evaluate(Engine(FakeBackend()), fixtures, compare_modes=True)
    assert report["summary"]["categorical"]["accuracy"] == 1
    assert report["summary"]["mode_comparison"]["changed_argmaxes"] == 0
    assert report["rows"][0]["response"]["timing"]["generated_tokens"] == 0
