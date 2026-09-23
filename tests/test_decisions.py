import math
from typing import Any

import pytest
from pydantic import ValidationError

from rizzo_flow.decisions import ABOVE, BELOW, UNKNOWN, candidates, decode, softmax
from rizzo_flow.responses import TypedAnswer
from rizzo_flow.schema import Question, Request


def question(kind: str = "score", **kwargs: Any) -> Question:
    data = {"type": kind, "instructions": "Evaluate the evidence", **kwargs}
    return Request.model_validate({"state": "Example", "questions": {"q": data}}).questions["q"]


def answer(q: Question, logits: list[float], temperature: float = 1.0) -> TypedAnswer:
    """decode() with the prompt facts every answer carries; irrelevant to these tests."""
    return decode(q, logits, prompt_sha256="0" * 64, input_tokens=7, temperature=temperature)


def test_score_expected_value_and_polarization() -> None:
    q = question(levels=["low", "medium", "high"], policy={"allow_abstain": False})
    result = answer(q, [math.log(p) for p in [0.05, 0.06, 0.89]])
    assert result.score == pytest.approx(1.84)
    polar = answer(q, [0, -1000, 0])
    center = answer(q, [-1000, 0, -1000])
    assert polar.score == center.score == 1
    assert polar.statistics_given_available.stddev == 1
    assert center.statistics_given_available.stddev == 0


def test_numeric_real_anchors_and_nonuniform_spacing() -> None:
    q = question(
        "numeric",
        unit="EUR",
        anchors=[
            {"value": v, "description": str(v)} for v in [180000, 225000, 275000, 325000, 375000]
        ],
        policy={"allow_abstain": False},
    )
    ps = [0.02, 0.10, 0.65, 0.21, 0.02]
    result = answer(q, [math.log(p) for p in ps] + [-1000, -1000])
    assert result.value == pytest.approx(280600)
    assert result.unit == "EUR"
    assert sum(result.probabilities.values()) == pytest.approx(1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_no_nonfinite_numbers(bad: float) -> None:
    with pytest.raises(ValueError):
        softmax([0, bad])
    with pytest.raises(ValidationError):
        question(
            "numeric",
            unit="x",
            anchors=[{"value": 0, "description": "x"}, {"value": bad, "description": "y"}],
        )
    with pytest.raises(ValidationError):
        Request.model_validate(
            {"state": {"x": bad}, "questions": {"q": {"type": "boolean", "instructions": "x"}}}
        )


def test_numeric_out_of_range_and_insufficient_are_not_clamped() -> None:
    q = question(
        "numeric",
        unit="kg",
        anchors=[{"value": 0, "description": "empty"}, {"value": 10, "description": "full"}],
    )
    cs = candidates(q)
    for candidate, status in [
        (ABOVE, "out_of_range"),
        (BELOW, "out_of_range"),
        (UNKNOWN, "insufficient_evidence"),
    ]:
        decoded = answer(q, [20 if c.id == candidate else 0 for c in cs])
        assert decoded.status == status
        assert decoded.value is None
        assert decoded.statistics_given_available is None


def test_abstention_uses_combined_unavailable_mass() -> None:
    q = question(
        "numeric",
        unit="x",
        anchors=[{"value": 1, "description": "one"}, {"value": 2, "description": "two"}],
    )
    a = answer(q, list(map(math.log, [0.30, 0.10, 0.20, 0.20, 0.20])))
    assert a.value is None


def test_boolean_unknown_is_not_false() -> None:
    q = question("boolean")
    a = answer(q, [0, 0, 20])
    assert a.value is None
    assert a.status == "insufficient_evidence"


def test_choice_rejects_duplicate_or_reserved_ids() -> None:
    for ids in [("same", "same"), ("__insufficient__", "valid")]:
        with pytest.raises(ValidationError):
            question("choice", options=[{"id": x, "description": x} for x in ids])


def test_low_probability_policy() -> None:
    q = question("boolean", policy={"allow_abstain": False, "min_top_probability": 0.9})
    assert answer(q, [0, 0]).status == "uncertain"


def test_invalid_shape_and_temperature() -> None:
    q = question("boolean")
    with pytest.raises(ValueError):
        answer(q, [0, 1])
    for t in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            softmax([1, 2], t)


def test_strict_request_and_numeric_order() -> None:
    with pytest.raises(ValidationError):
        question(
            "numeric",
            unit="x",
            anchors=[{"value": 2, "description": "a"}, {"value": 1, "description": "b"}],
        )
    with pytest.raises(ValidationError):
        question("boolean", unsupported=True)
