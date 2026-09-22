"""Pure, deterministic decoding of categorical logits into typed results."""

import math
from dataclasses import dataclass

from .schema import BooleanQuestion, ChoiceQuestion, NumericQuestion, Question, ScoreQuestion

UNKNOWN = "__insufficient__"
BELOW = "__below_range__"
ABOVE = "__above_range__"


@dataclass(frozen=True)
class Candidate:
    id: str
    description: str
    value: float | None = None


def candidates(question: Question) -> list[Candidate]:
    if isinstance(question, BooleanQuestion):
        result = [
            Candidate("false", question.false_description, 0),
            Candidate("true", question.true_description, 1),
        ]
    elif isinstance(question, ChoiceQuestion):
        result = [Candidate(o.id, o.description) for o in question.options]
    elif isinstance(question, ScoreQuestion):
        result = [Candidate(str(i), level, i) for i, level in enumerate(question.levels)]
    elif isinstance(question, NumericQuestion):
        result = [
            Candidate(
                str(i), f"Approximately {a.value:g} {question.unit}: {a.description}", a.value
            )
            for i, a in enumerate(question.anchors)
        ]
        result += [
            Candidate(BELOW, f"The value is below {question.anchors[0].value:g} {question.unit}."),
            Candidate(ABOVE, f"The value is above {question.anchors[-1].value:g} {question.unit}."),
        ]
    else:
        raise TypeError("Unsupported question")
    if question.policy.allow_abstain:
        result.append(
            Candidate(
                UNKNOWN,
                "Cannot determine the answer: the required information is not provided "
                "or is contradictory. A known value outside the numeric range is not "
                "missing information.",
            )
        )
    return result


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if len(logits) < 2 or not all(math.isfinite(x) for x in logits):
        raise ValueError("At least two finite logits are required")
    maximum = max(logits)
    weights = [math.exp((x - maximum) / temperature) for x in logits]
    total = sum(weights)
    return [x / total for x in weights]


def summarize(values: list[float], probabilities: list[float]) -> dict:
    mean = math.fsum(v * p for v, p in zip(values, probabilities, strict=True))
    variance = math.fsum(p * (v - mean) ** 2 for v, p in zip(values, probabilities, strict=True))

    def quantile(q):
        cumulative = 0.0
        for v, p in zip(values, probabilities, strict=True):
            cumulative += p
            if cumulative >= q:
                return v
        return values[-1]

    return {
        "mean": mean,
        "stddev": math.sqrt(variance),
        "median": quantile(0.5),
        "anchor_quantiles": {"p10": quantile(0.1), "p90": quantile(0.9)},
    }


def decode(question: Question, logits: list[float], temperature: float = 1.0) -> dict:
    choices = candidates(question)
    if len(logits) != len(choices):
        raise ValueError("Logit count does not match the declared candidates")
    ps = softmax(logits, temperature)
    distribution = {c.id: p for c, p in zip(choices, ps, strict=True)}
    unavailable_ids = {UNKNOWN, BELOW, ABOVE}
    valid = [(c, p) for c, p in zip(choices, ps, strict=True) if c.id not in unavailable_ids]
    unavailable = math.fsum(p for c, p in zip(choices, ps, strict=True) if c.id in unavailable_ids)
    available = math.fsum(p for _, p in valid)
    winner = choices[max(range(len(ps)), key=ps.__getitem__)].id
    top = max(ps)
    entropy = -math.fsum(p * math.log(p) for p in ps if p > 0)
    status = "ok"
    if winner in unavailable_ids or unavailable >= question.policy.max_unavailable_probability:
        status = (
            "out_of_range"
            if distribution.get(BELOW, 0) + distribution.get(ABOVE, 0)
            > distribution.get(UNKNOWN, 0)
            else "insufficient_evidence"
        )
    elif top < question.policy.min_top_probability:
        status = "uncertain"
    result = {
        "type": question.type,
        "status": status,
        "probabilities": distribution,
        "option_logits": {c.id: x for c, x in zip(choices, logits, strict=True)},
        "legend": {c.id: c.description for c in choices},
        "uncertainty": {
            "top_probability": top,
            "entropy_nats": entropy,
            "concentration": max(0.0, min(1.0, 1 - entropy / math.log(len(ps)))),
            "unavailable_probability": unavailable,
        },
        "probability_status": "uncalibrated_conditional_option_scores"
        if temperature == 1
        else "temperature_scaled_requires_held_out_validation",
        "temperature": temperature,
    }
    conditional = [p / available for _, p in valid] if available > 0 else None
    if question.type == "choice":
        result["choice"] = winner if status == "ok" else None
    elif question.type == "boolean":
        result["value"] = (winner == "true") if status == "ok" else None
        result["probability_true_given_available"] = conditional[1] if conditional else None
    else:
        values = [c.value for c, _ in valid]
        stats = summarize(values, conditional) if conditional and status == "ok" else None
        result["score" if question.type == "score" else "value"] = stats["mean"] if stats else None
        result["statistics_given_available"] = stats
        result["values"] = {c.id: c.value for c, _ in valid}
        result["support"] = [values[0], values[-1]]
        if question.type == "numeric":
            result["unit"] = question.unit
            result["range_probabilities"] = {
                "below": distribution[BELOW],
                "above": distribution[ABOVE],
            }
        else:
            result["normalized_score"] = stats["mean"] / values[-1] if stats else None
    return result
