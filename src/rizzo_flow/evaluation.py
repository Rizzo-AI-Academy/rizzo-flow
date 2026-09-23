"""Reproducible benchmark reports with coverage, proper scoring rules, and raw evidence.

The report is a file format: `results/` holds runs recorded months ago and `scripts/`
compares against them. The models below therefore declare their fields in the order the
recorded reports show, and what is genuinely absent from a run — a row's alternate-mode
response, a backend's peak memory — stays absent instead of turning into a null.
"""

import hashlib
import math
import statistics
from dataclasses import dataclass

from pydantic import model_serializer

from .engine import Engine
from .prompts import canonical
from .responses import Finite, Response, Timing
from .schema import Request, Strict

# statistics.mean over booleans returns an int when the average comes out whole, and the
# recorded reports show `1`, not `1.0`. Keep both spellings rather than widening to float.
Ratio = int | float


class Expectation(Strict):
    """What a fixture claims about one answer. Every field is optional on purpose."""

    label: str | None = None
    status: str | None = None
    value: float | None = None


class Fixture(Strict):
    id: str
    request: dict
    expected: dict[str, Expectation] = {}


@dataclass(frozen=True)
class CategoricalRow:
    """One labelled decision, kept only long enough to be averaged. Never serialized."""

    correct: bool
    accepted: bool
    nll: float
    brier: float
    confidence: float


class ReliabilityBin(Strict):
    lower: float
    count: int
    accuracy: Ratio
    mean_top_probability: float


class CategoricalSummary(Strict):
    rows: int
    accuracy: Ratio | None
    accepted_accuracy: Ratio | None
    nll: float | None
    brier: float | None
    ece_10_bins: float | None
    reliability_bins: list[ReliabilityBin]


class NumericGroup(Strict):
    rows: int
    answered: int
    mae_on_answered: float | None
    rmse_on_answered: float | None


class NumericSummary(Strict):
    rows: int
    by_type_unit_and_support: dict[str, NumericGroup]


class LatencySeconds(Strict):
    median: Finite
    p95: Finite


class ModeComparison(Strict):
    decisions: int
    changed_argmaxes: int
    max_probability_delta: Ratio
    median_seconds: dict[str, Finite | None]


class Summary(Strict):
    requests: int
    repeats: int
    warmup_excluded: bool
    decisions_per_second: Finite
    latency_seconds: LatencySeconds
    labeled_decision_coverage: Ratio | None
    status_accuracy: Ratio | None
    categorical: CategoricalSummary
    numeric: NumericSummary
    mode_comparison: ModeComparison | None


class EvidenceRow(Strict):
    id: str
    #: The fixture's own claim, echoed exactly as it was written, keys and all.
    expected: dict
    response: Response
    repeat_timings: list[Timing]
    alternate_mode_response: Response | None = None

    @model_serializer
    def without_an_unused_alternate(self) -> dict:
        row = {
            "id": self.id,
            "expected": self.expected,
            "response": self.response.model_dump(),
            "repeat_timings": [t.model_dump() for t in self.repeat_timings],
        }
        if self.alternate_mode_response is not None:
            row["alternate_mode_response"] = self.alternate_mode_response.model_dump()
        return row


class EvaluationReport(Strict):
    dataset_sha256: str
    summary: Summary
    rows: list[EvidenceRow]


def mean(values: list) -> float | None:
    return statistics.mean(values) if values else None


def evaluate(
    engine: Engine,
    fixtures: list[dict],
    repeats: int = 1,
    compare_modes: bool = False,
) -> EvaluationReport:
    if not fixtures or repeats < 1:
        raise ValueError("Provide fixtures and at least one repetition")
    # The hash covers the fixtures exactly as they were read, not as they were parsed.
    dataset_sha256 = hashlib.sha256(canonical(fixtures).encode()).hexdigest()
    cases = [Fixture.model_validate(f) for f in fixtures]
    # Explicit warmup is excluded from reported timings.
    engine.decide(cases[0].request)
    rows, latencies = [], []
    categorical: list[CategoricalRow] = []
    statuses, accepted = [], []
    numeric: dict[str, list[float | None]] = {}
    mode_times: dict[str, list[float]] = {"shared": [], "direct": []}
    decision_count = 0
    mode_deltas, changed = [], 0
    for raw, case in zip(fixtures, cases, strict=True):
        responses = [engine.decide(case.request) for _ in range(repeats)]
        latencies.extend(r.timing.total_seconds for r in responses)
        for r in responses:
            mode_times[r.mode].append(r.timing.total_seconds)
            decision_count += len(r.answers)
        response = responses[0]
        alternate = None
        for key, expected in case.expected.items():
            answer = response.answers[key]
            is_accepted = answer.status == "ok"
            accepted.append(is_accepted)
            if expected.status is not None:
                statuses.append(answer.status == expected.status)
            if expected.label is not None:
                ps = answer.probabilities
                if expected.label not in ps:
                    raise ValueError(
                        f"Unknown expected label {expected.label} in fixture {case.id}"
                    )
                predicted = max(ps, key=ps.get)
                categorical.append(
                    CategoricalRow(
                        correct=predicted == expected.label,
                        accepted=is_accepted,
                        nll=-math.log(max(ps[expected.label], 1e-300)),
                        brier=sum((p - (k == expected.label)) ** 2 for k, p in ps.items()),
                        confidence=max(ps.values()),
                    )
                )
            if expected.value is not None:
                if not math.isfinite(expected.value):
                    raise ValueError("Expected numeric targets must be finite")
                value = getattr(answer, answer.PRIMARY)
                group = canonical(
                    {
                        "type": answer.type,
                        "unit": getattr(answer, "unit", None),
                        "support": getattr(answer, "support", None),
                    }
                )
                numeric.setdefault(group, []).append(
                    None if value is None else float(value) - expected.value
                )
        if compare_modes:
            flipped = "direct" if response.mode == "shared" else "shared"
            alternate = engine.decide(
                Request.model_validate(case.request).model_copy(update={"mode": flipped})
            )
            mode_times[alternate.mode].append(alternate.timing.total_seconds)
            for key, answer in response.answers.items():
                a, b = answer.probabilities, alternate.answers[key].probabilities
                mode_deltas.append(max(abs(a[k] - b[k]) for k in a))
                changed += max(a, key=a.get) != max(b, key=b.get)
        rows.append(
            EvidenceRow(
                id=case.id,
                expected=raw.get("expected", {}),
                response=response,
                repeat_timings=[r.timing for r in responses],
                alternate_mode_response=alternate,
            )
        )

    bins, ece = [], 0.0
    for i in range(10):
        subset = [r for r in categorical if min(9, int(r.confidence * 10)) == i]
        if subset:
            accuracy = mean([r.correct for r in subset])
            confidence = mean([r.confidence for r in subset])
            ece += len(subset) / len(categorical) * abs(accuracy - confidence)
            bins.append(
                ReliabilityBin(
                    lower=i / 10,
                    count=len(subset),
                    accuracy=accuracy,
                    mean_top_probability=confidence,
                )
            )
    numeric_groups = {}
    for group, observations in numeric.items():
        errors = [x for x in observations if x is not None]
        numeric_groups[group] = NumericGroup(
            rows=len(observations),
            answered=len(errors),
            mae_on_answered=mean([abs(x) for x in errors]),
            rmse_on_answered=math.sqrt(mean([x * x for x in errors])) if errors else None,
        )
    sorted_times = sorted(latencies)
    summary = Summary(
        requests=len(cases),
        repeats=repeats,
        warmup_excluded=True,
        decisions_per_second=decision_count / sum(latencies),
        latency_seconds=LatencySeconds(
            median=statistics.median(latencies),
            p95=sorted_times[math.ceil(0.95 * len(sorted_times)) - 1],
        ),
        labeled_decision_coverage=mean(accepted),
        status_accuracy=mean(statuses),
        categorical=CategoricalSummary(
            rows=len(categorical),
            accuracy=mean([r.correct for r in categorical]),
            accepted_accuracy=mean([r.correct for r in categorical if r.accepted]),
            nll=mean([r.nll for r in categorical]),
            brier=mean([r.brier for r in categorical]),
            ece_10_bins=ece if categorical else None,
            reliability_bins=bins,
        ),
        numeric=NumericSummary(
            rows=sum(len(v) for v in numeric.values()),
            by_type_unit_and_support=numeric_groups,
        ),
        mode_comparison=ModeComparison(
            decisions=len(mode_deltas),
            changed_argmaxes=changed,
            max_probability_delta=max(mode_deltas, default=0),
            median_seconds={k: statistics.median(v) if v else None for k, v in mode_times.items()},
        )
        if compare_modes
        else None,
    )
    return EvaluationReport(dataset_sha256=dataset_sha256, summary=summary, rows=rows)
