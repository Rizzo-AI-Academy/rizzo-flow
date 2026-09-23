"""Compare distributions by semantic option ID across precision or perturbation reports.

Both inputs must be reports in the current `evaluation` format; an older run is rejected
outright rather than compared field by field against something it no longer resembles.
"""

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rizzo_flow.cli import write_json
from rizzo_flow.evaluation import EvaluationReport


@dataclass(frozen=True)
class Divergence:
    """One decision as the two reports each saw it."""

    id: str
    question: str
    changed_argmax: bool
    changed_status: bool
    max_probability_delta: float
    reference_probabilities: dict[str, float]
    candidate_probabilities: dict[str, float]


@dataclass(frozen=True)
class Totals:
    decisions: int
    changed_argmaxes: int
    changed_statuses: int
    max_probability_delta: float


@dataclass(frozen=True)
class Comparison:
    reference_dataset_sha256: str
    candidate_dataset_sha256: str
    summary: Totals
    rows: list[Divergence] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def compare(
    reference: EvaluationReport,
    candidate: EvaluationReport,
    perturbations: bool = False,
) -> Comparison:
    original = {row.id: row.response for row in reference.rows}
    rows = []
    for row in candidate.rows:
        key = row.id
        if perturbations:
            key = key.removesuffix("-reordered").removesuffix("-irrelevant")
        if key not in original:
            raise ValueError(f"Missing reference for {row.id}")
        for question_id, answer in row.response.answers.items():
            other = original[key].answers[question_id]
            a, b = other.probabilities, answer.probabilities
            if a.keys() != b.keys():
                raise ValueError(f"Candidate IDs changed for {key}/{question_id}")
            rows.append(
                Divergence(
                    id=row.id,
                    question=question_id,
                    changed_argmax=max(a, key=a.get) != max(b, key=b.get),
                    changed_status=other.status != answer.status,
                    max_probability_delta=max(abs(a[k] - b[k]) for k in a),
                    reference_probabilities=a,
                    candidate_probabilities=b,
                )
            )
    return Comparison(
        reference_dataset_sha256=reference.dataset_sha256,
        candidate_dataset_sha256=candidate.dataset_sha256,
        summary=Totals(
            decisions=len(rows),
            changed_argmaxes=sum(r.changed_argmax for r in rows),
            changed_statuses=sum(r.changed_status for r in rows),
            max_probability_delta=max((r.max_probability_delta for r in rows), default=0),
        ),
        rows=rows,
    )


def read_report(path: Path) -> EvaluationReport:
    return EvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--perturbations", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = compare(read_report(args.reference), read_report(args.candidate), args.perturbations)
    write_json(report.as_dict(), args.output)


if __name__ == "__main__":
    main()
