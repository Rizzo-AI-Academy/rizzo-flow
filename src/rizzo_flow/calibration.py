"""Temperature fitting on a separate labeled set; never claim validation from fit loss."""

import hashlib
import math
from typing import Annotated, Literal

from pydantic import Field

from .decisions import softmax
from .prompts import canonical
from .schema import Strict

QuestionType = Literal["boolean", "choice", "score", "numeric"]


class FitMetrics(Strict):
    """How much the fitted temperature moved the loss. Fit loss, never validation."""

    rows: int
    fit_nll_before: float
    fit_nll_after: float


class Calibration(Strict):
    version: Literal[1] = 1
    fingerprint: str
    dataset_sha256: str
    temperatures: dict[QuestionType, Annotated[float, Field(gt=0, allow_inf_nan=False)]]
    # extra='forbid' means a calibration file written with a different metric set is
    # rejected rather than half-read.
    fit_metrics: dict[QuestionType, FitMetrics]
    status: Literal["fitted_requires_held_out_validation"] = "fitted_requires_held_out_validation"

    @classmethod
    def from_file(cls, path):
        item = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if any(not math.isfinite(t) or t <= 0 for t in item.temperatures.values()):
            raise ValueError("Calibration temperatures must be finite and positive")
        return item


class LabeledLogits(Strict):
    type: QuestionType
    logits: list[float] = Field(min_length=2, max_length=26)
    label_index: int = Field(ge=0)


def fit_temperature(rows: list[dict], fingerprint: str) -> Calibration:
    if not rows:
        raise ValueError("Calibration requires labeled rows")
    groups = {}
    for raw in rows:
        row = LabeledLogits.model_validate(raw)
        softmax(row.logits)
        if row.label_index >= len(row.logits):
            raise ValueError("Label index is outside the candidate list")
        groups.setdefault(row.type, []).append(row)
    temperatures, metrics = {}, {}
    for kind, group in groups.items():
        if len(group) < 10:
            raise ValueError(f"Provide at least 10 calibration examples for {kind}")

        def loss(log_temperature, group=group):
            t = math.exp(log_temperature)
            total = 0.0
            for row in group:
                values = [x / t for x in row.logits]
                top = max(values)
                total += (
                    top + math.log(sum(math.exp(v - top) for v in values)) - values[row.label_index]
                )
            return total / len(group)

        # Positive one-dimensional search including T=1; bounded to avoid singular solutions.
        grid = [math.log(0.05) + i * math.log(400) / 240 for i in range(241)] + [0.0]
        best = min(grid, key=loss)
        temperatures[kind] = math.exp(best)
        metrics[kind] = FitMetrics(
            rows=len(group), fit_nll_before=loss(0), fit_nll_after=loss(best)
        )
    return Calibration(
        fingerprint=fingerprint,
        dataset_sha256=hashlib.sha256(canonical(rows).encode()).hexdigest(),
        temperatures=temperatures,
        fit_metrics=metrics,
    )
