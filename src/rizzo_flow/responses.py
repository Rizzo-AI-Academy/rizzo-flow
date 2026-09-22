"""Public output schema: numeric validity and nullability are checked before returning JSON."""

from typing import Annotated, Literal

from pydantic import Field, model_serializer, model_validator

from .schema import Strict

Finite = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Uncertainty(Strict):
    top_probability: Probability
    entropy_nats: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    concentration: Probability
    unavailable_probability: Probability


class Statistics(Strict):
    mean: Finite
    stddev: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    median: Finite
    anchor_quantiles: dict[Literal["p10", "p90"], Finite]


class Answer(Strict):
    status: Literal["ok", "insufficient_evidence", "out_of_range", "uncertain"]
    probabilities: dict[str, Probability]
    option_logits: dict[str, Finite]
    legend: dict[str, str]
    uncertainty: Uncertainty
    probability_status: Literal[
        "uncalibrated_conditional_option_scores",
        "temperature_scaled_requires_held_out_validation",
    ]
    temperature: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    prompt_sha256: str
    input_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_distribution(self):
        if abs(sum(self.probabilities.values()) - 1) > 1e-6:
            raise ValueError("Output probabilities must sum to one")
        if (
            self.probabilities.keys() != self.option_logits.keys()
            or self.probabilities.keys() != self.legend.keys()
        ):
            raise ValueError("Output candidate mappings must agree")
        field = (
            "choice" if hasattr(self, "choice") else "score" if hasattr(self, "score") else "value"
        )
        value = getattr(self, field)
        if (self.status == "ok") != (value is not None):
            raise ValueError("Only an accepted decision may have a non-null primary value")
        return self


class BooleanAnswer(Answer):
    type: Literal["boolean"]
    value: bool | None
    probability_true_given_available: Probability | None


class ChoiceAnswer(Answer):
    type: Literal["choice"]
    choice: str | None


class ScoreAnswer(Answer):
    type: Literal["score"]
    score: Finite | None
    normalized_score: Probability | None
    statistics_given_available: Statistics | None
    values: dict[str, Finite]
    support: list[Finite] = Field(min_length=2, max_length=2)


class NumericAnswer(Answer):
    type: Literal["numeric"]
    value: Finite | None
    unit: str
    statistics_given_available: Statistics | None
    values: dict[str, Finite]
    support: list[Finite] = Field(min_length=2, max_length=2)
    range_probabilities: dict[Literal["below", "above"], Probability]


TypedAnswer = Annotated[
    BooleanAnswer | ChoiceAnswer | ScoreAnswer | NumericAnswer, Field(discriminator="type")
]


class Timing(Strict):
    """Seconds and token counts of one request. The two backends count different
    peak-memory figures and the engine's own timings only exist once a request has run,
    so serialization drops whatever is absent instead of emitting nulls."""

    inference_seconds: Finite
    prefill_seconds: Finite
    shared_prefix_tokens: int
    evaluated_tokens_including_padding: int
    logical_input_tokens: int
    batches: int
    generated_tokens: int
    peak_mlx_bytes: int | None = None
    peak_device_bytes: int | None = None
    queue_seconds: Finite | None = None
    compile_seconds: Finite | None = None
    total_seconds: Finite | None = None

    @model_serializer
    def without_absent_counters(self) -> dict:
        return {
            name: getattr(self, name)
            for name in type(self).model_fields
            if getattr(self, name) is not None
        }


class Response(Strict):
    model: dict
    mode: Literal["shared", "direct"]
    answers: dict[str, TypedAnswer]
    calibration: dict | None
    timing: Timing
