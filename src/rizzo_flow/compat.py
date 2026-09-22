"""TypeSafe-compatible wire format (`POST /v1/systemone`) translated to native questions.

Only the interface matches the public TypeSafe docs. Answers come from the local Spark
checkpoint: the response `model` field always reports the local model, never a Jev version.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .metadata import Metadata
from .prompts import canonical
from .responses import Response
from .schema import MAX_SLOTS, Request

LOCAL_ALIAS = "rizzo-latest"
# Accepted so that clients written for the hosted API work unchanged against localhost.
FOREIGN_PREFIX = "jev-"
MAX_OPTIONS = MAX_SLOTS  # abstention is disabled here, so every letter is an option
MAX_LEVELS = 10

Structured = str | dict[str, JsonValue] | list[JsonValue]


class Wire(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoulCriteria(Wire):
    true: Structured | None = None
    false: Structured | None = None


class NoulQuestion(Wire):
    type: Literal["noul"]
    instructions: Structured
    criteria: NoulCriteria | None = None


class ChoiceQuestion(Wire):
    type: Literal["choice"]
    instructions: Structured
    criteria: dict[str, Structured | None] = Field(min_length=2, max_length=MAX_OPTIONS)


class ScoreQuestion(Wire):
    type: Literal["score"]
    instructions: Structured
    criteria: list[Structured] = Field(min_length=2, max_length=MAX_LEVELS)


WireQuestion = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(Wire):
    state: Structured
    model: str = Field(min_length=1, max_length=128)
    questions: dict[str, WireQuestion] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def nonblank_option_keys(self):
        for question in self.questions.values():
            if isinstance(question, ChoiceQuestion) and any(
                not key.strip() for key in question.criteria
            ):
                raise ValueError("Choice option keys must not be blank")
        return self


def model_name(metadata: Metadata) -> str:
    checkpoint = metadata.source.split("/")[-1].lower()  # spark-x2.5-4b
    return f"rizzo-{checkpoint}-{metadata.precision}"


def resolve_model(requested: str, metadata: Metadata) -> str:
    served = model_name(metadata)
    if requested in (LOCAL_ALIAS, served) or requested.startswith(FOREIGN_PREFIX):
        return served
    raise ValueError(
        f"Unknown model {requested!r}. Use {LOCAL_ALIAS!r}, {served!r} "
        f"or a {FOREIGN_PREFIX}* alias."
    )


def list_models(metadata: Metadata) -> dict:
    served = model_name(metadata)
    local = f"Local {metadata.source} scored with typed option logits."
    return {
        "models": [
            {"name": LOCAL_ALIAS, "description": local, "release_date": "2026-09-21"},
            {"name": served, "description": local, "release_date": "2026-09-21"},
            {
                "name": "jev-latest",
                "description": f"Compatibility alias: answered by {served}, not by TypeSafe Jev.",
                "release_date": "2026-09-21",
            },
        ]
    }


def text(value) -> str:
    return value.strip() if isinstance(value, str) else canonical(value)


def to_native(request: SystemOneRequest) -> tuple[Request, dict[str, list[str]]]:
    """Return the native request and, per choice question, option keys in slot order."""
    policy = {"allow_abstain": False}  # the wire format has no abstention outcome
    questions = {}
    options = {}
    for key, question in request.questions.items():
        native = {"instructions": text(question.instructions), "policy": policy}
        if isinstance(question, NoulQuestion):
            native["type"] = "boolean"
            criteria = question.criteria
            if criteria and criteria.true is not None:
                native["true_description"] = "Yes. " + text(criteria.true)
            if criteria and criteria.false is not None:
                native["false_description"] = "No. " + text(criteria.false)
        elif isinstance(question, ChoiceQuestion):
            # Option keys are free-form strings, so native IDs are positional.
            options[key] = list(question.criteria)
            native["type"] = "choice"
            native["options"] = [
                {
                    "id": f"o{index}",
                    "description": name if detail is None else f"{name}: {text(detail)}",
                }
                for index, (name, detail) in enumerate(question.criteria.items())
            ]
        else:
            native["type"] = "score"
            native["levels"] = [text(level) for level in question.criteria]
        questions[key] = native
    return Request.model_validate({"state": request.state, "questions": questions}), options


def confidence(probabilities) -> float:
    """Peak-over-uniform statistic from the public Confidence page; not a calibrated accuracy."""
    count = len(probabilities)
    return max(0.0, min(1.0, (count * max(probabilities) - 1) / (count - 1)))


def from_native(
    request: SystemOneRequest,
    response: Response,
    options: dict[str, list[str]],
    metadata: Metadata,
) -> dict:
    answers = {}
    for key, question in request.questions.items():
        native = response.answers[key]
        ps = native.probabilities
        if isinstance(question, NoulQuestion):
            answers[key] = {"type": "noul", "noul": ps["true"]}
        elif isinstance(question, ChoiceQuestion):
            named = {name: ps[f"o{index}"] for index, name in enumerate(options[key])}
            answers[key] = {
                "type": "choice",
                "choice": max(named, key=named.get),
                "probabilities": named,
                "confidence": confidence(list(named.values())),
            }
        else:
            answers[key] = {
                "type": "score",
                "score": native.score,
                "legend": {str(i): text(level) for i, level in enumerate(question.criteria)},
                "probabilities": ps,
                "confidence": confidence(list(ps.values())),
            }
    natives = list(response.answers.values())
    shared = response.timing.shared_prefix_tokens
    return {
        "model": model_name(metadata),
        "answers": answers,
        "usage": {
            # The shared state is evaluated once; nothing is ever generated.
            "input_tokens": sum(a.input_tokens for a in natives) - shared * (len(natives) - 1),
            "output_tokens": 0,
        },
        # Extension outside the TypeSafe contract; their SDKs ignore unknown fields.
        "x_rizzo": {
            "timing": response.timing.model_dump(),
            "probability_status": sorted({a.probability_status for a in natives}),
            "fingerprint": metadata.fingerprint,
        },
    }
