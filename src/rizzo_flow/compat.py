"""TypeSafe-compatible wire format (`POST /v1/systemone`) translated to native questions.

Only the interface matches the public TypeSafe docs. Answers come from the local Spark
checkpoint: the response `model` field always reports the local model, never a Jev version.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .metadata import Metadata
from .prompts import canonical
from .responses import Response, Timing
from .schema import MAX_SLOTS, Option, Policy, Request
from .schema import BooleanQuestion as NativeBoolean
from .schema import ChoiceQuestion as NativeChoice
from .schema import Question as NativeQuestion
from .schema import ScoreQuestion as NativeScore

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
    def nonblank_option_keys(self) -> Self:
        for question in self.questions.values():
            if isinstance(question, ChoiceQuestion) and any(
                not key.strip() for key in question.criteria
            ):
                raise ValueError("Choice option keys must not be blank")
        return self


class ModelCard(Wire):
    name: str
    description: str
    release_date: str


class ModelList(Wire):
    models: list[ModelCard]


class NoulResult(Wire):
    type: Literal["noul"]
    noul: float


class ChoiceResult(Wire):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreResult(Wire):
    type: Literal["score"]
    score: float | None
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


# An explicit union, never the shared base: annotating this with a base class would make
# Pydantic serialize only the base's fields and drop the rest without a word.
WireAnswer = Annotated[NoulResult | ChoiceResult | ScoreResult, Field(discriminator="type")]


class Usage(Wire):
    input_tokens: int
    output_tokens: int


class RizzoExtension(Wire):
    """Outside the TypeSafe contract; their SDKs ignore fields they do not know."""

    timing: Timing
    probability_status: list[str]
    fingerprint: str


class SystemOneResponse(Wire):
    model: str
    answers: dict[str, WireAnswer]
    usage: Usage
    x_rizzo: RizzoExtension


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


def list_models(metadata: Metadata) -> ModelList:
    served = model_name(metadata)
    local = f"Local {metadata.source} scored with typed option logits."
    return ModelList(
        models=[
            ModelCard(name=LOCAL_ALIAS, description=local, release_date="2026-09-21"),
            ModelCard(name=served, description=local, release_date="2026-09-21"),
            ModelCard(
                name="jev-latest",
                description=f"Compatibility alias: answered by {served}, not by TypeSafe Jev.",
                release_date="2026-09-21",
            ),
        ]
    )


def text(value: object) -> str:
    return value.strip() if isinstance(value, str) else canonical(value)


def no_abstention() -> Policy:
    """The wire format has no abstention outcome. One policy object per question."""
    return Policy(allow_abstain=False)


def to_native(request: SystemOneRequest) -> tuple[Request, dict[str, list[str]]]:
    """Return the native request and, per choice question, option keys in slot order."""
    questions: dict[str, NativeQuestion] = {}
    options: dict[str, list[str]] = {}
    for key, question in request.questions.items():
        instructions = text(question.instructions)
        if isinstance(question, NoulQuestion):
            criteria = question.criteria
            sides = {}
            if criteria and criteria.true is not None:
                sides["true_description"] = "Yes. " + text(criteria.true)
            if criteria and criteria.false is not None:
                sides["false_description"] = "No. " + text(criteria.false)
            questions[key] = NativeBoolean(
                type="boolean", instructions=instructions, policy=no_abstention(), **sides
            )
        elif isinstance(question, ChoiceQuestion):
            # Option keys are free-form strings, so native IDs are positional.
            options[key] = list(question.criteria)
            questions[key] = NativeChoice(
                type="choice",
                instructions=instructions,
                policy=no_abstention(),
                options=[
                    Option(
                        id=f"o{index}",
                        description=name if detail is None else f"{name}: {text(detail)}",
                    )
                    for index, (name, detail) in enumerate(question.criteria.items())
                ],
            )
        else:
            questions[key] = NativeScore(
                type="score",
                instructions=instructions,
                policy=no_abstention(),
                levels=[text(level) for level in question.criteria],
            )
    return Request(state=request.state, questions=questions), options


def confidence(probabilities: list[float]) -> float:
    """Peak-over-uniform statistic from the public Confidence page; not a calibrated accuracy."""
    count = len(probabilities)
    return max(0.0, min(1.0, (count * max(probabilities) - 1) / (count - 1)))


def from_native(
    request: SystemOneRequest,
    response: Response,
    options: dict[str, list[str]],
    metadata: Metadata,
) -> SystemOneResponse:
    answers: dict[str, WireAnswer] = {}
    for key, question in request.questions.items():
        native = response.answers[key]
        ps = native.probabilities
        if isinstance(question, NoulQuestion):
            answers[key] = NoulResult(type="noul", noul=ps["true"])
        elif isinstance(question, ChoiceQuestion):
            named = {name: ps[f"o{index}"] for index, name in enumerate(options[key])}
            answers[key] = ChoiceResult(
                type="choice",
                choice=max(named, key=named.get),
                probabilities=named,
                confidence=confidence(list(named.values())),
            )
        else:
            answers[key] = ScoreResult(
                type="score",
                score=native.score,
                legend={str(i): text(level) for i, level in enumerate(question.criteria)},
                probabilities=ps,
                confidence=confidence(list(ps.values())),
            )
    natives = list(response.answers.values())
    shared = response.timing.shared_prefix_tokens
    return SystemOneResponse(
        model=model_name(metadata),
        answers=answers,
        usage=Usage(
            # The shared state is evaluated once; nothing is ever generated.
            input_tokens=sum(a.input_tokens for a in natives) - shared * (len(natives) - 1),
            output_tokens=0,
        ),
        x_rizzo=RizzoExtension(
            timing=response.timing,
            probability_status=sorted({a.probability_status for a in natives}),
            fingerprint=metadata.fingerprint,
        ),
    )
