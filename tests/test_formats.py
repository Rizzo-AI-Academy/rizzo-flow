"""The refactor may not move a byte of what this project has already written down.

A fingerprint binds a calibration file to the weights, runtime and prompt that produced
it. Renaming a metadata field, adding one or dropping one changes that hash and quietly
invalidates every calibration on disk, so these tests rebuild the records from reports
recorded months ago and demand the same hash and the same JSON back.
"""

import json
import sys
from pathlib import Path

import pytest

from rizzo_flow.evaluation import EvaluationReport
from rizzo_flow.metadata import LlamaMetadata, MlxMetadata

REPORTS = Path(__file__).resolve().parents[1] / "results" / "semif-compare"
# `batch_size` is added by scripts/semif_compare.py, not by the backend.
NOT_FROM_THE_BACKEND = {"batch_size"}


def recorded(run: str) -> dict:
    report = json.loads((REPORTS / run / "report.json").read_text(encoding="utf-8"))
    return {k: v for k, v in report["model"].items() if k not in NOT_FROM_THE_BACKEND}


@pytest.mark.parametrize(
    ("run", "record"),
    [
        ("rizzo-q8_0-v3-llama-cuda", LlamaMetadata),
        ("rizzo-q8-v3-cuda", MlxMetadata),
    ],
)
def test_metadata_reproduces_a_recorded_run(run, record):
    model = recorded(run)
    metadata = record(**{k: v for k, v in model.items() if k != "fingerprint"})
    assert metadata.fingerprint == model["fingerprint"]
    # Same keys, same values and the same order the recorded report shows.
    assert metadata.as_dict() == model
    assert list(metadata.as_dict()) == list(model)


def test_the_two_backends_report_different_identities():
    llama = set(recorded("rizzo-q8_0-v3-llama-cuda"))
    mlx = set(recorded("rizzo-q8-v3-cuda"))
    assert llama - mlx == {
        "gguf_source",
        "gguf_revision",
        "runtime",
        "llama_cpp_release",
        "llama_cpp_commit",
        "device_name",
        "context_cells",
    }
    assert mlx - llama == {"runtime_revision", "quantization_group_size", "mlx", "mlx_lm"}


# Reports written by the current response layout. The `*-v2-validation` runs predate it —
# back then `Answer` declared `type` first — and `spark-bf16-validation` predates
# `decisions_per_second` entirely, so neither says anything about today's format.
CURRENT_FORMAT = [
    "llama-q8_0-cuda-validation/smoke.json",
    "llama-q8_0-cuda-validation/perturbations.json",
    "llama-q8_0-cuda-validation/long-state.json",
    "spark-bf16-final/smoke.json",
    "spark-bf16-final/perturbations.json",
    "spark-q8-final/smoke.json",
    "spark-q8-final/perturbations.json",
]


@pytest.mark.parametrize("name", CURRENT_FORMAT)
def test_report_survives_the_round_trip_unchanged(name):
    path = REPORTS.parent / name
    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert EvaluationReport.model_validate(recorded).model_dump() == recorded
    # Key order is part of the file, not only its content.
    assert json.dumps(EvaluationReport.model_validate(recorded).model_dump()) == json.dumps(
        recorded
    )


def test_the_comparison_script_reproduces_its_recorded_output():
    """scripts/compare_reports.py rebuilt results/precision-comparison.json, exactly."""
    sys.path.insert(0, str(REPORTS.parents[1] / "scripts"))
    from compare_reports import compare, read_report

    results = REPORTS.parent
    recorded = json.loads((results / "precision-comparison.json").read_text(encoding="utf-8"))
    rebuilt = compare(
        read_report(results / "spark-bf16-final" / "smoke.json"),
        read_report(results / "spark-q8-final" / "smoke.json"),
    ).as_dict()
    assert json.dumps(rebuilt) == json.dumps(recorded)
