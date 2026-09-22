"""The refactor may not move a byte of what this project has already written down.

A fingerprint binds a calibration file to the weights, runtime and prompt that produced
it. Renaming a metadata field, adding one or dropping one changes that hash and quietly
invalidates every calibration on disk, so these tests rebuild the records from reports
recorded months ago and demand the same hash and the same JSON back.
"""

import json
from pathlib import Path

import pytest

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
