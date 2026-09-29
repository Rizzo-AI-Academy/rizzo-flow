"""Real MAX runtime and real safetensors weights. Opt in: RIZZO_REAL=1 pytest -m integration.

Needs the max extra and `rizzo download --backend max` (any size). The first run compiles the
graphs, which takes minutes; MAX caches them. Agreement with llama.cpp is measured by
`scripts/backend_bench.py`, not here: two models rarely fit one GPU at once.
"""

import importlib.util
import json
import os
import string
from pathlib import Path

import pytest

from rizzo_flow.config import FLOW_CHECKPOINTS, MODELS

pytestmark = pytest.mark.integration


def smallest():
    """The smallest checkpoint directory on disk, fine-tune first, or None."""
    for specs in (FLOW_CHECKPOINTS, MODELS):
        found = [spec.path for spec in specs.values() if any(spec.path.glob("*.safetensors"))]
        if found:
            return min(found, key=lambda p: sum(f.stat().st_size for f in p.glob("*.safetensors")))
    return None


@pytest.fixture(scope="module")
def backend():
    if os.environ.get("RIZZO_REAL") != "1":
        pytest.skip("set RIZZO_REAL=1 to load real weights")
    if importlib.util.find_spec("max") is None:
        pytest.skip("install the max extra")
    if smallest() is None:
        pytest.skip("run `rizzo download --backend max` first")
    from rizzo_flow.backend_max import MaxBackend

    return MaxBackend.load(smallest())


def test_identity_is_pinned(backend):
    meta = backend.metadata
    assert meta["runtime"] == "max" and meta["precision"] == "bf16"
    assert meta["source"] in {spec.repo for spec in MODELS.values()}
    assert any(name.endswith(".safetensors") for name in meta["source_files"])
    fine_tune = smallest() in {spec.path for spec in FLOW_CHECKPOINTS.values()}
    assert (meta.get("weights") == "flow") == fine_tune


def test_answer_letters_are_single_tokens(backend):
    ids = [backend.tokenizer.encode(letter) for letter in string.ascii_uppercase]
    assert all(len(found) == 1 for found in ids) and len({found[0] for found in ids}) == 26


def test_shared_prefix_agrees_with_direct(backend):
    from rizzo_flow.engine import Engine

    engine = Engine(backend, ctx=4096)
    request = json.loads(Path("examples/ticket.json").read_text(encoding="utf-8"))
    shared = engine.decide({**request, "mode": "shared"})
    direct = engine.decide({**request, "mode": "direct"})
    assert shared["timing"]["shared_prefix_tokens"] > 0
    assert shared["timing"]["generated_tokens"] == 0
    for key, answer in shared["answers"].items():
        other = direct["answers"][key]
        assert answer["prompt_sha256"] == other["prompt_sha256"]
        mine, theirs = answer["probabilities"], other["probabilities"]
        assert max(mine, key=mine.get) == max(theirs, key=theirs.get)
        assert max(abs(mine[o] - theirs[o]) for o in theirs) < 0.05
