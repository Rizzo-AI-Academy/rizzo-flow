"""Command line: results are create-only, and a taken --output is refused before any work."""

import json
import sys
from pathlib import Path

import pytest
from test_service import FakeBackend

from rizzo_flow import cli, loader

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "decide": ROOT / "examples" / "ticket.json",
    "evaluate": ROOT / "benchmarks" / "smoke.jsonl",
}


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["rizzo", *argv])
    cli.main()


@pytest.mark.parametrize("command", ["decide", "evaluate"])
def test_a_taken_output_is_refused_before_the_model_loads(tmp_path, monkeypatch, capsys, command):
    taken = tmp_path / "report.json"
    taken.write_text('{"kept": true}\n', encoding="utf-8")

    def load_backend(*args, **kwargs):
        raise AssertionError("the model was loaded, and the run would be lost at the end")

    monkeypatch.setattr(loader, "load_backend", load_backend)
    with pytest.raises(SystemExit) as stopped:
        run(monkeypatch, command, str(INPUTS[command]), "--output", str(taken))
    assert stopped.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert taken.read_text(encoding="utf-8") == '{"kept": true}\n'


@pytest.mark.parametrize("command", ["decide", "evaluate"])
def test_a_free_output_is_written(tmp_path, monkeypatch, command):
    target = tmp_path / "new" / "report.json"
    monkeypatch.setattr(loader, "load_backend", lambda *args, **kwargs: FakeBackend())
    run(monkeypatch, command, str(INPUTS[command]), "--output", str(target))
    report = json.loads(target.read_text(encoding="utf-8"))
    assert "answers" in report if command == "decide" else "summary" in report
