"""CLI exit codes and file outputs exercise real command dispatch."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ember.base import SourceRecord
from ember.cli import arguments, main
from ember.playground.base import Parity
from ember.playground.breakout.base import Sources
from ember.providers.records import RunRecord
from ember.runner import Runner
from tests.providers.audit import EVENTS


def test_list_help_and_invalid_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    listed: dict[str, object] = json.loads(capsys.readouterr().out)
    assert listed["experiments"] == ["asterix", "breakout"]
    with pytest.raises(SystemExit) as exit_info:
        arguments(["--help"])
    assert exit_info.value.code == 0
    with pytest.raises(SystemExit):
        arguments(["replay"])


def test_evaluate_and_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    network: Path = tmp_path / "answer.md"
    network.write_text("```edits\nb3[:] = 0\n```", encoding="utf-8")
    command: list[str] = ["evaluate", "--network", str(network), "--episodes", "1", "--max-steps", "2"]
    assert main(command) == 1
    assert "Authors' evaluator" in capsys.readouterr().err
    report: Path = tmp_path / "evaluation.json"
    assert main([*command, "--reconstructed", "--output", str(report)]) == 0
    assert json.loads(report.read_text())["kind"] == "reconstructed"
    capsys.readouterr()
    html: Path = tmp_path / "replay.html"
    assert main(["replay", "--network", str(network), "--max-steps", "2", "--output", str(html)]) == 0
    assert html.is_file()
    recording: dict[str, object] = json.loads(capsys.readouterr().out)
    assert "frames" not in recording and "network_sha256" in recording
    assert main([*command, "--reconstructed", "--seed", "-1"]) == 1


def test_reference_and_run_dispatch(tmp_path: Path) -> None:
    source: SourceRecord = SourceRecord(commit="test", hashes={}, urls={})
    with patch.object(Sources, "fetch", return_value=source):
        assert main(["fetch", "--reference", str(tmp_path)]) == 0
    parity: Parity = Parity(
        source_commit="test",
        community_pr="test",
        applied=92,
        skipped=[],
        tensors_equal=True,
        reason="test",
        differences=[],
    )
    with patch.object(Sources, "parity", return_value=parity):
        assert main(["parity", "--reference", str(tmp_path)]) == 0
    with patch.object(Sources, "parity", return_value=parity.model_copy(update={"tensors_equal": False})):
        assert main(["parity", "--reference", str(tmp_path)]) == 1
    record: RunRecord = RunRecord(
        status="blocked",
        model="gpt-6-astra",
        cli_version="test",
        command=[],
        config=[],
        tool_inventory_empty=False,
    )
    with patch.object(Runner, "__call__", return_value=record):
        assert main(["run", "--output", str(tmp_path / "run")]) == 1


def test_audit_dispatch(tmp_path: Path) -> None:
    (tmp_path / "events.jsonl").write_text('{"type":"thread.started"}\n')
    (tmp_path / "answer.md").write_text("answer")
    (tmp_path / "inventory.json").write_text('[{"type":"message"}]')
    assert main(["audit", "--run", str(tmp_path)]) == 1
    assert main(["audit", "--run", str(tmp_path / "missing")]) == 1


def test_standalone_audit_checks_exact_context_and_saved_prompt(tmp_path: Path) -> None:
    reference: Path = Path(__file__).resolve().parents[1] / "resources"
    prompt: bytes = Sources(root=reference).verify_prompt()
    instructions: str = (reference / "breakout/instructions.md").read_text(encoding="utf-8").rstrip("\n")
    inventory: list[dict[str, object]] = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": instructions}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt.decode("utf-8")}]},
        {"type": "additional_tools", "tools": []},
    ]
    (tmp_path / "prompt.txt").write_bytes(prompt)
    (tmp_path / "answer.md").write_text("answer", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(event) for event in EVENTS), encoding="utf-8")
    (tmp_path / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    command: list[str] = ["audit", "--run", str(tmp_path), "--reference", str(reference)]
    assert main(command) == 0
    inventory.insert(
        1,
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "# AGENTS.md instructions"}],
        },
    )
    (tmp_path / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    assert main(command) == 1
    inventory.pop(1)
    (tmp_path / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    assert main(command) == 0
    (tmp_path / "prompt.txt").write_bytes(b"changed")
    assert main(command) == 1


def test_progress_uses_stderr_only_and_keeps_stdout_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    network: Path = tmp_path / "answer.md"
    network.write_text("```edits\nb3[:] = 0\n```", encoding="utf-8")
    replay: list[str] = ["replay", "--network", str(network), "--max-steps", "2", "--output", str(tmp_path / "r.html")]
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TTY_COMPATIBLE", "0")
    out: str
    err: str
    assert main(replay) == 0
    out, err = capsys.readouterr()
    assert err == ""
    monkeypatch.setenv("TTY_COMPATIBLE", "1")
    assert main(replay) == 0
    shown: str
    shown, err = capsys.readouterr()
    assert json.loads(shown) == json.loads(out) and "Recorded breakout" in err
    record: RunRecord = RunRecord(status="blocked", model="gpt-6-astra")
    with patch.object(Runner, "__call__", return_value=record):
        assert main(["run", "--output", str(tmp_path / "run")]) == 1
    out, err = capsys.readouterr()
    assert json.loads(out)["status"] == "blocked" and "ember run" in err and "blocked" in err
