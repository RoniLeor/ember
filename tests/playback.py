"""Saved-run inputs and browser opening preserve explicit execution boundaries."""

import argparse
import json
import webbrowser
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from ember.cli import arguments, main
from ember.providers.records import RunRecord


@pytest.fixture
def completed_run(tmp_path: Path) -> Path:
    record: RunRecord = RunRecord(status="completed", provider="claude", mode="comparison", model="test")
    (tmp_path / "run.json").write_text(record.model_dump_json(), encoding="utf-8")
    (tmp_path / "answer.md").write_text("```edits\nb3[:] = 0\n```", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("command", ["evaluate", "replay", "play"])
def test_saved_run_uses_real_network_and_opens_only_play(
    completed_run: Path, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output: Path = completed_run / ("evaluation.json" if command == "evaluate" else "replay.html")
    argv: list[str] = [command, "--run", str(completed_run), "--max-steps", "2", "--output", str(output)]
    if command == "evaluate":
        argv.extend(["--episodes", "1", "--reconstructed"])
    with patch("ember.cli.webbrowser.open", return_value=True) as browser:
        assert main(argv) == 0
    assert output.is_file()
    assert not (completed_run / "breakout.html").exists()
    result: dict[str, object] = json.loads(capsys.readouterr().out)
    assert result["network_sha256"] == sha256((completed_run / "answer.md").read_bytes()).hexdigest()
    if command == "play":
        browser.assert_called_once_with(output.resolve().as_uri())
    else:
        browser.assert_not_called()
    if command == "evaluate":
        assert result["evaluator_matches_authors"] is False


@pytest.mark.parametrize("status", ["prepared", "blocked", "failed"])
def test_incomplete_run_is_rejected_before_answer_parsing(
    completed_run: Path, status: str, capsys: pytest.CaptureFixture[str]
) -> None:
    record: Path = completed_run / "run.json"
    saved: RunRecord = RunRecord.model_validate_json(record.read_text())
    record.write_text(saved.model_copy(update={"status": status}).model_dump_json())
    (completed_run / "answer.md").write_text("not a network")
    output: Path = completed_run / "replay.html"
    with patch("ember.cli.webbrowser.open") as browser:
        assert main(["play", "--run", str(completed_run), "--output", str(output)]) == 1
    assert f"Run is {status}, not completed" in capsys.readouterr().err
    assert not output.exists()
    browser.assert_not_called()


@pytest.mark.parametrize("failure", ["missing-record", "invalid-record", "missing-answer", "answer-directory"])
def test_invalid_saved_run_is_rejected(completed_run: Path, failure: str, capsys: pytest.CaptureFixture[str]) -> None:
    if failure == "missing-record":
        (completed_run / "run.json").unlink()
    elif failure == "invalid-record":
        (completed_run / "run.json").write_text("{}")
    else:
        (completed_run / "answer.md").unlink()
        if failure == "answer-directory":
            (completed_run / "answer.md").mkdir()
    output: Path = completed_run / "replay.html"
    with patch("ember.cli.webbrowser.open") as browser:
        assert main(["play", "--run", str(completed_run), "--output", str(output)]) == 1
    assert "ember:" in capsys.readouterr().err
    assert not output.exists()
    browser.assert_not_called()


@pytest.mark.parametrize("command", ["evaluate", "replay", "play"])
def test_source_is_required_and_mutually_exclusive(command: str) -> None:
    with pytest.raises(SystemExit) as missing:
        arguments([command, "--output", "out"])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as ambiguous:
        arguments([command, "--network", "answer.md", "--run", "run", "--output", "out"])
    assert ambiguous.value.code == 2


@pytest.mark.parametrize("failure", [False, webbrowser.Error("browser unavailable")])
def test_browser_failure_keeps_saved_replay(
    completed_run: Path, failure: bool | webbrowser.Error, capsys: pytest.CaptureFixture[str]
) -> None:
    output: Path = completed_run / "with spaces.html"
    with patch(
        "ember.cli.webbrowser.open",
        return_value=False,
        side_effect=failure if isinstance(failure, Exception) else None,
    ):
        assert main(["play", "--run", str(completed_run), "--max-steps", "2", "--output", str(output)]) == 1
    assert output.is_file()
    assert "browser" in capsys.readouterr().err


def test_bad_network_never_opens_browser(tmp_path: Path) -> None:
    output: Path = tmp_path / "replay.html"
    with patch("ember.cli.webbrowser.open") as browser:
        assert main(["play", "--network", str(tmp_path / "missing.npz"), "--output", str(output)]) == 1
    assert not output.exists()
    browser.assert_not_called()


def test_unknown_provider_fails_before_creating_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output: Path = tmp_path / "run"
    assert main(["run", "--provider", "unknown", "--output", str(output)]) == 1
    assert "Unknown provider" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("command", ["replay", "play"])
def test_saved_run_defaults_output_to_its_folder(
    completed_run: Path, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output: Path = completed_run / "breakout.html"
    output.write_text("old replay", encoding="utf-8")
    with patch("ember.cli.webbrowser.open", return_value=True) as browser:
        assert main([command, "--run", str(completed_run), "--max-steps", "2"]) == 0
    result: dict[str, object] = json.loads(capsys.readouterr().out)
    assert result["replay"] == "breakout.html"
    assert 'id="episode"' in output.read_text()
    if command == "play":
        browser.assert_called_once_with(output.resolve().as_uri())
    else:
        browser.assert_not_called()


@pytest.mark.parametrize("command", ["replay", "play"])
def test_network_input_still_requires_explicit_output(command: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        arguments([command, "--network", str(tmp_path / "answer.md")])
    assert error.value.code == 2
    assert not (tmp_path / "breakout.html").exists()


@pytest.mark.parametrize("failure", ["failed", "blocked", "missing-record", "missing-answer"])
def test_default_output_is_not_created_for_invalid_run(completed_run: Path, failure: str) -> None:
    record: Path = completed_run / "run.json"
    if failure == "missing-record":
        record.unlink()
    elif failure == "missing-answer":
        (completed_run / "answer.md").unlink()
    else:
        saved: RunRecord = RunRecord.model_validate_json(record.read_text())
        record.write_text(saved.model_copy(update={"status": failure}).model_dump_json())
    with patch("ember.cli.webbrowser.open") as browser:
        assert main(["play", "--run", str(completed_run), "--max-steps", "2"]) == 1
    assert not (completed_run / "breakout.html").exists()
    browser.assert_not_called()


def test_evaluation_keeps_optional_output_unset(completed_run: Path) -> None:
    args: argparse.Namespace = arguments(["evaluate", "--run", str(completed_run)])
    assert args.output is None
