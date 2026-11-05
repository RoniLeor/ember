"""Provider selection and shared strict execution remain isolated from live models."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ember.cli import main
from ember.playground.breakout.base import Sources
from ember.providers import Claude, Codex, Provider, Registry, providers
from ember.providers.records import RunRecord
from ember.runner import Runner


def test_registry_registration_defaults_and_explicit_models(tmp_path: Path) -> None:
    registry: Registry = Registry()
    assert registry(Codex) is Codex
    assert registry(Claude) is Claude
    assert not Registry().entries
    sources: Sources = Sources(root=tmp_path)
    assert registry.get(name="codex", sources=sources).model == "gpt-6-astra"
    assert registry.get(name="claude", sources=sources).model == "claude-sonnet-5-5"
    assert Claude(sources=sources).model == "claude-sonnet-5-5"
    selected: Provider = registry.get(name="claude", sources=sources, model="claude-model-version")
    assert isinstance(selected, Claude) and selected.model == "claude-model-version"
    with pytest.raises(ValueError, match="duplicate"):
        registry(Claude)
    with pytest.raises(ValueError, match="Unknown"):
        registry.get(name="missing", sources=sources)
    with patch.object(Claude, "name", ""), pytest.raises(ValueError, match="Empty"):
        Registry()(Claude)


@pytest.mark.parametrize("model", ["", " ", "-flag", "model name", "model\n", "model\x00", "model\x7f", "model\x80"])
def test_invalid_model_is_rejected_before_creating_output(
    tmp_path: Path, model: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output: Path = tmp_path / "run"
    assert (
        main(["run", "--provider", "claude", f"--model={model}", "--reference", str(tmp_path), "--output", str(output)])
        == 1
    )
    assert "Model must be" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("model", [None, "opus", "sonnet"])
def test_claude_cli_records_selection_and_never_calls_model(tmp_path: Path, model: str | None) -> None:
    (tmp_path / "breakout").mkdir()
    (tmp_path / "breakout/instructions.md").write_text("instructions\n", encoding="utf-8")
    output: Path = tmp_path / "run"
    command: list[str] = [
        "run",
        "--provider",
        "claude",
        "--reference",
        str(tmp_path),
        "--output",
        str(output),
        "--execute",
    ]
    if model is not None:
        command.extend(["--model", model])
    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch(
            "ember.providers.claude.subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="2.test", stderr=""),
        ) as transport,
    ):
        assert main(command) == 1
    transport.assert_called_once_with(
        ["claude", "--version"], capture_output=True, encoding="utf-8", check=True, timeout=90
    )
    record: RunRecord = RunRecord.model_validate_json((output / "run.json").read_text())
    assert record.provider == "claude" and record.model == (model or "claude-sonnet-5-5") and record.status == "blocked"
    assert f"--model={model or 'claude-sonnet-5-5'}" in record.command
    assert not record.tool_inventory_empty and not record.prompt_context_matches
    assert record.config[record.config.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in record.config and "--disable-slash-commands" in record.config
    assert "--bare" not in record.config and "--dangerously-skip-permissions" not in record.config
    assert not (output / "answer.md").exists() and not (output / "inventory.json").exists()


def test_claude_execution_and_audit_cannot_be_forced(tmp_path: Path) -> None:
    provider: Claude = Claude(sources=Sources(root=tmp_path))
    forged: RunRecord = RunRecord(
        status="prepared", provider="claude", model="sonnet", tool_inventory_empty=True, prompt_context_matches=True
    )
    with (
        patch("ember.providers.claude.subprocess.run") as transport,
        pytest.raises(ValueError, match="no offline"),
    ):
        provider.execute(output=tmp_path, record=forged, prompt=b"prompt")
    transport.assert_not_called()
    with pytest.raises(ValueError, match="strict audit"):
        provider.audit(root=tmp_path)


def test_shared_gate_rejects_a_provider_with_false_evidence(tmp_path: Path) -> None:
    provider: Codex = Codex(sources=Sources(root=tmp_path))
    prepared: RunRecord = RunRecord(status="prepared", model=provider.model)
    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch.object(Codex, "prepare", return_value=prepared),
        patch.object(Codex, "execute") as execute,
    ):
        result: RunRecord = Runner(provider=provider)(output=tmp_path / "run", execute=True)
    assert result.status == "failed" and "preflight" in result.errors[0]
    execute.assert_not_called()


def test_shared_prepare_failure_keeps_provider_identity(tmp_path: Path) -> None:
    provider: Claude = Claude(sources=Sources(root=tmp_path), model="opus")
    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch.object(Claude, "prepare", side_effect=FileNotFoundError("claude missing")),
    ):
        result: RunRecord = Runner(provider=provider)(output=tmp_path / "run", execute=True)
    assert result.status == "failed" and result.provider == "claude" and result.model == "opus"
    assert "Claude is a different model family" in result.deviations[0]


def test_validation_precedes_directory_creation(tmp_path: Path) -> None:
    output: Path = tmp_path / "run"
    provider: Provider = providers.get(name="codex", sources=Sources(root=tmp_path))
    with patch.object(Sources, "verify_prompt", side_effect=ValueError("invalid prompt")), pytest.raises(ValueError):
        Runner(provider=provider)(output=output, execute=True)
    assert not output.exists()


def test_audit_provider_is_explicit_and_cannot_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    record: RunRecord = RunRecord(status="blocked", provider="claude", model="sonnet")
    (tmp_path / "run.json").write_text(record.model_dump_json(), encoding="utf-8")
    assert main(["audit", "--run", str(tmp_path)]) == 1
    assert "differs" in capsys.readouterr().err
    assert main(["audit", "--provider", "claude", "--run", str(tmp_path)]) == 1
    assert "strict audit" in capsys.readouterr().err
    assert main(["list"]) == 0
    listed: dict[str, object] = json.loads(capsys.readouterr().out)
    assert listed["providers"] == {"claude": "claude-sonnet-5-5", "codex": "gpt-6-astra"}
