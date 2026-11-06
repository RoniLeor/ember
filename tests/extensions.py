"""Alternate adapters exercise the shared pipeline without copying its behavior."""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest

from ember import (
    Breakout,
    Evaluation,
    EvaluationRequest,
    Evaluator,
    Module,
    Recording,
    Replay,
    Runner,
    RunRequest,
    Settings,
    experiment,
)
from ember.base import SourceRecord
from ember.cli import main
from ember.playground.base import GameSourceFactory, Parity
from ember.playground.breakout.base import FILES, Sources
from ember.playground.network import Weights
from ember.playground.policy import Loader
from ember.providers import Claude, Codex
from ember.providers.records import RunRecord
from tests.providers.audit import EVENTS

PROMPT: bytes = b"An alternate experiment prompt."
INSTRUCTIONS: bytes = b"Answer this alternate task."


class AlternateSources(Sources):
    """Reuse the verified source implementation with distinct pinned assets."""

    commit: ClassVar[str] = "alternate-source"
    directory: ClassVar[str] = "alternate"
    prompt_hash: ClassVar[str] = sha256(PROMPT).hexdigest()
    instructions_hash: ClassVar[str] = sha256(INSTRUCTIONS).hexdigest()


class AlternateWeights(Weights):
    """Reuse the production loader to test class-based selection."""


class AlternateGame(Breakout):
    """An isolated registration that shares only the environment and architecture."""

    name: ClassVar[str] = "local.alternate"
    sources: ClassVar[GameSourceFactory] = AlternateSources
    network: ClassVar[Loader] = AlternateWeights


@dataclass(frozen=True)
class WeightExplorer(Module[RunRequest, Evaluation]):
    """The documented extension composes production execution and scoring."""

    runner: Runner
    evaluator: Evaluator

    def forward(self, inputs: RunRequest) -> Evaluation:
        record: RunRecord = self.runner.run(inputs)
        if record.status != "completed":
            raise ValueError(f"Run is {record.status}, not completed")
        return self.evaluator.run(EvaluationRequest(network=inputs.output / "answer.md", reconstructed=True))


@pytest.fixture
def alternate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    assets: Path = tmp_path / "alternate"
    assets.mkdir()
    (assets / "prompt.txt").write_bytes(PROMPT)
    (assets / "instructions.md").write_bytes(INSTRUCTIONS)
    (tmp_path / "answer.md").write_text("```edits\nb3[:] = 0\n```", encoding="utf-8")
    template: Path = tmp_path / "viewer.html"
    template.write_text("alternate-viewer:__EPISODE__", encoding="utf-8")
    monkeypatch.setattr(AlternateSources, "assets", assets)
    monkeypatch.setattr(AlternateGame, "template", template)
    with patch.dict(experiment.games):
        experiment(AlternateGame)
        yield tmp_path


@pytest.mark.parametrize("failed", [False, True])
def test_runner_records_selected_prompt_and_source(alternate: Path, failed: bool) -> None:
    output: Path = alternate / "run"
    with patch(
        "ember.providers.claude.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="test", stderr=""),
        side_effect=OSError("CLI unavailable") if failed else None,
    ) as transport:
        assert (
            main(
                [
                    "run",
                    "--experiment",
                    AlternateGame.name,
                    "--provider",
                    "claude",
                    "--reference",
                    str(alternate),
                    "--output",
                    str(output),
                    "--execute",
                ]
            )
            == 1
        )
    record: RunRecord = RunRecord.model_validate_json((output / "run.json").read_text())
    assert record.source_commit == AlternateSources.commit
    assert record.prompt_sha256 == sha256(PROMPT).hexdigest()
    assert record.status == ("failed" if failed else "blocked")
    assert (output / "prompt.txt").read_bytes() == PROMPT
    transport.assert_called_once()
    assert transport.call_args.args[0] == ["claude", "--version"]
    if not failed:
        assert record.config[-1] == INSTRUCTIONS.decode()


def test_evaluation_and_replay_use_registered_adapters(alternate: Path) -> None:
    network: Path = alternate / "answer.md"
    settings: Settings = Settings(experiment=AlternateGame.name, episodes=1, max_steps=2)
    output: Path = alternate / "replay.html"
    with patch.object(AlternateWeights, "load", wraps=AlternateWeights.load) as loader:
        result: Evaluation = Evaluator(settings=settings)(network=network, reconstructed=True)
        recording: Recording = Replay(settings=settings)(network=network, output=output)
    assert loader.call_count == 2
    assert all(call.args == (network,) for call in loader.call_args_list)
    assert result.settings.experiment == AlternateGame.name
    assert result.network_sha256 == recording.network_sha256 == sha256(network.read_bytes()).hexdigest()
    assert output.read_text().startswith("alternate-viewer:")


def test_audit_uses_selected_source(alternate: Path) -> None:
    output: Path = alternate / "audit"
    output.mkdir()
    (output / "prompt.txt").write_bytes(PROMPT)
    (output / "answer.md").write_text("answer", encoding="utf-8")
    (output / "events.jsonl").write_text("\n".join(json.dumps(event) for event in EVENTS), encoding="utf-8")
    inventory: list[dict[str, object]] = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": INSTRUCTIONS.decode()}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": PROMPT.decode()}]},
        {"type": "additional_tools", "tools": []},
    ]
    (output / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    command: list[str] = [
        "audit",
        "--experiment",
        AlternateGame.name,
        "--reference",
        str(alternate),
        "--run",
        str(output),
    ]
    assert main(command) == 0
    (output / "prompt.txt").write_bytes(b"wrong game")
    assert main(command) == 1


def test_fetch_and_parity_dispatch_to_selected_source(alternate: Path) -> None:
    source: SourceRecord = SourceRecord(commit=AlternateSources.commit, hashes={}, urls={})
    parity: Parity = Parity(
        source_commit=AlternateSources.commit,
        applied=0,
        skipped=[],
        tensors_equal=True,
        reason="test",
        community_pr="test",
        differences=[],
    )
    with patch.object(AlternateSources, "fetch", return_value=source) as fetch:
        assert main(["fetch", "--experiment", AlternateGame.name, "--reference", str(alternate)]) == 0
    fetch.assert_called_once_with()
    with patch.object(AlternateSources, "parity", return_value=parity) as check:
        assert main(["parity", "--experiment", AlternateGame.name, "--reference", str(alternate)]) == 0
    check.assert_called_once_with()


def test_source_subclass_fetches_and_checks_its_own_directory(alternate: Path) -> None:
    reference: Path = Path(__file__).resolve().parents[1] / "resources/breakout"
    content: dict[str, bytes] = {name: (reference / name).read_bytes() for name in FILES}
    content["prompt.txt"] = PROMPT
    content["instructions.md"] = INSTRUCTIONS
    source: AlternateSources = AlternateSources(root=alternate)
    with patch("ember.playground.base.urlopen", side_effect=[BytesIO(value) for value in content.values()]):
        record: SourceRecord = source.fetch()
    assert record.commit == AlternateSources.commit
    assert "alternate/prompt.txt" in record.hashes and "breakout/prompt.txt" not in record.hashes
    assert set(record.hashes) == set(record.urls) == {f"alternate/{name}" for name in FILES}
    assert record.hashes == {f"alternate/{name}": sha256(data).hexdigest() for name, data in content.items()}
    assert all((alternate / "alternate" / name).read_bytes() == data for name, data in content.items())
    assert SourceRecord.model_validate_json((alternate / "alternate/manifest.json").read_text()) == record
    assert all(
        not (alternate / name).exists()
        for name in ("manifest.json", "runs", "research", "LICENSE-authors", "LICENSE-MinAtar")
    )
    assert source.parity().tensors_equal
    assert source.verify_prompt() == PROMPT
    assert not (alternate / "breakout").exists()


def test_runner_preserves_provenance_after_provider_replaces_record(alternate: Path) -> None:
    output: Path = alternate / "run"
    replacement: RunRecord = RunRecord(status="completed", provider="claude", mode="comparison", model="sonnet")
    with (
        patch(
            "ember.providers.claude.subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="test", stderr=""),
        ),
        patch.object(Claude, "execute", return_value=replacement),
    ):
        assert (
            main(
                [
                    "run",
                    "--experiment",
                    AlternateGame.name,
                    "--provider",
                    "claude",
                    "--mode",
                    "comparison",
                    "--reference",
                    str(alternate),
                    "--output",
                    str(output),
                    "--execute",
                ]
            )
            == 0
        )
    record: RunRecord = RunRecord.model_validate_json((output / "run.json").read_text())
    assert record.source_commit == AlternateSources.commit
    assert record.prompt_sha256 == sha256(PROMPT).hexdigest()


@pytest.mark.parametrize("command", ["fetch", "parity", "run", "audit"])
def test_unknown_experiment_has_no_side_effects(tmp_path: Path, command: str) -> None:
    output: Path = tmp_path / "absent"
    argv: list[str] = [command, "--experiment", "missing", "--reference", str(tmp_path)]
    if command == "run":
        argv.extend(["--output", str(output)])
    elif command == "audit":
        argv.extend(["--run", str(output)])
    assert main(argv) == 1
    assert not output.exists()


@pytest.mark.parametrize("execute", [False, True])
def test_module_extension_composes_execution_and_scoring(tmp_path: Path, execute: bool) -> None:
    output: Path = tmp_path / "run"
    provider: Codex = Codex(sources=Sources(root=tmp_path))
    runner: Runner = Runner(provider=provider)
    evaluator: Evaluator = Evaluator(settings=Settings(episodes=1, max_steps=2))
    module: WeightExplorer = WeightExplorer(runner=runner, evaluator=evaluator)
    request: RunRequest = RunRequest(output=output, execute=execute)
    prepared: RunRecord = RunRecord(
        status="prepared", model=provider.model, tool_inventory_empty=True, prompt_context_matches=True
    )

    def response(*, output: Path, record: RunRecord, prompt: bytes) -> RunRecord:
        assert prompt == b"prompt" and record.status == "prepared"
        (output / "answer.md").write_text("```edits\nb3[:] = 0\n```", encoding="utf-8")
        return record.model_copy(update={"status": "completed"})

    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch.object(Codex, "prepare", return_value=prepared),
        patch.object(Codex, "execute", side_effect=response) as transport,
        patch.object(Evaluator, "forward", wraps=evaluator.forward) as score,
    ):
        if execute:
            result: Evaluation = module(request)
            assert result.applied == 1 and result.settings is evaluator.settings
            assert result.network_sha256 == sha256((output / "answer.md").read_bytes()).hexdigest()
            transport.assert_called_once()
            score.assert_called_once_with(EvaluationRequest(network=output / "answer.md", reconstructed=True))
        else:
            with pytest.raises(ValueError, match="Run is prepared, not completed"):
                module.run(request)
            transport.assert_not_called()
            score.assert_not_called()
            assert not (output / "answer.md").exists()
    assert module.runner is runner and module.evaluator is evaluator
    assert RunRecord.model_validate_json((output / "run.json").read_text()).status == (
        "completed" if execute else "prepared"
    )
