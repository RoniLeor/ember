"""Explicit plugin imports use real decorators and restore all registration state."""

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from ember import experiment, provider
from ember.cli import main
from ember.playground.breakout.base import Sources
from ember.plugins import Plugins
from ember.providers.records import RunRecord

MODULE: str = "ember_local_test"
CODE: str = '''"""A local Breakout registration example."""
from typing import ClassVar
from ember import Breakout, experiment, provider
from ember.providers import Claude

@experiment
class LocalBreakout(Breakout):
    name: ClassVar[str] = "local.breakout"

@provider
class LocalClaude(Claude):
    name: ClassVar[str] = "local-claude"
'''


@pytest.fixture
def plugin_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setattr(experiment, "games", dict(experiment.games))
    monkeypatch.setattr(provider, "entries", dict(provider.entries))
    module: Path = tmp_path / f"{MODULE}.py"
    with patch.dict(sys.modules):
        yield module


def test_plugins_register_once_and_play_real_game(plugin_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plugin_file.write_text(CODE, encoding="utf-8")
    assert main(["--plugin", MODULE, "--plugin", MODULE, "list"]) == 0
    listed: dict[str, object] = json.loads(capsys.readouterr().out)
    assert listed["experiments"] == ["asterix", "breakout", "local.breakout"]
    assert listed["providers"] == {
        "claude": "claude-sonnet-5-5",
        "codex": "gpt-6-astra",
        "local-claude": "claude-sonnet-5-5",
    }
    network: Path = plugin_file.with_suffix(".md")
    network.write_text("```edits\nb3[:] = 0\n```", encoding="utf-8")
    output: Path = plugin_file.with_suffix(".html")
    assert (
        main(
            [
                "--plugin",
                MODULE,
                "replay",
                "--experiment",
                "local.breakout",
                "--network",
                str(network),
                "--max-steps",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()


def test_plugin_provider_uses_shared_strict_runner(plugin_file: Path) -> None:
    plugin_file.write_text(CODE, encoding="utf-8")
    reference: Path = plugin_file.parent / "reference"
    (reference / "breakout").mkdir(parents=True)
    (reference / "breakout/instructions.md").write_text("instructions", encoding="utf-8")
    output: Path = plugin_file.parent / "run"
    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch(
            "ember.providers.claude.subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="2.test", stderr=""),
        ) as transport,
    ):
        assert (
            main(
                [
                    "--plugin",
                    MODULE,
                    "run",
                    "--provider",
                    "local-claude",
                    "--reference",
                    str(reference),
                    "--output",
                    str(output),
                    "--execute",
                ]
            )
            == 1
        )
    transport.assert_called_once()
    assert transport.call_args.args[0] == ["claude", "--version"]
    record: RunRecord = RunRecord.model_validate_json((output / "run.json").read_text())
    assert record.provider == "local-claude" and record.status == "blocked"
    assert not (output / "answer.md").exists()


@pytest.mark.parametrize("name", ["", ".relative", "has space", "a/b", "a..b", "missing_ember_plugin"])
def test_invalid_or_missing_plugin_reports_error(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--plugin", name, "list"]) == 1
    assert repr(name) in capsys.readouterr().err


@pytest.mark.parametrize(
    "code",
    [
        "raise RuntimeError('broken plugin')",
        "broken (",
        "from missing_ember_dependency import X",
        "from ember import experiment, Breakout\nexperiment(Breakout)",
    ],
)
def test_plugin_failure_is_reported(plugin_file: Path, code: str, capsys: pytest.CaptureFixture[str]) -> None:
    plugin_file.write_text(code, encoding="utf-8")
    assert main(["--plugin", MODULE, "list"]) == 1
    assert f"Plugin {MODULE!r} failed" in capsys.readouterr().err


@pytest.mark.parametrize("exception", [SystemExit, KeyboardInterrupt])
def test_plugin_process_control_is_not_swallowed(plugin_file: Path, exception: type[BaseException]) -> None:
    plugin_file.write_text(f"raise {exception.__name__}", encoding="utf-8")
    with pytest.raises(exception):
        Plugins(modules=(MODULE,))()


@pytest.mark.parametrize(
    "argv", [["--help"], ["--plugin", MODULE, "run", "--help"], ["list", "--plugin", MODULE], ["--plugin"]]
)
def test_help_and_syntax_errors_do_not_import_plugins(plugin_file: Path, argv: list[str]) -> None:
    plugin_file.write_text("raise AssertionError('must not import')", encoding="utf-8")
    with pytest.raises(SystemExit) as result:
        main(argv)
    assert result.value.code == (0 if "--help" in argv else 2)
    assert MODULE not in sys.modules
