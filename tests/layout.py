"""Project-root defaults and installed entry points use the relocated package."""

import json
import subprocess
import sys
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path

import pytest

from ember.cli import arguments, main


@pytest.mark.parametrize("command", ["fetch", "parity", "run", "audit"])
def test_reference_defaults_to_project_root(command: str) -> None:
    argv: list[str] = [command]
    if command == "run":
        argv.extend(["--output", "example"])
    elif command == "audit":
        argv.extend(["--run", "example"])
    assert arguments(argv).reference == Path("resources")
    assert arguments([*argv, "--reference", "custom"]).reference == Path("custom")


def test_installed_command_targets_core() -> None:
    entries: list[EntryPoint] = list(entry_points(group="console_scripts", name="ember"))
    assert len(entries) == 1
    assert entries[0].value == "ember.cli:main"
    assert entries[0].load() is main


def test_module_entry_point_works_outside_project(tmp_path: Path) -> None:
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-m", "ember", "list"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    listed: dict[str, object] = json.loads(result.stdout)
    assert listed["experiments"] == ["breakout"]
    assert listed["providers"] == {"claude": "claude-sonnet-5-5", "codex": "gpt-6-astra"}
