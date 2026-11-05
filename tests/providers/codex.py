"""Codex comparisons use the retained execution policy and audit actual raw traces."""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from unittest.mock import patch

import pytest
from pydantic import JsonValue, TypeAdapter

from ember.models import Audit
from ember.playground.breakout.base import Sources
from ember.providers.codex import Codex
from ember.providers.records import RunRecord
from ember.providers.trace import TraceResult
from ember.runner import Runner
from tests.runner import EVENTS

from .trace import write_trace

FEATURES: tuple[str, ...] = (
    "code_mode",
    "collaboration_modes",
    "shell_tool",
    "unified_exec",
    "unified_exec_tty",
    "shell_snapshot",
    "view_image",
    "image_generation",
    "multi_agent",
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "code_mode_host",
    "skill_search",
    "skill_mcp_dependency_install",
    "sleep_tool",
    "tool_suggest",
    "goals",
    "hooks",
    "worktrees",
    "workspace_dependencies",
)


@dataclass
class Transport:
    """Write controlled process artifacts while keeping all audits real."""

    code: int = 0
    answer: bytes = b"answer\n"
    error: Exception | None = None
    model: str = "gpt-6-astra"
    trace: bool = True
    tool_call: bool = False
    exact: bool = False
    calls: int = 0

    def __call__(
        self,
        command: list[str],
        *,
        input: bytes | None = None,
        cwd: Path | None = None,
        stdout: BinaryIO | None = None,
        stderr: BinaryIO | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = False,
        encoding: str | None = None,
        check: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        self.calls += 1
        if command == ["codex", "--version"]:
            assert capture_output and encoding == "utf-8" and check
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="codex test")
        assert cwd is not None and stdout is not None and stderr is not None and env is not None
        assert input == b"prompt" and not check and timeout > 0
        assert command[1] == "exec" and command.count("--ignore-user-config") == 1
        assert 'approval_policy="never"' in command and 'sandbox_mode="read-only"' in command
        assert "features.hooks=false" in command
        if self.error is not None:
            raise self.error
        stdout.write(EVENTS)
        stderr.write(b"diagnostic")
        (cwd / "answer.md").write_bytes(self.answer)
        if self.trace:
            write_trace(cwd / "trace")
            request: Path = cwd / "trace/trace-one/payloads/request.json"
            request.write_text(request.read_text().replace("gpt-6-astra", self.model))
            if self.exact:
                request.write_text(
                    json.dumps(
                        {
                            "model": self.model,
                            "input": [
                                {"type": "additional_tools", "tools": []},
                                {
                                    "type": "message",
                                    "role": "developer",
                                    "content": [{"type": "input_text", "text": "instructions"}],
                                },
                                {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": "prompt"}],
                                },
                            ],
                        }
                    )
                )
            if self.tool_call:
                response: Path = cwd / "trace/trace-one/payloads/response.json"
                response.write_text(response.read_text().replace('"type": "reasoning"', '"type": "custom_tool_call"'))
        return subprocess.CompletedProcess(args=command, returncode=self.code)


@pytest.fixture
def provider(tmp_path: Path) -> Iterator[Codex]:
    reference: Path = tmp_path / "reference/breakout"
    reference.mkdir(parents=True)
    (reference / "instructions.md").write_text("instructions\n", encoding="utf-8")
    with patch.object(Sources, "verify_prompt", return_value=b"prompt"):
        yield Codex(sources=Sources(root=reference.parent))


def test_comparison_configuration_matches_retained_effective_values(
    provider: Codex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.toml").write_text("invalid = [")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config: list[str] = provider.comparison.configuration()
    argument: str
    actual: dict[str, JsonValue] = {}
    for argument in config[1::2]:
        actual[argument.partition("=")[0]] = TypeAdapter(JsonValue).validate_json(argument.partition("=")[2])
    expected: dict[str, JsonValue] = {
        "model": "gpt-6-astra",
        "model_reasoning_effort": "xhigh",
        "model_reasoning_summary": "detailed",
        "model_context_window": 872000,
        "model_auto_compact_token_limit": 1000000,
        "model_instructions_file": str(provider.sources.instructions_path.resolve()),
        "web_search": "disabled",
        "agents.enabled": False,
        "approval_policy": "never",
        "sandbox_mode": "read-only",
    }
    expected.update({f"features.{name}": False for name in FEATURES})
    assert len(FEATURES) == 26 and len(config) == 2 * len(actual)
    assert config[::2] == ["-c"] * len(actual) and actual == expected
    with pytest.raises(ValueError):
        provider.configuration()


def test_preparation_never_probes_or_executes_and_uses_mode_deviations(provider: Codex, tmp_path: Path) -> None:
    output: Path = tmp_path / "run"
    transport: Transport = Transport()
    with patch("ember.providers.codex.subprocess.run", side_effect=transport):
        record: RunRecord = Runner(provider=provider, mode="comparison")(output=output, execute=False)
    assert record.status == "prepared" and record.mode == "comparison" and transport.calls == 1
    assert not record.tool_inventory_empty and not record.prompt_context_matches and record.preflight_returncode is None
    assert record.command.count("--ignore-user-config") == 1 and record.command[-1] == "-"
    assert not (output / "inventory.json").exists() and not (output / "answer.md").exists()
    assert any("hooks are disabled" in value for value in record.deviations)
    assert not any("hooks remain enabled" in value or "are inherited" in value for value in record.deviations)


@pytest.mark.parametrize("failure", [OSError("missing executable"), subprocess.TimeoutExpired(cmd="codex", timeout=1)])
def test_preparation_failure_retains_comparison_policy(provider: Codex, tmp_path: Path, failure: Exception) -> None:
    with patch("ember.providers.codex.subprocess.run", side_effect=failure):
        record: RunRecord = Runner(provider=provider, mode="comparison")(output=tmp_path / "run", execute=True)
    assert record.status == "failed" and record.mode == "comparison" and record.errors
    assert any("hooks are disabled" in value for value in record.deviations)


def test_execution_saves_trace_evidence_and_reaudit_uses_recorded_model(provider: Codex, tmp_path: Path) -> None:
    selected: Codex = Codex(sources=provider.sources, model="custom-model")
    output: Path = tmp_path / "run"
    transport: Transport = Transport(model=selected.model)
    with patch("ember.providers.codex.subprocess.run", side_effect=transport):
        record: RunRecord = Runner(provider=selected, mode="comparison")(output=output, execute=True)
    assert record.status == "completed" and record.returncode == 0 and transport.calls == 2
    assert record.audit is not None and record.audit.passed and record.audit.mode == "comparison"
    assert not record.tool_inventory_empty and not record.prompt_context_matches
    evidence: TraceResult = TraceResult.model_validate_json((output / "trace-audit.json").read_text())
    assert evidence.advertised_tools == ["functions.exec"] and evidence.tool_call_count == 0
    assert evidence.inference_count == 1 and evidence.extra_message_count == 1
    assert provider.audit(root=output, mode="comparison").passed


def test_actual_exact_request_sets_true_run_and_audit_flags(provider: Codex, tmp_path: Path) -> None:
    output: Path = tmp_path / "run"
    with patch("ember.providers.comparison.subprocess.run", side_effect=Transport(exact=True)):
        record: RunRecord = Runner(provider=provider, mode="comparison")(output=output, execute=True)
    assert record.status == "completed" and record.tool_inventory_empty and record.prompt_context_matches
    audit: Audit = provider.audit(root=output, mode="comparison")
    assert audit.passed and audit.tool_inventory_empty and audit.prompt_context_matches


def test_comparison_preserves_registered_provider_identity(provider: Codex, tmp_path: Path) -> None:
    output: Path = tmp_path / "run"
    with (
        patch.object(Codex, "name", "local-codex"),
        patch("ember.providers.comparison.subprocess.run", side_effect=Transport()),
    ):
        record: RunRecord = Runner(provider=provider, mode="comparison")(output=output, execute=True)
        assert record.status == "completed" and record.provider == "local-codex"
        assert provider.audit(root=output, mode="comparison").passed


@pytest.mark.parametrize("failure", ["tool-call", "missing-trace", "invalid-utf8", "nonzero", "os-error", "timeout"])
def test_execution_failures_retain_process_outcome(provider: Codex, tmp_path: Path, failure: str) -> None:
    transport: Transport = Transport(
        code=3 if failure == "nonzero" else 0,
        answer=b"\xff" if failure == "invalid-utf8" else b"answer",
        trace=failure != "missing-trace",
        tool_call=failure == "tool-call",
        error=OSError("process failed")
        if failure == "os-error"
        else (subprocess.TimeoutExpired(cmd="codex", timeout=1) if failure == "timeout" else None),
    )
    output: Path = tmp_path / "run"
    with patch("ember.providers.codex.subprocess.run", side_effect=transport):
        record: RunRecord = Runner(provider=provider, mode="comparison")(output=output, execute=True)
    assert record.status == "failed" and record.wall_seconds is not None and record.wall_seconds >= 0
    assert record.returncode == (None if failure in ("os-error", "timeout") else transport.code)
    assert RunRecord.model_validate_json((output / "run.json").read_text()) == record
    if failure in ("tool-call", "missing-trace"):
        evidence: TraceResult = TraceResult.model_validate_json((output / "trace-audit.json").read_text())
        assert evidence.errors and evidence.tool_call_count is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "claude"),
        ("model", "changed"),
        ("status", "blocked"),
        ("config", []),
        ("command", ["changed"]),
    ],
)
def test_changed_prepared_record_never_starts_execution(
    provider: Codex, tmp_path: Path, field: str, value: object
) -> None:
    with patch(
        "ember.providers.codex.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="test"),
    ):
        record: RunRecord = provider.prepare(output=tmp_path, prompt=b"prompt", mode="comparison")
    with patch("ember.providers.codex.subprocess.run") as process, pytest.raises(ValueError):
        provider.execute(output=tmp_path, record=record.model_copy(update={field: value}), prompt=b"prompt")
    process.assert_not_called()


def test_changed_prompt_never_starts_execution(provider: Codex, tmp_path: Path) -> None:
    with patch(
        "ember.providers.codex.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="test"),
    ):
        record: RunRecord = provider.prepare(output=tmp_path, prompt=b"prompt", mode="comparison")
    with patch("ember.providers.codex.subprocess.run") as process, pytest.raises(ValueError, match="prompt"):
        provider.execute(output=tmp_path, record=record, prompt=b"changed")
    process.assert_not_called()


def test_reaudit_supports_relocated_answer_and_instruction_paths(provider: Codex, tmp_path: Path) -> None:
    output: Path = tmp_path / "run"
    with patch("ember.providers.codex.subprocess.run", side_effect=Transport()):
        Runner(provider=provider, mode="comparison")(output=output, execute=True)
    relocated: Path = output.rename(tmp_path / "moved")
    reference: Path = provider.sources.root.rename(tmp_path / "moved-reference")
    current: Codex = Codex(sources=Sources(root=reference))
    assert current.audit(root=relocated, mode="comparison").passed


@pytest.mark.parametrize(
    "failure",
    ["mode", "provider", "returncode", "prompt", "trace", "events", "command", "config", "duplicate-key", "odd-config"],
)
def test_saved_audit_rejects_changed_or_incomplete_evidence(provider: Codex, tmp_path: Path, failure: str) -> None:
    output: Path = tmp_path / "run"
    with patch("ember.providers.codex.subprocess.run", side_effect=Transport()):
        record: RunRecord = Runner(provider=provider, mode="comparison")(output=output, execute=True)
    if failure in ("mode", "provider", "returncode"):
        record = record.model_copy(update={failure: {"mode": "strict", "provider": "claude", "returncode": 1}[failure]})
    elif failure == "command":
        record = record.model_copy(update={"command": ["codex", "exec"]})
    elif failure == "config":
        record = record.model_copy(update={"config": ["-c", 'approval_policy="on-request"']})
    elif failure == "duplicate-key":
        record = record.model_copy(update={"config": [*record.config, "-c", 'model="gpt-6-astra"']})
    elif failure == "odd-config":
        record = record.model_copy(update={"config": [*record.config, "-c"]})
    elif failure == "prompt":
        (output / "prompt.txt").write_bytes(b"changed")
    elif failure == "trace":
        (output / "trace/trace-one/payloads/response.json").unlink()
    else:
        (output / "events.jsonl").write_text("{}")
    (output / "run.json").write_text(record.model_dump_json(), encoding="utf-8")
    if failure in ("trace", "events"):
        audit: Audit = provider.audit(root=output, mode="comparison")
        assert not audit.passed and audit.errors
    else:
        with pytest.raises(ValueError):
            provider.audit(root=output, mode="comparison")
