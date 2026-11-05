"""No model calls in tests: transport doubles exercise the production gates."""

import json
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from operator import methodcaller
from pathlib import Path
from typing import BinaryIO
from unittest.mock import Mock, patch

import pytest

from ember.models import RunMode
from ember.module import Context
from ember.playground.breakout.base import Sources
from ember.providers.codex import Codex, UserConfig
from ember.providers.records import RunRecord
from ember.runner import Runner, RunRequest

EMPTY_INVENTORY: str = json.dumps(
    [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "instructions"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "prompt"}]},
        {"type": "additional_tools", "tools": []},
    ]
)
EVENTS: bytes = (
    b'{"type":"thread.started"}\n'
    b'{"type":"turn.started"}\n'
    b'{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}\n'
    b'{"type":"turn.completed"}'
)


@dataclass
class ExecutionTransport:
    """Write process output at the transport boundary, leaving file I/O real."""

    code: int = 0
    answer: bytes = b"answer\n"

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
    ) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
        if command[1] == "--version":
            assert capture_output and encoding == "utf-8" and check
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="codex test", stderr="")
        if command[1:3] == ["debug", "prompt-input"]:
            assert command[-1] == "prompt" and cwd is not None and capture_output
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=EMPTY_INVENTORY, stderr="")
        assert cwd is not None and stdout is not None and stderr is not None and env is not None
        assert input == b"prompt" and not check and timeout > 0
        assert command[-1] == "-" and command[1] == "exec"
        assert Path(command[command.index("-o") + 1]) == cwd / "answer.md"
        assert env["CODEX_ROLLOUT_TRACE_ROOT"] == str(cwd / "trace")
        stdout.write(EVENTS)
        stderr.write(b"diagnostic")
        (cwd / "answer.md").write_bytes(self.answer)
        return subprocess.CompletedProcess(args=command, returncode=self.code)


def test_configuration_retains_hooks_and_disables_named_servers(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('[mcp_servers.test-server]\nurl="https://example.org"')
    runner: Codex = Codex(sources=Sources(root=tmp_path))
    with patch("ember.providers.codex.os.environ", {"CODEX_HOME": str(tmp_path)}):
        flags: list[str] = runner.configuration()
    assert "mcp_servers.test-server.enabled=false" in flags
    assert 'model_reasoning_effort="xhigh"' in flags
    assert "features.hooks=false" not in flags and "--ignore-user-config" not in flags
    assert "test.server" in repr(UserConfig(mcp_servers={"test.server": {}}))


def test_missing_configuration_and_unsupported_server_name(tmp_path: Path) -> None:
    runner: Codex = Codex(sources=Sources(root=tmp_path))
    with patch("ember.providers.codex.os.environ", {"CODEX_HOME": str(tmp_path)}):
        assert 'model="gpt-6-astra"' in runner.configuration()
        (tmp_path / "config.toml").write_text('[mcp_servers."test.server"]\nurl="https://example.org"')
        with pytest.raises(ValueError, match="MCP server names"):
            runner.configuration()


@pytest.mark.parametrize(
    "inventory,code,status",
    [
        (EMPTY_INVENTORY, 0, "prepared"),
        (EMPTY_INVENTORY.replace('"prompt"', '"changed"'), 0, "blocked"),
        ('[{"type":"additional_tools","tools":[{"name":"exec"}]}]', 0, "blocked"),
        ("malformed", 0, "blocked"),
        ('[{"type":"message"}]', 0, "blocked"),
        (EMPTY_INVENTORY, 1, "blocked"),
    ],
)
def test_preflight_saves_evidence_and_never_calls_model(
    tmp_path: Path,
    inventory: str,
    code: int,
    status: str,
) -> None:
    runner: Runner = Runner(provider=Codex(sources=Sources(root=tmp_path)))
    transport: Mock = Mock(
        side_effect=[
            subprocess.CompletedProcess(args=[], returncode=0, stdout="codex 0.test\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=code, stdout=inventory, stderr="notice"),
        ]
    )
    (tmp_path / "breakout").mkdir()
    (tmp_path / "breakout/instructions.md").write_text("instructions\n", encoding="utf-8")
    output: Path = tmp_path / "run"
    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch.object(Codex, "configuration", return_value=[]),
        patch("ember.providers.codex.subprocess.run", transport),
    ):
        result: RunRecord = runner(output=output, execute=status == "blocked")
    assert result.status == status and transport.call_count == 2
    assert result.preflight_returncode == code and result.preflight_error == "notice"
    assert (output / "inventory.json").read_text() == inventory
    assert json.loads((output / "run.json").read_text())["status"] == status
    assert not (output / "answer.md").exists()
    with pytest.raises(FileExistsError), patch.object(Sources, "verify_prompt", return_value=b"prompt"):
        runner(output=output, execute=False)


def test_execution_rejects_failed_gate(tmp_path: Path) -> None:
    record: RunRecord = RunRecord(
        status="blocked",
        model="gpt-6-astra",
        cli_version="test",
        command=[],
        config=[],
        tool_inventory_empty=False,
    )
    with (
        patch("ember.providers.codex.subprocess.run") as transport,
        pytest.raises(ValueError, match="preflight"),
    ):
        Codex(sources=Sources(root=tmp_path)).execute(output=tmp_path, record=record, prompt=b"prompt")
    transport.assert_not_called()
    record = record.model_copy(update={"status": "prepared", "tool_inventory_empty": True})
    with (
        patch("ember.providers.codex.subprocess.run") as transport,
        pytest.raises(ValueError, match="preflight"),
    ):
        Codex(sources=Sources(root=tmp_path)).execute(output=tmp_path, record=record, prompt=b"prompt")
    transport.assert_not_called()


def test_execution_timeout_retains_failure_record(tmp_path: Path) -> None:
    record: RunRecord = RunRecord(
        status="prepared",
        model="gpt-6-astra",
        cli_version="test",
        command=["codex"],
        config=[],
        tool_inventory_empty=True,
        prompt_context_matches=True,
    )
    with patch("ember.providers.codex.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=1)):
        result: RunRecord = Codex(sources=Sources(root=tmp_path)).execute(
            output=tmp_path,
            record=record,
            prompt=b"prompt",
        )
    assert result.status == "failed" and "timed out" in result.errors[0]
    assert (tmp_path / "events.jsonl").is_file()


@pytest.mark.parametrize(
    "code,answer,status", [(0, b"answer\n", "completed"), (1, b"answer\n", "failed"), (0, b"\xff", "failed")]
)
def test_execution_success_nonzero_exit_and_invalid_utf8(tmp_path: Path, code: int, answer: bytes, status: str) -> None:
    boundary: ExecutionTransport = ExecutionTransport(code=code, answer=answer)
    transport: Mock = Mock(side_effect=boundary)
    output: Path = tmp_path / "run"
    runner: Runner = Runner(provider=Codex(sources=Sources(root=tmp_path)))
    (tmp_path / "breakout").mkdir()
    (tmp_path / "breakout/instructions.md").write_text("instructions\n", encoding="utf-8")
    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch.object(Codex, "configuration", return_value=[]),
        patch("ember.providers.codex.subprocess.run", transport),
    ):
        result: RunRecord = runner(output=output, execute=True)
    assert result.status == status and transport.call_count == 3
    assert transport.call_args.args[0] == result.command
    assert json.loads((output / "run.json").read_text())["status"] == status
    if status == "completed":
        assert result.audit is not None and result.audit.passed
    assert not result.exact_author_reproduction


@pytest.mark.parametrize(
    "error", [FileNotFoundError("codex"), subprocess.TimeoutExpired(cmd="codex", timeout=1), ValueError("bad probe")]
)
def test_preflight_exceptions_save_failed_outcome(tmp_path: Path, error: Exception) -> None:
    output: Path = tmp_path / "run"
    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch.object(Codex, "prepare", side_effect=error),
    ):
        result: RunRecord = Runner(provider=Codex(sources=Sources(root=tmp_path)))(output=output, execute=True)
    assert result.status == "failed" and result.errors
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


def test_execution_os_error_saves_failed_outcome(tmp_path: Path) -> None:
    prepared: RunRecord = RunRecord(
        status="prepared", model="gpt-6-astra", tool_inventory_empty=True, prompt_context_matches=True
    )
    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch.object(Codex, "prepare", return_value=prepared),
        patch("ember.providers.codex.subprocess.run", side_effect=OSError("process failed")),
    ):
        result: RunRecord = Runner(provider=Codex(sources=Sources(root=tmp_path)))(
            output=tmp_path / "run", execute=True
        )
    assert result.status == "failed" and result.errors == ["process failed"]


@pytest.mark.parametrize("entrypoint", ["legacy", "request", "run"])
def test_module_entrypoints_preserve_run_record(tmp_path: Path, entrypoint: str) -> None:
    output: Path = tmp_path / "run"
    provider: Codex = Codex(sources=Sources(root=tmp_path))
    runner: Runner = Runner(provider=provider)
    request: RunRequest = RunRequest(output=output, execute=False)
    prepared: RunRecord = RunRecord(status="prepared", model=provider.model)
    result: RunRecord
    with (
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch.object(Codex, "prepare", return_value=prepared) as prepare,
        patch.object(Codex, "execute") as execute,
    ):
        if entrypoint == "legacy":
            result = runner(output=output, execute=False)
        elif entrypoint == "request":
            result = runner(request)
        else:
            result = runner.run(request)
    expected: RunRecord = prepared.model_copy(
        update={"source_commit": provider.sources.commit, "prompt_sha256": sha256(b"prompt").hexdigest()}
    )
    assert result == expected
    assert RunRecord.model_validate_json((output / "run.json").read_text()) == expected
    assert json.loads((output / "run.json").read_text()) == expected.model_dump(mode="json")
    assert (output / "prompt.txt").read_bytes() == b"prompt"
    assert runner.provider is provider and runner.mode == runner.ctx.value == "strict"
    prepare.assert_called_once_with(output=output, prompt=b"prompt", mode="strict")
    execute.assert_not_called()


def test_shared_mode_context_preserves_gate_and_restores(tmp_path: Path) -> None:
    context: Context[RunMode] = Context(value="comparison")
    runner: Runner = Runner(provider=Codex(sources=Sources(root=tmp_path)), context=context)
    request: RunRequest = RunRequest(output=tmp_path / "run", execute=False)
    assert runner.ctx is context and runner.mode == "strict"
    with patch.object(Codex, "modes", ("strict",)), pytest.raises(ValueError, match="does not support mode comparison"):
        runner.run(request)
    assert not request.output.exists()
    with (
        context.override("strict"),
        patch.object(Sources, "verify_prompt", return_value=b"prompt"),
        patch.object(Codex, "prepare", return_value=RunRecord(status="prepared", model=runner.provider.model)),
    ):
        assert runner.run(request).mode == "strict"
    assert context.value == "comparison"


def test_runner_captures_mode_before_provider_changes_context(tmp_path: Path) -> None:
    context: Context[RunMode] = Context(value="strict")
    runner: Runner = Runner(provider=Codex(sources=Sources(root=tmp_path)), context=context)
    scopes: ExitStack
    with ExitStack() as scopes:

        def prepare(*, output: Path, prompt: bytes, mode: RunMode) -> RunRecord:
            assert output == tmp_path / "run" and prompt == b"prompt" and mode == "strict"
            scopes.enter_context(context.override("comparison"))
            return RunRecord(status="prepared", model=runner.provider.model)

        with (
            patch.object(Sources, "verify_prompt", return_value=b"prompt"),
            patch.object(Codex, "prepare", side_effect=prepare),
            patch.object(Codex, "execute") as execute,
        ):
            result: RunRecord = runner.run(RunRequest(output=tmp_path / "run", execute=True))
        assert context.value == "comparison"
        assert result.mode == "strict" and result.status == "failed" and "preflight" in result.errors[0]
        execute.assert_not_called()
    assert context.value == "strict"


def test_run_facade_rejects_missing_and_mixed_arguments(tmp_path: Path) -> None:
    runner: Runner = Runner(provider=Codex(sources=Sources(root=tmp_path)))
    request: RunRequest = RunRequest(output=tmp_path / "run", execute=False)
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__")(runner)
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__", output=request.output)(runner)
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__", execute=False)(runner)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", request, output=request.output)(runner)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", request, execute=False)(runner)
    assert not request.output.exists()
