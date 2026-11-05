"""Comparison transport and evidence tests never call a model."""

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from unittest.mock import patch

import pytest

from ember.cli import main
from ember.playground.breakout.base import Sources
from ember.providers import Claude, Codex
from ember.providers.events import Transcript
from ember.providers.records import RunRecord
from ember.runner import Runner

LOG: str = """{"type":"system","subtype":"hook_started","session_id":"s"}
{"type":"system","subtype":"init","session_id":"s","tools":[],"model":"resolved-model"}
{"type":"rate_limit_event","session_id":"s","rate_limit_info":{"status":"allowed"}}
{"type":"assistant","session_id":"s","message":{"id":"m","role":"assistant","content":[{"type":"thinking","thinking":"reasoning"},{"type":"redacted_thinking","data":"opaque"}]}}
{"type":"assistant","session_id":"s","message":{"id":"m","role":"assistant","content":[{"type":"text","text":"answer"}]}}
{"type":"result","subtype":"success","session_id":"s","num_turns":1,"is_error":false,"stop_reason":"end_turn","result":"answer","permission_denials":[],"usage":{"input_tokens":12,"output_tokens":3,"cache_creation":{"ephemeral_5m_input_tokens":0},"service_tier":"standard"}}
"""


@dataclass
class Transport:
    """Write controlled evidence at the subprocess boundary."""

    content: bytes = LOG.encode()
    code: int = 0
    error: Exception | None = None
    calls: int = 0

    def __call__(
        self,
        command: list[str],
        *,
        input: bytes | None = None,
        cwd: Path | None = None,
        stdout: BinaryIO | None = None,
        stderr: BinaryIO | None = None,
        capture_output: bool = False,
        encoding: str | None = None,
        check: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        self.calls += 1
        if command == ["claude", "--version"]:
            assert capture_output and encoding == "utf-8" and check
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="test")
        assert input == b"prompt" and cwd is not None and stdout is not None and stderr is not None
        assert not check and timeout > 0 and "prompt" not in command
        assert command[command.index("--tools") + 1] == ""
        assert "--strict-mcp-config" in command and "--no-session-persistence" in command
        assert command[command.index("--system-prompt") + 1] == "instructions"
        stdout.write(self.content)
        stderr.write(b"diagnostic")
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(args=command, returncode=self.code)


@pytest.fixture
def provider(tmp_path: Path) -> Iterator[Claude]:
    (tmp_path / "breakout").mkdir()
    (tmp_path / "breakout/instructions.md").write_text("instructions\n", encoding="utf-8")
    with patch.object(Sources, "verify_prompt", return_value=b"prompt"):
        yield Claude(sources=Sources(root=tmp_path), model="opus")


def test_comparison_preparation_never_executes(provider: Claude, tmp_path: Path) -> None:
    transport: Transport = Transport()
    with patch("ember.providers.claude.subprocess.run", side_effect=transport):
        record: RunRecord = Runner(provider=provider, mode="comparison")(output=tmp_path / "run", execute=False)
    assert transport.calls == 1 and record.status == "prepared" and record.mode == "comparison"
    assert not record.tool_inventory_empty and not record.prompt_context_matches and not record.errors
    assert any("CLAUDE.md" in deviation for deviation in record.deviations)
    assert not (tmp_path / "run/answer.md").exists()


def test_comparison_cli_execution_and_explicit_audit(provider: Claude, tmp_path: Path) -> None:
    output: Path = tmp_path / "run"
    transport: Transport = Transport()
    run: list[str] = ["run", "--provider", "claude", "--model", "opus", "--mode", "comparison"]
    run.extend(["--reference", str(tmp_path), "--output", str(output), "--execute"])
    with patch("ember.providers.claude.subprocess.run", side_effect=transport):
        assert main(run) == 0
    record: RunRecord = RunRecord.model_validate_json((output / "run.json").read_bytes())
    assert transport.calls == 2 and record.status == "completed" and record.mode == "comparison"
    assert record.audit is not None and record.audit.passed and record.audit.mode == "comparison"
    assert not record.audit.tool_inventory_empty and not record.audit.prompt_context_matches
    assert record.audit.usage == {"input_tokens": 12, "output_tokens": 3}
    assert not record.exact_author_reproduction and record.model == "opus"
    assert (output / "answer.md").read_bytes() == b"answer"
    assert (output / "events.jsonl").read_text() == LOG
    command: list[str] = ["audit", "--provider", "claude", "--reference", str(tmp_path), "--run", str(output)]
    assert main(command) == 1
    assert main([*command, "--mode", "comparison"]) == 0
    (output / "answer.md").write_text("changed", encoding="utf-8")
    assert main([*command, "--mode", "comparison"]) == 1
    (output / "prompt.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prompt differs"):
        provider.audit(root=output, mode="comparison")


@pytest.mark.parametrize(
    "content,code,error",
    [
        (LOG.encode(), 2, None),
        (LOG.encode(), 0, subprocess.TimeoutExpired(cmd="claude", timeout=1)),
        (LOG.encode(), 0, OSError("spawn failed")),
        (b"not json", 0, None),
        (b"\xff", 0, None),
        (b"", 0, None),
    ],
)
def test_execution_failures_retain_evidence(
    provider: Claude,
    tmp_path: Path,
    content: bytes,
    code: int,
    error: Exception | None,
) -> None:
    transport: Transport = Transport(content=content, code=code, error=error)
    output: Path = tmp_path / "run"
    with patch("ember.providers.claude.subprocess.run", side_effect=transport):
        record: RunRecord = Runner(provider=provider, mode="comparison")(output=output, execute=True)
    assert transport.calls == 2 and record.status == "failed" and record.mode == "comparison"
    assert record.errors and record.returncode == (None if error else code)
    assert record.wall_seconds >= 0 and not (output / "answer.md").exists()
    assert (output / "events.jsonl").read_bytes() == content
    assert (output / "stderr.log").read_bytes() == b"diagnostic"


@pytest.mark.parametrize(
    "old,new",
    [
        ('"type":"text"', '"type":"tool_use"'),
        ('"type":"text"', '"type":"server_tool_use"'),
        ('"type":"assistant"', '"type":"user"'),
        ('"type":"rate_limit_event"', '"type":"unknown"'),
        ('"subtype":"hook_started"', '"subtype":"unknown"'),
        ('"tools":[]', '"tools":["Bash"]'),
        ('"tools":[]', '"missing_tools":[]'),
        ('"num_turns":1', '"num_turns":2'),
        ('"num_turns":1', '"missing_turns":1'),
        ('"is_error":false', '"is_error":true'),
        ('"is_error":false', '"missing_is_error":false'),
        ('"stop_reason":"end_turn"', '"stop_reason":"max_tokens"'),
        ('"permission_denials":[]', '"permission_denials":[{}]'),
        ('"permission_denials":[]', '"missing_denials":[]'),
        ('"result":"answer"', '"result":"other"'),
        ('"result":"answer"', '"result":""'),
        ('"type":"result"', '"parent_tool_use_id":"tool","type":"result"'),
        ('"type":"result"', '"deferred_tool_use":{},"type":"result"'),
        ('"type":"assistant"', '"aborted":true,"type":"assistant"'),
        ('"type":"assistant"', '"error":"overloaded","type":"assistant"'),
        ('"role":"assistant"', '"role":"user"'),
        (
            '"id":"m","role":"assistant","content":[{"type":"text"',
            '"id":"second","role":"assistant","content":[{"type":"text"',
        ),
        (
            '"type":"result","subtype":"success","session_id":"s"',
            '"type":"result","subtype":"success","session_id":"other"',
        ),
    ],
)
def test_invalid_evidence_cannot_pass(old: str, new: str) -> None:
    content: str = LOG.replace(old, new)
    assert content != LOG
    try:
        transcript: Transcript = Transcript.read(content)
    except ValueError:
        return
    assert not transcript(answer="answer").passed


@pytest.mark.parametrize("kind", ["empty", "missing_init", "missing_result", "duplicate_result", "order", "extra_text"])
def test_missing_duplicate_and_reordered_events(kind: str) -> None:
    lines: list[str] = LOG.splitlines()
    if kind == "empty":
        lines = []
    elif kind == "missing_init":
        lines.pop(1)
    elif kind == "missing_result":
        lines.pop()
    elif kind == "duplicate_result":
        lines.append(lines[-1])
    elif kind == "order":
        lines[1], lines[4] = lines[4], lines[1]
    else:
        lines.insert(4, lines[4])
    transcript: Transcript = Transcript.read("\n".join(lines))
    assert not transcript(answer="answer").passed
    if kind in ("empty", "missing_result"):
        assert transcript.answer == ""


def test_default_legacy_and_unsupported_modes(provider: Claude, tmp_path: Path) -> None:
    record: RunRecord = RunRecord(status="blocked", model="sonnet")
    assert record.mode == "strict"
    assert not Transcript.read(LOG)(answer="changed").passed
    codex: Codex = Codex(sources=provider.sources)
    with patch.object(Codex, "modes", ("strict",)):
        with pytest.raises(ValueError, match="does not support"), patch("subprocess.run") as transport:
            Runner(provider=codex, mode="comparison")(output=tmp_path / "unsupported", execute=True)
        transport.assert_not_called()
        assert not (tmp_path / "unsupported").exists()
        with pytest.raises(ValueError, match="strict mode"):
            codex.prepare(output=tmp_path, prompt=b"prompt", mode="comparison")
        with pytest.raises(ValueError, match="strict mode"):
            codex.audit(root=tmp_path, mode="comparison")
        with pytest.raises(ValueError, match="strict mode"):
            codex.execute(output=tmp_path, record=record.model_copy(update={"mode": "comparison"}), prompt=b"prompt")
    with pytest.raises(OSError):
        provider.audit(root=tmp_path, mode="comparison")


@pytest.mark.parametrize("failure", ["missing_cli", "mode", "provider"])
def test_prepare_failure_keeps_requested_label(provider: Claude, tmp_path: Path, failure: str) -> None:
    prepared: RunRecord = RunRecord(status="prepared", provider="claude", mode="comparison", model="opus")
    if failure == "mode":
        prepared = prepared.model_copy(update={"mode": "strict"})
    elif failure == "provider":
        prepared = prepared.model_copy(update={"provider": "codex"})
    with (
        patch.object(
            Claude,
            "prepare",
            return_value=prepared,
            side_effect=FileNotFoundError("missing") if failure == "missing_cli" else None,
        ),
        patch.object(Claude, "execute") as execute,
    ):
        record: RunRecord = Runner(provider=provider, mode="comparison")(output=tmp_path / "run", execute=True)
    assert record.status == "failed" and record.mode == "comparison" and record.provider == "claude"
    execute.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "blocked"),
        ("provider", "codex"),
        ("model", "other"),
        ("command", ["unsafe"]),
    ],
)
def test_prepared_command_cannot_be_replaced(provider: Claude, tmp_path: Path, field: str, value: object) -> None:
    record: RunRecord = RunRecord(
        status="prepared",
        provider="claude",
        mode="comparison",
        model="opus",
        command=["claude", *provider.configuration()],
    ).model_copy(update={field: value})
    with patch("subprocess.run") as transport, pytest.raises(ValueError):
        provider.execute(output=tmp_path, record=record, prompt=b"prompt")
    transport.assert_not_called()


def test_saved_audit_rejects_failed_process_and_wrong_mode(provider: Claude, tmp_path: Path) -> None:
    record: RunRecord = RunRecord(status="failed", provider="claude", mode="comparison", model="opus", returncode=2)
    (tmp_path / "run.json").write_text(record.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="successful process"):
        provider.audit(root=tmp_path, mode="comparison")
    (tmp_path / "run.json").write_text(record.model_copy(update={"returncode": 0, "mode": "strict"}).model_dump_json())
    with pytest.raises(ValueError, match="matching Claude"):
        provider.audit(root=tmp_path, mode="comparison")


@pytest.mark.parametrize("kind", ["hook_progress", "hook_response", "rate_limit_event"])
def test_metadata_is_accepted_before_but_not_after_completion(kind: str) -> None:
    event: str = (
        '{"type":"rate_limit_event","session_id":"s"}'
        if kind == "rate_limit_event"
        else '{"type":"system","subtype":"' + kind + '","session_id":"s"}'
    )
    lines: list[str] = LOG.splitlines()
    assert Transcript.read("\n".join([*lines[:-1], event, lines[-1]]))(answer="answer").passed
    assert not Transcript.read("\n".join([*lines, event]))(answer="answer").passed


def test_missing_session_or_answer_fails_closed(provider: Claude, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Transcript.read(LOG.replace('"session_id":"s",', "", 1))
    record: RunRecord = RunRecord(status="failed", provider="claude", mode="comparison", model="opus", returncode=0)
    (tmp_path / "run.json").write_text(record.model_dump_json(), encoding="utf-8")
    (tmp_path / "prompt.txt").write_bytes(b"prompt")
    (tmp_path / "events.jsonl").write_text(LOG, encoding="utf-8")
    with pytest.raises(OSError):
        provider.audit(root=tmp_path, mode="comparison")


def test_cli_rejects_unsupported_provider_mode_before_creating_folder(tmp_path: Path) -> None:
    output: Path = tmp_path / "unsupported"
    with patch.object(Codex, "modes", ("strict",)):
        assert main(["run", "--provider", "codex", "--mode", "comparison", "--output", str(output)]) == 1
    assert not output.exists()


@pytest.mark.parametrize(
    "metadata,accepted",
    [
        ('"subtype":"ui_invalidate","event":"ui.render"', True),
        ('"subtype":"ui_invalidate"', False),
        ('"subtype":"ui_invalidate","event":"ui.other"', False),
        ('"subtype":"unknown","event":"ui.render"', False),
    ],
)
def test_ui_refresh_requires_the_exact_event_pair(metadata: str, accepted: bool) -> None:
    event: str = '{"type":"system","session_id":"s",' + metadata + "}"
    transcript: Transcript = Transcript.read(event + "\n" + LOG)
    assert transcript(answer="answer").passed is accepted
    if accepted:
        assert transcript.events[0].event == "ui.render"
    assert not Transcript.read(LOG + event + "\n")(answer="answer").passed


def test_ui_refresh_rejects_non_string_event() -> None:
    with pytest.raises(ValueError):
        Transcript.read('{"type":"system","subtype":"ui_invalidate","event":1,"session_id":"s"}')
