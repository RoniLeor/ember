"""The retained Codex comparison policy and its actual-request evidence checks."""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import JsonValue, TypeAdapter

from ember.base import Source
from ember.models import Audit

from .audit import EventAudit
from .records import RunRecord
from .trace import TraceAudit, TraceResult

RUN_TIMEOUT: int = 1_800
PREFLIGHT_TIMEOUT: int = 90
EXEC_ARGS: tuple[str, ...] = ("exec", "--strict-config", "--skip-git-repo-check", "--json", "--ignore-user-config")
DISABLED: tuple[str, ...] = (
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


@dataclass(frozen=True)
class CodexComparison:
    """Keep comparison policy separate from the unchanged strict provider path."""

    sources: Source
    name: str
    model: str
    executable: str
    deviations: ClassVar[tuple[str, ...]] = (
        "User config is ignored; hooks are disabled; approval and sandbox are explicitly never/read-only.",
        "Global instruction discovery can add context and advertise tools.",
        "The CLI and runtime context can differ from the authors' request.",
        "A completed comparison requires zero attempted tool calls verified from the raw trace.",
        "The authors' evaluator is unavailable.",
    )

    def configuration(self) -> list[str]:
        """Match effective manual overrides without reading the ignored user configuration."""
        values: dict[str, JsonValue] = {
            "model": self.model,
            "model_reasoning_effort": "xhigh",
            "model_reasoning_summary": "detailed",
            "model_context_window": 872000,
            "model_auto_compact_token_limit": 1000000,
            "model_instructions_file": str(self.sources.instructions_path.resolve()),
            "web_search": "disabled",
            "agents.enabled": False,
            "approval_policy": "never",
            "sandbox_mode": "read-only",
        }
        values.update({f"features.{name}": False for name in DISABLED})
        return [argument for key, value in values.items() for argument in ("-c", f"{key}={json.dumps(value)}")]

    def command(self, *, output: Path, config: list[str]) -> list[str]:
        """Build one command without redundant model or policy overrides."""
        return [self.executable, *EXEC_ARGS, *config, "-o", str((output / "answer.md").resolve()), "-"]

    def prepare(self, *, output: Path, prompt: bytes) -> RunRecord:
        """Prepare direct execution; an offline probe cannot reproduce ignored user configuration."""
        record: RunRecord = RunRecord(
            status="failed",
            provider=self.name,
            mode="comparison",
            model=self.model,
            deviations=list(self.deviations),
        )
        try:
            if prompt != self.sources.verify_prompt():
                raise ValueError("Prompt differs from the pinned research prompt")
            config: list[str] = self.configuration()
            version: subprocess.CompletedProcess[str] = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                encoding="utf-8",
                check=True,
                timeout=PREFLIGHT_TIMEOUT,
            )
            return record.model_copy(
                update={
                    "status": "prepared",
                    "cli_version": version.stdout.strip(),
                    "config": config,
                    "command": self.command(output=output, config=config),
                }
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return record.model_copy(update={"errors": [str(error)]})

    def execute(self, *, output: Path, record: RunRecord, prompt: bytes) -> RunRecord:
        """Execute only a matching prepared comparison and retain process failures."""
        self.prepared(output=output, record=record, prompt=prompt)
        started: float = time.monotonic()
        code: int | None = None
        errors: list[str] = []
        environment: dict[str, str] = {**os.environ, "CODEX_ROLLOUT_TRACE_ROOT": str((output / "trace").resolve())}
        try:
            with (output / "events.jsonl").open("wb") as events, (output / "stderr.log").open("wb") as stderr:
                process: subprocess.CompletedProcess[bytes] = subprocess.run(
                    record.command,
                    input=prompt,
                    cwd=output,
                    stdout=events,
                    stderr=stderr,
                    env=environment,
                    check=False,
                    timeout=RUN_TIMEOUT,
                )
                code = process.returncode
                if code != 0:
                    errors.append(f"Codex exited with code {code}")
        except subprocess.TimeoutExpired:
            errors.append("Model request timed out; partial output is not accepted")
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(str(error))
        return self.finish(output=output, record=record, prompt=prompt, started=started, code=code, errors=errors)

    def finish(
        self, *, output: Path, record: RunRecord, prompt: bytes, started: float, code: int | None, errors: list[str]
    ) -> RunRecord:
        """Keep return code and elapsed time even when raw evidence cannot be audited."""
        audit: Audit = Audit(
            passed=False,
            mode="comparison",
            tool_inventory_empty=False,
            prompt_context_matches=False,
            turn_completed=False,
            answer_matches=False,
            errors=[],
        )
        try:
            answer_path: Path = output / "answer.md"
            answer: str = answer_path.read_text(encoding="utf-8") if answer_path.exists() else ""
            trace: TraceResult = self.trace(root=output, record=record, prompt=prompt, answer=answer)
            (output / "trace-audit.json").write_text(trace.model_dump_json(indent=2), encoding="utf-8")
            audit = self.events(root=output, answer=answer, trace=trace)
        except (OSError, ValueError) as error:
            errors.append(str(error))
        return record.model_copy(
            update={
                "status": "completed" if code == 0 and audit.passed and not errors else "failed",
                "wall_seconds": time.monotonic() - started,
                "returncode": code,
                "audit": audit,
                "tool_inventory_empty": audit.tool_inventory_empty,
                "prompt_context_matches": audit.prompt_context_matches,
                "errors": [*errors, *audit.errors],
            }
        )

    def audit(self, *, root: Path) -> Audit:
        """Recheck saved semantic policy and raw trace using the recorded model."""
        record: RunRecord = RunRecord.model_validate_json((root / "run.json").read_text(encoding="utf-8"))
        self.saved(root=root, record=record)
        prompt: bytes = self.sources.verify_prompt()
        if (root / "prompt.txt").read_bytes() != prompt:
            raise ValueError("Saved run prompt differs from the pinned research prompt")
        answer: str = (root / "answer.md").read_text(encoding="utf-8")
        trace: TraceResult = self.trace(root=root, record=record, prompt=prompt, answer=answer)
        return self.events(root=root, answer=answer, trace=trace)

    def prepared(self, *, output: Path, record: RunRecord, prompt: bytes) -> None:
        """Reject changed prepared records before starting a process."""
        if (
            record.provider != self.name
            or record.model != self.model
            or record.mode != "comparison"
            or record.status != "prepared"
        ):
            raise ValueError("Comparison requires a matching prepared provider, model, mode, and status")
        config: list[str] = self.configuration()
        if record.config != config or record.command != self.command(output=output, config=config):
            raise ValueError("Comparison command or configuration differs from preparation")
        if prompt != self.sources.verify_prompt():
            raise ValueError("Comparison prompt differs from the pinned research prompt")

    def trace(self, *, root: Path, record: RunRecord, prompt: bytes, answer: str) -> TraceResult:
        """Read actual request evidence using the recorded model identity."""
        return TraceAudit()(
            root=root / "trace",
            model=record.model,
            prompt=prompt,
            instructions=self.sources.instructions_path.read_text(encoding="utf-8").rstrip("\n"),
            answer=answer,
        )

    @staticmethod
    def events(*, root: Path, answer: str, trace: TraceResult) -> Audit:
        """Require one completed JSONL turn and verified zero-call raw trace evidence."""
        audit: Audit = EventAudit()(
            content=(root / "events.jsonl").read_text(encoding="utf-8"),
            answer=answer,
            inventory_empty=trace.tool_inventory_empty,
            context_matches=trace.prompt_context_matches,
            mode="comparison",
        )
        errors: list[str] = [*audit.errors, *trace.errors]
        return audit.model_copy(update={"passed": not errors, "errors": errors})

    def saved(self, *, root: Path, record: RunRecord) -> None:
        """Validate recorded policy values while allowing relocated evidence paths."""
        if record.provider != self.name or record.mode != "comparison" or record.returncode != 0:
            raise ValueError("Comparison audit requires a matching Codex record and successful process exit")
        actual: dict[str, JsonValue] = self.values(record.config)
        expected: dict[str, JsonValue] = self.values(self.configuration())
        expected["model"] = record.model
        path: JsonValue = actual.pop("model_instructions_file", None)
        if not isinstance(path, str) or not path:
            raise ValueError("Missing recorded instruction path")
        expected.pop("model_instructions_file")
        if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("Recorded comparison configuration differs from the required policy")
        command: list[str] = list(record.command)
        if not command or Path(command[0]).name != Path(self.executable).name or command.count("-o") != 1:
            raise ValueError("Invalid recorded comparison command")
        position: int = command.index("-o") + 1
        if position >= len(command) or not command[position]:
            raise ValueError("Missing recorded answer path")
        command[0] = self.executable
        command[position] = str((root / "answer.md").resolve())
        if command != self.command(output=root, config=record.config):
            raise ValueError("Recorded comparison command differs from its configuration")

    @staticmethod
    def values(arguments: list[str]) -> dict[str, JsonValue]:
        """Parse generated CLI overrides without accepting duplicate policy keys."""
        if len(arguments) % 2:
            raise ValueError("Invalid comparison configuration arguments")
        values: dict[str, JsonValue] = {}
        index: int
        for index in range(0, len(arguments), 2):
            key: str
            separator: str
            value: str
            key, separator, value = arguments[index + 1].partition("=")
            if arguments[index] != "-c" or not key or not separator or key in values:
                raise ValueError("Invalid or duplicate comparison configuration key")
            values[key] = TypeAdapter(JsonValue).validate_json(value)
        return values
