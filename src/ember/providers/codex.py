"""Strict Codex execution and explicit traced comparison dispatch."""

import json
import os
import re
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ember.base import Source
from ember.models import Audit, RunMode

from .audit import EventAudit, PromptContext, RunAudit, ToolInventory
from .base import providers
from .comparison import CodexComparison
from .records import RunRecord

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
    "multi_agent_v2",
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
    "worktrees",
    "workspace_dependencies",
    "memories",
)
RUN_TIMEOUT: int = 1_800
PREFLIGHT_TIMEOUT: int = 90
EXEC_ARGS: tuple[str, ...] = ("exec", "--strict-config", "--skip-git-repo-check", "--json")


class UserConfig(BaseModel):
    """Read only MCP server names; never save credentials or full user configuration."""

    model_config = ConfigDict(extra="ignore")
    mcp_servers: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"UserConfig(servers={list(self.mcp_servers)})"


@providers
@dataclass(frozen=True)
class Codex:
    """Use the existing authenticated CLI and preserve its security controls."""

    name: ClassVar[str] = "codex"
    default_model: ClassVar[str] = "gpt-6-astra"
    modes: ClassVar[tuple[RunMode, ...]] = ("strict", "comparison")
    deviations: ClassVar[tuple[str, ...]] = (
        "User and managed hooks remain enabled; the authors disabled hooks.",
        "CLI versions and injected runtime instructions may differ.",
        "The authors' evaluator is unavailable.",
        "Approval policy and sandbox mode are inherited; the authors used never/read-only.",
        "Additional isolation settings disable memories, multi_agent_v2, and project document loading.",
    )

    sources: Source
    executable: str = "codex"
    model: str = "gpt-6-astra"

    @property
    def comparison(self) -> CodexComparison:
        """Select the fixed internal comparison policy without another public runner."""
        return CodexComparison(sources=self.sources, name=self.name, model=self.model, executable=self.executable)

    def configuration(self) -> list[str]:
        home: Path = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
        path: Path = home / "config.toml"
        config: UserConfig = UserConfig.model_validate(
            tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
        if any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in config.mcp_servers):
            raise ValueError("MCP server names must use letters, digits, underscores or hyphens for safe CLI overrides")
        values: dict[str, JsonValue] = {
            "model": self.model,
            "model_reasoning_effort": "xhigh",
            "model_reasoning_summary": "detailed",
            "model_context_window": 872000,
            "model_auto_compact_token_limit": 1000000,
            "model_instructions_file": str(self.sources.instructions_path.resolve()),
            "web_search": "disabled",
            "agents.enabled": False,
            "project_doc_max_bytes": 0,
        }
        values.update({f"features.{name}": False for name in DISABLED})
        values.update({f"mcp_servers.{name}.enabled": False for name in config.mcp_servers})
        return [argument for key, value in values.items() for argument in ("-c", f"{key}={json.dumps(value)}")]

    def prepare(self, *, output: Path, prompt: bytes, mode: RunMode = "strict") -> RunRecord:
        if mode == "comparison" and mode in self.modes:
            return self.comparison.prepare(output=output, prompt=prompt)
        if mode not in self.modes:
            raise ValueError("Codex supports strict mode only")
        config: list[str] = self.configuration()
        command: list[str] = [self.executable, *EXEC_ARGS, *config, "-o", str((output / "answer.md").resolve()), "-"]
        version: subprocess.CompletedProcess[str] = subprocess.run(
            [self.executable, "--version"], capture_output=True, encoding="utf-8", check=True, timeout=PREFLIGHT_TIMEOUT
        )
        probe: subprocess.CompletedProcess[str] = self.probe(output=output, config=config, prompt=prompt)
        errors: list[str] = []
        empty: bool = False
        context_matches: bool = False
        try:
            empty = probe.returncode == 0 and ToolInventory()(probe.stdout)
            context_matches = probe.returncode == 0 and PromptContext()(
                content=probe.stdout,
                instructions=self.sources.instructions_path.read_text(encoding="utf-8").rstrip("\n"),
                prompt=prompt.decode("utf-8"),
            )
        except ValueError as error:
            errors.append(str(error))
        if not empty:
            errors.append("Preflight did not establish an empty model tool inventory; no model request was made.")
        if not context_matches:
            errors.append("Extra instructions or changed prompt detected; no model request was made.")
        return RunRecord(
            status="prepared" if empty and context_matches else "blocked",
            provider=self.name,
            model=self.model,
            deviations=list(self.deviations),
            cli_version=version.stdout.strip(),
            command=command,
            config=config,
            tool_inventory_empty=empty,
            prompt_context_matches=context_matches,
            preflight_returncode=probe.returncode,
            preflight_error=probe.stderr,
            errors=errors,
        )

    def probe(self, *, output: Path, config: list[str], prompt: bytes) -> subprocess.CompletedProcess[str]:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [self.executable, "debug", "prompt-input", *config, prompt.decode("utf-8")],
            cwd=output,
            capture_output=True,
            encoding="utf-8",
            check=False,
            timeout=PREFLIGHT_TIMEOUT,
        )
        (output / "inventory.json").write_text(result.stdout, encoding="utf-8")
        (output / "preflight.log").write_text(result.stderr, encoding="utf-8")
        return result

    def execute(self, *, output: Path, record: RunRecord, prompt: bytes) -> RunRecord:
        if record.mode == "comparison" and record.mode in self.modes:
            return self.comparison.execute(output=output, record=record, prompt=prompt)
        if record.mode != "strict":
            raise ValueError("Codex supports strict mode only")
        if record.status != "prepared" or not record.tool_inventory_empty or not record.prompt_context_matches:
            raise ValueError("Model execution requires a successful no-tools preflight")
        started: float = time.monotonic()
        code: int | None = None
        errors: list[str] = []
        environment: dict[str, str] = {**os.environ, "CODEX_ROLLOUT_TRACE_ROOT": str((output / "trace").resolve())}
        with (output / "events.jsonl").open("wb") as events, (output / "stderr.log").open("wb") as stderr:
            try:
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
            except subprocess.TimeoutExpired:
                errors.append("Model request timed out; partial output is not accepted")
        answer_path: Path = output / "answer.md"
        answer: str = answer_path.read_text(encoding="utf-8") if answer_path.exists() else ""
        audit: Audit = EventAudit()(
            content=(output / "events.jsonl").read_text(encoding="utf-8"),
            answer=answer,
            inventory_empty=record.tool_inventory_empty,
            context_matches=record.prompt_context_matches,
        )
        return RunRecord(
            **record.model_dump(exclude={"status", "wall_seconds", "returncode", "audit", "errors"}),
            status="completed" if code == 0 and audit.passed else "failed",
            wall_seconds=time.monotonic() - started,
            returncode=code,
            audit=audit,
            errors=errors + audit.errors,
        )

    def audit(self, *, root: Path, mode: RunMode = "strict") -> Audit:
        if mode == "comparison" and mode in self.modes:
            return self.comparison.audit(root=root)
        if mode not in self.modes:
            raise ValueError("Codex supports strict mode only")
        return RunAudit(sources=self.sources)(root=root)
