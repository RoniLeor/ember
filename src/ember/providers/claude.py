"""Claude comparison execution with explicit limits on prompt and tool evidence."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from ember.base import Source
from ember.models import Audit, RunMode

from .base import providers
from .events import Transcript
from .records import RunRecord

PREFLIGHT_TIMEOUT: int = 90
RUN_TIMEOUT: int = 1_800
UNVERIFIED: str = (
    "Claude CLI provides no offline rendered prompt/tool inventory. "
    "Exact context and unavailable tools cannot be verified before execution; no model request was made."
)


@providers
@dataclass(frozen=True)
class Claude:
    """Run labelled comparisons without substituting flags for independent evidence."""

    name: ClassVar[str] = "claude"
    default_model: ClassVar[str] = "claude-sonnet-5-5"
    modes: ClassVar[tuple[RunMode, ...]] = ("strict", "comparison")
    deviations: ClassVar[tuple[str, ...]] = (
        "Claude is a different model family from the authors' Astra run.",
        "Claude uses high effort; the authors' Codex run used xhigh and specific context limits.",
        "User and managed hooks remain enabled and can add context.",
        "CLAUDE.md files and auto-memory can be injected as additional context.",
        "CLI flags declare disabled tools; no rendered offline inventory establishes this.",
        "The authors' evaluator is unavailable.",
    )

    sources: Source
    model: str = "claude-sonnet-5-5"
    executable: str = "claude"

    def configuration(self) -> list[str]:
        instructions: str = self.sources.instructions_path.read_text(encoding="utf-8").rstrip("\n")
        return [
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            f"--model={self.model}",
            "--effort",
            "high",
            "--tools",
            "",
            "--disallowedTools",
            "mcp__*",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--disable-slash-commands",
            "--permission-prompts",
            "none",
            "--system-prompt",
            instructions,
        ]

    def prepare(self, *, output: Path, prompt: bytes, mode: RunMode = "strict") -> RunRecord:
        if mode not in self.modes:
            raise ValueError(f"Unsupported Claude mode: {mode}")
        config: list[str] = self.configuration()
        version: subprocess.CompletedProcess[str] = subprocess.run(
            [self.executable, "--version"], capture_output=True, encoding="utf-8", check=True, timeout=PREFLIGHT_TIMEOUT
        )
        return RunRecord(
            status="prepared" if mode == "comparison" else "blocked",
            provider=self.name,
            mode=mode,
            model=self.model,
            cli_version=version.stdout.strip(),
            command=[self.executable, *config],
            config=config,
            deviations=list(self.deviations),
            errors=[UNVERIFIED] if mode == "strict" else [],
        )

    def execute(self, *, output: Path, record: RunRecord, prompt: bytes) -> RunRecord:
        if record.mode != "comparison":
            raise ValueError(UNVERIFIED)
        command: list[str] = [self.executable, *self.configuration()]
        if record.status != "prepared" or record.provider != self.name or record.model != self.model:
            raise ValueError("Claude comparison requires a matching prepared record")
        if record.command != command or prompt != self.sources.verify_prompt():
            raise ValueError("Claude command or prompt differs from the prepared comparison")
        started: float = time.monotonic()
        code: int | None = None
        errors: list[str] = []
        with (output / "events.jsonl").open("wb") as events, (output / "stderr.log").open("wb") as stderr:
            try:
                process: subprocess.CompletedProcess[bytes] = subprocess.run(
                    command, input=prompt, cwd=output, stdout=events, stderr=stderr, check=False, timeout=RUN_TIMEOUT
                )
                code = process.returncode
                if code != 0:
                    errors.append(f"Claude exited with code {code}")
            except (OSError, subprocess.SubprocessError) as error:
                errors.append(f"Claude request failed: {error}")
        record = record.model_copy(
            update={
                "status": "failed",
                "returncode": code,
                "wall_seconds": time.monotonic() - started,
                "errors": errors,
            }
        )
        return self.finish(output=output, record=record)

    @staticmethod
    def finish(*, output: Path, record: RunRecord) -> RunRecord:
        try:
            transcript: Transcript = Transcript.read((output / "events.jsonl").read_text(encoding="utf-8"))
            audit: Audit = transcript(answer=transcript.answer)
        except (OSError, ValueError) as error:
            return record.model_copy(update={"errors": [*record.errors, str(error)]})
        completed: bool = record.returncode == 0 and audit.passed and not record.errors
        if completed:
            (output / "answer.md").write_bytes(transcript.answer.encode("utf-8"))
        return record.model_copy(
            update={
                "status": "completed" if completed else "failed",
                "audit": audit,
                "errors": [*record.errors, *audit.errors],
            }
        )

    def audit(self, *, root: Path, mode: RunMode = "strict") -> Audit:
        if mode != "comparison":
            raise ValueError("Claude strict audit is unavailable without independently verified prompt/tool evidence")
        record: RunRecord = RunRecord.model_validate_json((root / "run.json").read_text(encoding="utf-8"))
        if record.provider != self.name or record.mode != mode or record.returncode != 0:
            raise ValueError("Comparison audit requires a matching Claude record and successful process exit")
        if (root / "prompt.txt").read_bytes() != self.sources.verify_prompt():
            raise ValueError("Saved run prompt differs from the pinned research prompt")
        return Transcript.read((root / "events.jsonl").read_text(encoding="utf-8"))(
            answer=(root / "answer.md").read_bytes().decode("utf-8")
        )
