"""One callable lifecycle shared by every registered model provider."""

import subprocess
from dataclasses import InitVar, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol, overload

from .models import RunMode
from .module import Context, Module
from .providers import Provider
from .providers.records import RunRecord


class Writer(Protocol):
    """A callable experiment runner, independent of its CLI provider."""

    def __call__(self, *, output: Path, execute: bool) -> RunRecord: ...


@dataclass(frozen=True)
class RunRequest:
    """Trusted paths and execution intent for one model request."""

    output: Path
    execute: bool


@dataclass(frozen=True)
class Runner(Module[RunRequest, RunRecord]):
    """Own prompt validation, artifacts, strict gates and failure recording."""

    provider: Provider
    mode: RunMode = "strict"
    context: InitVar[Context[RunMode] | None] = field(default=None, kw_only=True)
    ctx: Context[RunMode] = field(init=False, repr=False, compare=False)

    def __post_init__(self, context: Context[RunMode] | None) -> None:
        object.__setattr__(self, "ctx", Context(value=self.mode) if context is None else context)

    @overload
    def __call__(self, inputs: RunRequest) -> RunRecord: ...

    @overload
    def __call__(self, *, output: Path, execute: bool) -> RunRecord: ...

    def __call__(
        self, inputs: RunRequest | None = None, *, output: Path | None = None, execute: bool | None = None
    ) -> RunRecord:
        if inputs is None:
            if output is None or execute is None:
                raise TypeError("output and execute are required")
            inputs = RunRequest(output=output, execute=execute)
        elif output is not None or execute is not None:
            raise TypeError("Use a run request or keyword arguments, not both")
        return self.run(inputs)

    def forward(self, inputs: RunRequest) -> RunRecord:
        mode: RunMode = self.ctx.value
        output: Path = inputs.output
        if mode not in self.provider.modes:
            raise ValueError(f"Provider {self.provider.name} does not support mode {mode}")
        prompt: bytes = self.provider.sources.verify_prompt()
        source_commit: str = self.provider.sources.commit
        prompt_digest: str = sha256(prompt).hexdigest()
        output.mkdir(parents=True, exist_ok=False)
        (output / "prompt.txt").write_bytes(prompt)
        record: RunRecord = RunRecord(
            status="failed",
            provider=self.provider.name,
            model=self.provider.model,
            source_commit=source_commit,
            prompt_sha256=prompt_digest,
            mode=mode,
            deviations=list(self.provider.deviations),
        )
        try:
            prepared: RunRecord = self.provider.prepare(output=output, prompt=prompt, mode=mode)
            if prepared.mode != mode or prepared.provider != self.provider.name:
                raise ValueError("Prepared record differs from the requested provider or mode")
            record = prepared.model_copy(update={"source_commit": source_commit, "prompt_sha256": prompt_digest})
            self.save(output=output, record=record)
            if inputs.execute and record.status == "prepared":
                if mode == "strict" and (not record.tool_inventory_empty or not record.prompt_context_matches):
                    raise ValueError("Model execution requires a successful no-tools and exact-context preflight")
                record = self.provider.execute(output=output, record=record, prompt=prompt)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            record = record.model_copy(update={"status": "failed", "errors": [*record.errors, str(error)]})
        record = record.model_copy(update={"source_commit": source_commit, "prompt_sha256": prompt_digest})
        self.save(output=output, record=record)
        return record

    @staticmethod
    def save(*, output: Path, record: RunRecord) -> None:
        (output / "run.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
