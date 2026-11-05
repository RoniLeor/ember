"""Provider-neutral records for experiment execution."""

from typing import Literal

from pydantic import Field

from ember.models import Audit, Record, RunMode


class RunRecord(Record):
    """Execution outcome. A blocked preflight is not a model response."""

    status: Literal["prepared", "blocked", "failed", "completed"]
    provider: str = "codex"
    mode: RunMode = "strict"
    model: str
    cli_version: str = "unknown"
    source_commit: str = ""
    prompt_sha256: str = ""
    command: list[str] = Field(default_factory=list)
    config: list[str] = Field(default_factory=list)
    tool_inventory_empty: bool = False
    prompt_context_matches: bool = False
    preflight_returncode: int | None = None
    preflight_error: str = ""
    wall_seconds: float = 0
    returncode: int | None = None
    audit: Audit | None = None
    errors: list[str] = Field(default_factory=list)
    exact_author_reproduction: Literal[False] = False
    deviations: list[str] = Field(default_factory=list)
