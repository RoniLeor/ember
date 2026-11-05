"""Shared immutable records, run modes, and audit results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RunMode = Literal["strict", "comparison"]


class Record(BaseModel):
    """Immutable records reject misspelled fields and non-finite numbers."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Audit(Record):
    """Evidence for completion and tool use, kept separate from availability."""

    passed: bool
    mode: RunMode = "strict"
    tool_inventory_empty: bool
    prompt_context_matches: bool
    turn_completed: bool
    answer_matches: bool
    errors: list[str]
    usage: dict[str, int] = Field(default_factory=dict)
