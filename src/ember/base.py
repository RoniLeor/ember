"""Verified prompt inputs and generic source provenance."""

from pathlib import Path
from typing import ClassVar, Protocol

from .models import Record


class SourceRecord(Record):
    """Exact downloaded bytes and their immutable upstream location."""

    commit: str
    hashes: dict[str, str]
    urls: dict[str, str]


class Source(Protocol):
    """Verify pinned inputs and expose the instructions used by providers."""

    @property
    def root(self) -> Path: ...

    commit: ClassVar[str]

    @property
    def instructions_path(self) -> Path: ...

    def verify_prompt(self) -> bytes: ...
