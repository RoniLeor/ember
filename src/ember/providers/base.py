"""Typed provider contract and class registration decorator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, TypeVar

from ember.base import Source
from ember.models import Audit, RunMode

from .records import RunRecord


class Provider(Protocol):
    """Keep CLI commands and evidence interpretation within each provider."""

    name: ClassVar[str]
    default_model: ClassVar[str]
    deviations: ClassVar[tuple[str, ...]]
    modes: ClassVar[tuple[RunMode, ...]]

    @property
    def sources(self) -> Source: ...

    @property
    def model(self) -> str: ...

    def prepare(self, *, output: Path, prompt: bytes, mode: RunMode = "strict") -> RunRecord: ...

    def execute(self, *, output: Path, record: RunRecord, prompt: bytes) -> RunRecord: ...

    def audit(self, *, root: Path, mode: RunMode = "strict") -> Audit: ...


class Factory(Protocol):
    """A provider class accepts the same two construction arguments."""

    @property
    def name(self) -> str: ...

    @property
    def default_model(self) -> str: ...

    def __call__(self, *, sources: Source, model: str) -> Provider: ...


ProviderType = TypeVar("ProviderType", bound=Factory)


@dataclass
class Registry:
    """Register provider classes without wrapping their behavior."""

    entries: dict[str, Factory] = field(default_factory=dict)

    def __call__(self, provider: ProviderType) -> ProviderType:
        if not provider.name or provider.name in self.entries:
            raise ValueError(f"Empty or duplicate provider name: {provider.name!r}")
        self.entries[provider.name] = provider
        return provider

    def get(self, *, name: str, sources: Source, model: str | None = None) -> Provider:
        if name not in self.entries:
            raise ValueError(f"Unknown provider: {name}. Available: {', '.join(self.entries)}")
        provider: Factory = self.entries[name]
        selected: str = provider.default_model if model is None else model
        if (
            not selected
            or selected.startswith("-")
            or any(character.isspace() or not character.isprintable() for character in selected)
        ):
            raise ValueError(
                "Model must be a nonempty CLI identifier without whitespace, control characters or leading '-'"
            )
        return provider(sources=sources, model=selected)


providers: Registry = Registry()
