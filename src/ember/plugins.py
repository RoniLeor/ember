"""Explicit imports for local experiment and provider registration."""

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Plugins:
    """Load named Python modules; never discover or reload user code implicitly."""

    modules: tuple[str, ...]

    def __call__(self) -> None:
        name: str
        for name in self.modules:
            if not all(part.isidentifier() for part in name.split(".")):
                raise ValueError(f"Invalid plugin module name: {name!r}")
            try:
                importlib.import_module(name)
            except Exception as error:
                raise ValueError(f"Plugin {name!r} failed: {error}") from error
