"""Game adapters and an instance-based registration decorator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, TypeVar

import numpy as np
from minatar import Environment
from numpy.typing import NDArray

from .base import GameSourceFactory
from .policy import Array, Loader


class Game(Protocol):
    """Boundary for adding an environment and its observation mapping."""

    name: ClassVar[str]
    sources: ClassVar[GameSourceFactory]
    network: ClassVar[Loader]
    template: ClassVar[Path]
    packages: ClassVar[tuple[str, ...]]

    def __call__(self, *, seed: int) -> Environment: ...

    def encode(self, *, state: NDArray[np.bool_]) -> Array: ...


GameType = TypeVar("GameType", bound=type[Game])


@dataclass
class Registry:
    """Use an instance as a decorator; duplicate experiment names are errors."""

    games: dict[str, type[Game]] = field(default_factory=dict)

    def __call__(self, game: GameType) -> GameType:
        if not game.name or game.name in self.games:
            raise ValueError(f"Empty or duplicate experiment name: {game.name!r}")
        self.games[game.name] = game
        return game

    def get(self, name: str) -> Game:
        try:
            return self.games[name]()
        except KeyError as error:
            raise ValueError(f"Unknown experiment: {name}. Available: {', '.join(self.games)}") from error


experiments: Registry = Registry()
