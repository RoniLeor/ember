"""Game adapters and an instance-based registration decorator."""

from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, TypeVar, overload

import numpy as np
from minatar import Environment
from numpy.typing import NDArray

from ember.module import Context, Module

from .base import GameSourceFactory
from .network import Weights
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


STICKY_ACTION_PROBABILITY: float = 0.1


@dataclass(frozen=True)
class MinAtarGame(Module[int, Environment]):
    """Create seeded official environments with scoped sticky-action settings."""

    environment: ClassVar[str]
    network: ClassVar[Loader] = Weights
    packages: ClassVar[tuple[str, ...]] = ()

    context: InitVar[Context[float] | None] = field(default=None, kw_only=True)
    ctx: Context[float] = field(init=False, repr=False, compare=False)

    def __post_init__(self, context: Context[float] | None) -> None:
        object.__setattr__(self, "ctx", Context(value=STICKY_ACTION_PROBABILITY) if context is None else context)

    @overload
    def __call__(self, inputs: int) -> Environment: ...

    @overload
    def __call__(self, *, seed: int) -> Environment: ...

    def __call__(self, inputs: int | None = None, *, seed: int | None = None) -> Environment:
        if inputs is None:
            if seed is None:
                raise TypeError("seed is required")
            inputs = seed
        elif seed is not None:
            raise TypeError("Use a seed input or keyword, not both")
        return self.run(inputs)

    def forward(self, inputs: int) -> Environment:
        probability: float = self.ctx.value
        environment: Environment = Environment(env_name=self.environment, sticky_action_prob=probability)
        environment.seed(seed=inputs)
        return environment

    @abstractmethod
    def encode(self, *, state: NDArray[np.bool_]) -> Array:
        """Map a game's native channels to the task's network input."""
        ...
