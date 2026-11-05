"""Breakout rules, observation mapping, and experiment registration."""

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import ClassVar, overload

import numpy as np
from minatar import Environment
from numpy.typing import NDArray

from ember.module import Context, Module

from .base import GameSourceFactory, Sources
from .games import experiments
from .network import Weights
from .observation import to_inputs
from .policy import Array, Loader

STICKY_ACTION_PROBABILITY: float = 0.1
OBSERVATION_SHAPE: tuple[int, int, int] = (10, 10, 4)


@experiments
@dataclass(frozen=True)
class Breakout(Module[int, Environment]):
    """Official MinAtar 1.0.15 rules and the authors' six-channel input."""

    name: ClassVar[str] = "breakout"
    sources: ClassVar[GameSourceFactory] = Sources
    network: ClassVar[Loader] = Weights
    template: ClassVar[Path] = Path(__file__).with_name("viewer.html")
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
        environment: Environment = Environment(env_name="breakout", sticky_action_prob=probability)
        environment.seed(seed=inputs)
        return environment

    def encode(self, *, state: NDArray[np.bool_]) -> Array:
        if state.shape != OBSERVATION_SHAPE:
            raise ValueError(f"Expected Breakout state (10, 10, 4), got {state.shape}")
        return to_inputs(state=state).astype(np.float32)
