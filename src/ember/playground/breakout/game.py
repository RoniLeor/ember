"""Breakout rules, observation mapping, and experiment registration."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ember.playground.base import GameSourceFactory
from ember.playground.games import MinAtarGame, experiments
from ember.playground.policy import Array

from .base import Sources
from .observation import to_inputs

OBSERVATION_SHAPE: tuple[int, int, int] = (10, 10, 4)


@experiments
@dataclass(frozen=True)
class Breakout(MinAtarGame):
    """Official MinAtar 1.0.15 rules and the authors' six-channel input."""

    name: ClassVar[str] = "breakout"
    environment: ClassVar[str] = "breakout"
    sources: ClassVar[GameSourceFactory] = Sources
    template: ClassVar[Path] = Path(__file__).with_name("viewer.html")

    def encode(self, *, state: NDArray[np.bool_]) -> Array:
        if state.shape != OBSERVATION_SHAPE:
            raise ValueError(f"Expected Breakout state (10, 10, 4), got {state.shape}")
        return to_inputs(state=state).astype(np.float32)
