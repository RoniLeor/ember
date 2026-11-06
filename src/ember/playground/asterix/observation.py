"""Map the MinAtar Asterix state to the 600 network inputs from the task."""

import numpy as np
from numpy.typing import NDArray

from ember.playground.tensors import FloatArray

PLAYER_SOURCE: int = 0
ENEMY_SOURCE: int = 1
TRAIL_SOURCE: int = 2
GOLD_SOURCE: int = 3

PLAYER_INPUT: int = 0
REWARD_INPUT: int = 2
DANGER_INPUT: int = 1
TRAIL_INPUT: int = 4


def to_inputs(state: NDArray[np.bool_]) -> FloatArray:
    """Return inputs where index = (row * 10 + column) * 6 + channel."""
    grid: FloatArray = np.zeros(shape=(10, 10, 6))
    grid[:, :, PLAYER_INPUT] = state[:, :, PLAYER_SOURCE]
    grid[:, :, REWARD_INPUT] = state[:, :, GOLD_SOURCE]
    grid[:, :, DANGER_INPUT] = state[:, :, ENEMY_SOURCE]
    grid[:, :, TRAIL_INPUT] = state[:, :, TRAIL_SOURCE]
    return grid.ravel()
