"""Map the MinAtar Breakout state to the 600 network inputs from the task."""

import numpy as np
from numpy.typing import NDArray

from ember.playground.tensors import FloatArray

PADDLE_SOURCE: int = 0
BALL_SOURCE: int = 1
TRAIL_SOURCE: int = 2
BRICK_SOURCE: int = 3

PLAYER_INPUT: int = 0
REWARD_INPUT: int = 2
PROJECTILE_INPUT: int = 3
TRAIL_INPUT: int = 4


def to_inputs(state: NDArray[np.bool_]) -> FloatArray:
    """Return inputs where index = (row * 10 + column) * 6 + channel."""
    grid: FloatArray = np.zeros(shape=(10, 10, 6))
    grid[:, :, PLAYER_INPUT] = state[:, :, PADDLE_SOURCE]
    grid[:, :, REWARD_INPUT] = state[:, :, BRICK_SOURCE]
    grid[:, :, PROJECTILE_INPUT] = state[:, :, BALL_SOURCE]
    grid[:, :, TRAIL_INPUT] = state[:, :, TRAIL_SOURCE]
    return grid.ravel()
