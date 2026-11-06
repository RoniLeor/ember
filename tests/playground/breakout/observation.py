"""Preserve the original Breakout observation behavior."""

import numpy as np
from minatar import Environment

from ember.playground.breakout.observation import to_inputs
from ember.playground.tensors import FloatArray


def input_index(row: int, column: int, channel: int) -> int:
    return (row * 10 + column) * 6 + channel


def test_real_reset_state_maps_to_the_task_layout() -> None:
    env: Environment = Environment(env_name="breakout")
    env.seed(seed=0)
    env.reset()
    inputs: FloatArray = to_inputs(state=env.state())
    grid: FloatArray = inputs.reshape(10, 10, 6)
    assert inputs.shape == (600,)
    assert inputs[input_index(row=9, column=4, channel=0)] == 1
    assert grid[:, :, 0].sum() == 1
    assert grid[:, :, 2].sum() == 30
    assert grid[1:4, :, 2].all()
    assert not grid[:, :, 1].any()
    assert not grid[:, :, 5].any()
    assert grid[:, :, 3].sum() == 1
    assert (grid[:, :, 3] == grid[:, :, 4]).all()
    assert grid[3, 0, 3] == 1 or grid[3, 9, 3] == 1


def test_trail_and_ball_use_separate_channels_after_a_step() -> None:
    env: Environment = Environment(env_name="breakout", sticky_action_prob=0.0)
    env.seed(seed=0)
    env.reset()
    env.act(0)
    grid: FloatArray = to_inputs(state=env.state()).reshape(10, 10, 6)
    ball_row: int = int(np.argwhere(grid[:, :, 3])[0][0])
    trail_row: int = int(np.argwhere(grid[:, :, 4])[0][0])
    assert ball_row == 4
    assert trail_row == 3
