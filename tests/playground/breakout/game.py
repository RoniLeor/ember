"""Registration isolation and the official Breakout observation contract."""

from operator import methodcaller

import numpy as np
import pytest
from minatar import Environment
from numpy.typing import NDArray

from ember.module import Context
from ember.playground.breakout import Breakout
from ember.playground.breakout.policy import Array


def test_observation_channels_and_invalid_shape() -> None:
    state: NDArray[np.bool_] = np.zeros((10, 10, 4), dtype=np.bool_)
    state[9, 5, 0] = True
    state[8, 2, 1] = True
    state[7, 1, 2] = True
    state[1, 4, 3] = True
    inputs: Array = Breakout().encode(state=state)
    assert inputs.dtype == np.float32
    assert np.flatnonzero(inputs).tolist() == [86, 430, 495, 570]
    assert inputs.sum() == 4
    with pytest.raises(ValueError, match="Expected"):
        Breakout().encode(state=np.zeros((10, 10, 3), dtype=np.bool_))


def test_environment_seed_and_sticky_state() -> None:
    left: Environment = Breakout()(seed=19)
    right: Environment = Breakout()(seed=19)
    assert left.sticky_action_prob == 0.1
    left.reset()
    right.reset()
    assert np.array_equal(left.state(), right.state())
    left.last_action = 3
    left.reset()
    assert left.last_action == 3
    left.sticky_action_prob = 1.0
    left.act(1)
    assert left.last_action == 3


def test_module_dispatch_and_shared_sticky_context() -> None:
    context: Context[float] = Context(value=0.1)
    game: Breakout = Breakout(context=context)
    assert game.ctx is context and game == Breakout()
    assert repr(game) == "Breakout()"
    legacy: Environment = game(seed=0)
    typed: Environment = game(0)
    dispatched: Environment = game.run(0)
    legacy.reset()
    typed.reset()
    dispatched.reset()
    np.testing.assert_array_equal(legacy.state(), typed.state())
    np.testing.assert_array_equal(legacy.state(), dispatched.state())
    with context.override(1.0):
        assert game.run(0).sticky_action_prob == 1.0
        assert Breakout().run(0).sticky_action_prob == 0.1
    assert game.run(0).sticky_action_prob == 0.1


def test_game_facade_rejects_missing_and_mixed_seed() -> None:
    game: Breakout = Breakout()
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__")(game)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", 0, seed=0)(game)
