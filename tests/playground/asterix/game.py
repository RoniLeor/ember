"""Asterix observation mapping, stochastic seeding, and official game transitions."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from minatar import Environment
from minatar.environments import asterix
from numpy.typing import NDArray

from ember import Asterix, Context, experiment
from ember.playground.asterix.observation import to_inputs
from ember.playground.policy import Array


def test_distinct_source_channels_map_to_literal_task_indices() -> None:
    state: NDArray[np.bool_] = np.zeros(shape=(10, 10, 4), dtype=np.bool_)
    state[1, 2, 0] = True
    state[3, 4, 1] = True
    state[5, 6, 2] = True
    state[7, 8, 3] = True
    original: NDArray[np.bool_] = state.copy()
    inputs: Array = Asterix().encode(state=state)
    assert inputs.shape == (600,) and inputs.dtype == np.float32
    assert np.flatnonzero(inputs).tolist() == [72, 205, 340, 470]
    assert np.flatnonzero(to_inputs(state=state)).tolist() == [72, 205, 340, 470]
    assert not inputs.reshape(10, 10, 6)[:, :, [3, 5]].any()
    np.testing.assert_array_equal(state, original)


def test_empty_and_overlapping_observations_preserve_all_channels() -> None:
    state: NDArray[np.bool_] = np.zeros(shape=(10, 10, 4), dtype=np.bool_)
    assert not Asterix().encode(state=state).any()
    state[0, 0, :] = True
    state[9, 9, :] = True
    assert np.flatnonzero(Asterix().encode(state=state)).tolist() == [0, 1, 2, 4, 594, 595, 596, 598]


@pytest.mark.parametrize("shape", [(10, 10, 3), (10, 10, 5), (9, 10, 4), (600,), (0, 10, 4)])
def test_invalid_observation_shapes_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="Expected Asterix state"):
        Asterix().encode(state=np.zeros(shape=shape, dtype=np.bool_))


def test_seeded_transitions_and_scoped_sticky_actions() -> None:
    context: Context[float] = Context(value=0.1)
    game: Asterix = Asterix(context=context)
    assert game.ctx is context and repr(game) == "Asterix()"
    assert isinstance(experiment.get("asterix"), Asterix)
    left: Environment = game(seed=19)
    right: Environment = game(seed=19)
    other: Environment = game(seed=20)
    left.reset()
    right.reset()
    other.reset()
    diverged: bool = False
    step: int
    for step in range(250):
        action: int = step % 6
        assert left.act(action) == right.act(action)
        other.act(action)
        np.testing.assert_array_equal(left.state(), right.state())
        diverged = diverged or not np.array_equal(left.state(), other.state())
        if left.env.terminal:
            left.reset()
            right.reset()
        if other.env.terminal:
            other.reset()
    assert diverged and left.env.ramping
    left.last_action = 3
    left.reset()
    assert left.last_action == 3
    left.sticky_action_prob = 1.0
    left.act(1)
    assert left.last_action == 3
    with context.override(0.0):
        assert game(seed=0).sticky_action_prob == 0.0
    assert game(seed=0).sticky_action_prob == 0.1


@pytest.mark.parametrize("gold", [True, False])
def test_official_treasure_reward_and_enemy_death(gold: bool) -> None:
    game: Asterix = Asterix(context=Context(value=0.0))
    environment: Environment = game(seed=0)
    environment.reset()
    environment.env.entities[4] = [6, 5, True, gold]
    reward: float
    terminal: bool
    reward, terminal = environment.act(3)
    assert reward == (1.0 if gold else 0.0)
    assert terminal is not gold


def test_prompt_embeds_the_installed_official_rules() -> None:
    prompt: str = Path(__file__).resolve().parents[3].joinpath("resources/asterix/prompt.txt").read_text()
    assert inspect.getsource(asterix).strip() in prompt
