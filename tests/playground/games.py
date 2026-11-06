"""MinAtar registry isolation and registration errors."""

import pytest
from minatar import Environment

from ember.module import Context
from ember.playground.asterix import Asterix
from ember.playground.breakout import Breakout
from ember.playground.games import MinAtarGame, Registry


def test_registry_isolated_duplicate_and_unknown() -> None:
    registry: Registry = Registry()
    assert registry(Breakout) is Breakout
    assert isinstance(registry.get("breakout"), Breakout)
    assert not Registry().games
    with pytest.raises(ValueError, match="duplicate"):
        registry(Breakout)
    with pytest.raises(ValueError, match="Unknown"):
        registry.get("exp1.breakout")


def test_shared_adapter_requires_a_concrete_observation_mapping() -> None:
    with pytest.raises(TypeError, match="abstract.*encode"):
        type.__call__(MinAtarGame)


@pytest.mark.parametrize("game_type", [Asterix, Breakout])
def test_shared_adapter_dispatch_uses_the_concrete_environment(game_type: type[MinAtarGame]) -> None:
    context: Context[float] = Context(value=0.0)
    game: MinAtarGame = game_type(context=context)
    assert game.ctx is context
    environment: Environment = MinAtarGame.forward(game, 7)
    assert environment.game_name() == game.environment
    assert environment.sticky_action_prob == 0.0
    with context.override(1.0):
        assert MinAtarGame.__call__(game, seed=7).sticky_action_prob == 1.0
    assert MinAtarGame.__call__(game, 7).sticky_action_prob == 0.0
