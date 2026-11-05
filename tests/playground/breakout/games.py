"""MinAtar registry isolation and registration errors."""

import pytest

from ember.playground.breakout import Breakout
from ember.playground.breakout.games import Registry


def test_registry_isolated_duplicate_and_unknown() -> None:
    registry: Registry = Registry()
    assert registry(Breakout) is Breakout
    assert isinstance(registry.get("breakout"), Breakout)
    assert not Registry().games
    with pytest.raises(ValueError, match="duplicate"):
        registry(Breakout)
    with pytest.raises(ValueError, match="Unknown"):
        registry.get("exp1.breakout")
