"""Every built-in game configures its shared resource and runtime boundaries."""

from pathlib import Path

import pytest

from ember.playground.games import Game, experiments


@pytest.mark.parametrize("name", ["asterix", "breakout"])
def test_builtin_source_configuration_is_complete(name: str) -> None:
    game: Game = experiments.get(name)
    source: object = game.sources(root=Path("resources"))
    attribute: str
    for attribute in (
        "commit",
        "repository",
        "directory",
        "assets",
        "prompt_hash",
        "instructions_hash",
        "answer_hash",
        "network_hash",
        "files",
    ):
        assert getattr(source, attribute), attribute
    assert getattr(source, "directory") == name
    assets: Path = getattr(source, "assets")
    assert assets.name == name
    assert game.template.is_file()
    assert game.network is experiments.get("breakout").network
