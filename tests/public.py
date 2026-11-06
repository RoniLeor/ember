"""Public exports preserve component identities alongside typed module APIs."""

import importlib

import pytest

import ember
import ember.playground
import ember.providers


@pytest.mark.parametrize(
    ("name", "module", "attribute"),
    [
        ("experiment", "playground.games", "experiments"),
        ("provider", "providers", "providers"),
        ("Game", "playground.games", "Game"),
        ("Asterix", "playground.asterix.game", "Asterix"),
        ("Breakout", "playground.breakout.game", "Breakout"),
        ("Provider", "providers", "Provider"),
        ("Runner", "runner", "Runner"),
        ("RunRequest", "runner", "RunRequest"),
        ("RunRecord", "providers.records", "RunRecord"),
        ("Evaluator", "playground.eval", "Evaluator"),
        ("EvaluationRequest", "playground.eval", "EvaluationRequest"),
        ("Player", "playground.eval", "Player"),
        ("PlayRequest", "playground.eval", "PlayRequest"),
        ("Replay", "playground.action", "Replay"),
        ("ReplayRequest", "playground.action", "ReplayRequest"),
        ("Module", "module", "Module"),
        ("Context", "module", "Context"),
        ("Sources", "playground.breakout.base", "Sources"),
        ("Source", "base", "Source"),
        ("Settings", "playground.models", "Settings"),
        ("Device", "playground.models", "Device"),
        ("RunMode", "models", "RunMode"),
        ("Audit", "models", "Audit"),
        ("Evaluation", "playground.models", "Evaluation"),
        ("Recording", "playground.models", "Recording"),
    ],
)
def test_exports_are_original_components(name: str, module: str, attribute: str) -> None:
    assert name in ember.__all__
    assert getattr(ember, name) is getattr(importlib.import_module(f"ember.{module}"), attribute)


def test_public_decorators_do_not_shadow_submodules() -> None:
    assert ember.playground is importlib.import_module("ember.playground")
    assert ember.providers is importlib.import_module("ember.providers")
    assert len(ember.__all__) == len(set(ember.__all__))
