"""Public exports preserve component identities alongside typed module APIs."""

import importlib

import pytest

import ember
import ember.playground
import ember.providers


@pytest.mark.parametrize(
    ("name", "module", "attribute"),
    [
        ("experiment", "playground.breakout.games", "experiments"),
        ("provider", "providers", "providers"),
        ("Game", "playground.breakout.games", "Game"),
        ("Breakout", "playground.breakout.game", "Breakout"),
        ("Provider", "providers", "Provider"),
        ("Runner", "runner", "Runner"),
        ("RunRequest", "runner", "RunRequest"),
        ("RunRecord", "providers.records", "RunRecord"),
        ("Evaluator", "playground.breakout.eval", "Evaluator"),
        ("EvaluationRequest", "playground.breakout.eval", "EvaluationRequest"),
        ("Player", "playground.breakout.eval", "Player"),
        ("PlayRequest", "playground.breakout.eval", "PlayRequest"),
        ("Replay", "playground.breakout.action", "Replay"),
        ("ReplayRequest", "playground.breakout.action", "ReplayRequest"),
        ("Module", "module", "Module"),
        ("Context", "module", "Context"),
        ("Sources", "playground.breakout.base", "Sources"),
        ("Source", "base", "Source"),
        ("Settings", "playground.breakout.models", "Settings"),
        ("Device", "playground.breakout.models", "Device"),
        ("RunMode", "models", "RunMode"),
        ("Audit", "models", "Audit"),
        ("Evaluation", "playground.breakout.models", "Evaluation"),
        ("Recording", "playground.breakout.models", "Recording"),
    ],
)
def test_exports_are_original_components(name: str, module: str, attribute: str) -> None:
    assert name in ember.__all__
    assert getattr(ember, name) is getattr(importlib.import_module(f"ember.{module}"), attribute)


def test_public_decorators_do_not_shadow_submodules() -> None:
    assert ember.playground is importlib.import_module("ember.playground")
    assert ember.providers is importlib.import_module("ember.providers")
    assert len(ember.__all__) == len(set(ember.__all__))
