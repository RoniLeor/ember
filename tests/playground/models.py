"""Settings reject invalid boundaries and retain explicit serialization contracts."""

import pytest
from pydantic import ValidationError

from ember.playground.models import Frame, Settings


@pytest.mark.parametrize(
    "field,value",
    [
        ("episodes", 0),
        ("seed", -1),
        ("seed", 2**32),
        ("max_steps", 0),
        ("max_steps", 10001),
        ("device", "gpu"),
        ("experiment", ""),
        ("typo", 1),
    ],
)
def test_invalid_settings(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: value})


def test_defaults_and_immutable_models() -> None:
    settings: Settings = Settings()
    assert settings.episodes == 1000 and settings.max_steps == 10000 and settings.device == "cpu"
    assert settings.experiment == "breakout"
    assert Settings.model_validate_json(settings.model_dump_json()) == settings
    with pytest.raises(ValidationError):
        settings.seed = 1  # ty: ignore[invalid-assignment]  # Exercise frozen-model validation at runtime.
    with pytest.raises(ValidationError):
        Frame(cells=[0], score=0, action=0)
    with pytest.raises(ValidationError):
        Frame(cells=[0] * 100, score=float("nan"), action=0)


def test_legacy_frames_remain_readable_without_lossless_layers() -> None:
    frame: Frame = Frame.model_validate({"cells": [0] * 100, "score": 0, "action": 0})
    assert frame.layers is None


@pytest.mark.parametrize("layers", [[0], [0] * 101, [-1] * 100])
def test_invalid_lossless_layers_are_rejected(layers: list[int]) -> None:
    with pytest.raises(ValidationError):
        Frame(cells=[0] * 100, layers=layers, score=0, action=0)
