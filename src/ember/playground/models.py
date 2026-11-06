"""Validated gameplay settings, constants, and saved episode results."""

from typing import Annotated, Literal

from pydantic import Field

from ember.models import Record

from .config import DEFAULT_EXPERIMENT

Device = Literal["cpu", "cuda", "mps"]
ACTIONS: int = 6
CELLS: int = 100
STEP_LIMIT: int = 10_000
EPISODES: int = 1_000


class Settings(Record):
    """Local evaluation settings; these never change the research prompt."""

    experiment: str = Field(default=DEFAULT_EXPERIMENT, min_length=1)
    device: Device = "cpu"
    episodes: int = Field(default=EPISODES, ge=1)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    max_steps: int = Field(default=STEP_LIMIT, ge=1, le=STEP_LIMIT)


class Frame(Record):
    """An observed frame. Action is sampled, before sticky replacement; initial action is a placeholder."""

    cells: list[int] = Field(min_length=CELLS, max_length=CELLS)
    layers: list[Annotated[int, Field(ge=0)]] | None = Field(default=None, min_length=CELLS, max_length=CELLS)
    score: float
    action: int = Field(ge=0, le=ACTIONS - 1)


class Episode(Record):
    """One episode, including the reason it stopped."""

    score: float
    steps: int = Field(ge=1)
    capped: bool
    frames: list[Frame] = Field(default_factory=list)


class Evaluation(Record):
    """A reconstructed diagnostic, never an author-certified score."""

    kind: Literal["reconstructed"] = "reconstructed"
    evaluator_matches_authors: Literal[False] = False
    settings: Settings
    mean: float
    standard_error: float
    mean_length: float
    capped_fraction: float
    scores: list[float]
    network_sha256: str
    versions: dict[str, str]
    applied: int | None
    skipped: list[str]


class Recording(Episode):
    """One recorded episode with the exact input identity and execution settings."""

    settings: Settings
    network_sha256: str
    versions: dict[str, str]
    applied: int | None
    skipped: list[str]
    replay: str
