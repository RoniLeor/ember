"""Inference contracts and MinAtar action sampling."""

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .models import ACTIONS, Device

Array = NDArray[np.float32]


class Policy(Protocol):
    """Map one observation to MinAtar action logits."""

    def __call__(self, inputs: Array) -> Array: ...


class Network(Protocol):
    """Validated network provenance and device-specific inference."""

    @property
    def source_sha256(self) -> str: ...

    @property
    def applied(self) -> int | None: ...

    @property
    def skipped(self) -> tuple[str, ...]: ...

    def policy(self, device: Device) -> Policy: ...


class Loader(Protocol):
    """A network class loads text or archive bytes under its own contract."""

    def load(self, path: Path) -> Network: ...


def sample(*, logits: Array, rng: np.random.Generator) -> int:
    """Categorical softmax sampling; never select an action with argmax alone."""
    if logits.shape != (ACTIONS,) or not np.isfinite(logits).all():
        raise ValueError("Expected six finite action logits")
    values: NDArray[np.float64] = logits.astype(np.float64)
    probabilities: NDArray[np.float64] = np.exp(values - values.max())
    return int(rng.choice(a=ACTIONS, p=probabilities / probabilities.sum()))
