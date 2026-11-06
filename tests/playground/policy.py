"""Shared MinAtar sampling distribution and invalid-logit boundaries."""

import numpy as np
import pytest

from ember.playground.policy import Array, sample


def test_sampling_distribution_and_invalid_logits() -> None:
    rng: np.random.Generator = np.random.default_rng(seed=42)
    logits: Array = np.array([0, 1, -2, 2, 0, -1], dtype=np.float32)
    actions: list[int] = [sample(logits=logits, rng=rng) for _ in range(6000)]
    expected: Array = np.exp(logits) / np.exp(logits).sum()
    np.testing.assert_allclose(np.bincount(actions, minlength=6) / len(actions), expected, atol=0.02)
    assert sample(logits=np.array([1e30, 0, 0, 0, 0, 0], dtype=np.float32), rng=rng) == 0
    with pytest.raises(ValueError):
        sample(logits=np.zeros(5, dtype=np.float32), rng=rng)
    with pytest.raises(ValueError):
        sample(logits=np.full(6, np.nan, dtype=np.float32), rng=rng)
