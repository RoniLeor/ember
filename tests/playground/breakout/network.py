"""Production parser boundaries, finite logits, sampling, and backend agreement."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ember.module import Module
from ember.playground.breakout.network import KEYS, SHAPES, NumpyPolicy, TorchPolicy, Weights, torch
from ember.playground.breakout.policy import Array


@pytest.fixture
def weights(tmp_path: Path) -> Weights:
    path: Path = tmp_path / "answer.md"
    path.write_text("```edits\nb3[1] = 2\nb3[3] = -1\n```", encoding="utf-8")
    return Weights.load(path)


def test_reply_loading_and_numpy(weights: Weights) -> None:
    assert weights.applied == 2 and weights.skipped == ()
    policy: NumpyPolicy = NumpyPolicy(weights=weights)
    np.testing.assert_array_equal(policy(np.zeros(600, dtype=np.float32)), [0, 2, 0, -1, 0, 0])


def test_numpy_module_dispatch_retains_weights(weights: Weights) -> None:
    policy: Module[Array, Array] = weights.policy("cpu")
    inputs: Array = np.zeros(600, dtype=np.float32)
    expected: Array = policy.forward(inputs)
    np.testing.assert_array_equal(policy(inputs), expected)
    np.testing.assert_array_equal(policy.run(inputs), expected)
    assert isinstance(policy, NumpyPolicy) and policy.weights is weights


def test_a_zero_network_gives_zero_logits(tmp_path: Path) -> None:
    path: Path = tmp_path / "answer.md"
    path.write_text("```edits\n```\n", encoding="utf-8")
    logits: Array = Weights.load(path).policy("cpu")(np.ones(600, dtype=np.float32))
    assert logits.dtype == np.float32 and logits.tolist() == [0.0] * 6


def test_weight_rows_are_sources_and_columns_are_targets(tmp_path: Path) -> None:
    path: Path = tmp_path / "answer.md"
    path.write_text("```edits\nW1[7, 0] = 20\nW2[0, 1] = 20\nW3[1, 2] = 5\n```\n", encoding="utf-8")
    inputs: Array = np.zeros(600, dtype=np.float32)
    inputs[7] = 1.0
    logits: Array = Weights.load(path).policy("cpu")(inputs)
    assert logits[2] == 5.0
    assert np.count_nonzero(logits) == 1


def test_the_reversed_w2_entry_does_not_connect_the_units(tmp_path: Path) -> None:
    path: Path = tmp_path / "answer.md"
    path.write_text("```edits\nW1[7, 0] = 20\nW2[1, 0] = 20\nW3[1, 2] = 5\n```\n", encoding="utf-8")
    inputs: Array = np.zeros(600, dtype=np.float32)
    inputs[7] = 1.0
    assert Weights.load(path).policy("cpu")(inputs)[2] == 0.0


def test_biases_add_at_each_layer(tmp_path: Path) -> None:
    path: Path = tmp_path / "answer.md"
    path.write_text("```edits\nb2[3] = 20\nW3[3, 4] = 2\nb3[4] = 1\n```\n", encoding="utf-8")
    assert Weights.load(path).policy("cpu")(np.zeros(600, dtype=np.float32))[4] == 3.0


def test_skipped_edits_and_incomplete_last_block(tmp_path: Path) -> None:
    path: Path = tmp_path / "reply.md"
    path.write_text("```edits\nb1[0] 2\nb3[5] = 3\n```", encoding="utf-8")
    parsed: Weights = Weights.load(path)
    assert len(parsed.skipped) == 1 and parsed.tensors["b3"][5] == 3
    path.write_text("```edits\nb3[5] = 3\n```\n```edits\nb3[0] = 2", encoding="utf-8")
    with pytest.raises(ValueError):
        Weights.load(path)


def test_archive_validation(tmp_path: Path) -> None:
    path: Path = tmp_path / "net.npz"
    arrays: dict[str, Array] = {KEYS[name]: np.zeros(shape, dtype=np.float32) for name, shape in SHAPES.items()}
    np.savez(path, allow_pickle=False, **arrays)
    assert Weights.load(path).tensors["W1"].shape == (600, 32)
    arrays[KEYS["b3"]] = np.zeros(5, dtype=np.float32)
    np.savez(path, allow_pickle=False, **arrays)
    with pytest.raises(ValueError, match="Invalid shape"):
        Weights.load(path)
    arrays[KEYS["b3"]] = np.full(6, np.inf, dtype=np.float32)
    np.savez(path, allow_pickle=False, **arrays)
    with pytest.raises(ValueError, match="non-finite"):
        Weights.load(path)
    np.savez(path, wrong=np.zeros(1))
    with pytest.raises(ValueError, match="six published"):
        Weights.load(path)


def test_optional_torch_and_no_silent_fallback(weights: Weights) -> None:
    with patch("ember.playground.breakout.network.torch", None):
        with pytest.raises(ValueError, match="uv sync"):
            weights.policy("mps")
        with pytest.raises(ValueError, match="uv sync"):
            TorchPolicy(weights=weights, device="cpu")
    if torch is not None:
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            pytest.raises(ValueError, match="unavailable"),
        ):
            weights.policy("cuda")
        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            pytest.raises(ValueError, match="unavailable"),
        ):
            weights.policy("mps")
        policy: TorchPolicy = TorchPolicy(weights=weights, device="cpu")
        assert "cpu" in repr(policy)
        np.testing.assert_allclose(
            policy(np.ones(600, dtype=np.float32)), weights.policy("cpu")(np.ones(600, dtype=np.float32))
        )
        inputs: Array = np.ones(600, dtype=np.float32)
        tensors: dict[str, object] = dict(policy.tensors)
        np.testing.assert_array_equal(policy.run(inputs), policy.forward(inputs))
        assert all(policy.tensors[name] is tensor for name, tensor in tensors.items())
        with patch("ember.playground.breakout.network.torch", None), pytest.raises(ValueError, match="unavailable"):
            policy(np.ones(600, dtype=np.float32))
