"""Validated network loading and interchangeable inference backends."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ember.module import Module

from .models import Device
from .policy import Array
from .utils import ParsedReply, read_reply

if TYPE_CHECKING:
    from torch import Tensor  # pyright: ignore[reportMissingImports]  # Optional GPU extra.

try:
    import torch  # pyright: ignore[reportMissingImports]  # Optional GPU extra.
except ImportError:
    torch = None  # type: ignore[assignment]  # Optional module; backend entry points check availability.

KEYS: dict[str, str] = {
    "W1": "actor/dense1/kernel",
    "b1": "actor/dense1/bias",
    "W2": "actor/dense2/kernel",
    "b2": "actor/dense2/bias",
    "W3": "actor/logits/kernel",
    "b3": "actor/logits/bias",
}
SHAPES: dict[str, tuple[int, ...]] = {
    "W1": (600, 32),
    "b1": (32,),
    "W2": (32, 32),
    "b2": (32,),
    "W3": (32, 6),
    "b3": (6,),
}


@dataclass(frozen=True)
class Weights:
    """Network arrays, validated before an inference backend receives them."""

    tensors: dict[str, Array]
    skipped: tuple[str, ...] = ()
    applied: int | None = None
    source_sha256: str = ""

    @classmethod
    def load(cls, path: Path) -> "Weights":
        tensors: dict[str, Array]
        skipped: tuple[str, ...] = ()
        applied: int | None = None
        content: bytes = path.read_bytes()
        if path.suffix == ".npz":
            with np.load(file=BytesIO(content), allow_pickle=False) as archive:
                if set(archive.files) != set(KEYS.values()):
                    raise ValueError("Network archive must contain exactly the six published tensor names")
                tensors = {name: np.asarray(archive[key], dtype=np.float32) for name, key in KEYS.items()}
        else:
            parsed: ParsedReply = read_reply(reply=content.decode("utf-8"))
            tensors = {name: value.astype(np.float32) for name, value in parsed.network.tensors().items()}
            skipped = tuple(parsed.skipped)
            applied = parsed.applied
        name: str
        tensor: Array
        for name, tensor in tensors.items():
            if tensor.shape != SHAPES[name] or not np.isfinite(tensor).all():
                raise ValueError(f"Invalid shape or non-finite values in {name}")
        return cls(tensors=tensors, skipped=skipped, applied=applied, source_sha256=sha256(content).hexdigest())

    def policy(self, device: Device) -> Module[Array, Array]:
        if device == "cpu":
            return NumpyPolicy(weights=self)
        if torch is None:
            raise ValueError("GPU support requires: uv sync --extra gpu")
        available: bool = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
        if not available:
            raise ValueError(f"Requested device {device!r} is unavailable; no CPU fallback was used")
        return TorchPolicy(weights=self, device=device)


@dataclass(frozen=True)
class NumpyPolicy(Module[Array, Array]):
    """Float32 sequential reference for the published float32 archive format."""

    weights: Weights

    def forward(self, inputs: Array) -> Array:
        w: dict[str, Array] = self.weights.tensors
        hidden: Array = np.tanh(inputs @ w["W1"] + w["b1"])
        hidden = np.tanh(hidden @ w["W2"] + w["b2"])
        return hidden @ w["W3"] + w["b3"]


class TorchPolicy(Module[Array, Array]):
    """Optional CUDA or Apple Metal inference; environment stepping stays on CPU."""

    def __init__(self, *, weights: Weights, device: str) -> None:
        """Copy immutable inference tensors to the selected device once."""
        if torch is None:
            raise ValueError("GPU support requires: uv sync --extra gpu")
        self.device: str = device
        self.tensors: dict[str, Tensor] = {
            name: torch.tensor(value, dtype=torch.float32, device=device) for name, value in weights.tensors.items()
        }

    def __repr__(self) -> str:
        return f"TorchPolicy(device={self.device!r})"

    def forward(self, inputs: Array) -> Array:
        if torch is None:
            raise ValueError("PyTorch is unavailable")
        w: dict[str, Tensor] = self.tensors
        x: Tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            hidden: Tensor = torch.tanh(x @ w["W1"] + w["b1"])
            hidden = torch.tanh(hidden @ w["W2"] + w["b2"])
            return (hidden @ w["W3"] + w["b3"]).cpu().numpy()
