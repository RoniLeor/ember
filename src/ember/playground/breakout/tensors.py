"""Float64 tensor storage for the published Breakout edit format."""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Tensors:
    """Float64 edit state; runtime policies perform inference in float32."""

    w1: FloatArray
    b1: FloatArray
    w2: FloatArray
    b2: FloatArray
    w3: FloatArray
    b3: FloatArray

    @classmethod
    def zeros(cls) -> Self:
        return cls(
            w1=np.zeros(shape=(600, 32)),
            b1=np.zeros(shape=32),
            w2=np.zeros(shape=(32, 32)),
            b2=np.zeros(shape=32),
            w3=np.zeros(shape=(32, 6)),
            b3=np.zeros(shape=6),
        )

    def tensors(self) -> dict[str, FloatArray]:
        return {
            "W1": self.w1,
            "b1": self.b1,
            "W2": self.w2,
            "b2": self.b2,
            "W3": self.w3,
            "b3": self.b3,
        }
