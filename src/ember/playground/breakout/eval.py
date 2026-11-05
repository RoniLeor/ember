"""Reconstructed evaluation and recording, using official environment transitions."""

from dataclasses import InitVar, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import overload

import numpy as np
from minatar import Environment
from numpy.typing import NDArray

from ember.module import Context, Module

from .games import Game, experiments
from .models import Episode, Evaluation, Frame, Settings
from .policy import Network, Policy, sample


@dataclass(frozen=True)
class PlayRequest:
    """One environment and random stream, reused explicitly between episodes."""

    environment: Environment
    rng: np.random.Generator
    record: bool = False


@dataclass(frozen=True)
class Player(Module[PlayRequest, Episode]):
    """Play one episode. Reuse the environment to retain MinAtar sticky-action state."""

    game: Game
    policy: Policy
    max_steps: int
    context: InitVar[Context[int] | None] = field(default=None, kw_only=True)
    ctx: Context[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self, context: Context[int] | None) -> None:
        object.__setattr__(self, "ctx", Context(value=self.max_steps) if context is None else context)

    @overload
    def __call__(self, inputs: PlayRequest) -> Episode: ...

    @overload
    def __call__(self, *, environment: Environment, rng: np.random.Generator, record: bool = False) -> Episode: ...

    def __call__(
        self,
        inputs: PlayRequest | None = None,
        *,
        environment: Environment | None = None,
        rng: np.random.Generator | None = None,
        record: bool | None = None,
    ) -> Episode:
        if inputs is None:
            if environment is None or rng is None:
                raise TypeError("environment and rng are required")
            inputs = PlayRequest(environment=environment, rng=rng, record=False if record is None else record)
        elif environment is not None or rng is not None or record is not None:
            raise TypeError("Use a play request or keyword arguments, not both")
        return self.run(inputs)

    def forward(self, inputs: PlayRequest) -> Episode:
        max_steps: int = self.ctx.value
        environment: Environment = inputs.environment
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        environment.reset()
        total: float = 0.0
        frames: list[Frame] = []
        terminal: bool = False
        step: int = 0
        action: int = 0
        if inputs.record:
            frames.append(self.frame(environment=environment, score=total, action=action))
        for step in range(1, max_steps + 1):
            action = sample(logits=self.policy(self.game.encode(state=environment.state())), rng=inputs.rng)
            reward: float
            reward, terminal = environment.act(action)
            total += reward
            if inputs.record:
                frames.append(self.frame(environment=environment, score=total, action=action))
            if terminal:
                break
        return Episode(score=total, steps=step, capped=not terminal, frames=frames)

    @staticmethod
    def frame(*, environment: Environment, score: float, action: int) -> Frame:
        state: NDArray[np.bool_] = environment.state()
        cells: NDArray[np.int64] = np.max(state * np.arange(1, state.shape[2] + 1), axis=2)
        return Frame(cells=cells.ravel().tolist(), score=score, action=action)


@dataclass(frozen=True)
class EvaluationRequest:
    """A network input and the explicit reconstructed-evaluator gate."""

    network: Path
    reconstructed: bool = False


@dataclass(frozen=True)
class Evaluator(Module[EvaluationRequest, Evaluation]):
    """Diagnostic implementation. The authors' actual scoring program is unavailable."""

    settings: Settings
    context: InitVar[Context[Settings] | None] = field(default=None, kw_only=True)
    ctx: Context[Settings] = field(init=False, repr=False, compare=False)

    def __post_init__(self, context: Context[Settings] | None) -> None:
        object.__setattr__(self, "ctx", Context(value=self.settings) if context is None else context)

    @overload
    def __call__(self, inputs: EvaluationRequest) -> Evaluation: ...

    @overload
    def __call__(self, *, network: Path, reconstructed: bool = False) -> Evaluation: ...

    def __call__(
        self,
        inputs: EvaluationRequest | None = None,
        *,
        network: Path | None = None,
        reconstructed: bool | None = None,
    ) -> Evaluation:
        if inputs is None:
            if network is None:
                raise TypeError("network is required")
            inputs = EvaluationRequest(network=network, reconstructed=False if reconstructed is None else reconstructed)
        elif network is not None or reconstructed is not None:
            raise TypeError("Use an evaluation request or keyword arguments, not both")
        return self.run(inputs)

    def forward(self, inputs: EvaluationRequest) -> Evaluation:
        settings: Settings = self.ctx.value
        if not inputs.reconstructed:
            raise ValueError("Authors' evaluator is unavailable. Use --reconstructed only for a labelled diagnostic.")
        game: Game = experiments.get(settings.experiment)
        weights: Network = game.network.load(inputs.network)
        player: Player = Player(game=game, policy=weights.policy(settings.device), max_steps=settings.max_steps)
        environment: Environment = game(seed=settings.seed)
        rng: np.random.Generator = np.random.default_rng(seed=settings.seed)
        episodes: list[Episode] = [player(environment=environment, rng=rng) for _ in range(settings.episodes)]
        scores: list[float] = [episode.score for episode in episodes]
        error: float = float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
        packages: list[str] = ["numpy", "minatar", "pydantic", *game.packages, "ember-playground"]
        if settings.device != "cpu":
            packages.append("torch")
        return Evaluation(
            settings=settings,
            mean=float(np.mean(scores)),
            standard_error=error,
            mean_length=float(np.mean([episode.steps for episode in episodes])),
            capped_fraction=sum(episode.capped for episode in episodes) / len(episodes),
            scores=scores,
            network_sha256=weights.source_sha256,
            versions={name: version(name) for name in packages},
            applied=weights.applied,
            skipped=list(weights.skipped),
        )
