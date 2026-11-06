"""Standalone replay of recorded official MinAtar frames, without browser physics."""

from dataclasses import InitVar, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import overload

import numpy as np
from minatar import Environment

from ember.module import Context, Module

from .eval import Observer, Player
from .games import Game, experiments
from .models import Episode, Recording, Settings
from .policy import Network


@dataclass(frozen=True)
class ReplayRequest:
    """A network input and the explicit HTML destination."""

    network: Path
    output: Path


@dataclass(frozen=True)
class Replay(Module[ReplayRequest, Recording]):
    """Record once, then show the same frames on every operating system."""

    settings: Settings
    observer: Observer | None = field(default=None, kw_only=True, compare=False)
    context: InitVar[Context[Settings] | None] = field(default=None, kw_only=True)
    ctx: Context[Settings] = field(init=False, repr=False, compare=False)

    def __post_init__(self, context: Context[Settings] | None) -> None:
        object.__setattr__(self, "ctx", Context(value=self.settings) if context is None else context)

    @overload
    def __call__(self, inputs: ReplayRequest) -> Recording: ...

    @overload
    def __call__(self, *, network: Path, output: Path) -> Recording: ...

    def __call__(
        self, inputs: ReplayRequest | None = None, *, network: Path | None = None, output: Path | None = None
    ) -> Recording:
        if inputs is None:
            if network is None or output is None:
                raise TypeError("network and output are required")
            inputs = ReplayRequest(network=network, output=output)
        elif network is not None or output is not None:
            raise TypeError("Use a replay request or keyword arguments, not both")
        return self.run(inputs)

    def forward(self, inputs: ReplayRequest) -> Recording:
        settings: Settings = self.ctx.value
        game: Game = experiments.get(settings.experiment)
        weights: Network = game.network.load(inputs.network)
        environment: Environment = game(seed=settings.seed)
        player: Player = Player(
            game=game, policy=weights.policy(settings.device), max_steps=settings.max_steps, observer=self.observer
        )
        episode: Episode = player(
            environment=environment,
            rng=np.random.default_rng(seed=settings.seed),
            record=True,
        )
        template: str = game.template.read_text(encoding="utf-8")
        packages: list[str] = ["numpy", "minatar", *game.packages, "ember-playground"]
        if settings.device != "cpu":
            packages.append("torch")
        recording: Recording = Recording(
            **episode.model_dump(),
            settings=settings.model_copy(update={"episodes": 1}),
            network_sha256=weights.source_sha256,
            versions={name: version(name) for name in packages},
            applied=weights.applied,
            skipped=list(weights.skipped),
            replay=inputs.output.name,
        )
        payload: str = recording.model_dump_json().replace("<", "\\u003c")
        inputs.output.write_text(template.replace("__EPISODE__", payload), encoding="utf-8")
        return recording
