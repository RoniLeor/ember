"""Episode boundaries, deterministic official rules, and diagnostic-only scoring."""

from collections.abc import Callable
from contextlib import ExitStack
from operator import methodcaller
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from minatar import Environment

from ember.module import Context
from ember.playground.breakout import Breakout
from ember.playground.breakout.action import Replay, ReplayRequest
from ember.playground.breakout.eval import EvaluationRequest, Evaluator, Player, PlayRequest
from ember.playground.breakout.models import Episode, Evaluation, Recording, Settings
from ember.playground.breakout.network import Weights


@pytest.fixture
def network(tmp_path: Path) -> Path:
    path: Path = tmp_path / "reply.md"
    path.write_text("```edits\nb3[:] = 0\n```", encoding="utf-8")
    return path


def test_terminal_on_last_step_is_not_capped(network: Path) -> None:
    environment: Environment = Mock(spec=Environment)
    environment.state.return_value = np.zeros((10, 10, 4), dtype=np.bool_)
    environment.act.side_effect = [(1.0, False), (2.0, True)]
    player: Player = Player(game=Breakout(), policy=Weights.load(network).policy("cpu"), max_steps=2)
    result: Episode = player(environment=environment, rng=np.random.default_rng(seed=1), record=True)
    assert result.score == 3 and result.steps == 2 and not result.capped
    assert len(result.frames) == 3
    assert result.frames[0].score == 0 and result.frames[-1].score == 3
    environment.reset.assert_called_once()
    assert environment.act.call_count == 2


def test_cap_and_invalid_limit(network: Path) -> None:
    environment: Environment = Mock(spec=Environment)
    environment.state.return_value = np.zeros((10, 10, 4), dtype=np.bool_)
    environment.act.return_value = (0.0, False)
    weights: Weights = Weights.load(network)
    result: Episode = Player(game=Breakout(), policy=weights.policy("cpu"), max_steps=3)(
        environment=environment,
        rng=np.random.default_rng(seed=0),
    )
    assert result.capped and result.steps == 3 and not result.frames
    with pytest.raises(ValueError):
        Player(game=Breakout(), policy=weights.policy("cpu"), max_steps=0)(
            environment=environment,
            rng=np.random.default_rng(seed=0),
        )


def test_official_gate_and_repeatable_diagnostic(network: Path) -> None:
    evaluator: Evaluator = Evaluator(settings=Settings(episodes=3, max_steps=20))
    with pytest.raises(ValueError, match="Authors' evaluator"):
        evaluator(network=network)
    first: Evaluation = evaluator(network=network, reconstructed=True)
    second: Evaluation = evaluator(network=network, reconstructed=True)
    assert first == second and first.kind == "reconstructed" and not first.evaluator_matches_authors
    assert first.versions["minatar"] == "1.0.15" and len(first.network_sha256) == 64
    assert len(first.scores) == 3 and first.mean_length <= 20
    assert (
        Evaluator(settings=Settings(episodes=1, max_steps=1))(
            network=network,
            reconstructed=True,
        ).standard_error
        == 0
    )


def test_replay_and_skipped_edits(network: Path, tmp_path: Path) -> None:
    output: Path = tmp_path / "replay.html"
    replay: Replay = Replay(settings=Settings(max_steps=5))
    episode: Recording = replay(network=network, output=output)
    text: str = output.read_text()
    assert "__EPISODE__" not in text and episode.model_dump_json() in text
    assert len(episode.frames) == episode.steps + 1 and "RECORDED PLAY" in text
    assert episode.settings.episodes == 1 and "ember-playground" in episode.versions
    assert "spark" not in episode.versions
    assert episode.applied == 1 and len(episode.network_sha256) == 64
    network.write_text("```edits\ninvalid\nb3[:] = 0\n```", encoding="utf-8")
    partial: Recording = replay(network=network, output=output)
    assert partial.frames == episode.frames and partial.skipped == ["invalid"] and partial.applied == 1
    result: Evaluation = Evaluator(settings=Settings(episodes=1, max_steps=5))(network=network, reconstructed=True)
    assert result.skipped == ["invalid"] and result.applied == 1 and result.scores == [episode.score]


def test_evaluation_hash_tracks_loaded_bytes(network: Path) -> None:
    original: Weights = Weights.load(network)
    network.write_text("```edits\nb3[:] = 1\n```", encoding="utf-8")
    with patch.object(Weights, "load", return_value=original):
        result: Evaluation = Evaluator(settings=Settings(episodes=1, max_steps=1))(network=network, reconstructed=True)
    assert result.network_sha256 == original.source_sha256
    assert result.network_sha256 != Weights.load(network).source_sha256


@pytest.mark.parametrize("record", [False, True])
def test_player_request_and_run_match_legacy_call(network: Path, record: bool) -> None:
    player: Player = Player(game=Breakout(), policy=Weights.load(network).policy("cpu"), max_steps=3)
    legacy: Episode = player(environment=Breakout()(seed=2), rng=np.random.default_rng(seed=2), record=record)
    request: Episode = player(
        PlayRequest(environment=Breakout()(seed=2), rng=np.random.default_rng(seed=2), record=record)
    )
    dispatched: Episode = player.run(
        PlayRequest(environment=Breakout()(seed=2), rng=np.random.default_rng(seed=2), record=record)
    )
    assert legacy == request == dispatched
    assert player.max_steps == player.ctx.value == 3


def test_player_context_changes_cap_and_validates_before_reset(network: Path) -> None:
    environment: Environment = Mock(spec=Environment)
    environment.state.return_value = np.zeros((10, 10, 4), dtype=np.bool_)
    environment.act.return_value = (0.0, False)
    context: Context[int] = Context(value=2)
    player: Player = Player(game=Breakout(), policy=Weights.load(network).policy("cpu"), max_steps=1, context=context)
    request: PlayRequest = PlayRequest(environment=environment, rng=np.random.default_rng(seed=0))
    assert player.ctx is context and player.max_steps == 1
    assert player.run(request).steps == 2
    environment.reset.reset_mock()
    with context.override(0), pytest.raises(ValueError, match="max_steps must be positive"):
        player.run(request)
    environment.reset.assert_not_called()
    assert context.value == 2


def test_evaluation_and_replay_share_scoped_settings(network: Path, tmp_path: Path) -> None:
    default: Settings = Settings(episodes=2, seed=1, max_steps=3)
    scoped: Settings = Settings(episodes=1, seed=2, max_steps=2)
    context: Context[Settings] = Context(value=default)
    evaluator: Evaluator = Evaluator(settings=default, context=context)
    replay: Replay = Replay(settings=default, context=context)
    evaluation_request: EvaluationRequest = EvaluationRequest(network=network, reconstructed=True)
    replay_request: ReplayRequest = ReplayRequest(network=network, output=tmp_path / "replay.html")
    assert evaluator.ctx is replay.ctx is context
    baseline: Evaluation = evaluator.run(evaluation_request)
    with context.override(scoped):
        result: Evaluation = evaluator(evaluation_request)
        assert result == evaluator(network=network, reconstructed=True)
        recording: Recording = replay(replay_request)
        assert recording == replay.run(replay_request) == replay(network=network, output=replay_request.output)
        assert result.settings == recording.settings == scoped
        assert len(result.scores) == 1 and recording.steps <= 2
    assert evaluator.ctx.value == replay.ctx.value == default
    assert evaluator.settings is replay.settings is default
    assert baseline.settings == default and len(baseline.scores) == 2
    assert evaluator.run(evaluation_request) == baseline


@pytest.mark.parametrize("entrypoint", ["evaluate", "replay"])
def test_orchestration_captures_settings_before_loader_changes_context(
    network: Path, tmp_path: Path, entrypoint: str
) -> None:
    settings: Settings = Settings(episodes=2, seed=1, max_steps=3)
    changed: Settings = Settings(episodes=1, seed=2, max_steps=1)
    context: Context[Settings] = Context(value=settings)
    original: Callable[[Path], Weights] = Weights.load
    scopes: ExitStack
    with ExitStack() as scopes:

        def load(path: Path) -> Weights:
            assert path == network
            scopes.enter_context(context.override(changed))
            return original(path)

        with patch.object(Weights, "load", side_effect=load):
            if entrypoint == "evaluate":
                evaluation: Evaluation = Evaluator(settings=settings, context=context).run(
                    EvaluationRequest(network=network, reconstructed=True)
                )
                assert evaluation.settings == settings and len(evaluation.scores) == 2
            else:
                recording: Recording = Replay(settings=settings, context=context).run(
                    ReplayRequest(network=network, output=tmp_path / "replay.html")
                )
                assert recording.settings == settings.model_copy(update={"episodes": 1})
                assert recording.steps == settings.max_steps
        assert context.value == changed
    assert context.value == settings


def test_typed_evaluator_gate_precedes_lookup_and_loading(tmp_path: Path) -> None:
    evaluator: Evaluator = Evaluator(settings=Settings(experiment="missing", episodes=1, max_steps=1))
    with pytest.raises(ValueError, match="Authors' evaluator"):
        evaluator.run(EvaluationRequest(network=tmp_path / "missing"))


def test_play_facade_rejects_missing_and_mixed_arguments(network: Path) -> None:
    player: Player = Player(game=Breakout(), policy=Weights.load(network).policy("cpu"), max_steps=1)
    request: PlayRequest = PlayRequest(environment=Breakout()(seed=0), rng=np.random.default_rng(seed=0))
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__")(player)
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__", environment=request.environment)(player)
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__", rng=request.rng)(player)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", request, environment=request.environment)(player)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", request, rng=request.rng)(player)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", request, record=False)(player)


def test_evaluation_and_replay_facades_reject_missing_and_mixed_arguments(network: Path, tmp_path: Path) -> None:
    settings: Settings = Settings(episodes=1, max_steps=1)
    evaluator: Evaluator = Evaluator(settings=settings)
    evaluation_request: EvaluationRequest = EvaluationRequest(network=network, reconstructed=True)
    replay: Replay = Replay(settings=settings)
    replay_request: ReplayRequest = ReplayRequest(network=network, output=tmp_path / "replay.html")
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__")(evaluator)
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__", reconstructed=False)(evaluator)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", evaluation_request, network=network)(evaluator)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", evaluation_request, reconstructed=False)(evaluator)
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__")(replay)
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__", network=network)(replay)
    with pytest.raises(TypeError, match="required"):
        methodcaller("__call__", output=replay_request.output)(replay)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", replay_request, network=network)(replay)
    with pytest.raises(TypeError, match="not both"):
        methodcaller("__call__", replay_request, output=replay_request.output)(replay)
    assert not replay_request.output.exists()
