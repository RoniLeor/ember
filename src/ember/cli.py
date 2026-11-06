"""Command-line boundaries for preparation, generation, audit, evaluation and replay."""

import argparse
import json
import subprocess
import sys
import webbrowser
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console

from .console import replaying, watch
from .models import Audit
from .playground.action import Replay
from .playground.base import GameSource, Parity
from .playground.config import DEFAULT_EXPERIMENT
from .playground.eval import Evaluator
from .playground.games import experiments
from .playground.models import Settings
from .plugins import Plugins
from .providers import Provider, providers
from .providers.records import RunRecord
from .runner import Runner, Writer

COMMAND_HELP: dict[str, str] = {
    "fetch": "Download pinned research inputs",
    "parity": "Verify published text weights against the NPZ network",
    "run": "Prepare or execute one audited model request",
    "evaluate": "Score a network with the reconstructed evaluator",
    "replay": "Save a standalone recorded episode",
    "play": "Record an episode and open it in your browser",
}


def arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI syntax; Pydantic validates numeric settings at the command boundary."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import a registration module (repeatable; before the command)",
    )
    commands: argparse._SubParsersAction = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List registered experiments and providers")
    name: str
    for name in ("fetch", "parity", "run"):
        command: argparse.ArgumentParser = commands.add_parser(name, help=COMMAND_HELP[name])
        command.add_argument(
            "--reference", type=Path, default=Path("resources"), help="Root of per-game resource folders"
        )
        command.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--provider", default="codex", help="Registered provider name (see ember list)")
            command.add_argument("--model", help="Model name or alias; defaults to the provider's model")
            command.add_argument("--mode", choices=("strict", "comparison"), default="strict")
            command.add_argument(
                "--execute", action="store_true", help="Make one request under the selected audit mode"
            )
    playback_arguments(commands=commands)
    audit: argparse.ArgumentParser = commands.add_parser(
        "audit", help="Check saved run evidence without calling a model"
    )
    audit.add_argument("--run", type=Path, required=True)
    audit.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    audit.add_argument("--reference", type=Path, default=Path("resources"), help="Root of per-game resource folders")
    audit.add_argument("--provider", default="codex", help="Registered provider name (see ember list)")
    audit.add_argument("--mode", choices=("strict", "comparison"), default="strict")
    args: argparse.Namespace = parser.parse_args(argv)
    if args.command in ("replay", "play") and args.output is None:
        args.output = replay_path(run=args.run, experiment=args.experiment, parser=parser)
    return args


def replay_path(*, run: Path | None, experiment: str, parser: argparse.ArgumentParser) -> Path:
    """Resolve a safe default replay path or report the existing CLI syntax error."""
    if run is None:
        parser.error("--output is required with --network")
    if Path(experiment).name != experiment or "\\" in experiment or experiment in (".", ".."):
        parser.error("Experiment must be a file name to use the default replay output")
    return run / f"{experiment}.html"


def playback_arguments(*, commands: argparse._SubParsersAction) -> None:
    """Share network input and device settings across scoring and playback commands."""
    name: str
    for name in ("evaluate", "replay", "play"):
        command: argparse.ArgumentParser = commands.add_parser(name, help=COMMAND_HELP[name])
        source: argparse._MutuallyExclusiveGroup = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--network", type=Path, help="Text weight edits or an NPZ network")
        source.add_argument("--run", type=Path, help="Use answer.md from a completed Ember run")
        command.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
        command.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--max-steps", type=int, default=10_000)
        command.add_argument(
            "--output",
            type=Path,
            help="Optional JSON destination; the result is also printed"
            if name == "evaluate"
            else "HTML destination; with --run, defaults to RUN/<experiment>.html (replaces an existing file)",
        )
        if name == "evaluate":
            command.add_argument("--episodes", type=int, default=1_000)
            command.add_argument("--reconstructed", action="store_true")
        else:
            command.set_defaults(episodes=1)


def network_path(args: argparse.Namespace) -> Path:
    """Resolve explicit network bytes or an answer from a completed run record."""
    if args.network is not None:
        return args.network
    root: Path = args.run.resolve()
    record: RunRecord = RunRecord.model_validate_json((root / "run.json").read_text(encoding="utf-8"))
    if record.status != "completed":
        raise ValueError(f"Run is {record.status}, not completed: {root}")
    answer: Path = root / "answer.md"
    if not answer.is_file():
        raise ValueError(f"Completed run has no answer file: {answer}")
    return answer


def dispatch(*, args: argparse.Namespace, console: Console) -> BaseModel | dict[str, object]:
    """Keep transport and game execution separate from command parsing; progress goes to the console."""
    if args.command == "list":
        return {
            "experiments": list(experiments.games),
            "providers": {name: provider.default_model for name, provider in providers.entries.items()},
        }
    if args.command in ("fetch", "parity", "run"):
        sources: GameSource = experiments.get(args.experiment).sources(root=args.reference.resolve())
        if args.command == "fetch":
            return sources.fetch()
        if args.command == "parity":
            return sources.parity()
        selected: Provider = providers.get(name=args.provider, sources=sources, model=args.model)
        writer: Writer = Runner(provider=selected, mode=args.mode)
        output: Path = args.output.resolve()
        return watch(
            console=console,
            output=output,
            execute=args.execute,
            title=f"{selected.name} · {selected.model} · {args.mode}",
            work=lambda: writer(output=output, execute=args.execute),
        )
    if args.command == "audit":
        provider: Provider = providers.get(
            name=args.provider, sources=experiments.get(args.experiment).sources(root=args.reference.resolve())
        )
        saved: Path = args.run / "run.json"
        if saved.exists():
            record: RunRecord = RunRecord.model_validate_json(saved.read_text(encoding="utf-8"))
            if record.provider != args.provider or record.mode != args.mode:
                raise ValueError("Selected audit provider or mode differs from the saved run")
        return provider.audit(root=args.run, mode=args.mode)
    settings: Settings = Settings.model_validate(
        {name: getattr(args, name) for name in ("experiment", "device", "seed", "episodes", "max_steps")}
    )
    if args.command == "evaluate":
        return Evaluator(settings=settings)(network=network_path(args), reconstructed=args.reconstructed)
    with replaying(console=console, experiment=settings.experiment, max_steps=settings.max_steps) as progress:
        return Replay(settings=settings, observer=progress)(network=network_path(args), output=args.output)


def main(argv: list[str] | None = None) -> int:
    """Return nonzero for blocked, failed or invalid work; never print success on errors."""
    args: argparse.Namespace = arguments(argv)
    try:
        Plugins(modules=tuple(args.plugin))()
        result: BaseModel | dict[str, object] = dispatch(args=args, console=Console(stderr=True))
        text: str = (
            result.model_dump_json(indent=2, exclude={"frames"})
            if isinstance(result, BaseModel)
            else json.dumps(result, indent=2)
        )
        if args.command == "evaluate" and args.output is not None:
            args.output.write_text(text, encoding="utf-8")
        print(text)
        if args.command == "play" and args.output is not None and not webbrowser.open(args.output.resolve().as_uri()):
            raise OSError(f"Replay saved at {args.output.resolve()}, but no browser opened. Open the file manually.")
        if isinstance(result, RunRecord) and result.status in ("blocked", "failed"):
            return 1
        if isinstance(result, Audit) and not result.passed:
            return 1
        if isinstance(result, Parity) and (not result.tensors_equal or result.skipped):
            return 1
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, webbrowser.Error) as error:
        print(f"ember: {error}", file=sys.stderr)
        return 1
