# Ember technical guide

Ember is a typed Python playground for weight-writing experiments with MinAtar games. Breakout is the first game.
Run all commands from the project root. The [README](../README.md) shows the setup and the main examples.

## Commands

Each command has `--help`. Put each `--plugin MODULE` option before the command name.
All commands except `list` accept `--experiment NAME`. The default experiment is `breakout`.

| Command | What it does |
| --- | --- |
| `list` | Shows the experiments and the default model of each provider. |
| `fetch` | Downloads the pinned research inputs. |
| `parity` | Compares the published text weights with the published NPZ network. |
| `run` | Prepares one model request. With `--execute`, it also sends the request if the mode checks pass. |
| `audit` | Checks the evidence of an executed run. It does not call a model. |
| `evaluate` | Scores a network with the reconstructed evaluator. You must add `--reconstructed`. |
| `replay` | Records one episode and saves it as an HTML file. |
| `play` | Records one episode, saves it, and opens it in the browser. |

- `evaluate`, `replay`, and `play` read `--network FILE` or `--run DIR`. `--run` uses `DIR/answer.md` from a completed run.
- `--run` does not prove that the run passed `audit`. The author folder has no `run.json`, so use its `hand.npz` with `--network`.
- For `replay` and `play`, `--output` is necessary with `--network`. With `--run`, the default output is `DIR/breakout.html`.
- Ember does not make the parent folder of `--output`, and it replaces an existing file. For the saved author and Astra runs, use a new file name.
- The defaults are `--device cpu`, `--seed 0`, and `--max-steps 10000`. By default, `evaluate` plays 1,000 episodes (`--episodes N`).
- Each `run` call must use a new output folder. This rule also applies when you do not add `--execute`.
- Ember sends a Claude request only with `--mode comparison`. In the default strict mode, Ember sends no Claude request.
- `audit` must use the provider and the mode of the run, for example `--provider codex --mode comparison`.
- Before you use `--device mps` or `--device cuda`, run `make sync UV_SYNC_ARGS="--extra gpu"`.
- If the device is not available, the command fails. Ember does not use CPU inference in its place.
- The exit code is 0 for success, 1 for a failure, and 2 for an incorrect command.

## Python API

`Runner`, `Player`, `Evaluator`, `Replay`, and `Breakout` are `Module[Input, Output]` components.
A call `component(inputs)` calls `run(inputs)`, and `run` calls `forward(inputs)`.
These components read their configuration from a `Context`. If you give `context=`, the component uses the context value.
To share one configuration, give the same `Context` to more than one component.

```python
from pathlib import Path

from ember import Context, Evaluation, EvaluationRequest, Evaluator, Settings

settings: Settings = Settings(episodes=100)
ctx: Context[Settings] = Context(value=settings)
evaluate: Evaluator = Evaluator(settings=settings, context=ctx)
network: Path = Path("resources/breakout/runs/author/hand.npz")

with ctx.override(Settings(episodes=2, max_steps=10)):
    result: Evaluation = evaluate(EvaluationRequest(network=network, reconstructed=True))
```

The `with` block changes the configuration only inside the block. Use frozen values, such as `Settings`.

## Extend

A plugin is a Python module that registers classes. `--plugin MODULE` imports it before the command runs.

1. Save this file as `local.py` in the project root:

   ```python
   from typing import ClassVar

   from ember import Breakout, experiment


   @experiment
   class LocalBreakout(Breakout):
       name: ClassVar[str] = "local.breakout"
   ```

2. Run `uv run python -m ember --plugin local list`. The output shows `local.breakout`.

- A new game is an experiment class. It follows the `Game` contract in `src/ember/playground/breakout/games.py`.
- A new provider uses `@ember.provider` and the `Provider` contract in `src/ember/providers/base.py`.
- Two experiments or two providers with the same name cause an error. Ember does not search folders for plugins.
- `python -m ember --plugin NAME` can import `NAME.py` from the current folder. The `ember` script imports only installed modules.
- A plugin runs as trusted Python code in the Ember process. It is not a model tool.

## Checks

Run `make check PYTEST_ARGS=` before delivery. It runs Ruff, Pyright, ty, mypy, and all tests.
[AGENTS.md](../AGENTS.md) gives the other targets and the rules for checks.

## More information

- [Breakout](../src/ember/playground/breakout/README.md): the network, the parser, the rules, and the pinned inputs.
- [Research protocol](protocol.md): the providers, the audit modes, the evidence, and the saved results.
