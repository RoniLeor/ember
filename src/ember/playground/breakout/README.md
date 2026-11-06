# Breakout

This package defines experiment `breakout`.
Shared gameplay contracts, policies, records, evaluation, and replay live in `ember.playground`.
Generic model execution lives in the Ember core.

## Files

- `../games.py`: game contract and experiment registration.
- `../models.py`: settings and gameplay records.
- `../policy.py`: inference contracts and action sampling.
- `../eval.py`: episode execution and reconstructed evaluation.
- `../action.py`: episode recording and HTML export.
- `game.py`: registration, official environment creation, observation encoding, and adapter selection.
- `../network.py`: validated weight loading and NumPy/PyTorch inference.
- `../utils/`: safe text-edit grammar and ordered application.
- `../tensors.py`: float64 parser tensor storage.
- `observation.py`: the task’s six-channel input mapping.
- `NOTICE.md`: provenance for the absorbed local parser and mapping.
- `base.py`: pinned source identity, downloads, prompt verification, and published-weight parity.
- `prompt.txt` and `instructions.md`: exact copies of the researchers' task and instruction.
- `viewer.html`: presentation of recorded environment frames. It does not implement separate physics.

## Rules

Use one fresh response with the exact task prompt. Do not train the network or provide model feedback.
All 20,486 parameters start at zero. The architecture is 600 → 32 → 32 → 6, with tanh hidden layers.
The parser reads the final `edits` block as data and rejects an incomplete block. It does not execute generated Python.
Unreadable statements are skipped and reported. Supported statements apply in order.

The environment uses MinAtar 1.0.15 with sticky actions 0.1. Evaluation samples from softmax logits.
The default diagnostic uses 1,000 episodes, seed 0, and a 10,000-step limit.
The original evaluator is unavailable; the evaluator reports a labelled reconstruction.

The runtime prompt and instruction come from `resources/breakout/` by default.
Use `--reference PATH` to select another resource root with a `breakout/` subfolder.
Published answer and weights live in `resources/breakout/runs/author/`; the latest local result lives in `runs/astra-20261104T181934Z/` there.
The `research/` subfolder retains the original configuration and multi-game published scores. Model runs do not read these records.
The bundled copies are verified mirrors. Both copies must match the pinned hashes before a model run.
Preserve their bytes and line endings. Prompt SHA-256:

```text
766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876
```

## Commands

Run these from the project root:

```sh
uv run ember fetch --experiment breakout
uv run ember parity --experiment breakout
uv run ember run --experiment breakout --provider codex --model gpt-6-astra --output resources/breakout/runs/breakout-check
uv run ember evaluate --experiment breakout --network resources/breakout/runs/author/hand.npz --reconstructed --output resources/breakout/runs/author/diagnostic.json
uv run ember play --experiment breakout --network resources/breakout/runs/author/hand.npz --output resources/breakout/runs/author/author-preview.html
```

`run` above prepares only. Add `--execute` to request execution after the strict gates pass.
After an executed run completes, check its evidence with `uv run ember audit --experiment breakout --run RUN`.
Use `replay` instead of `play` for HTML export without browser opening.
The retained Astra folder includes `run.json`. Use `--run resources/breakout/runs/astra-20261104T181934Z` for its saved answer.
Add `--output` with a new file name to keep the saved `breakout.html`.

Source: [research logs at the pinned commit](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch).
Logs and data retain the authors' CC BY 4.0 attribution. The prompt contains MinAtar code under GPL-3.0.
Both license files are included beside these assets.
