# Asterix

Experiment `asterix` uses official MinAtar 1.0.15 with difficulty ramping and sticky actions 0.1.
Collect treasure for one point. Contact with an enemy ends the episode. Rows 1 through 8 are playable.

The task uses 600 inputs and a 600 → 32 → 32 → 6 network with tanh hidden layers.
Input channels are player, danger, reward, empty projectile, trail, and empty gauge.
The source channels are player 0, enemy 1, trail 2, and gold 3.
Fire acts as no-op. The viewer uses recorded channel masks to show overlapping objects.

Pinned research commit: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
Resources and runs live under `resources/asterix/`; bundled prompt and instructions must match pinned bytes.
The original published mean is 53.178 across 1,000 episodes. Ember scores are reconstructed diagnostics.
The original evaluator is unavailable; matching weights or nearby scores do not prove evaluator equivalence.

```sh
uv run ember parity --experiment asterix
uv run ember evaluate --experiment asterix --network resources/asterix/runs/author/hand.npz --reconstructed
uv run ember run --experiment asterix --provider codex --model gpt-6-astra --mode comparison --output resources/asterix/runs/NEW --execute
uv run ember audit --experiment asterix --provider codex --mode comparison --run resources/asterix/runs/NEW
uv run ember replay --experiment asterix --run resources/asterix/runs/NEW
```

Comparison mode permits advertised tools but rejects any attempted tool call. It does not prove an empty tool inventory or exact model context.
