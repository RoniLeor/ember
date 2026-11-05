# Research protocol and audit

Run commands below from the project root. Saved results record the versions and paths used at the time.

## Run the one-shot experiment

Install and authenticate the selected CLI first. The default provider is `codex`, with model `gpt-6-astra`.
Each output directory must be new. This protects existing evidence.

```sh
uv run ember run --output resources/breakout/runs/breakout-check
uv run ember run --output resources/breakout/runs/breakout-new --execute
```

Both commands verify the pinned prompt and instructions, then inspect Codex's model-visible tool inventory.
Without `--execute`, the command only prepares the run.
With `--execute`, it makes one fresh request only when an explicit empty tool inventory is present.
Missing or unrecognized inventory evidence blocks execution.
The context must contain only the pinned instructions and prompt, in order. Extra messages or changed text block execution.
The instructions comparison permits the trailing newline removed by the CLI. The prompt comparison remains exact.
It uses `xhigh`, detailed summaries, context window 872000, and auto-compaction limit 1000000.
It provides no answer, score, feedback, or second turn.
User and managed security hooks remain enabled. This differs from the authors' hook configuration.
Sandbox and approval settings remain inherited. All known deviations are listed in `run.json`.
The output explicitly marks exact author reproduction as unestablished.
Use `--model NAME` to adapt the transport; changing the model changes the experiment.

### Select a provider and model

Both providers use the same command and callable runner:

```sh
uv run ember list
uv run ember run --provider codex --model gpt-6-astra --output resources/breakout/runs/astra-check
uv run ember run --provider claude --model claude-sonnet-5-5 --output resources/breakout/runs/sonnet-check
uv run ember run --provider claude --model opus --output resources/breakout/runs/opus-check
```

Omit `--model` to use `gpt-6-astra` for Codex or `claude-sonnet-5-5` for Claude.
Model identifiers pass unchanged to the selected CLI. Aliases such as `sonnet` can resolve differently over time.
Empty names, whitespace, control characters, and leading `-` are rejected before a run folder is created.

**Strict mode is the default. Claude cannot execute a strictly verified experiment.**
Claude Code 2.1.317 provides tool-disabling flags, but no offline rendered prompt/tool inventory was found.
In strict mode, the adapter runs only `claude --version`, records the intended command, and returns `blocked`.
This includes `--execute`: strict mode sends no Claude inference request.
Declared flags do not prove unavailable tools. Hooks, CLAUDE.md files, and auto-memory can add context.
Claude also differs from the authors' model family, effort, and context settings.
Earlier Claude and Codex preflight records are preserved in the local archive outside this project.

The run folder records `provider` as well as `model`. Audit selection is explicit:

```sh
uv run ember audit --provider codex --run resources/breakout/runs/breakout-new
```

An audit needs an executed run. A prepared or blocked run has no answer to check.
An audit rejects a different provider recorded in `run.json`. Older records without that field remain Codex records.
Claude strict audit returns an explicit verification error.

### Labelled Claude comparisons

Use `--mode comparison` to allow Claude execution with declared tool-disabling flags.
Without `--execute`, this only prepares the command.

```sh
uv run ember run --provider claude --model claude-sonnet-5-5 --mode comparison --output resources/breakout/runs/sonnet-comparison --execute
uv run ember audit --provider claude --mode comparison --run resources/breakout/runs/sonnet-comparison
uv run ember replay --network resources/breakout/runs/sonnet-comparison/answer.md --output resources/breakout/runs/sonnet-comparison/breakout.html
```

Both run and audit records carry `mode: comparison`. Tool inventory and exact-context evidence remain false.
The comparison audit checks the declared empty tool list and observed events, not independently verified availability.
It requires one session, one assistant message, one text block, and one successful completed turn.
Thinking blocks may accompany that text block. The answer must match the result and assistant text exactly.
Tool blocks, permission denials, unknown event types, incomplete turns, and failed processes are rejected.
Known hook lifecycle and rate-limit events are retained as metadata. They do not establish prompt equivalence.
Plugins can also inject context, including AGENTS.md instructions. Hooks inherit the launch environment.
The command preserves security hooks, sends the pinned prompt on stdin, and disables session persistence.
No resume or model retry is initiated by Ember. Raw events and stderr remain available after failures.
Comparison audits require a saved run record and explicit `--mode comparison`; strict audits reject that label.
Legacy records without a mode remain strict. No comparison is an exact reproduction of the authors' experiment.
If parsing fails, no answer file is accepted. Standalone audit then requires manual recovery after a reviewed parser fix.
Preserve the original failed record and raw logs during recovery; do not request another model answer.

Earlier Claude comparisons and their raw evidence are preserved in the local archive outside this project.

### Labelled Codex comparisons

Codex also supports `--mode comparison` through the existing `Runner` module and Python subprocess transport.
It restores the retained manual run's effective policy: ignored user configuration, disabled hooks, and explicit `never`/`read-only` settings.
The command disables the 26 features listed in `resources/breakout/research/codex.toml`.
It uses the selected model (default `gpt-6-astra`), `xhigh`, detailed summaries, context 872000, and compaction limit 1000000.
It sends the pinned prompt on stdin and selects the pinned instruction file once. It passes no `gpt-6-luna` or `medium` values.
The strict path retains its existing configuration and preflight behavior.

```sh
uv run ember run --provider codex --model gpt-6-astra --mode comparison --output resources/breakout/runs/astra-comparison-new --execute
uv run ember audit --provider codex --mode comparison --run resources/breakout/runs/astra-comparison-new
```

Preparation checks pinned inputs and the CLI version, then records the command without an inference request.
The offline `debug prompt-input` command cannot apply `--ignore-user-config`; comparison preparation therefore does not use that probe.
Prepared availability and exact-context flags are false because they are not yet established.
Execution fills those flags from the actual request trace. Global instructions can add messages and advertise tools despite disabled features.

`trace-audit.json` records advertised tool names, extra message count, actual context flags, inference count, and attempted tool-call count.
A completed run requires one matching inference request/response pair, the recorded model, pinned input blocks, and a nonempty final answer.
That answer must equal `answer.md`, apart from trailing newlines.
Only reasoning and final assistant text are accepted as response items. Tool-call items fail even when execution was rejected.
Explicit tool events in raw protocol or history evidence also fail. Missing or invalid evidence leaves call counts unestablished.
The JSONL audit separately requires one completed turn and a matching answer; JSONL alone cannot establish zero actual tool calls.
The saved run record, raw trace, prompt, and pinned instructions remain authoritative during re-audit.
Re-audit supports relocated evidence paths and compares the trace model with the saved model identity.

This comparison does not claim unavailable tools, exact historical context, or author-certified evaluation.
Use a fresh timestamped run directory for each request. Preserve the retained Astra and author reference artifacts.

For strict Codex runs, the folder records the prompt, model, command, CLI version, inventory, errors, and outcome.
A completed request also records the event stream, answer, usage, stderr, and optional CLI trace.
The audit rejects tool calls, unknown events, incomplete turns, multiple answers, and answer-file mismatches.
The standalone audit also verifies the saved prompt and model context against the pinned reference files.
Use `audit --reference PATH` when the resource root is elsewhere. Breakout files must be in its `breakout/` subfolder.
Its tool-availability evidence comes from the preflight inventory. A trace is retained for further request-level inspection.
The audit cannot independently prove the model or settings used. Those are recorded in `run.json` and any available CLI trace.
The saved preflight is not cryptographically linked to the execution log.

```sh
uv run ember audit --run resources/breakout/runs/breakout-new
```

## Retained results

The retained reference results include a completed local Astra run and the author's baseline.
They are under `resources/breakout/runs/`. Earlier evidence is archived outside the project at
`~/Library/Application Support/Ember/archives/runs-20261104T105814Z/`.
The archive manifest records file hashes, permissions, symlinks, and retained file locations. These archives are private and are not part of the public repository.

### Author baseline

`author/answer.md` and `author/hand.npz` are pinned upstream artifacts.
`ember parity` checks them against the SHA-256 values in `src/ember/playground/breakout/base.py`. `manifest.json` records the same values.
`author/evaluation.json`, `author/replay.json`, and `author/breakout.html` are saved local diagnostics using those published weights.
These local diagnostics are reconstructed results, not the authors' original evaluation or pinned upstream files.
The diagnostic mean is 502.2 with standard error 7.845337 across 1,000 CPU episodes, seed 0.
Its replay scores 491 across 7,284 steps and ends naturally. The authors' published mean is 517.411.

### Latest local Astra result

`astra-20261104T181934Z/` preserves the fixed Ember comparison run from 2026-11-04.
`run.json` records Codex CLI 0.166.0, model `gpt-6-astra`, and successful completion in 707.62 seconds.
The request used `xhigh`, detailed summaries, context window 872000, and auto-compaction limit 1000000.
It ignored user configuration, disabled hooks, and used approval policy `never` with sandbox mode `read-only`.

`trace-audit.json` verifies one inference and zero attempted tool calls. Saved-file re-audit also passed.
Raw traces, event logs, and preflight context are not published. Re-auditing this retained run requires the original local evidence.
Four tool schemas were advertised, and global instruction discovery added two messages.
This comparison does not establish unavailable tools or exact author reproduction.
All 182 weight edits applied without skipped statements.
The answer SHA-256 is `3978dce63679c99ea15ef631eb80178a2f319bf2d1fa56ab6869af6fa30aa424`.

`evaluation.json` records a reconstructed mean of 530.13, with standard error 14.866304, across 100 CPU episodes, seed 0.
This smaller diagnostic reached the 10,000-step cap in 76% of episodes. It does not establish author score parity.
`breakout.html` replays one episode scoring 652 at the same step cap.
`verification.json` records checks, independent review, re-audit, and diagnostic evidence.
The run directory retains mode 0700 and contains raw local request context.

```sh
uv run ember audit --provider codex --mode comparison --run resources/breakout/runs/astra-20261104T181934Z
```

The earlier manual Astra run and blocked strict attempt are archived at
`~/Library/Application Support/Ember/archives/runs-20261104T184521Z/`.
Its manifest records original paths and verified file hashes. Only the fixed Astra run and author baseline remain in this checkout.

## Evaluation

The authors specify 1,000 episodes, sampled softmax actions, sticky actions 0.1, and a 10,000-step episode limit.
Their published Breakout mean is 517.411, with standard error 7.41695.

The original evaluator was not found in the pinned repository release.
[Community PR #1](https://github.com/RikiyaT/lm_understand-logs/pull/1) provides a reconstruction, not the original scorer.
It is open and has an unresolved parser review finding. This playground does not execute its code.
Matching weights or obtaining a nearby score does not prove evaluator equivalence.

Official scoring fails explicitly. Optional diagnostics require `--reconstructed`:

```sh
uv run ember evaluate --network resources/breakout/runs/author/hand.npz --reconstructed --output resources/breakout/runs/author/diagnostic.json
```

Every diagnostic records `kind: reconstructed` and `evaluator_matches_authors: false`.
It also records settings, versions, network hash, scores, standard error, capped fraction, and applied or skipped edits.
Replay metadata records settings, versions, network hash, and applied or skipped edits for one episode.
It does not record `kind` or `evaluator_matches_authors`. The HTML also embeds every recorded frame.
The default is 1,000 episodes. Smaller values of `--episodes` or `--max-steps` are smoke checks only.
The diagnostic uses float32 inference, NumPy categorical sampling, and the same seed for separate game and action generators.
It preserves MinAtar's previous action across episode resets. The authors' seeding and sampling implementation are unknown.
The existing parser computes edits in float64 and then converts to float32. Published Breakout tensors match exactly;
this does not establish parity for every possible reply. Unreadable statements are skipped, as the prompt requires.
The remaining edits still apply. Diagnostics and replay metadata report skipped statements.

## Sources and licenses

Research source: [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
Assets are pinned to [`4d09ba6`](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch).
`resources/breakout/manifest.json` records source URLs and SHA-256 hashes. Its file keys are relative to the `resources/` root.
Runtime prompt and instructions stay at the game root. Author answer and weights are under `breakout/runs/author/`.
`breakout/research/evaluation.json` is the intact upstream score archive covering several games.
`breakout/research/codex.toml` is the published harness configuration. Model runs do not load either research file.
The manifest covers the eight pinned upstream artifacts, not locally generated diagnostics or Astra request records.
Logs, data, and networks retain CC BY 4.0 attribution to the original authors.
The prompt contains MinAtar source under GPL-3.0. Both authoritative license files are in `resources/breakout/`.
