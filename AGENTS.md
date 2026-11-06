# Ember agent guidance

Apply the user's global guidance. Use `pyproject.toml` as the authority for checks and test collection.
Run commands from the project root. Keep changes scoped to the request.

## Commands and timing

| When | Command | Purpose |
| --- | --- | --- |
| First setup or after dependency changes | `make sync` | Install locked dependencies and development tools. Preserve installed optional packages. |
| Work needs PyTorch or GPU checks | `make sync UV_SYNC_ARGS="--extra gpu"` | Install the optional inference backend. Linux and Windows select CUDA wheels. |
| After Python edits | `make check-lint` | Check formatting, imports, lint rules, and types with Pyright, ty, and mypy. |
| During implementation | `make test PYTEST_ARGS="tests/playground/models.py -k invalid"` | Run focused tests against the affected production code. Replace the example selection. |
| After finishing code changes and before delivery or any commit | `make check PYTEST_ARGS=` | Require passing static checks and the complete test suite. Clear focused test selections. |
| Investigate test failures | `make test-verbose PYTEST_ARGS="tests/playground/models.py"` | Show individual test results and failure details. |
| Formatting needs correction | `make format` | Rewrite Python formatting. Review the resulting changes. |
| Lint findings have automatic fixes | `make fix` | Format Python and apply Ruff's safe fixes. Review every change and rerun checks. |
| Package metadata, assets, or build settings change | `make build` | Build the wheel. Inspect its contents for required package files. |

Check targets use `uv run --no-sync`. Run `make sync` first; checks do not install missing tools.
`make sync` uses `--inexact` to retain an installed GPU backend when no extra is requested.
After changing dependencies, run `uv lock`, then `make sync`. Keep `uv.lock` consistent with `pyproject.toml`.
Use `make help` to list targets. No service or cloud setup is needed for these checks.
Setup and isolated wheel builds can download dependencies. Static checks and tests use the installed environment.

## Required completion and commit checks

- After the final code or configuration edit, run `make check PYTEST_ARGS=` before reporting completion.
- Before creating any commit, run `make check PYTEST_ARGS=` against the exact files to be committed.
- This command must pass formatting, lint, Pyright, ty, mypy, and the complete pytest suite.
- Focused tests and earlier passing runs do not replace this final check.
- If code or configuration changes after a passing check, rerun the command before completion or commit.
- Fix failures and rerun the command. Do not bypass checks, reduce test selection, or weaken tests.
- If checks cannot run or still fail, report the blocker. Do not commit or claim verification is complete.
- Report the check result and test count in the completion message.
- These rules do not authorize a commit. Commit only when the user requests it.

## Scope and evidence

- Preserve `python_files = ["*.py"]` and `testpaths = ["tests"]`. Existing tests do not use `test_` filenames.
- Keep the configured Ruff rules. Do not copy service paths, exclusions, or unrelated settings from other projects.
- Keep all three type checkers. Their checks overlap but can find different errors.
- Keep missing-import exemptions limited to untyped MinAtar and optional PyTorch in mypy.
- ty permits unresolved optional `torch` imports. Pyright exemptions apply only to the two optional imports in the backend.
- Keep the optional PyTorch fallback's line-specific mypy exemption. Backend entry points check availability before use.
- Preserve the frozen-model mutation test. Its line-specific ty exemption allows an intentional invalid operation for runtime validation.
- Add isolated tests for new functions and methods. Never weaken tests to hide a defect.
- Keep module imports at the top. Use single-dot imports within a package and absolute imports across package boundaries.
- Use typed, focused callable components. Add no speculative wrappers, options, or abstractions.
- Obtain independent read-only reviews before implementation and before delivery of code changes.
- Format and fix targets rewrite files. Review the Git diff and preserve unrelated work before broad edits.
- Tools can create `.ruff_cache`, `.mypy_cache`, `.pytest_cache`, `__pycache__`, and `dist`. Do not remove saved research evidence.
- Preserve pinned files and saved run evidence under `resources/` unless the user requests changes there.
- Do not use model execution, research-input downloads, or long evaluations as routine checks. They require task-specific authorization.
- Do not replace or delete files in the retained author and Astra runs unless the user requests it. Use a new output directory for each authorized experiment.
- Save replay, preview, and evaluation outputs in the corresponding `resources/<game>/runs/<run>/` directory, not in the project root. Use a file name that does not exist in that directory.
- Passing static checks or tests does not establish model quality, score parity, CUDA support, or performance.
- Report commands, results, and missing evidence. State whether optional backend tests actually ran.

For documentation-only delivery, check the text and referenced commands. The required check still applies before any commit.

- Keep raw model traces, event logs, local credentials, and runtime caches out of commits. Publish only reviewed run metadata and summaries.
