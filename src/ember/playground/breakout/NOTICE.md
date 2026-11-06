# Local code provenance

The Breakout parser, tensor state, and observation mapping came from this project's local
`breakout/candidate/src/spark` copy. They are local reconstruction code, not the authors' evaluator.
No license grant was found in that candidate package. This notice does not assign one.

The original migration preserved parsing, arithmetic, errors, and observation mapping.
It changed package imports, renamed `Network` to `Tensors`, and added documentation and typing updates.
The original 105 parser, tensor, and observation test cases moved into `tests/playground/`. Breakout observation tests remain in `tests/playground/breakout/`.
Later cleanup removed the unused float64 inference helper.
Its four tests now load edited weights and exercise float32 runtime inference.
Parser edits still apply in float64.

The existing `LICENSE-authors` and `LICENSE-MinAtar` files retain their original scope.
Their bytes match the pinned research resources. They do not establish a license for the local Spark code.

## Source mapping

Hashes below identify the original files before migration. Destination paths are relative to this package. Shared components now live one directory above it.

| Original Spark path | Ember path | Original SHA-256 |
| --- | --- | --- |
| `edits/__init__.py` | `../utils/__init__.py` | `a924373d646dfdf7b3a623e65a39b23cf4eee2befe2d6d68336b7f46941cce96` |
| `edits/apply.py` | `../utils/apply.py` | `81b22e7b7bcdfd7c62f4bf5def0c88403064db9c4ce6335084597d20620279e2` |
| `edits/grammar.py` | `../utils/grammar.py` | `d4eef8b1727d39c4cb74fac9a3f3e433d5e08591e7b5c2643898c38db54d8c08` |
| `edits/parse.py` | `../utils/parse.py` | `bfa12f8575be00d3b7a035dcf54c532e8f852615813e4bf80757c10ba9307f1d` |
| `network.py` | `../tensors.py` | `278364bc6140335dad20816cab5c09e0f8e347d4e206b4fa3d944028065b3022` |
| `observation.py` | `observation.py` | `420eb176780713f1450c2584d143cd8dbe62d64215befd809f702c4074a47246` |

The original `edits/apply.py`, `edits/grammar.py`, `edits/parse.py`, and `observation.py`
hashes match those recorded in `model-comparison-20261104.json`, preserved in the local runs archive outside this project.
Saved reports retain their historical Spark versions and source hashes.
New reports record Ember and omit Spark.

## Removed folder

The outer `breakout/` folder was archived outside this project before removal.
The local archive is stored under `~/Library/Application Support/Ember/archives/`, with a verified file-hash manifest.
It preserves the old prompts, scripts, tests, package metadata, and preparation notes.
The archived candidate prompt matches the pinned task prompt; the outer prompt was an earlier reconstruction.
The active package does not require that archive or a `vendor/spark` copy.
