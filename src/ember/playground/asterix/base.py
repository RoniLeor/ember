"""Pinned Asterix source identity; shared code verifies and loads the resources."""

from pathlib import Path
from typing import ClassVar

from ember.playground.base import PinnedSources

COMMIT: str = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
PROMPT_HASH: str = "9e8243d5545956e42b8bd130bceba6ed90ecb807768be3ae0f498f765197f5da"
INSTRUCTIONS_HASH: str = "b33fe0624137cd6f190e42a8659046baf16261f3c29cd6455734523ba473afeb"
NETWORK_HASH: str = "4c537e227824318c650d98e1e52de5a106071699213b7551e0810af71e9cdf1e"
ANSWER_HASH: str = "d4efb2cabc57190a8d900d66911570103223db91294825896aecc7e277a3cd16"
FILES: dict[str, str] = {
    "prompt.txt": "exp5_write_from_scratch/runs/asterix/message.txt",
    "instructions.md": "exp5_write_from_scratch/runs/asterix/instructions.md",
    "runs/author/answer.md": "exp5_write_from_scratch/runs/asterix/reply/answer.md",
    "runs/author/hand.npz": "exp5_write_from_scratch/networks/asterix/hand.npz",
    "research/codex.toml": "harness/codex_config.toml",
    "research/evaluation.json": "exp5_write_from_scratch/eval/minatar_eval.json",
    "LICENSE-authors": "LICENSE",
    "LICENSE-MinAtar": "exp5_write_from_scratch/LICENSE-MinAtar",
}


class Sources(PinnedSources):
    """Verified Asterix prompt, published answer, and network archive."""

    commit: ClassVar[str] = COMMIT
    directory: ClassVar[str] = "asterix"
    assets: ClassVar[Path] = Path(__file__).parent
    prompt_hash: ClassVar[str] = PROMPT_HASH
    instructions_hash: ClassVar[str] = INSTRUCTIONS_HASH
    answer_hash: ClassVar[str] = ANSWER_HASH
    network_hash: ClassVar[str] = NETWORK_HASH
    files: ClassVar[dict[str, str]] = FILES
