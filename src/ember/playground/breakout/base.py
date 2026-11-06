"""Pinned Breakout source identity; shared code verifies and loads the resources."""

from pathlib import Path
from typing import ClassVar

from ember.playground.base import PinnedSources

COMMIT: str = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
PROMPT_HASH: str = "766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876"
INSTRUCTIONS_HASH: str = "b33fe0624137cd6f190e42a8659046baf16261f3c29cd6455734523ba473afeb"
NETWORK_HASH: str = "6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a"
ANSWER_HASH: str = "a30bdda1acef79b507498edcef7e48d4fff16aa85783741925da9d5d4f42faf0"
FILES: dict[str, str] = {
    "prompt.txt": "exp5_write_from_scratch/runs/breakout/message.txt",
    "instructions.md": "exp5_write_from_scratch/runs/breakout/instructions.md",
    "runs/author/answer.md": "exp5_write_from_scratch/runs/breakout/reply/answer.md",
    "runs/author/hand.npz": "exp5_write_from_scratch/networks/breakout/hand.npz",
    "research/codex.toml": "harness/codex_config.toml",
    "research/evaluation.json": "exp5_write_from_scratch/eval/minatar_eval.json",
    "LICENSE-authors": "LICENSE",
    "LICENSE-MinAtar": "exp5_write_from_scratch/LICENSE-MinAtar",
}


class Sources(PinnedSources):
    """Verified Breakout prompt, published answer, and network archive."""

    commit: ClassVar[str] = COMMIT
    directory: ClassVar[str] = "breakout"
    assets: ClassVar[Path] = Path(__file__).parent
    prompt_hash: ClassVar[str] = PROMPT_HASH
    instructions_hash: ClassVar[str] = INSTRUCTIONS_HASH
    answer_hash: ClassVar[str] = ANSWER_HASH
    network_hash: ClassVar[str] = NETWORK_HASH
    files: ClassVar[dict[str, str]] = FILES
