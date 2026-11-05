"""Breakout source contracts, provenance, and pinned input verification."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Literal, Protocol
from urllib.request import urlopen

import numpy as np

from ember.base import Source, SourceRecord
from ember.models import Record

from .network import Weights

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


class Parity(Record):
    """Weight equality and evaluator provenance are independent claims."""

    source_commit: str
    applied: int | None
    skipped: list[str]
    tensors_equal: bool
    evaluator_matches_authors: Literal[False] = False
    reason: str
    community_pr: str
    differences: list[str]


class GameSource(Source, Protocol):
    """Add research download and weight-parity operations to verified inputs."""

    def fetch(self) -> SourceRecord: ...

    def parity(self) -> Parity: ...


class GameSourceFactory(Protocol):
    """A game-source class accepts one reference-cache location."""

    def __call__(self, *, root: Path) -> GameSource: ...


@dataclass(frozen=True)
class Sources:
    """Keep each game's resources together without overwriting different files."""

    root: Path
    commit: ClassVar[str] = COMMIT
    repository: ClassVar[str] = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs"
    directory: ClassVar[str] = "breakout"
    assets: ClassVar[Path] = Path(__file__).parent
    prompt_hash: ClassVar[str] = PROMPT_HASH
    instructions_hash: ClassVar[str] = INSTRUCTIONS_HASH
    files: ClassVar[dict[str, str]] = FILES

    @property
    def instructions_path(self) -> Path:
        """Keep provider commands pointed at the verified reference cache."""
        return self.root / self.directory / "instructions.md"

    def fetch(self) -> SourceRecord:
        hashes: dict[str, str] = {}
        urls: dict[str, str] = {}
        relative: str
        upstream: str
        for relative, upstream in self.files.items():
            url: str = f"{self.repository}/{self.commit}/{upstream}"
            local: str = f"{self.directory}/{relative}"
            with urlopen(url=url, timeout=30) as response:
                data: bytes = response.read()
            destination: Path = self.root / local
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError(f"Existing reference differs: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            hashes[local] = sha256(data).hexdigest()
            urls[local] = url
        record: SourceRecord = SourceRecord(commit=self.commit, hashes=hashes, urls=urls)
        self.verify_prompt()
        self.parity()
        (self.root / self.directory / "manifest.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    def verify_prompt(self) -> bytes:
        prompt: bytes = (self.root / self.directory / "prompt.txt").read_bytes()
        instructions: bytes = self.instructions_path.read_bytes()
        if sha256(prompt).hexdigest() != self.prompt_hash or sha256(instructions).hexdigest() != self.instructions_hash:
            raise ValueError("Prompt or instructions differ from the pinned authors' bytes")
        packaged_prompt: bytes = (self.assets / "prompt.txt").read_bytes()
        packaged_instructions: bytes = (self.assets / "instructions.md").read_bytes()
        if (
            sha256(packaged_prompt).hexdigest() != self.prompt_hash
            or sha256(packaged_instructions).hexdigest() != self.instructions_hash
        ):
            raise ValueError("Packaged prompt or instructions differ from the pinned authors' bytes")
        return prompt

    def parity(self) -> Parity:
        author: Path = self.root / self.directory / "runs/author"
        expected: dict[str, str] = {"answer.md": ANSWER_HASH, "hand.npz": NETWORK_HASH}
        name: str
        digest: str
        for name, digest in expected.items():
            if sha256((author / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Published reference changed: {name}")
        parsed: Weights = Weights.load(author / "answer.md")
        published: Weights = Weights.load(author / "hand.npz")
        return Parity(
            source_commit=self.commit,
            community_pr="https://github.com/RikiyaT/lm_understand-logs/pull/1",
            applied=parsed.applied,
            skipped=list(parsed.skipped),
            tensors_equal=all(np.array_equal(value, published.tensors[key]) for key, value in parsed.tensors.items()),
            reason="Original scorer not present in the pinned release; community PR #1 remains a reconstruction.",
            differences=[
                "Original sampler and seeding scheme are unknown.",
                "This diagnostic uses float32 inference and NumPy categorical softmax sampling.",
                "Environment and sampler use the same seed; last_action survives reset, as in official MinAtar.",
                "Ember's parser applies edits in float64, then converts to float32; general parser parity is unproven.",
                "CUDA and MPS can change floating-point results; no cross-device score equivalence is claimed.",
            ],
        )
