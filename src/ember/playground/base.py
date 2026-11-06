"""Shared game-source contracts and verified pinned research resource I/O."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Literal, Protocol
from urllib.request import urlopen

import numpy as np

from ember.base import Source, SourceRecord
from ember.models import Record

from .network import Weights


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
class PinnedSources:
    """Keep each game's resources together without overwriting different files."""

    root: Path
    commit: ClassVar[str]
    repository: ClassVar[str] = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs"
    directory: ClassVar[str]
    assets: ClassVar[Path]
    prompt_hash: ClassVar[str]
    instructions_hash: ClassVar[str]
    answer_hash: ClassVar[str]
    network_hash: ClassVar[str]
    files: ClassVar[dict[str, str]]

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
        expected: dict[str, str] = {"answer.md": self.answer_hash, "hand.npz": self.network_hash}
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
