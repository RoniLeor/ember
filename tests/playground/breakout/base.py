"""Pinned downloads preserve changed files and verify input bytes."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ember.base import SourceRecord
from ember.playground.breakout.base import FILES, Parity, Sources


def test_fetch_pins_urls_and_keeps_existing_data(tmp_path: Path) -> None:
    sources: Sources = Sources(root=tmp_path)
    response: Mock = Mock(side_effect=[BytesIO(b"data") for _ in FILES])
    with (
        patch("ember.playground.breakout.base.urlopen", response),
        patch.object(Sources, "verify_prompt", return_value=b"data"),
        patch.object(Sources, "parity"),
    ):
        record: SourceRecord = sources.fetch()
    assert len(record.hashes) == len(FILES) and (tmp_path / "breakout/manifest.json").is_file()
    assert set(FILES) == {
        "prompt.txt",
        "instructions.md",
        "runs/author/answer.md",
        "runs/author/hand.npz",
        "research/codex.toml",
        "research/evaluation.json",
        "LICENSE-authors",
        "LICENSE-MinAtar",
    }
    assert set(record.hashes) == set(record.urls) == {f"breakout/{name}" for name in FILES}
    assert {path.name for path in tmp_path.iterdir()} == {"breakout"}
    assert record.hashes == {name: sha256((tmp_path / name).read_bytes()).hexdigest() for name in record.hashes}
    assert SourceRecord.model_validate_json((tmp_path / "breakout/manifest.json").read_text()) == record
    assert all(record.commit in url for url in record.urls.values())
    response.assert_called()
    (tmp_path / "breakout/prompt.txt").write_bytes(b"unique existing content")
    with (
        patch("ember.playground.breakout.base.urlopen", return_value=BytesIO(b"different")),
        pytest.raises(ValueError, match="differs"),
    ):
        sources.fetch()
    assert (tmp_path / "breakout/prompt.txt").read_bytes() == b"unique existing content"


def test_prompt_hash_validation(tmp_path: Path) -> None:
    (tmp_path / "breakout").mkdir()
    (tmp_path / "breakout/prompt.txt").write_bytes(b"prompt")
    (tmp_path / "breakout/instructions.md").write_bytes(b"instructions")
    sources: Sources = Sources(root=tmp_path)
    with pytest.raises(ValueError, match="differ"):
        sources.verify_prompt()
    with patch("ember.playground.breakout.base.sha256") as digest:
        digest.return_value.hexdigest.side_effect = [
            "766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876",
            "b33fe0624137cd6f190e42a8659046baf16261f3c29cd6455734523ba473afeb",
            "766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876",
            "b33fe0624137cd6f190e42a8659046baf16261f3c29cd6455734523ba473afeb",
        ]
        assert sources.verify_prompt() == b"prompt"


@pytest.mark.parametrize("filename", ["answer.md", "hand.npz"])
def test_parity_requires_published_bytes(tmp_path: Path, filename: str) -> None:
    author: Path = tmp_path / "breakout/runs/author"
    author.mkdir(parents=True)
    reference: Path = Path(__file__).resolve().parents[3] / "resources/breakout/runs/author"
    name: str
    for name in ("answer.md", "hand.npz"):
        (author / name).write_bytes((reference / name).read_bytes())
    (author / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match=f"Published reference changed: {filename}"):
        Sources(root=tmp_path).parity()


def test_runtime_inputs_do_not_require_saved_results_or_research(tmp_path: Path) -> None:
    runtime: Path = tmp_path / "breakout"
    runtime.mkdir()
    name: str
    for name in ("prompt.txt", "instructions.md"):
        (runtime / name).write_bytes((Sources.assets / name).read_bytes())
    sources: Sources = Sources(root=tmp_path)
    assert sources.verify_prompt() == (Sources.assets / "prompt.txt").read_bytes()
    assert sources.instructions_path.read_bytes() == (Sources.assets / "instructions.md").read_bytes()
    assert not (runtime / "runs").exists()
    assert not (runtime / "research").exists()


def test_actual_published_reference_parity() -> None:
    sources: Sources = Sources(root=Path(__file__).resolve().parents[3] / "resources")
    result: Parity = sources.parity()
    assert result.tensors_equal and not result.evaluator_matches_authors
    assert result.applied == 92 and not result.skipped
    assert sources.verify_prompt().startswith(b"YOUR TASK: write, from scratch,")
    record: SourceRecord = SourceRecord.model_validate_json((sources.root / "breakout/manifest.json").read_text())
    assert record.commit == sources.commit
    assert record.hashes == {
        f"breakout/{name}": sha256((sources.root / "breakout" / name).read_bytes()).hexdigest() for name in FILES
    }
    assert record.urls == {
        f"breakout/{name}": f"{sources.repository}/{sources.commit}/{upstream}" for name, upstream in FILES.items()
    }


def test_failed_fetch_verification_does_not_write_manifest(tmp_path: Path) -> None:
    with (
        patch("ember.playground.breakout.base.urlopen", side_effect=[BytesIO(b"data") for _ in FILES]),
        pytest.raises(ValueError, match="differ"),
    ):
        Sources(root=tmp_path).fetch()
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "breakout/manifest.json").exists()


@pytest.mark.parametrize("filename", ["prompt.txt", "instructions.md"])
def test_packaged_inputs_match_reference_and_reject_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    reference: Path = Path(__file__).resolve().parents[3] / "resources"
    sources: Sources = Sources(root=reference)
    assert sources.instructions_path == reference / "breakout/instructions.md"
    name: str
    for name in ("prompt.txt", "instructions.md"):
        content: bytes = (reference / "breakout" / name).read_bytes()
        assert (sources.assets / name).read_bytes() == content
        (tmp_path / name).write_bytes(content)
    monkeypatch.setattr(Sources, "assets", tmp_path)
    assert sources.verify_prompt() == (reference / "breakout/prompt.txt").read_bytes()
    (tmp_path / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Packaged prompt or instructions differ"):
        sources.verify_prompt()
