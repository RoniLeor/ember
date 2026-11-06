"""Asterix pinned bytes, download provenance, and published tensor equality."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from ember.base import SourceRecord
from ember.playground.asterix.base import FILES, Sources
from ember.playground.base import Parity

ROOT: Path = Path(__file__).resolve().parents[3] / "resources"


def test_published_prompt_manifest_and_tensor_parity() -> None:
    sources: Sources = Sources(root=ROOT)
    result: Parity = sources.parity()
    assert result.tensors_equal and result.applied == 55 and not result.skipped
    assert not result.evaluator_matches_authors
    assert sources.verify_prompt() == (sources.assets / "prompt.txt").read_bytes()
    record: SourceRecord = SourceRecord.model_validate_json((ROOT / "asterix/manifest.json").read_text())
    assert record.commit == sources.commit
    assert record.hashes == {
        f"asterix/{name}": sha256((ROOT / "asterix" / name).read_bytes()).hexdigest() for name in FILES
    }
    assert record.urls == {
        f"asterix/{name}": f"{sources.repository}/{sources.commit}/{upstream}" for name, upstream in FILES.items()
    }


@pytest.mark.parametrize("filename", ["prompt.txt", "instructions.md"])
def test_packaged_and_runtime_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    runtime: Path = tmp_path / "asterix"
    runtime.mkdir()
    assets: Path = tmp_path / "assets"
    assets.mkdir()
    name: str
    for name in ("prompt.txt", "instructions.md"):
        content: bytes = (Sources.assets / name).read_bytes()
        (runtime / name).write_bytes(content)
        (assets / name).write_bytes(content)
    monkeypatch.setattr(Sources, "assets", assets)
    sources: Sources = Sources(root=tmp_path)
    sources.verify_prompt()
    (runtime / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Prompt or instructions differ"):
        sources.verify_prompt()
    (runtime / filename).write_bytes((assets / filename).read_bytes())
    (assets / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Packaged prompt or instructions differ"):
        sources.verify_prompt()


def test_fetch_uses_asterix_urls_and_preserves_conflicting_files(tmp_path: Path) -> None:
    payloads: list[BytesIO] = [BytesIO((ROOT / "asterix" / name).read_bytes()) for name in FILES]
    with patch("ember.playground.base.urlopen", side_effect=payloads) as transport:
        record: SourceRecord = Sources(root=tmp_path).fetch()
    assert record == SourceRecord.model_validate_json((ROOT / "asterix/manifest.json").read_text())
    assert [call.kwargs["url"] for call in transport.call_args_list] == list(record.urls.values())
    (tmp_path / "asterix/prompt.txt").write_bytes(b"unique")
    with (
        patch("ember.playground.base.urlopen", return_value=BytesIO(b"different")),
        pytest.raises(ValueError, match="Existing reference differs"),
    ):
        Sources(root=tmp_path).fetch()
    assert (tmp_path / "asterix/prompt.txt").read_bytes() == b"unique"
