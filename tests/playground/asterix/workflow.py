"""Asterix uses the real shared CLI, evaluator, and lossless replay template."""

import json
from pathlib import Path

from ember import Asterix
from ember.cli import main
from ember.playground.models import Recording

ROOT: Path = Path(__file__).resolve().parents[3]


def test_author_policy_cli_evaluation(tmp_path: Path) -> None:
    network: Path = ROOT / "resources/asterix/runs/author/hand.npz"
    output: Path = tmp_path / "evaluation.json"
    assert main(["parity", "--experiment", "asterix", "--reference", str(ROOT / "resources")]) == 0
    assert (
        main(
            [
                "evaluate",
                "--experiment",
                "asterix",
                "--network",
                str(network),
                "--episodes",
                "2",
                "--max-steps",
                "3",
                "--reconstructed",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    data: dict[str, object] = json.loads(output.read_text())
    assert data["scores"] == [0.0, 0.0]
    assert data["evaluator_matches_authors"] is False


def test_author_policy_cli_replay(tmp_path: Path) -> None:
    network: Path = ROOT / "resources/asterix/runs/author/hand.npz"
    replay: Path = tmp_path / "asterix.html"
    assert (
        main(
            [
                "replay",
                "--experiment",
                "asterix",
                "--network",
                str(network),
                "--max-steps",
                "3",
                "--output",
                str(replay),
            ]
        )
        == 0
    )
    template: str = Asterix.template.read_text()
    assert template.count("__EPISODE__") == 1
    rendered: str = replay.read_text()
    assert "__EPISODE__" not in rendered and "ember / asterix" in rendered
    payload: str = rendered.split('<script id="episode" type="application/json">', 1)[1].split("</script>", 1)[0]
    recording: Recording = Recording.model_validate_json(payload)
    assert recording.settings.experiment == "asterix" and recording.steps == 3
    assert len(recording.frames) == 4 and recording.frames[0].layers is not None
    assert "frame.layers" in template and "mask & 1" in template
