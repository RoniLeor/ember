"""The run panel infers stages from event files and leaves the run folder to the runner."""

import io
from pathlib import Path

import pytest
from rich.console import Console

from ember.console.events import Tail, Thinking
from ember.console.run import STAGES, Board, current, marker, render, watch
from ember.providers.records import RunRecord


def terminal() -> tuple[Console, io.StringIO]:
    stream: io.StringIO = io.StringIO()
    return Console(file=stream, force_terminal=True, color_system=None, width=100), stream


def text(renderable: object) -> str:
    console: Console
    stream: io.StringIO
    console, stream = terminal()
    console.print(renderable)
    return stream.getvalue()


def test_current_stage_follows_events() -> None:
    assert current(Thinking()) == 0
    assert current(Thinking(started=True)) == 1
    assert current(Thinking(started=True, answered=True)) == 2


def test_marker_shows_done_running_waiting_and_outcome() -> None:
    assert marker(index=0, active=1, elapsed=0, status=None).plain == "✓"
    assert marker(index=2, active=1, elapsed=0, status=None).plain == "○"
    assert marker(index=1, active=1, elapsed=0.1, status=None).plain == "⠙"
    assert marker(index=1, active=1, elapsed=0, status="completed").plain == "✓"
    assert marker(index=0, active=0, elapsed=0, status="blocked").plain == "✗"


def test_render_shows_thoughts_tokens_and_status() -> None:
    state: Thinking = Thinking(started=True, answered=True, thoughts=3, tokens=1234)
    state.recent.append("Designing weights")
    output: str = text(render(title="codex · m · strict", stages=STAGES, state=state, elapsed=75, status="failed"))
    assert "codex · m · strict" in output and "3 thoughts · 1,234 tokens" in output
    assert "› Designing weights" in output and "01:15" in output and "failed" in output and "✗" in output


def test_render_marks_unreached_stages_skipped_and_clamps_stage() -> None:
    blocked: str = text(render(title="t", stages=STAGES, state=Thinking(), elapsed=0, status="blocked"))
    assert blocked.count("skipped") == 1 and "0 thoughts" in blocked
    started: Thinking = Thinking(started=True)
    prepared: str = text(render(title="t", stages=STAGES[:1], state=started, elapsed=0, status="prepared"))
    assert "Preflight" in prepared and "Model thinking" not in prepared and "✓" in prepared


def test_board_reads_events_for_each_frame(tmp_path: Path) -> None:
    path: Path = tmp_path / "events.jsonl"
    board: Board = Board(title="t", stages=STAGES, tail=Tail(path=path))
    assert "○" in text(board())
    path.write_text('{"type": "item.completed", "item": {"type": "reasoning", "text": "**Idea**"}}\n')
    assert "› Idea" in text(board()) and "1 thought " in text(board())


def test_watch_without_terminal_only_runs_the_work(tmp_path: Path) -> None:
    stream: io.StringIO = io.StringIO()
    record: RunRecord = RunRecord(status="blocked", model="m")
    result: RunRecord = watch(
        console=Console(file=stream), output=tmp_path / "run", execute=True, title="t", work=lambda: record
    )
    assert result is record and stream.getvalue() == "" and not (tmp_path / "run").exists()


def test_watch_shows_final_status_and_leaves_folder_creation_to_work(tmp_path: Path) -> None:
    console: Console
    stream: io.StringIO
    console, stream = terminal()
    output: Path = tmp_path / "run"

    def work() -> RunRecord:
        output.mkdir(exist_ok=False)
        (output / "events.jsonl").write_text('{"type": "turn.completed", "usage": {"output_tokens": 9}}\n')
        return RunRecord(status="completed", model="m")

    result: RunRecord = watch(console=console, output=output, execute=True, title="codex", work=work)
    assert result.status == "completed"
    assert "ember run" in stream.getvalue() and "completed" in stream.getvalue() and "9 tokens" in stream.getvalue()


def test_watch_marks_failure_and_reraises_work_errors(tmp_path: Path) -> None:
    console: Console
    stream: io.StringIO
    console, stream = terminal()

    def work() -> RunRecord:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        watch(console=console, output=tmp_path / "run", execute=False, title="t", work=work)
    assert "failed" in stream.getvalue() and "✗" in stream.getvalue()
