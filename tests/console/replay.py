"""The replay bar follows observed steps and fills when the episode ends."""

import io

import pytest
from rich.console import Console
from rich.progress import Task

from ember.console.replay import ReplayProgress, replaying


def test_observer_updates_steps_and_score() -> None:
    stream: io.StringIO = io.StringIO()
    console: Console = Console(file=stream, force_terminal=True, color_system=None, width=120)
    with replaying(console=console, experiment="breakout", max_steps=100) as progress:
        progress(step=7, score=3.0)
        task: Task = progress.progress.tasks[0]
        assert task.completed == 7 and task.fields["score"] == 3.0 and task.total == 100
    assert task.total == 7 and task.description == "Recorded breakout"
    assert "7/7" in stream.getvalue() and "score 3" in stream.getvalue()


def test_failure_keeps_the_partial_bar() -> None:
    console: Console = Console(file=io.StringIO(), force_terminal=True, color_system=None)
    progress: ReplayProgress
    with pytest.raises(ValueError), replaying(console=console, experiment="seaquest", max_steps=10) as progress:
        progress(step=2, score=0.0)
        raise ValueError("bad network")
    task: Task = progress.progress.tasks[0]
    assert task.total == 10 and task.description == "Playing seaquest"


def test_non_terminal_console_prints_nothing() -> None:
    stream: io.StringIO = io.StringIO()
    with replaying(console=Console(file=stream), experiment="breakout", max_steps=5) as progress:
        progress(step=5, score=1.0)
    assert stream.getvalue() == ""
