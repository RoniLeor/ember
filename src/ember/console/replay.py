"""A terminal progress bar for one recorded replay episode."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


@dataclass(frozen=True)
class ReplayProgress:
    """Update the bar with each played step; this is a Player observer."""

    progress: Progress
    task: TaskID

    def __call__(self, *, step: int, score: float) -> None:
        self.progress.update(task_id=self.task, completed=step, score=score)


@contextmanager
def replaying(*, console: Console, experiment: str, max_steps: int) -> Generator[ReplayProgress, None, None]:
    """Show steps and score on a terminal; fill the bar when the episode ends early."""
    progress: Progress = Progress(
        SpinnerColumn(finished_text="[green]✓[/]"),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("steps · score [cyan]{task.fields[score]:g}[/]"),
        TimeElapsedColumn(),
        console=console,
        disable=not console.is_terminal,
        redirect_stdout=False,
        redirect_stderr=False,
    )
    task: TaskID = progress.add_task(description=f"Playing {experiment}", total=max_steps, score=0.0)
    with progress:
        yield ReplayProgress(progress=progress, task=task)
        steps: float = progress.tasks[0].completed
        progress.update(task_id=task, total=steps, description=f"Recorded {experiment}")
