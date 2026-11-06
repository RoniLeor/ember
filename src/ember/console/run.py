"""Live run stages on the terminal: preflight, model thinking, and audit."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ember.providers.records import RunRecord

from .events import Tail, Thinking

STAGES: tuple[str, ...] = ("Preflight", "Model thinking", "Audit")
FRAMES: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
PASSED: tuple[str, ...] = ("completed", "prepared")
COLORS: dict[str, str] = {"completed": "green", "prepared": "cyan", "blocked": "yellow", "failed": "red"}


def current(state: Thinking) -> int:
    """Infer the active stage: events mean the model started; an answer means the audit runs."""
    if not state.started:
        return 0
    return 2 if state.answered else 1


def marker(*, index: int, active: int, elapsed: float, status: str | None) -> Text:
    """Show a done, running, failed, waiting or skipped stage."""
    if index < active:
        return Text("✓", style="green")
    if index > active:
        return Text("○", style="dim")
    if status is None:
        return Text(FRAMES[int(elapsed * 10) % len(FRAMES)], style="bold cyan")
    return Text("✓", style="green") if status in PASSED else Text("✗", style="bold red")


def render(
    *, title: str, stages: tuple[str, ...], state: Thinking, elapsed: float, status: str | None
) -> RenderableType:
    """Draw one frame of the run panel from explicit state."""
    active: int = min(current(state), len(stages) - 1)
    table: Table = Table.grid(padding=(0, 1))
    index: int
    name: str
    for index, name in enumerate(stages):
        detail: str = ""
        if index == 1:
            detail = f"{state.thoughts} thought{'' if state.thoughts == 1 else 's'}" + (
                f" · {state.tokens:,} tokens" if state.tokens else ""
            )
        elif index > active and status is not None:
            detail = "skipped"
        table.add_row(
            marker(index=index, active=active, elapsed=elapsed, status=status), name, Text(detail, style="dim")
        )
    thoughts: list[Text] = [
        Text(f"  › {headline}", style="italic dim", no_wrap=True, overflow="ellipsis") for headline in state.recent
    ]
    minutes: int
    seconds: int
    minutes, seconds = divmod(int(elapsed), 60)
    subtitle: str = f"{minutes:02d}:{seconds:02d}"
    if status is not None:
        subtitle += f" · [{COLORS.get(status, 'white')}]{status}[/]"
    return Panel(Group(table, *thoughts), title=f"[bold]ember run[/] · {title}", subtitle=subtitle)


@dataclass
class Board:
    """Live renderable state; the final status is set once the run returns."""

    title: str
    stages: tuple[str, ...]
    tail: Tail
    started: float = field(default_factory=time.monotonic)
    status: str | None = None

    def __call__(self) -> RenderableType:
        return render(
            title=self.title,
            stages=self.stages,
            state=self.tail(),
            elapsed=time.monotonic() - self.started,
            status=self.status,
        )


def watch(*, console: Console, output: Path, execute: bool, title: str, work: Callable[[], RunRecord]) -> RunRecord:
    """Run the work under a live panel on a terminal; otherwise run it without output."""
    if not console.is_terminal:
        return work()
    board: Board = Board(title=title, stages=STAGES if execute else STAGES[:1], tail=Tail(path=output / "events.jsonl"))
    with Live(
        get_renderable=board, console=console, refresh_per_second=8, redirect_stdout=False, redirect_stderr=False
    ):
        try:
            record: RunRecord = work()
        except BaseException:
            board.status = "failed"
            raise
        board.status = record.status
    return record
