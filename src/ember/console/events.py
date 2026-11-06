"""Lenient reading of model event logs for live display only, never for audit evidence."""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

RECENT: int = 6


class Part(BaseModel):
    """A Codex item or a Claude content block; unknown fields are ignored."""

    model_config = ConfigDict(extra="ignore")
    type: str = ""
    text: str = ""
    thinking: str = ""


class Message(BaseModel):
    """The content blocks of one Claude assistant message."""

    model_config = ConfigDict(extra="ignore")
    content: list[Part] = Field(default_factory=list)


class Usage(BaseModel):
    """Token counts that Codex and Claude report when the answer is complete."""

    model_config = ConfigDict(extra="ignore")
    output_tokens: int = 0


class Line(BaseModel):
    """One JSONL event from Codex (`item`) or Claude (`message`)."""

    model_config = ConfigDict(extra="ignore")
    type: str = ""
    item: Part | None = None
    message: Message | None = None
    usage: Usage | None = None

    def blocks(self) -> list[Part]:
        """Return the completed parts that this event carries."""
        if self.type == "item.completed" and self.item is not None:
            return [self.item]
        if self.type == "assistant" and self.message is not None:
            return self.message.content
        return []


def headlines(text: str) -> list[str]:
    """Return the non-empty lines of a thought without Markdown bold markers."""
    cleaned: list[str] = [line.strip().strip("*").strip() for line in text.splitlines()]
    return [line for line in cleaned if line]


@dataclass
class Thinking:
    """Display state built from the events seen so far."""

    started: bool = False
    answered: bool = False
    thoughts: int = 0
    tokens: int = 0
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=RECENT))

    def add(self, line: Line) -> None:
        """Count thoughts and record the final answer event."""
        part: Part
        for part in line.blocks():
            if part.type in ("reasoning", "thinking"):
                self.thoughts += 1
                self.recent.extend(headlines(part.text or part.thinking))
        if line.type in ("turn.completed", "result"):
            self.answered = True
            if line.usage is not None:
                self.tokens = line.usage.output_tokens


@dataclass
class Tail:
    """Read only complete lines appended since the last call; never create the file."""

    path: Path
    state: Thinking = field(default_factory=Thinking)
    offset: int = 0

    def __call__(self) -> Thinking:
        if not self.path.is_file():
            return self.state
        self.state.started = True
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk: bytes = stream.read()
        except OSError:
            return self.state
        end: int = chunk.rfind(b"\n") + 1
        self.offset += end
        raw: bytes
        for raw in chunk[:end].splitlines():
            try:
                self.state.add(Line.model_validate_json(raw))
            except ValidationError:
                continue
        return self.state
