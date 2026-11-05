"""Fail-closed event audit and independent model-visible tool inventory checks."""

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from ember.base import Source
from ember.models import Audit, RunMode

NOTICE: str = (
    "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; "
    "enable `features.code_mode_host` and install `codex-code-mode-host`."
)


class Item(BaseModel):
    """Known event fields; provider metadata is retained by the raw log."""

    model_config = ConfigDict(extra="ignore", strict=True)
    type: str
    text: str = ""
    message: str = ""


class Event(Item):
    """One JSONL event from codex exec."""

    item: Item | None = None
    usage: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True)
class ToolInventory:
    """Check Codex's rendered prompt, not the model's claim about available tools."""

    def __call__(self, content: str) -> bool:
        items: list[dict[str, JsonValue]] = TypeAdapter(list[dict[str, JsonValue]]).validate_json(content)
        if not items:
            raise ValueError("Empty prompt inventory is not evidence")
        inventories: int = 0
        messages: int = 0
        item: dict[str, JsonValue]
        for item in items:
            kind: JsonValue = item.get("type")
            if kind == "additional_tools":
                if item.get("tools") != []:
                    return False
                inventories += 1
            elif kind == "message":
                messages += 1
            else:
                raise ValueError(f"Unknown prompt inventory item: {kind}")
        return inventories == 1 and messages > 0


@dataclass(frozen=True)
class PromptContext:
    """Require only the pinned instructions and prompt, in their original order."""

    def __call__(self, *, content: str, instructions: str, prompt: str) -> bool:
        items: list[dict[str, JsonValue]] = TypeAdapter(list[dict[str, JsonValue]]).validate_json(content)
        actual: list[dict[str, JsonValue]] = [
            {"role": item.get("role"), "content": item.get("content")}
            for item in items
            if item.get("type") == "message"
        ]
        expected: list[dict[str, JsonValue]] = [
            {"role": "developer", "content": [{"type": "input_text", "text": instructions}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ]
        return actual == expected


@dataclass(frozen=True)
class EventAudit:
    """Check one complete turn, one answer, and strict-mode prompt and tool gates."""

    def __call__(
        self, *, content: str, answer: str, inventory_empty: bool, context_matches: bool, mode: RunMode = "strict"
    ) -> Audit:
        errors: list[str] = []
        events: list[Event] = []
        try:
            events = [Event.model_validate_json(line) for line in content.splitlines()]
        except ValidationError as error:
            errors.append(f"Invalid event: {error}")
        state: TurnState = TurnState(errors=errors)
        if not events or events[0].type != "thread.started":
            errors.append("Missing initial thread.started")
        if mode == "strict" and not inventory_empty:
            errors.append("Empty model tool inventory was not established")
        if mode == "strict" and not context_matches:
            errors.append("Model context differs from the pinned instructions and prompt")
        event: Event
        for event in events[1:]:
            state(event)
        finished: bool = bool(events) and events[-1].type == "turn.completed" and state.started == state.completed == 1
        matched: bool = (
            len(state.answers) == 1 and bool(answer.strip()) and state.answers[0].rstrip("\n") == answer.rstrip("\n")
        )
        if not finished:
            errors.append("Expected exactly one complete turn")
        if not matched:
            errors.append("Expected one nonempty answer matching the answer file")
        return Audit(
            passed=not errors,
            mode=mode,
            tool_inventory_empty=inventory_empty,
            prompt_context_matches=context_matches,
            turn_completed=finished,
            answer_matches=matched,
            errors=errors,
            usage=state.usage,
        )


@dataclass(frozen=True)
class RunAudit:
    """Recheck a saved run against pinned inputs, without trusting its prior result."""

    sources: Source

    def __call__(self, *, root: Path) -> Audit:
        prompt: bytes = self.sources.verify_prompt()
        if (root / "prompt.txt").read_bytes() != prompt:
            raise ValueError("Saved run prompt differs from the pinned research prompt")
        inventory: str = (root / "inventory.json").read_text(encoding="utf-8")
        return EventAudit()(
            content=(root / "events.jsonl").read_text(encoding="utf-8"),
            answer=(root / "answer.md").read_text(encoding="utf-8"),
            inventory_empty=ToolInventory()(inventory),
            context_matches=PromptContext()(
                content=inventory,
                instructions=self.sources.instructions_path.read_text(encoding="utf-8").rstrip("\n"),
                prompt=prompt.decode("utf-8"),
            ),
        )


@dataclass
class TurnState:
    """State transitions for a single fresh turn; allocated separately for each audit."""

    errors: list[str]
    started: int = 0
    completed: int = 0
    active: bool = False
    answers: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def __call__(self, event: Event) -> None:
        if event.type == "turn.started" and not self.active and self.started == 0:
            self.started += 1
            self.active = True
        elif event.type == "turn.completed" and self.active and self.completed == 0:
            self.completed += 1
            self.active = False
            self.usage = event.usage
        elif event.type == "item.completed" and event.item is not None:
            item: Item = event.item
            if item.type == "error" and self.started == 0 and item.message == NOTICE:
                return
            if not self.active or item.type not in ("reasoning", "agent_message"):
                self.errors.append(f"Unexpected item: {item.type}")
            elif item.type == "agent_message":
                self.answers.append(item.text)
        else:
            self.errors.append(f"Unexpected event: {event.type}")
