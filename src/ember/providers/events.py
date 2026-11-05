"""Read Claude JSONL evidence without claiming independently verified tool availability."""

from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue

from ember.models import Audit, Record


class Block(Record):
    """Only text and reasoning blocks are accepted; tool blocks fail validation."""

    model_config = ConfigDict(extra="ignore", strict=True)
    type: Literal["text", "thinking", "redacted_thinking"]
    text: str = ""


class Message(Record):
    """All emitted blocks must belong to one assistant message."""

    model_config = ConfigDict(extra="ignore", strict=True)
    id: str = Field(min_length=1)
    role: Literal["assistant"]
    content: list[Block] = Field(min_length=1)


class Event(Record):
    """Known evidence fields; other provider metadata remains in the raw log."""

    model_config = ConfigDict(extra="ignore", strict=True)
    type: Literal["system", "assistant", "result", "rate_limit_event"]
    session_id: str = Field(min_length=1)
    subtype: str = ""
    event: str = ""
    message: Message | None = None
    tools: list[str] | None = None
    parent_tool_use_id: None = None
    agent_id: None = None
    deferred_tool_use: None = None
    error: str | None = None
    aborted: bool = False
    result: str = ""
    is_error: bool | None = None
    num_turns: int | None = None
    stop_reason: str | None = None
    permission_denials: list[JsonValue] | None = None
    usage: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True)
class Transcript:
    """Check one reply using declared settings and observed events only."""

    events: list[Event]

    @classmethod
    def read(cls, content: str) -> "Transcript":
        return cls(events=[Event.model_validate_json(line) for line in content.splitlines()])

    @property
    def answer(self) -> str:
        return self.events[-1].result if self.events and self.events[-1].type == "result" else ""

    def validate(self) -> list[str]:
        initial: list[Event] = [event for event in self.events if event.type == "system" and event.subtype == "init"]
        assistants: list[Event] = [event for event in self.events if event.type == "assistant"]
        results: list[Event] = [event for event in self.events if event.type == "result"]
        messages: list[Message] = [event.message for event in assistants if event.message is not None]
        checks: dict[str, bool] = {
            "Expected one init declaring empty tools": len(initial) == 1 and initial[0].tools == [],
            "Unknown system event": all(
                event.subtype in ("init", "hook_started", "hook_progress", "hook_response")
                or (event.subtype, event.event) == ("ui_invalidate", "ui.render")
                for event in self.events
                if event.type == "system"
            ),
            "Expected one session": len({event.session_id for event in self.events}) == 1,
            "Expected one assistant message": bool(messages)
            and len(messages) == len(assistants)
            and len({message.id for message in messages}) == 1,
            "Failed or aborted assistant": all(event.error is None and not event.aborted for event in assistants),
            "Invalid event order": [event.type for event in self.events if event.type in ("assistant", "result")]
            == ["assistant"] * len(assistants) + ["result"]
            and bool(initial)
            and bool(assistants)
            and self.events.index(initial[0]) < self.events.index(assistants[0]),
            "Permission denial or missing denial evidence": len(results) == 1 and results[0].permission_denials == [],
        }
        return [message for message, passed in checks.items() if not passed]

    def __call__(self, *, answer: str) -> Audit:
        results: list[Event] = [event for event in self.events if event.type == "result"]
        texts: list[str] = [
            block.text
            for event in self.events
            if event.type == "assistant" and event.message is not None
            for block in event.message.content
            if block.type == "text"
        ]
        finished: bool = len(results) == 1 and bool(self.events) and self.events[-1] == results[0]
        finished = finished and results[0].subtype == "success" and results[0].is_error is False
        finished = finished and results[0].num_turns == 1 and results[0].stop_reason == "end_turn"
        matched: bool = len(texts) == 1 and bool(answer.strip()) and texts[0] == answer == self.answer
        errors: list[str] = self.validate()
        if not finished:
            errors.append("Expected one successful completed turn")
        if not matched:
            errors.append("Expected one text block matching result and answer")
        usage: dict[str, int] = (
            {key: value for key, value in results[0].usage.items() if type(value) is int} if len(results) == 1 else {}
        )
        return Audit(
            passed=not errors,
            mode="comparison",
            tool_inventory_empty=False,
            prompt_context_matches=False,
            turn_completed=finished,
            answer_matches=matched,
            errors=errors,
            usage=usage,
        )
