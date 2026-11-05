"""Verify one Codex inference and reject observed tool-call attempts."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ember.models import Record

from .audit import PromptContext

PINNED_MESSAGES: int = 2
CALL_EVENTS: tuple[str, ...] = (
    "tool_call",
    "tool_output",
    "function_call",
    "function_output",
    "custom_tool",
    "command_execution",
    "exec_command",
    "web_search",
    "shell_command",
    "mcp_call",
    "computer_call",
    "local_shell_call",
    "file_search_call",
    "image_generation_call",
)


class TraceData(BaseModel):
    """Validate evidence fields; raw files retain unrelated trace metadata."""

    model_config = ConfigDict(extra="ignore", strict=True)


class Text(TraceData):
    """Retain text metadata when comparing the exact model context."""

    model_config = ConfigDict(extra="allow", strict=True)
    type: Literal["input_text", "output_text"]
    text: str


class Tool(TraceData):
    """Advertised tool definitions can contain nested namespaces."""

    type: str
    name: str = ""
    tools: list["Tool"] = Field(default_factory=list)


class InputItem(TraceData):
    """Fresh model inputs contain messages and one tool declaration."""

    model_config = ConfigDict(extra="allow", strict=True)
    type: Literal["message", "additional_tools"]
    role: str = ""
    content: list[Text] = Field(default_factory=list)
    tools: list[Tool] | None = None


class Reference(TraceData):
    """A payload reference must remain inside its trace directory."""

    path: str


class Item(TraceData):
    """An observed history item exposes its operation type."""

    type: str


class Payload(TraceData):
    """Fields needed to pair inference evidence and inspect observed calls."""

    type: str
    event_type: str = ""
    item_type: str = ""
    item: Item | None = None
    inference_call_id: str = ""
    response_id: str = ""
    request_payload: Reference | None = None
    response_payload: Reference | None = None
    event_payload: Reference | None = None


class Entry(TraceData):
    """A trace entry wraps an observed payload."""

    payload: Payload


class Manifest(TraceData):
    """Locate the event log within the trace directory."""

    schema_version: Literal[2]
    raw_event_log: Literal["trace.jsonl"]


class Request(TraceData):
    """The actual model and rendered input sent for inference."""

    model: str
    input: list[InputItem]


class OutputItem(TraceData):
    """Only reasoning and assistant text are accepted as inference output."""

    type: Literal["reasoning", "message"]
    role: str = ""
    content: list[Text] = Field(default_factory=list)


class Response(TraceData):
    """Tool-call items fail validation regardless of execution status."""

    response_id: str
    output_items: list[OutputItem] = Field(min_length=1)


class TraceResult(Record):
    """Separate advertised tools, verified usage, and incomplete evidence."""

    advertised_tools: list[str] = Field(default_factory=list)
    tool_inventory_empty: bool = False
    prompt_context_matches: bool = False
    extra_message_count: int | None = None
    inference_count: int | None = None
    tool_call_count: int | None = None
    errors: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Inference:
    """One validated pair of request and response references."""

    request: Reference
    response: Reference
    response_id: str


@dataclass(frozen=True)
class TraceAudit:
    """Check actual inference output without counting advertised tools as calls."""

    def __call__(self, *, root: Path, model: str, prompt: bytes, instructions: str, answer: str) -> TraceResult:
        result: TraceResult = TraceResult()
        try:
            manifests: list[Path] = list(root.rglob("manifest.json"))
            if len(manifests) != 1:
                raise ValueError("Expected exactly one Codex trace manifest")
            manifest: Manifest = Manifest.model_validate_json(
                self.read(root=root, path=str(manifests[0].relative_to(root)))
            )
            directory: Path = manifests[0].parent
            entries: list[Entry] = [
                Entry.model_validate_json(line)
                for line in self.read(root=directory, path=manifest.raw_event_log).splitlines()
            ]
            inference: Inference = self.inference(entries)
            self.events(entries=entries, root=directory)
            request: Request = Request.model_validate_json(self.read(root=directory, path=inference.request.path))
            result = self.inputs(items=request.input, prompt=prompt.decode("utf-8"), instructions=instructions)
            if request.model != model:
                raise ValueError("Trace model differs from the run record")
            response: Response = Response.model_validate_json(self.read(root=directory, path=inference.response.path))
            if response.response_id != inference.response_id:
                raise ValueError("Inference response IDs do not match")
            self.answer(response=response, answer=answer)
            return result.model_copy(update={"inference_count": 1, "tool_call_count": 0})
        except (OSError, ValueError) as error:
            return result.model_copy(update={"errors": [str(error)]})

    @staticmethod
    def inference(entries: list[Entry]) -> Inference:
        """Require one ordered inference pair with matching call identity."""
        payloads: list[Payload] = [entry.payload for entry in entries]
        starts: list[Payload] = [payload for payload in payloads if payload.type == "inference_started"]
        ends: list[Payload] = [payload for payload in payloads if payload.type == "inference_completed"]
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError("Expected exactly one started and completed inference")
        start: Payload = starts[0]
        end: Payload = ends[0]
        if not start.inference_call_id or start.inference_call_id != end.inference_call_id:
            raise ValueError("Inference call IDs do not match")
        if payloads.index(start) >= payloads.index(end):
            raise ValueError("Inference completion precedes its request")
        if start.request_payload is None or end.response_payload is None or not end.response_id:
            raise ValueError("Missing inference request, response, or response ID")
        return Inference(request=start.request_payload, response=end.response_payload, response_id=end.response_id)

    def events(self, *, entries: list[Entry], root: Path) -> None:
        """Reject explicit call attempts in trace, protocol, and history events."""
        entry: Entry
        for entry in entries:
            payload: Payload = entry.payload
            observed: list[Payload] = [payload]
            if payload.event_payload is not None:
                observed.append(Payload.model_validate_json(self.read(root=root, path=payload.event_payload.path)))
            item: Payload
            for item in observed:
                kinds: list[str] = [item.type, item.event_type, item.item_type]
                if item.item is not None:
                    kinds.append(item.item.type)
                if any(marker in kind.lower() for kind in kinds for marker in CALL_EVENTS):
                    raise ValueError(f"Observed tool-call event: {', '.join(kinds)}")

    def inputs(self, *, items: list[InputItem], prompt: str, instructions: str) -> TraceResult:
        """Require pinned inputs while recording additional context and tool names."""
        declarations: list[InputItem] = [item for item in items if item.type == "additional_tools"]
        messages: list[InputItem] = [item for item in items if item.type == "message"]
        if len(declarations) != 1 or declarations[0].tools is None or not messages:
            raise ValueError("Expected one tool declaration and prompt messages")
        if any(
            item.role not in ("developer", "user", "assistant")
            or not item.content
            or any(block.type != "input_text" for block in item.content)
            for item in messages
        ):
            raise ValueError("Invalid prompt message")
        if sum(block.text == prompt for item in messages if item.role == "user" for block in item.content) != 1:
            raise ValueError("Expected the exact user prompt once")
        if not instructions or not any(
            instructions == block.text for item in messages if item.role == "developer" for block in item.content
        ):
            raise ValueError("Pinned instructions are missing from developer input")
        return TraceResult(
            advertised_tools=self.tools(declarations[0].tools),
            tool_inventory_empty=not declarations[0].tools,
            prompt_context_matches=PromptContext()(
                content=json.dumps([item.model_dump(mode="json", exclude_unset=True) for item in items]),
                instructions=instructions,
                prompt=prompt,
            ),
            extra_message_count=len(messages) - PINNED_MESSAGES,
        )

    def tools(self, tools: list[Tool], prefix: str = "") -> list[str]:
        """Flatten advertised namespaces without treating availability as usage."""
        names: list[str] = []
        tool: Tool
        for tool in tools:
            name: str = ".".join(part for part in (prefix, tool.name or tool.type) if part)
            names.extend(self.tools(tool.tools, prefix=name) if tool.tools else [name])
        return sorted(set(names))

    @staticmethod
    def answer(*, response: Response, answer: str) -> None:
        """Match one final assistant output against the saved answer."""
        messages: list[OutputItem] = [item for item in response.output_items if item.type == "message"]
        if len(messages) != 1 or response.output_items[-1].type != "message":
            raise ValueError("Expected exactly one final assistant message")
        message: OutputItem = messages[0]
        if (
            message.role != "assistant"
            or not message.content
            or any(block.type != "output_text" for block in message.content)
        ):
            raise ValueError("Expected assistant output text")
        text: str = "".join(block.text for block in message.content)
        if not answer.strip() or text.rstrip("\n") != answer.rstrip("\n"):
            raise ValueError("Trace answer differs from the answer file")

    @staticmethod
    def read(*, root: Path, path: str) -> str:
        """Read only evidence contained within the selected trace directory."""
        candidate: Path = (root / path).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise ValueError("Trace payload path escapes its directory")
        return candidate.read_text(encoding="utf-8")
