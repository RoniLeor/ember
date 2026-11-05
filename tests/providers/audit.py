"""The audit rejects malformed, incomplete, multi-turn and tool-using runs."""

import json

import pytest

from ember.models import Audit
from ember.providers.audit import NOTICE, Event, EventAudit, PromptContext, ToolInventory, TurnState

EVENTS: list[dict[str, object]] = [
    {"type": "thread.started", "thread_id": "test"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"type": "reasoning", "text": "reason"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
    {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 2}},
]


def test_complete_answer_and_usage() -> None:
    content: str = "\n".join(json.dumps(event) for event in EVENTS)
    audit: EventAudit = EventAudit()
    result: Audit = audit(content=content, answer="answer\n", inventory_empty=True, context_matches=True)
    assert result.passed and result.usage["output_tokens"] == 2
    assert audit(content=content, answer="answer", inventory_empty=True, context_matches=True) == result
    assert not audit(content=content, answer="answer", inventory_empty=False, context_matches=True).passed
    assert not audit(content=content, answer="different", inventory_empty=True, context_matches=True).passed
    assert not audit(content=content, answer="answer", inventory_empty=True, context_matches=False).passed


@pytest.mark.parametrize("kind", ["command_execution", "mcp_tool_call", "web_search", "error", "unknown"])
def test_tool_and_unknown_item_rejected(kind: str) -> None:
    events: list[dict[str, object]] = [*EVENTS[:2], {"type": "item.completed", "item": {"type": kind}}, *EVENTS[2:]]
    result: Audit = EventAudit()(
        content="\n".join(json.dumps(event) for event in events),
        answer="answer",
        inventory_empty=True,
        context_matches=True,
    )
    assert not result.passed


@pytest.mark.parametrize("content", ["", "null", "[]", "{}", "not-json", "\n", '{"type":"thread.started"}\n\n'])
def test_invalid_event_data(content: str) -> None:
    assert not EventAudit()(content=content, answer="answer", inventory_empty=True, context_matches=True).passed


@pytest.mark.parametrize("kind", ["item.started", "item.updated", "turn.failed", "error", "thread.started"])
def test_unknown_event_rejected(kind: str) -> None:
    events: list[dict[str, object]] = [*EVENTS[:2], {"type": kind}, *EVENTS[2:]]
    assert not EventAudit()(
        content="\n".join(json.dumps(event) for event in events),
        answer="answer",
        inventory_empty=True,
        context_matches=True,
    ).passed


def test_notice_order_incomplete_duplicate_and_empty_answer() -> None:
    notice: dict[str, object] = {"type": "item.completed", "item": {"type": "error", "message": NOTICE}}
    valid: list[dict[str, object]] = [EVENTS[0], notice, *EVENTS[1:]]
    assert EventAudit()(
        content="\n".join(json.dumps(event) for event in valid),
        answer="answer",
        inventory_empty=True,
        context_matches=True,
    ).passed
    invalid: list[list[dict[str, object]]] = [
        EVENTS[:-1],
        EVENTS[1:],
        [*EVENTS, *EVENTS[1:]],
        [*EVENTS[:2], notice, *EVENTS[2:]],
        [*EVENTS[:-1], EVENTS[3], EVENTS[-1]],
        [EVENTS[0], EVENTS[3], *EVENTS[1:]],
        [*EVENTS, EVENTS[2]],
    ]
    sequence: list[dict[str, object]]
    for sequence in invalid:
        assert not EventAudit()(
            content="\n".join(json.dumps(event) for event in sequence),
            answer="answer",
            inventory_empty=True,
            context_matches=True,
        ).passed
    assert not EventAudit()(
        content="\n".join(json.dumps(event) for event in EVENTS),
        answer="",
        inventory_empty=True,
        context_matches=True,
    ).passed


def test_inventory_and_state_models() -> None:
    inventory: ToolInventory = ToolInventory()
    assert inventory('[{"type":"message"},{"type":"additional_tools","tools":[]}]')
    assert not inventory('[{"type":"message","role":"developer"}]')
    assert not inventory('[{"type":"additional_tools","tools":[]}]')
    assert not inventory(
        '[{"type":"message"},{"type":"additional_tools","tools":[]},{"type":"additional_tools","tools":[]}]'
    )
    assert not inventory('[{"type":"additional_tools","tools":[{"name":"exec"}]}]')
    assert not inventory('[{"type":"additional_tools"}]')
    with pytest.raises(ValueError):
        inventory("[]")
    with pytest.raises(ValueError):
        inventory('[{"type":"unknown"}]')
    with pytest.raises(ValueError):
        inventory("{}")
    state: TurnState = TurnState(errors=[])
    state(Event(type="item.completed"))
    assert state.errors


def test_prompt_context_rejects_extra_instructions_and_changed_prompt() -> None:
    messages: list[dict[str, object]] = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "instructions"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "prompt"}]},
        {"type": "additional_tools", "tools": []},
    ]
    check: PromptContext = PromptContext()
    assert check(content=json.dumps(messages), instructions="instructions", prompt="prompt")
    assert not check(content=json.dumps(messages), instructions="instructions", prompt="changed")
    assert not check(content=json.dumps(messages), instructions="changed", prompt="prompt")
    extra: dict[str, object] = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "# AGENTS.md instructions"}],
        "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["agents_md.instructions"]},
    }
    assert not check(
        content=json.dumps([messages[0], extra, *messages[1:]]), instructions="instructions", prompt="prompt"
    )
    assert not check(content=json.dumps(messages[::-1]), instructions="instructions", prompt="prompt")
    assert not check(content="[]", instructions="instructions", prompt="prompt")
