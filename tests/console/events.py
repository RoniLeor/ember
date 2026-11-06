"""Display parsing reads Codex and Claude events leniently and only complete lines."""

import json
from pathlib import Path
from unittest.mock import patch

from ember.console.events import RECENT, Line, Tail, Thinking, headlines

CODEX: list[dict[str, object]] = [
    {"type": "thread.started", "thread_id": "t"},
    {"type": "item.started", "item": {"type": "reasoning", "text": "**Ignored until complete**"}},
    {"type": "item.completed", "item": {"type": "reasoning", "text": "**Designing weights**\n**Tracking ball**"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
    {"type": "turn.completed", "usage": {"output_tokens": 35823, "input_tokens": 1}},
]


def lines(events: list[dict[str, object]]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


def test_headlines_drop_bold_markers_and_blank_lines() -> None:
    assert headlines("**One**\n\n  **Two.**  \n***\nplain") == ["One", "Two.", "plain"]
    assert headlines("") == []


def test_blocks_accept_only_completed_items_and_assistant_messages() -> None:
    assert Line.model_validate({"type": "item.started", "item": {"type": "reasoning"}}).blocks() == []
    assert [part.type for part in Line.model_validate(CODEX[2]).blocks()] == ["reasoning"]
    claude: Line = Line.model_validate({"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}})
    assert [part.text for part in claude.blocks()] == ["x"]
    assert Line.model_validate({"type": "assistant"}).blocks() == []


def test_thinking_counts_codex_reasoning_and_usage() -> None:
    state: Thinking = Thinking()
    event: dict[str, object]
    for event in CODEX:
        state.add(Line.model_validate(event))
    assert state.thoughts == 1 and state.answered and state.tokens == 35823
    assert list(state.recent) == ["Designing weights", "Tracking ball"]


def test_thinking_reads_claude_thinking_and_result() -> None:
    state: Thinking = Thinking()
    state.add(
        Line.model_validate({"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "Plan"}]}})
    )
    assert state.thoughts == 1 and not state.answered
    state.add(Line.model_validate({"type": "result", "usage": {"output_tokens": 12}}))
    assert state.answered and state.tokens == 12 and list(state.recent) == ["Plan"]


def test_thinking_keeps_only_recent_headlines() -> None:
    state: Thinking = Thinking()
    text: str = "\n".join(f"**Step {index}**" for index in range(RECENT + 2))
    state.add(Line.model_validate({"type": "item.completed", "item": {"type": "reasoning", "text": text}}))
    assert len(state.recent) == RECENT and state.recent[-1] == f"Step {RECENT + 1}"


def test_tail_never_creates_a_missing_file_or_folder(tmp_path: Path) -> None:
    tail: Tail = Tail(path=tmp_path / "run" / "events.jsonl")
    state: Thinking = tail()
    assert not state.started and state.thoughts == 0
    assert not (tmp_path / "run").exists()


def test_tail_waits_for_complete_lines_and_skips_invalid_ones(tmp_path: Path) -> None:
    path: Path = tmp_path / "events.jsonl"
    complete: str = lines(CODEX[:3])
    path.write_text("not json\n[1]\n" + complete + '{"type": "turn.comp', encoding="utf-8")
    tail: Tail = Tail(path=path)
    state: Thinking = tail()
    assert state.started and state.thoughts == 1 and not state.answered
    with path.open("a", encoding="utf-8") as stream:
        stream.write('leted", "usage": {"output_tokens": 5}}\n')
    assert tail() is state
    assert state.answered and state.tokens == 5 and state.thoughts == 1
    assert tail().thoughts == 1


def test_tail_reads_claude_stream_json(tmp_path: Path) -> None:
    path: Path = tmp_path / "events.jsonl"
    thinking: dict[str, object] = {"type": "thinking", "thinking": "Plan the gates\nthen weights"}
    claude: list[dict[str, object]] = [
        {"type": "system", "session_id": "s"},
        {"type": "assistant", "message": {"content": [thinking, {"type": "text", "text": "edits"}]}},
        {"type": "result", "usage": {"output_tokens": 42}},
    ]
    path.write_text(lines(claude), encoding="utf-8")
    state: Thinking = Tail(path=path)()
    assert state.thoughts == 1 and state.answered and state.tokens == 42
    assert list(state.recent) == ["Plan the gates", "then weights"]


def test_tail_keeps_state_when_the_file_cannot_be_read(tmp_path: Path) -> None:
    path: Path = tmp_path / "events.jsonl"
    path.write_text(lines(CODEX), encoding="utf-8")
    tail: Tail = Tail(path=path)
    with patch.object(Path, "open", side_effect=PermissionError("denied")):
        assert tail() is tail.state
    assert tail.state.started and tail.offset == 0 and tail.state.thoughts == 0
    assert tail().thoughts == 1
