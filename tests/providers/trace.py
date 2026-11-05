"""Synthetic Codex traces distinguish advertised tools from actual call attempts."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from ember.providers.trace import TraceAudit, TraceResult

INPUTS: list[JsonValue] = [
    {
        "type": "additional_tools",
        "tools": [{"type": "namespace", "name": "functions", "tools": [{"type": "custom", "name": "exec"}]}],
    },
    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "runtime context"}]},
    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "instructions"}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "prompt"}]},
]


def write_trace(root: Path) -> None:
    """Write synthetic raw evidence for trace and transport boundary tests."""
    directory: Path = root / "trace-one"
    payloads: Path = directory / "payloads"
    payloads.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"schema_version": 2, "raw_event_log": "trace.jsonl"}))
    (payloads / "request.json").write_text(json.dumps({"model": "gpt-6-astra", "input": INPUTS}))
    (payloads / "response.json").write_text(
        json.dumps(
            {
                "response_id": "response-one",
                "output_items": [
                    {"type": "reasoning", "summary": []},
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
                ],
            }
        )
    )
    entries: list[dict[str, JsonValue]] = [
        {"payload": {"type": "rollout_started"}},
        {
            "payload": {
                "type": "inference_started",
                "inference_call_id": "call-one",
                "request_payload": {"path": "payloads/request.json"},
            }
        },
        {"payload": {"type": "history_item_accepted", "item_id": "message-one"}},
        {
            "payload": {
                "type": "inference_completed",
                "inference_call_id": "call-one",
                "response_id": "response-one",
                "response_payload": {"path": "payloads/response.json"},
            }
        },
        {"payload": {"type": "rollout_ended"}},
    ]
    (directory / "trace.jsonl").write_text("\n".join(json.dumps(entry) for entry in entries))


@pytest.fixture
def trace(tmp_path: Path) -> Path:
    write_trace(tmp_path)
    return tmp_path


def test_one_inference_records_advertised_tools_without_calls(trace: Path) -> None:
    result: TraceResult = TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer\n"
    )
    assert not result.errors and result.inference_count == 1 and result.tool_call_count == 0
    assert result.advertised_tools == ["functions.exec"]
    assert not result.tool_inventory_empty and not result.prompt_context_matches
    assert result.extra_message_count == 1


def test_exact_context_and_empty_tools_are_reported_truthfully(trace: Path) -> None:
    path: Path = trace / "trace-one/payloads/request.json"
    path.write_text(
        json.dumps(
            {
                "model": "gpt-6-astra",
                "input": [
                    {"type": "additional_tools", "tools": []},
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "instructions"}],
                    },
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "prompt"}]},
                ],
            }
        )
    )
    result: TraceResult = TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer"
    )
    assert not result.errors and result.tool_inventory_empty and result.prompt_context_matches
    assert not result.advertised_tools


@pytest.mark.parametrize(
    "kind", ["function_call", "custom_tool_call", "web_search_call", "mcp_call", "computer_call", "unknown"]
)
def test_every_unaccepted_output_item_rejects_completion(trace: Path, kind: str) -> None:
    path: Path = trace / "trace-one/payloads/response.json"
    path.write_text(path.read_text().replace('"type": "reasoning"', f'"type": "{kind}", "status": "failed"'))
    result: TraceResult = TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer"
    )
    assert result.errors and kind in result.errors[0]
    assert result.tool_call_count is None and result.inference_count is None


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "tool_call_started"},
        {"type": "protocol_event_observed", "event_type": "exec_command_begin"},
        {"type": "history_item_accepted", "item_type": "custom_tool_call"},
        {"type": "history_item_accepted", "item": {"type": "function_call"}},
        {"type": "history_item_accepted", "item": {"type": "computer_call"}},
        {"type": "history_item_accepted", "item_type": "local_shell_call"},
        {"type": "protocol_event_observed", "event_type": "file_search_call"},
        {"type": "image_generation_call"},
    ],
)
def test_call_attempts_in_events_reject_completion(trace: Path, payload: dict[str, JsonValue]) -> None:
    path: Path = trace / "trace-one/trace.jsonl"
    path.write_text(path.read_text() + "\n" + json.dumps({"payload": payload}))
    result: TraceResult = TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer"
    )
    assert result.errors and "tool-call event" in result.errors[0] and result.tool_call_count is None


def test_referenced_protocol_calls_reject_but_unrelated_metadata_is_allowed(trace: Path) -> None:
    path: Path = trace / "trace-one/trace.jsonl"
    metadata: Path = trace / "trace-one/payloads/event.json"
    metadata.write_text(json.dumps({"type": "new_diagnostic_metadata", "version": 99}))
    path.write_text(
        path.read_text()
        + "\n"
        + json.dumps({"payload": {"type": "protocol_event_observed", "event_payload": {"path": "payloads/event.json"}}})
    )
    assert not TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer"
    ).errors
    metadata.write_text(json.dumps({"type": "item_completed", "item": {"type": "custom_tool_call"}}))
    assert TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer"
    ).errors


@pytest.mark.parametrize(
    ("relative", "before", "after"),
    [
        ("manifest.json", '"schema_version": 2', '"schema_version": 3'),
        ("manifest.json", '"schema_version"', '"unknown_version"'),
        (
            "trace.jsonl",
            '"inference_call_id": "call-one", "response_id"',
            '"inference_call_id": "different", "response_id"',
        ),
        ("trace.jsonl", '"response_id": "response-one"', '"response_id": "different"'),
        ("trace.jsonl", '"inference_started"', '"unrelated_metadata"'),
        ("trace.jsonl", '"inference_completed"', '"unrelated_metadata"'),
        ("trace.jsonl", '"request_payload"', '"missing_request"'),
        ("trace.jsonl", '"response_payload"', '"missing_response"'),
        ("payloads/request.json", '"gpt-6-astra"', '"different-model"'),
        ("payloads/request.json", '"prompt"', '"changed prompt"'),
        ("payloads/request.json", '"instructions"', '"changed"'),
        ("payloads/request.json", '"instructions"', '"Do not instructions"'),
        ("payloads/request.json", '"additional_tools"', '"function_call"'),
        ("payloads/request.json", '"tools": [{"type": "namespace"', '"tools": null, "ignored": [{"type": "namespace"'),
        ("payloads/request.json", '"role": "user"', '"role": "tool"'),
        ("payloads/request.json", '"input_text"', '"output_text"'),
        ("payloads/response.json", '"role": "assistant"', '"role": "user"'),
        ("payloads/response.json", '"output_text"', '"input_text"'),
        ("payloads/response.json", '"answer"', '"different"'),
    ],
)
def test_invalid_trace_boundaries_fail_closed(trace: Path, relative: str, before: str, after: str) -> None:
    path: Path = trace / "trace-one" / relative
    content: str = path.read_text()
    assert before in content
    path.write_text(content.replace(before, after))
    result: TraceResult = TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer"
    )
    assert result.errors and result.tool_call_count is None


@pytest.mark.parametrize(
    "relative", ["manifest.json", "trace.jsonl", "payloads/request.json", "payloads/response.json"]
)
def test_missing_or_malformed_files_do_not_claim_zero_calls(trace: Path, relative: str) -> None:
    path: Path = trace / "trace-one" / relative
    path.write_text("not json")
    assert TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer"
    ).errors
    path.unlink()
    result: TraceResult = TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer"
    )
    assert result.errors and result.tool_call_count is None


def test_payload_paths_cannot_escape_through_parent_or_symlink(trace: Path) -> None:
    outside: Path = trace / "outside.json"
    outside.write_text("{}")
    root: Path = trace / "trace-one"
    with pytest.raises(ValueError, match="escapes"):
        TraceAudit.read(root=root, path="../outside.json")
    (root / "escape.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        TraceAudit.read(root=root, path="escape.json")


@pytest.mark.parametrize("kind", ["duplicate-inference", "duplicate-manifest", "reverse", "empty"])
def test_ambiguous_and_incomplete_traces_fail(trace: Path, kind: str) -> None:
    path: Path = trace / "trace-one/trace.jsonl"
    lines: list[str] = path.read_text().splitlines()
    if kind == "duplicate-inference":
        lines.append(lines[1])
    elif kind == "duplicate-manifest":
        (trace / "manifest.json").write_text('{"raw_event_log":"trace.jsonl"}')
    elif kind == "reverse":
        lines.reverse()
    else:
        lines = []
    path.write_text("\n".join(lines))
    assert TraceAudit()(
        root=trace, model="gpt-6-astra", prompt=b"prompt", instructions="instructions", answer="answer"
    ).errors
