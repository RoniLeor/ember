"""Preserve the original Breakout applying behavior."""

import pytest

from ember.playground.utils import ParsedReply, read_reply


def fenced(*lines: str) -> str:
    return "```edits\n" + "\n".join(lines) + "\n```\n"


def test_every_form_from_the_task_examples() -> None:
    parsed: ParsedReply = read_reply(
        reply=fenced(
            "W1[12, 0] = 2.5",
            "W1[0:6, 3] = [1, 0, -1, 0, 2, 0]",
            "b1[0:4] = -1",
            "W1[3:600:60, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]",
            "W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]",
            "W3[0, 2] = 3",
            "W3[:, 2] *= 2",
        )
    )
    assert parsed.skipped == []
    assert parsed.applied == 7
    assert parsed.network.w1[12, 0] == 2.5
    assert parsed.network.w1[:6, 3].tolist() == [1, 0, -1, 0, 2, 0]
    assert parsed.network.b1[:5].tolist() == [-1, -1, -1, -1, 0]
    assert parsed.network.w1[483, 0] == 1 and parsed.network.w1[543, 0] == 1 and parsed.network.w1[423, 0] == 0
    assert parsed.network.w2[0, 2] == 0.5 and parsed.network.w2[1, 1] == 2
    assert parsed.network.w3[0, 2] == 6


def test_add_and_subtract_apply_in_order() -> None:
    parsed: ParsedReply = read_reply(reply=fenced("b1[0] = 5", "b1[0] += 2", "b1[0] -= 4", "b1[0] = 9", "b1[0] -= 1"))
    assert parsed.network.b1[0] == 8


def test_a_multi_line_list_applies_as_one_statement() -> None:
    parsed: ParsedReply = read_reply(reply=fenced("W2[0:2, 0:3] = [[1, -1, 0.5],  # first row", "  [0, 2, 0]]"))
    assert parsed.skipped == []
    assert parsed.applied == 1
    assert parsed.network.w2[1, 1] == 2


@pytest.mark.parametrize(
    argnames="statement",
    argvalues=[
        "W1[599, 31] = 1",
        "b1[31] = 1",
        "b1[0:32] = 1",
        "b1[32:] = 1",
        "b1[0] = 1000000",
        "b1[0] = -1e6",
        "b1[0] = .5",
        "b1[5:5] = 1",
    ],
)
def test_boundary_statements_are_read(statement: str) -> None:
    assert read_reply(reply=fenced(statement)).skipped == []


@pytest.mark.parametrize(
    argnames="statement",
    argvalues=[
        "W1[600, 0] = 1",
        "W1[0, 32] = 1",
        "b1[32] = 1",
        "b1[0:33] = 1",
        "b1[40:] = 1",
        "b1[-1] = 1",
        "W1[0, 0, 0] = 1",
        "W1[0] = 1",
        "W1[0:6:0, 0] = 1",
        "W1[0:2, 0] = [[1], [2, 3]]",
        "W2[0:2, 0:3] = [1, 2, 3]",
        "b1[0:6] = [5]",
        "b1[0] = [1]",
        "b1[0] = 1000000.5",
        "b1[0] = 1e999",
        "b1[0] /= 2",
        "Q1[0] = 1",
        "b1[0] = True",
        "b1[0] = 1j",
        "b1[0] = 0x10",
        "b1[0] = 1_000",
        "b1[0] = 1 + 1",
        "b1[0] = 1; b1[1] = 2",
        "b1[0] = b1[1] = 1",
        "b1[0]",
        "x = 1",
        "b1[0] = [1,",
        "not python at all",
    ],
)
def test_unreadable_statement_is_skipped_and_leaves_the_network_unchanged(
    statement: str,
) -> None:
    parsed: ParsedReply = read_reply(reply=fenced(statement, "b3[5] = 7"))
    assert parsed.skipped == [statement]
    assert parsed.applied == 1
    assert parsed.network.b3[5] == 7
    assert all(not tensor.any() for name, tensor in parsed.network.tensors().items() if name != "b3")


@pytest.mark.filterwarnings("ignore:overflow encountered in multiply")
def test_a_tensor_that_is_not_finite_is_an_error() -> None:
    with pytest.raises(expected_exception=ValueError, match="not finite"):
        read_reply(reply=fenced("b1[0] = 1000000", *["b1[0] *= 1000000"] * 60))


def test_a_reply_with_no_block_is_an_error() -> None:
    with pytest.raises(expected_exception=ValueError, match="no complete edits block"):
        read_reply(reply="The design is good, but I wrote no weights.")


def test_a_zero_network_results_from_an_empty_block() -> None:
    parsed: ParsedReply = read_reply(reply="```edits\n```\n")
    assert parsed.applied == 0
    assert all(not tensor.any() for tensor in parsed.network.tensors().values())
