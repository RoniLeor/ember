"""Preserve the original Breakout parsing behavior."""

import pytest

from ember.playground.breakout.utils.parse import has_open_bracket, last_block, split_statements


def test_last_block_uses_the_last_of_two_blocks() -> None:
    reply: str = "```edits\nb1[0] = 1\n```\ntext\n```edits\nb1[0] = 2\n```\n"
    assert last_block(reply=reply) == "b1[0] = 2\n"


def test_a_second_opening_fence_does_not_merge_two_blocks() -> None:
    reply: str = "```edits\nb1[0] = 1\n\nprose\n```edits\nb1[1] = 2\n```\n"
    assert last_block(reply=reply) == "b1[1] = 2\n"


@pytest.mark.parametrize(
    argnames=("text", "expected"),
    argvalues=[("b1[0] = [1,", True), ("b1[0] = [1]", False), ("b1[0] = 1", False), ("", False)],
)
def test_has_open_bracket(text: str, expected: bool) -> None:
    assert has_open_bracket(text=text) is expected


@pytest.mark.parametrize(
    argnames="reply",
    argvalues=[
        "```edits\nb1[0] = 1\n```\n```edits\nb1[0] = 2\n",
        "```edits\nb1[0] = 1\n```\n```edits",
        "```edits\nb1[0] = 1\n```\n```edits\nb1[0] = 2\n```not a fence\n",
    ],
)
def test_an_incomplete_last_block_never_falls_back_to_an_earlier_block(reply: str) -> None:
    with pytest.raises(expected_exception=ValueError, match="no closing fence"):
        last_block(reply=reply)


def test_a_body_line_with_backticks_stays_in_the_last_block() -> None:
    reply: str = "```edits\nb1[0] = 1\n```\ntext\n```edits\nb1[1] = 2\n```not a fence\nb1[2] = 3\n```\n"
    assert last_block(reply=reply) == "b1[1] = 2\n```not a fence\nb1[2] = 3\n"


def test_a_closing_fence_may_have_trailing_spaces_and_prose_may_follow() -> None:
    assert last_block(reply="before\n```edits  \nb1[0] = 1\n```  \nafter\n```python\nx\n```\n") == "b1[0] = 1\n"


@pytest.mark.parametrize(
    argnames="reply",
    argvalues=["", "no block here", "```python\nb1[0] = 1\n```", "```edits\nb1[0] = 1\n"],
)
def test_last_block_raises_without_a_complete_block(reply: str) -> None:
    with pytest.raises(expected_exception=ValueError, match="no complete edits block"):
        last_block(reply=reply)


def test_last_block_reads_windows_line_endings() -> None:
    assert last_block(reply="```edits\r\nb1[0] = 1\r\n```\r\n") == "b1[0] = 1\n"


def test_an_empty_block_has_no_statements() -> None:
    assert split_statements(block=last_block(reply="```edits\n```")) == []


def test_split_joins_a_multi_line_list() -> None:
    block: str = "W2[0:2, 0:3] = [[1, -1, 0.5],\n  [0, 2, 0]]\nb1[0] = 1\n"
    assert split_statements(block=block) == [
        "W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]",
        "b1[0] = 1",
    ]


def test_split_ignores_brackets_and_text_in_comments() -> None:
    block: str = "b1[0] = 1  # note [ here\n# whole comment [\nb1[1] = 2\n"
    assert split_statements(block=block) == ["b1[0] = 1", "b1[1] = 2"]


def test_an_unclosed_bracket_does_not_swallow_the_next_statement() -> None:
    assert split_statements(block="b1[0:2] = [1,\nb1[3] = 7\n") == [
        "b1[0:2] = [1,",
        "b1[3] = 7",
    ]
