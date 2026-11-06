"""Preserve the original Breakout grammar behavior."""

import ast

import pytest

from ember.playground.utils.grammar import (
    DECIMAL,
    INTEGER,
    Unreadable,
    read_axis,
    read_bound,
    read_index,
    read_number,
    read_parts,
    read_value,
)


def expression(source: str) -> ast.expr:
    return ast.parse(source=source, mode="eval").body


def subscript(source: str) -> ast.expr:
    node: ast.expr = expression(source=source)
    assert isinstance(node, ast.Subscript)
    return node.slice


@pytest.mark.parametrize(argnames="text", argvalues=["1", "-1", "2.5", ".5", "1e3", "-1E-2", "5."])
def test_read_number_accepts_plain_decimals(text: str) -> None:
    assert read_number(node=expression(source=text), source=text, pattern=DECIMAL) == text


@pytest.mark.parametrize(argnames="text", argvalues=["True", "1j", "0x10", "1_000", "1 + 1", "None"])
def test_read_number_rejects_other_expressions(text: str) -> None:
    with pytest.raises(expected_exception=Unreadable):
        read_number(node=expression(source=text), source=text, pattern=DECIMAL)


def test_read_value_keeps_the_list_nesting() -> None:
    text: str = "[[1, -2.5], [3e2, .5]]"
    assert read_value(node=expression(source=text), source=text) == [[1.0, -2.5], [300.0, 0.5]]


def test_read_bound_uses_the_default_only_without_a_node() -> None:
    assert read_bound(node=None, source="", default=7) == 7
    assert read_bound(node=expression(source="3"), source="3", default=7) == 3


def test_read_bound_rejects_a_negative_number() -> None:
    with pytest.raises(expected_exception=Unreadable):
        read_bound(node=expression(source="-1"), source="-1", default=7)


def test_integer_pattern_rejects_a_fraction() -> None:
    with pytest.raises(expected_exception=Unreadable):
        read_number(node=expression(source="1.5"), source="1.5", pattern=INTEGER)


@pytest.mark.parametrize(
    argnames=("text", "expected"),
    argvalues=[
        ("x[:]", slice(0, 10, 1)),
        ("x[2:8:3]", slice(2, 8, 3)),
        ("x[10:]", slice(10, 10, 1)),
        ("x[0:10]", slice(0, 10, 1)),
        ("x[9]", 9),
    ],
)
def test_read_axis_accepts_ranges_inside_the_tensor(text: str, expected: int | slice) -> None:
    assert read_axis(node=subscript(source=text), source=text, size=10) == expected


@pytest.mark.parametrize(argnames="text", argvalues=["x[10]", "x[-1]", "x[1:11]", "x[11:]", "x[::0]", "x[0:5:-1]"])
def test_read_axis_rejects_ranges_outside_the_tensor(text: str) -> None:
    with pytest.raises(expected_exception=Unreadable):
        read_axis(node=subscript(source=text), source=text, size=10)


def test_read_index_gives_one_entry_for_each_axis() -> None:
    text: str = "x[1, 2:4]"
    assert read_index(node=subscript(source=text), source=text, shape=(5, 5)) == (1, slice(2, 4, 1))
    assert read_index(node=subscript(source="x[1]"), source="x[1]", shape=(5,)) == (1,)


@pytest.mark.parametrize(argnames=("text", "shape"), argvalues=[("x[1]", (5, 5)), ("x[1, 2]", (5,))])
def test_read_index_rejects_the_wrong_number_of_axes(text: str, shape: tuple[int, ...]) -> None:
    with pytest.raises(expected_exception=Unreadable):
        read_index(node=subscript(source=text), source=text, shape=shape)


def test_read_parts_separates_plain_and_augmented_assignments() -> None:
    assert read_parts(statement="b1[0] = 1")[2] is None
    assert isinstance(read_parts(statement="b1[0] += 1")[2], ast.Add)


@pytest.mark.parametrize(argnames="statement", argvalues=["b1[0]", "a = b = 1", "a = 1; b = 2", ""])
def test_read_parts_rejects_other_statements(statement: str) -> None:
    with pytest.raises(expected_exception=Unreadable):
        read_parts(statement=statement)
