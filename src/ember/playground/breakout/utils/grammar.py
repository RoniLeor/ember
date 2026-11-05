"""Read the parts of one edits statement. Anything outside the grammar raises Unreadable or SyntaxError."""

import ast
import re

DECIMAL: re.Pattern[str] = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
INTEGER: re.Pattern[str] = re.compile(r"\d+")

type Index = tuple[int | slice, ...]
type Nested = float | list["Nested"]


class Unreadable(ValueError):
    """The statement does not follow the edits grammar."""


def read_number(node: ast.expr, source: str, pattern: re.Pattern[str]) -> str:
    text: str | None = ast.get_source_segment(source=source, node=node)
    if text is None or not pattern.fullmatch(text):
        raise Unreadable(f"Not a plain number: {text}")
    return text


def read_value(node: ast.expr, source: str) -> Nested:
    if isinstance(node, ast.List):
        return [read_value(node=item, source=source) for item in node.elts]
    return float(read_number(node=node, source=source, pattern=DECIMAL))


def read_bound(node: ast.expr | None, source: str, default: int) -> int:
    return default if node is None else int(read_number(node=node, source=source, pattern=INTEGER))


def read_axis(node: ast.expr, source: str, size: int) -> int | slice:
    if isinstance(node, ast.Slice):
        start: int = read_bound(node=node.lower, source=source, default=0)
        stop: int = read_bound(node=node.upper, source=source, default=size)
        step: int = read_bound(node=node.step, source=source, default=1)
        if step == 0 or start > size or stop > size:
            raise Unreadable("The slice has a zero step, or it is outside the tensor.")
        return slice(start, stop, step)
    index: int = int(read_number(node=node, source=source, pattern=INTEGER))
    if index >= size:
        raise Unreadable("The index is outside the tensor.")
    return index


def read_index(node: ast.expr, source: str, shape: tuple[int, ...]) -> Index:
    axes: list[ast.expr] = node.elts if isinstance(node, ast.Tuple) else [node]
    if len(axes) != len(shape):
        raise Unreadable("The statement must give one index for each tensor axis.")
    return tuple(read_axis(node=axis, source=source, size=size) for axis, size in zip(axes, shape, strict=True))


def read_parts(statement: str) -> tuple[ast.expr, ast.expr, ast.operator | None]:
    body: list[ast.stmt] = ast.parse(statement).body
    if len(body) != 1:
        raise Unreadable("A line must hold exactly one statement.")
    node: ast.stmt = body[0]
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return node.targets[0], node.value, None
    if isinstance(node, ast.AugAssign):
        return node.target, node.value, node.op
    raise Unreadable("The statement is not an assignment to one tensor range.")
