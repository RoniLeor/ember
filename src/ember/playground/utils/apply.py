"""Apply edits statements to a zero network. A statement that cannot be read is skipped and reported."""

import ast
from dataclasses import dataclass

import numpy as np

from ember.playground.tensors import FloatArray, Tensors

from .grammar import Index, Unreadable, read_index, read_parts, read_value
from .parse import last_block, split_statements

MAX_MAGNITUDE: float = 1e6


@dataclass(frozen=True)
class ParsedReply:
    network: Tensors
    applied: int
    skipped: list[str]


def apply_statement(statement: str, tensors: dict[str, FloatArray]) -> None:
    target: ast.expr
    value_node: ast.expr
    operator: ast.operator | None
    target, value_node, operator = read_parts(statement=statement)
    if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)):
        raise Unreadable("The target is not a tensor range.")
    tensor: FloatArray | None = tensors.get(target.value.id)
    if tensor is None:
        raise Unreadable(f"Unknown tensor: {target.value.id}")
    index: Index = read_index(node=target.slice, source=statement, shape=tensor.shape)
    value: FloatArray = np.array(object=read_value(node=value_node, source=statement), dtype=np.float64)
    if value.shape not in ((), np.shape(tensor[index])):
        raise Unreadable("The value is not one number and does not have the exact shape of the range.")
    if value.size and not np.abs(value).max() <= MAX_MAGNITUDE:
        raise Unreadable("A value is larger than the limit.")
    if operator is None:
        tensor[index] = value
    elif isinstance(operator, ast.Add):
        tensor[index] += value
    elif isinstance(operator, ast.Sub):
        tensor[index] -= value
    elif isinstance(operator, ast.Mult):
        tensor[index] *= value
    else:
        raise Unreadable("The only operators are =, +=, -=, and *=.")


def read_reply(reply: str) -> ParsedReply:
    """Build the network from the last edits block. Statements apply in order."""
    network: Tensors = Tensors.zeros()
    tensors: dict[str, FloatArray] = network.tensors()
    statements: list[str] = split_statements(block=last_block(reply=reply))
    skipped: list[str] = []
    statement: str
    for statement in statements:
        try:
            apply_statement(statement=statement, tensors=tensors)
        except (SyntaxError, ValueError):
            skipped.append(statement)
    if not all(np.isfinite(tensor).all() for tensor in tensors.values()):
        raise ValueError("The edits give a tensor with a value that is not finite.")
    return ParsedReply(network=network, applied=len(statements) - len(skipped), skipped=skipped)
