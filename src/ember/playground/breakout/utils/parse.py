"""Find the last edits block in a reply and split it into statements."""

import re

OPENING_FENCE: re.Pattern[str] = re.compile(pattern=r"^```edits[ \t]*$", flags=re.MULTILINE)
CLOSING_FENCE: re.Pattern[str] = re.compile(pattern=r"^```[ \t]*$", flags=re.MULTILINE)
STATEMENT_START: re.Pattern[str] = re.compile(r"[A-Za-z_]\w*\s*\[")


def last_block(reply: str) -> str:
    """Return the body of the last edits block. If that block is not complete, raise: never use an earlier block."""
    text: str = reply.replace("\r\n", "\n")
    openings: list[re.Match[str]] = list(OPENING_FENCE.finditer(text))
    if not openings:
        raise ValueError("The reply has no complete edits block.")
    body_start: int = openings[-1].end() + 1
    closing: re.Match[str] | None = CLOSING_FENCE.search(text, pos=body_start)
    if closing is None:
        raise ValueError("The reply has no complete edits block: the last block has no closing fence.")
    return text[body_start : closing.start()]


def has_open_bracket(text: str) -> bool:
    return text.count("[") > text.count("]")


def split_statements(block: str) -> list[str]:
    """Join the lines of a long list. A line that starts with a tensor index always starts a new statement."""
    statements: list[str] = []
    line: str
    for line in block.splitlines():
        code: str = line.split(sep="#", maxsplit=1)[0].strip()
        if not code:
            continue
        if statements and has_open_bracket(text=statements[-1]) and not STATEMENT_START.match(code):
            statements[-1] += " " + code
        else:
            statements.append(code)
    return statements
