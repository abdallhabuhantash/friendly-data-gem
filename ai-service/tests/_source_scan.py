"""Test helper: scan CODE only, ignoring docstrings and comments.

Static guarantees must be about behaviour, not prose. Explanatory docstrings are
allowed (and encouraged) to say things like "book is NOT paper" — they must not
make a source-text assertion fail.
"""

from __future__ import annotations

import ast
import io
import pathlib
import tokenize


def code_text(path: pathlib.Path | str) -> str:
    """Returns the module source with comments and docstrings removed."""
    source = pathlib.Path(path).read_text(encoding="utf-8")

    # Strip comments via tokenize.
    pieces: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        pieces.append(token.string if token.type != tokenize.NL else "\n")
    stripped = "\n".join(pieces)

    # Strip docstrings via AST re-render of the original source.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n" + stripped_literals(stripped)


def stripped_literals(text: str) -> str:
    """Keeps the comment-free token stream available for coarse checks."""
    return text
