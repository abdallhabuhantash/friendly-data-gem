"""Test helper: scan CODE only, ignoring docstrings and comments.

Static guarantees must be about behaviour, not prose. Explanatory docstrings are
allowed (and encouraged) to say things like "book is NOT paper" — they must not
make a source-text assertion fail.
"""

from __future__ import annotations

import ast
import pathlib


def code_text(path: pathlib.Path | str) -> str:
    """Returns the module source with all docstrings and comments removed.

    Re-rendering the AST drops comments automatically; docstrings are removed
    explicitly. String/other literals used by real code are preserved, so a
    prohibited mapping such as ``"book": "paper"`` is still caught.
    """
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = list(getattr(node, "body", []))
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
