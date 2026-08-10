"""Test-environment stubs for heavy runtime-only dependencies.

The pure-logic test suite must run without OpenCV, Ultralytics or the Supabase
client installed (those only exist on the deployment host). Missing modules are
replaced by permissive stubs so production modules can be imported and their
logic exercised directly. Installed real packages are never shadowed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

_OPTIONAL_MODULES = (
    "cv2",
    "ultralytics",
    "supabase",
    "requests",
    "httpx",
)


class _StubModule(types.ModuleType):
    def __getattr__(self, name):  # noqa: ANN001 - permissive stub
        value = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


def _install_stubs() -> None:
    for name in _OPTIONAL_MODULES:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = _StubModule(name)


_install_stubs()
