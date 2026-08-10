"""Logging with hard secret redaction.

Secrets are scrubbed at the formatter level so an accidental log call can never
leak a service-role key, a camera password or a Telegram token.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Iterable

from .camera.source_builder import redact

_REDACTED = "***REDACTED***"
_TOKEN_PATTERNS = [
    re.compile(r"bot\d{6,}:[A-Za-z0-9_-]{20,}"),  # Telegram bot token
    re.compile(r"\beyJ[A-Za-z0-9_\-\.]{20,}"),  # JWT-shaped keys
    re.compile(r"\bsb_secret_[A-Za-z0-9_\-]+"),
]


class RedactingFilter(logging.Filter):
    """Strips credentials from every emitted record."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s and len(s) >= 8]

    def _scrub(self, text: str) -> str:
        cleaned = redact(text)
        for secret in self._secrets:
            cleaned = cleaned.replace(secret, _REDACTED)
        for pattern in _TOKEN_PATTERNS:
            cleaned = pattern.sub(_REDACTED, cleaned)
        return cleaned

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = self._scrub(str(record.msg))
            if record.args:
                record.args = tuple(
                    self._scrub(arg) if isinstance(arg, str) else arg for arg in record.args
                )
        except Exception:  # never break logging
            pass
        return True


def configure_logging(level: str = "INFO", secrets: Iterable[str] = ()) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    handler.addFilter(RedactingFilter(secrets))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)