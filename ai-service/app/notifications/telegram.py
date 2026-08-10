"""Telegram Bot API provider. Optional and disabled by default."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from .base import NotificationPayload, format_message

logger = logging.getLogger(__name__)


class TelegramProvider:
    """Sends a photo with caption when a local snapshot exists, else text."""

    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 15.0) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout

    @property
    def _base(self) -> str:
        return f"https://api.telegram.org/bot{self._token}"

    def send(self, payload: NotificationPayload) -> bool:
        caption = format_message(payload)
        snapshot = Path(payload.snapshot_file) if payload.snapshot_file else None
        if snapshot and snapshot.exists():
            if self._send_photo(snapshot, caption):
                return True
            logger.warning("Telegram photo send failed; falling back to text")
        return self._send_text(caption)

    def _send_photo(self, snapshot: Path, caption: str) -> bool:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base}/sendPhoto",
                    data={"chat_id": self._chat_id, "caption": caption[:1024]},
                    files={"photo": (snapshot.name, snapshot.read_bytes(), "image/jpeg")},
                )
            return self._ok(response)
        except Exception as exc:
            logger.warning("Telegram photo request error: %s", type(exc).__name__)
            return False

    def _send_text(self, message: str) -> bool:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base}/sendMessage",
                    json={"chat_id": self._chat_id, "text": message},
                )
            return self._ok(response)
        except Exception as exc:
            logger.warning("Telegram text request error: %s", type(exc).__name__)
            return False

    @staticmethod
    def _ok(response) -> bool:  # noqa: ANN001
        # Never log the response URL or body: both can contain the bot token.
        if response.status_code != 200:
            logger.warning("Telegram API returned HTTP %s", response.status_code)
            return False
        try:
            return bool(response.json().get("ok"))
        except Exception:
            return False