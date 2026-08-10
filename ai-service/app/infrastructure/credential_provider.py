"""Camera credential resolution.

Credentials never leave this process, never reach the API layer and are never
logged. The default source is a git-ignored local JSON file on the Windows
machine running the service.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

Credentials = tuple[Optional[str], Optional[str]]


class CredentialSource(Protocol):
    def get(self, camera_id: str) -> Credentials: ...


class FileCredentialProvider:
    """Reads `secrets/cameras.json` and caches it until the file changes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime: float | None = None
        self._data: dict[str, dict[str, str]] = {}

    def _load(self) -> None:
        if not self._path.exists():
            self._data = {}
            return
        mtime = self._path.stat().st_mtime
        if mtime == self._mtime:
            return
        try:
            parsed = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = parsed if isinstance(parsed, dict) else {}
            self._mtime = mtime
            logger.info("Loaded camera credentials for %d camera(s)", len(self._data))
        except (OSError, json.JSONDecodeError) as exc:
            # Log the failure kind only, never the file content.
            logger.error("Unable to read camera credentials file: %s", type(exc).__name__)
            self._data = {}

    def get(self, camera_id: str) -> Credentials:
        self._load()
        entry = self._data.get(camera_id) or {}
        return entry.get("username"), entry.get("password")


class SupabaseCredentialProvider:
    """Optional service-role-only fallback to public.camera_credentials."""

    def __init__(self, repository) -> None:  # noqa: ANN001 - avoids import cycle
        self._repository = repository

    def get(self, camera_id: str) -> Credentials:
        try:
            return self._repository.camera_credentials(camera_id)
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Credential lookup failed for camera %s: %s", camera_id, type(exc).__name__)
            return (None, None)


class ChainedCredentialProvider:
    """Local file first, optional database fallback second."""

    def __init__(self, sources: list[CredentialSource]) -> None:
        self._sources = sources

    def get(self, camera_id: str) -> Credentials:
        for source in self._sources:
            username, password = source.get(camera_id)
            if username or password:
                return username, password
        return (None, None)