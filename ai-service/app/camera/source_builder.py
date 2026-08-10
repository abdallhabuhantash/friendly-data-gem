"""Builds RTSP/demo capture sources from generic camera configuration.

No vendor-specific assumptions are made: `stream_path` and `channel` come from
the console exactly as the administrator configured them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

from ..domain.models import CameraConfig, SourceType

_CREDENTIAL_URL = re.compile(r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<creds>[^/@\s]+)@")


def redact(value: Optional[str]) -> str:
    """Removes any embedded credentials before a URL reaches a log line."""
    if not value:
        return ""
    return _CREDENTIAL_URL.sub(lambda match: f"{match.group('scheme')}***:***@", value)


@dataclass(frozen=True)
class CaptureSource:
    """A resolved capture target. `url` may contain credentials: never log it."""

    url: str
    kind: str  # "rtsp" | "file"
    loop: bool = False

    @property
    def safe_url(self) -> str:
        return redact(self.url)


def _normalise_path(camera: CameraConfig) -> str:
    path = (camera.stream_path or "").strip()
    if path and not path.startswith("/"):
        path = f"/{path}"
    return path


def build_rtsp_url(
    camera: CameraConfig,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """Constructs the RTSP URL. Credentials come only from local secure config."""
    auth = ""
    if username:
        auth = f"{quote(username, safe='')}:{quote(password or '', safe='')}@"
    return f"rtsp://{auth}{camera.host}:{int(camera.rtsp_port)}{_normalise_path(camera)}"


def build_source(
    camera: CameraConfig,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    demo_video_path: Optional[str] = None,
    demo_loop: bool = True,
) -> Optional[CaptureSource]:
    """Resolves the capture source, or None when the camera cannot be opened."""
    if camera.source_type is SourceType.DEMO:
        if not demo_video_path:
            return None
        return CaptureSource(url=demo_video_path, kind="file", loop=demo_loop)

    if not camera.host:
        return None
    return CaptureSource(
        url=build_rtsp_url(camera, username=username, password=password), kind="rtsp"
    )