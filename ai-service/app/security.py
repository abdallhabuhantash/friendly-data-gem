"""Shared secret verification for operational endpoints."""

from __future__ import annotations

import hmac
from typing import Optional


def verify_service_key(expected: Optional[str], provided: Optional[str]) -> bool:
    """Constant-time comparison of the `X-Service-Key` header.

    An unset expected key means the deployment did not enable stream auth; the
    endpoint stays closed rather than silently public.
    """
    if not expected:
        return False
    if not provided:
        return False
    return hmac.compare_digest(str(expected), str(provided))