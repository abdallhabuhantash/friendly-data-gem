"""Pure, dependency-free statistics helpers for the runtime benchmark.

Truthfulness rules
------------------
* An unavailable measurement is ``None``. Zero is never used as a stand-in for
  "we could not measure this".
* No division by zero: zero frames or zero elapsed time yields ``None`` FPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

__all__ = [
    "LatencySummary",
    "fps",
    "mean",
    "median",
    "percentage_change",
    "percentile",
]


def _clean(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def mean(values: Sequence[float]) -> Optional[float]:
    data = _clean(values)
    if not data:
        return None
    return sum(data) / len(data)


def median(values: Sequence[float]) -> Optional[float]:
    data = sorted(_clean(values))
    if not data:
        return None
    middle = len(data) // 2
    if len(data) % 2:
        return data[middle]
    return (data[middle - 1] + data[middle]) / 2.0


def percentile(values: Sequence[float], percent: float) -> Optional[float]:
    """Deterministic linear-interpolation percentile (``percent`` in 0..100)."""
    if not 0.0 <= float(percent) <= 100.0:
        raise ValueError("percent must be within 0..100")
    data = sorted(_clean(values))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * (float(percent) / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(data) - 1)
    weight = position - lower
    return data[lower] + (data[upper] - data[lower]) * weight


def fps(frames: int, elapsed_seconds: float) -> Optional[float]:
    """Measured throughput. ``None`` when it cannot be computed truthfully."""
    if int(frames) <= 0:
        return None
    if float(elapsed_seconds) <= 0.0:
        return None
    return float(frames) / float(elapsed_seconds)


def percentage_change(baseline: Optional[float], current: Optional[float]) -> Optional[float]:
    """Signed percentage change from ``baseline`` to ``current``."""
    if baseline is None or current is None:
        return None
    if float(baseline) == 0.0:
        return None
    return (float(current) - float(baseline)) / float(baseline) * 100.0


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Immutable summary of one measured duration series (milliseconds)."""

    count: int = 0
    mean_ms: Optional[float] = None
    median_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    max_ms: Optional[float] = None

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "LatencySummary":
        data = _clean(samples)
        if not data:
            return cls()
        return cls(
            count=len(data),
            mean_ms=mean(data),
            median_ms=median(data),
            p95_ms=percentile(data, 95.0),
            max_ms=max(data),
        )

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_ms": _round(self.mean_ms),
            "median_ms": _round(self.median_ms),
            "p95_ms": _round(self.p95_ms),
            "max_ms": _round(self.max_ms),
        }


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)
