"""Benchmark tooling (measurement only).

Importing this package must never load a model, contact a camera or download
weights. Heavy dependencies (torch, ultralytics, OpenCV) are imported lazily
inside the functions that actually execute a benchmark.
"""

__all__ = ["statistics", "runtime_benchmark"]
