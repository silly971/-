"""Changepoint detection toolbox package."""

from .algorithms import DETECTORS, DetectionResult
from .gui import ChangepointToolbox, main

__all__ = [
    "DETECTORS",
    "DetectionResult",
    "ChangepointToolbox",
    "main",
]
