"""
Feature-engineering package.

Expose the single public helper:

    >>> from src.features import build_features
"""

from __future__ import annotations

from .engineering import build_features

__all__: list[str] = ["build_features"]
