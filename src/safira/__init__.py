"""SAFIRA-SSA public API."""

from __future__ import annotations

from .constants import PROJECT_LONG_NAME, PROJECT_NAME
from .sai import build_composite_sai, get_standard_country_code

__all__ = [
    "PROJECT_LONG_NAME",
    "PROJECT_NAME",
    "build_composite_sai",
    "get_standard_country_code",
]

__version__ = "0.1.0"
