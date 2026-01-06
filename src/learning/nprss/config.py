# src/learning/nprss/config.py
"""
NPRSS Configuration

Settings for the learning system components.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from functools import lru_cache


@dataclass
class NPRSSSettings:
    """NPRSS system settings."""

    # FSRS Parameters
    fsrs_request_retention: float = 0.9
    fsrs_maximum_interval: int = 365
    fsrs_weights: List[float] = field(default_factory=lambda: [
        0.4, 0.6, 2.4, 5.8, 4.93,
        0.94, 0.86, 0.01, 1.49, 0.14,
        0.94, 2.18, 0.05, 0.34, 1.26,
        0.29, 2.61
    ])

    # R-Level Schedule (days)
    r_level_days: List[int] = field(default_factory=lambda: [
        1, 3, 7, 14, 30, 60, 120
    ])

    # Card Generation
    max_cards_per_chunk: int = 20
    max_cards_per_document: int = 200
    min_quality_score: float = 0.6
    min_confidence: float = 0.7

    # Socratic Mode
    socratic_max_hints: int = 3
    socratic_default_top_k: int = 5

    # Session Defaults
    default_session_cards: int = 20
    max_session_duration_minutes: int = 60


# Singleton instance
_settings: Optional[NPRSSSettings] = None


@lru_cache()
def get_nprss_settings() -> NPRSSSettings:
    """Get cached NPRSS settings instance."""
    global _settings
    if _settings is None:
        _settings = NPRSSSettings()
    return _settings


def configure_nprss(
    fsrs_request_retention: float = None,
    fsrs_maximum_interval: int = None,
    fsrs_weights: List[float] = None,
    **kwargs
) -> NPRSSSettings:
    """
    Configure NPRSS settings.

    Args:
        fsrs_request_retention: Target retention rate (0-1)
        fsrs_maximum_interval: Max days between reviews
        fsrs_weights: FSRS algorithm weights (17 values)
        **kwargs: Other settings

    Returns:
        Updated settings instance
    """
    global _settings

    if _settings is None:
        _settings = NPRSSSettings()

    if fsrs_request_retention is not None:
        _settings.fsrs_request_retention = fsrs_request_retention
    if fsrs_maximum_interval is not None:
        _settings.fsrs_maximum_interval = fsrs_maximum_interval
    if fsrs_weights is not None:
        _settings.fsrs_weights = fsrs_weights

    for key, value in kwargs.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)

    # Clear cache
    get_nprss_settings.cache_clear()

    return _settings
