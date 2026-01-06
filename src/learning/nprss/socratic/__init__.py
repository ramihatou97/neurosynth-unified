# src/learning/nprss/socratic/__init__.py
"""
Socratic Learning Mode

Implements guided learning through questioning rather than direct answers.
Based on research showing 77.8% learner preference for guided discovery.
"""

from .prompts import (
    SOCRATIC_SYSTEM_PROMPT,
    SOCRATIC_TEMPLATES,
    HINT_PROMPT,
    REVEAL_ANSWER_PROMPT
)
from .engine import SocraticEngine

__all__ = [
    'SOCRATIC_SYSTEM_PROMPT',
    'SOCRATIC_TEMPLATES',
    'HINT_PROMPT',
    'REVEAL_ANSWER_PROMPT',
    'SocraticEngine'
]
