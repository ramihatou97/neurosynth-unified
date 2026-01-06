# src/learning/nprss/repositories/__init__.py
"""
NPRSS Repository Layer

Provides database access for all learning system entities.
Follows NeuroSynth's existing BaseRepository pattern.
"""

from .base import BaseRepository
from .learning_item import LearningItemRepository
from .user_state import UserLearningStateRepository
from .review_history import ReviewHistoryRepository
from .study_session import StudySessionRepository
from .milestone import MilestoneRepository
from .socratic import SocraticPromptRepository, SocraticResponseRepository

__all__ = [
    'BaseRepository',
    'LearningItemRepository',
    'UserLearningStateRepository',
    'ReviewHistoryRepository',
    'StudySessionRepository',
    'MilestoneRepository',
    'SocraticPromptRepository',
    'SocraticResponseRepository',
]
