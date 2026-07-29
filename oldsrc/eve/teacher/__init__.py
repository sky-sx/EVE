"""
EVE Teacher Layer — offline labeling and reward computation (cold path).

All teacher components run on the cold path and must never block the
hot path. The TeacherOrchestra coordinates fast (rule-based) and slow
(deferred) teachers, with a RewardOracle for ground-truth rewards.
"""

from .teacher_orchestra import TeacherOrchestra
from .fast_teacher import FastTeacher
from .slow_teacher import SlowTeacher
from .reward_oracle import RewardOracle

__all__ = [
    "TeacherOrchestra",
    "FastTeacher",
    "SlowTeacher",
    "RewardOracle",
]
