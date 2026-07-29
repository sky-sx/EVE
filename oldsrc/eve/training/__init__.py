"""
EVE Training Pipeline — cold-path offline learning.

All training modules operate on the cold path and must never block
the hot path. Uses numpy-based gradient-free optimization only.
Deterministic and synthetic-safe by default.
"""

from .dataset_builder import DatasetBuilder
from .trainer_farm import TrainerFarm
from .policy_trainer import PolicyTrainer
from .attention_trainer import AttentionTrainer
from .retina_trainer import RetinaTrainer
from .micro_skill_trainer import MicroSkillTrainer

__all__ = [
    "DatasetBuilder",
    "TrainerFarm",
    "PolicyTrainer",
    "AttentionTrainer",
    "RetinaTrainer",
    "MicroSkillTrainer",
]
