"""EVE Dock — 训练机床，负责 TNN 训练全流程。"""
from eve.dock.order import TrainingOrder, TrainingResult
from eve.dock.trainer import Trainer

__all__ = ["Trainer", "TrainingOrder", "TrainingResult"]
