"""
Trainer Farm — manages multiple independent trainers.

Cold-path registry for training modules. Runs all registered
trainers in sequence and tracks per-trainer metrics.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


class TrainerProtocol(Protocol):
    """Minimal protocol that trainer objects must satisfy."""
    def train(self, dataset: dict, epochs: int) -> dict: ...


@dataclass
class TrainerFarm:
    """Registry-style manager for multiple concurrent trainers.

    Each trainer is registered by name. Training runs all registered
    trainers sequentially on the same dataset.

    Attributes:
        checkpoint_root: Directory for trainer checkpoints.
    """

    checkpoint_root: Path = Path("training_checkpoints")

    _trainers: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _metrics: dict[str, dict] = field(default_factory=dict, init=False, repr=False)

    def register(self, name: str, trainer: object) -> None:
        """Register a trainer instance under a name.

        Args:
            name: Unique name for the trainer.
            trainer: Any object with a train(dataset, epochs) -> dict method.
        """
        ...

    def train_all(self, job_spec: dict) -> dict:
        """Run all registered trainers using the job specification.

        Args:
            job_spec: dict with dataset, epochs, config.

        Returns:
            model_candidate: dict mapping trainer name to best model checkpoint.
        """
        ...

    def train_single(self, job_spec: dict) -> dict:
        """Train a single named trainer using the job specification.

        Args:
            job_spec: dict with name (trainer name), dataset, epochs, config.

        Returns:
            dict with training results for the named trainer.
        """
        ...

    def get_trainer(self, name: str) -> object:
        """Return the trainer registered under name.

        Args:
            name: Trainer name.

        Returns:
            The registered trainer object.

        Raises:
            KeyError: If name is not registered.
        """
        ...

    def list_trainers(self) -> list[str]:
        """Return sorted list of registered trainer names.

        Returns:
            list[str] of trainer names.
        """
        ...

    def get_metrics(self, name: str) -> dict:
        """Return the last training metrics for a trainer.

        Args:
            name: Trainer name.

        Returns:
            dict of training metrics.

        Raises:
            KeyError: If name is not registered or hasn't been trained yet.
        """
        ...

    def save_all(self) -> None:
        """Save checkpoints for all registered trainers."""
        ...
