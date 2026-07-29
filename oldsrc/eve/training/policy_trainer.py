"""
Policy Trainer — gradient-free policy optimization.

Trains a policy using random search / evolutionary optimization.
Fully numpy-based, no torch. Cold-path only.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PolicyTrainer:
    """Gradient-free policy trainer using random search.

    Optimizes a linear weight matrix W that maps state to action via
    a = tanh(W @ state). Evaluates candidates by total reward over the
    dataset.

    Attributes:
        state_dim: Dimension of input state vector.
        action_dim: Dimension of output action vector.
        population_size: Number of candidates per generation.
        noise_scale: Standard deviation of perturbation noise.
        top_fraction: Fraction of population kept as elites.
        seed: Random seed for reproducibility.
    """

    state_dim: int = 8
    action_dim: int = 4
    population_size: int = 32
    noise_scale: float = 0.1
    top_fraction: float = 0.25
    seed: int = 42

    _weights: np.ndarray = field(init=False, repr=False)
    _best_weights: np.ndarray = field(init=False, repr=False)
    _best_score: float = field(default=-np.inf, init=False, repr=False)
    _rng: np.random.RandomState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ...

    def train(self, dataset: dict, epochs: int = 100) -> dict:
        """Run random-search training for multiple epochs.

        Args:
            dataset: Dataset dict as produced by DatasetBuilder.build().
            epochs: Number of training epochs.

        Returns:
            dict with keys: best_score, best_weights_norm, epoch_scores,
            final_score.
        """
        ...

    def get_best_weights(self) -> dict:
        """Return the best weight matrix and score found so far.

        Returns:
            dict with keys: weights (ndarray), score (float).
        """
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save best weights and trainer state to a .npz file.

        Args:
            path: File path for the checkpoint.
        """
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load best weights and trainer state from a .npz file.

        Args:
            path: File path to the checkpoint.
        """
        ...
