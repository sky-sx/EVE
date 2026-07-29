"""
Attention Trainer — optimizes attention / saliency parameters.

Trains an attention vector that weights which state dimensions
matter most for action selection. Uses random search over the
projected action space. Cold-path only.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class AttentionTrainer:
    """Trains attention weights over state dimensions.

    The attention vector a (shape state_dim,) produces a weighted
    state via element-wise multiplication: s_weighted = a * s.
    The quality of the attention is measured by reward alignment.

    Attributes:
        state_dim: Dimension of the state vector.
        population_size: Candidates per generation.
        noise_scale: Perturbation magnitude per step.
        top_fraction: Fraction kept as elites.
        seed: Random seed.
    """

    state_dim: int = 8
    population_size: int = 32
    noise_scale: float = 0.1
    top_fraction: float = 0.25
    seed: int = 42

    _params: np.ndarray = field(init=False, repr=False)
    _best_params: np.ndarray = field(init=False, repr=False)
    _best_score: float = field(default=-np.inf, init=False, repr=False)
    _rng: np.random.RandomState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ...

    def train(self, states: np.ndarray, targets: np.ndarray, epochs: int = 10) -> dict:
        """Train attention parameters via random search.

        Args:
            states: np.ndarray — state vectors.
            targets: np.ndarray — target attention maps.
            epochs: Number of training epochs.

        Returns:
            dict with best_score, epoch_scores, best_params_norm.
        """
        ...

    def get_best_params(self) -> dict:
        """Return the best attention parameters found.

        Returns:
            dict with keys: attention (ndarray), score (float).
        """
        ...
