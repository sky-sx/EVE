"""Neural attention network that computes saliency from HotState."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(slots=True)
class AttentionState:
    """Saliency and focus derived from HotState attention.

    Attributes:
        saliency: Saliency map as (H, W) float32 numpy array.
        focus_x: Normalized x-coordinate of peak attention ([0, 1]).
        focus_y: Normalized y-coordinate of peak attention ([0, 1]).
        attention_level: Global attention level scalar ([0, 1]).
    """

    saliency: np.ndarray
    focus_x: float
    focus_y: float
    attention_level: float


class AttentionPolicyNet(nn.Module):
    """Neural attention network that computes saliency from HotState.

    Uses center-surround contrast and motion-based heuristics to produce
    a downsampled saliency map and focus point.

    Attributes:
        scale: Downsampling factor for the saliency map (default 8).
    """

    def __init__(self, scale: int = 8) -> None:
        """Initialize the attention policy network.

        Args:
            scale: Downsampling factor for the saliency map.
        """
        super().__init__()
        ...

    def compute_saliency(
        self,
        frame: np.ndarray,
        state_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute a saliency/attention map from a synthetic frame.

        Args:
            frame: Input frame as (H, W, C) numpy array (uint8 or float).
            state_vector: HotState vector (1D float array).

        Returns:
            Saliency map as (H//scale, W//scale) float32 numpy array,
            normalized to [0, 1].
        """
        ...

    def forward(
        self,
        frame: np.ndarray,
        state_vector: np.ndarray,
    ) -> AttentionState:
        """Forward pass: compute saliency and produce AttentionState.

        Args:
            frame: Input frame as (H, W, C) numpy array (uint8 or float).
            state_vector: HotState vector (1D float array).

        Returns:
            AttentionState with saliency map, focus point, and attention level.
        """
        ...
