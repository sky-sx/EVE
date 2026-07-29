"""Micro-skill adapter module — maps HotState to policy bias vector."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class MicroSkillAdapter(nn.Module):
    """Adapter between HotState and policy bias layer.

    Maps high-level state features into a policy bias vector that
    shifts the output distribution of the policy network.

    Attributes:
        skills: Registered micro-skill mappings keyed by name.
    """

    def __init__(self) -> None:
        """Initialize with default micro-skills."""
        super().__init__()
        ...

    def adapt(self, state: np.ndarray) -> np.ndarray:
        """Convert a HotState vector into a policy bias vector.

        Args:
            state: HotState vector (1D float numpy array).

        Returns:
            Policy bias vector as np.ndarray.
        """
        ...

    def register_skill(self, name: str, mapping: dict[str, Any]) -> None:
        """Register a new micro-skill mapping.

        Args:
            name: Action type name (e.g. "scroll", "key_press").
            mapping: Dict with keys: motor_type, params_template.
        """
        ...

    @property
    def skill_names(self) -> list[str]:
        """List of registered skill names."""
        ...
