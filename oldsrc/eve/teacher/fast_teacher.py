"""
Fast Teacher — rule-based fast labeling for the cold path.

Fast teacher with optional learned heuristics (混合 NN/rule).
Handles simple reward/label decisions using deterministic heuristics.
Returns None for cases it cannot handle, which then fall through to
the slow teacher.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FastTeacher(nn.Module):
    """Rule-based teacher that labels simple scenarios instantly.

    Fast teacher with optional learned heuristics (混合 NN/rule).
    Uses a small set of heuristics to produce {reward, label, confidence}
    triples. When no heuristic matches, returns None to signal that the
    slow teacher should take over.
    """

    def __init__(
        self,
        proximity_threshold: float = 0.1,
        idle_penalty: float = -0.01,
    ) -> None:
        super().__init__()
        self.proximity_threshold = proximity_threshold
        self.idle_penalty = idle_penalty

    def label(self, state: dict, action: dict) -> dict | None:
        """Attempt to label *action* in the context of *state*.

        Args:
            state: The state dict.
            action: The action dict with at least an "action" key.

        Returns:
            A dict with reward, label, confidence keys when a rule
            matches, otherwise None.
        """
        ...
