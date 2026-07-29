"""
Slow Teacher — deferred deep reasoning teacher for the cold path.

Slow deep reasoning teacher, works on full episodes (混合 NN/rule).
Simulates deeper analysis with a configurable processing delay and
rule-based reasoning. Never called from the hot path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SlowTeacher(nn.Module):
    """Deferred teacher that labels complex scenarios.

    Slow deep reasoning teacher, works on full episodes (混合 NN/rule).
    Produces richer output than FastTeacher: reward, label, confidence,
    and a *reasoning* string. A configurable delay simulates the cost
    of deep analysis.
    """

    def __init__(self, processing_delay_s: float = 0.1) -> None:
        super().__init__()
        self.processing_delay_s = processing_delay_s

    def label(self, episode: dict) -> dict:
        """Label an entire *episode* with deeper analysis.

        Always returns a result—this is the fallback after FastTeacher
        returns None. The simulated delay makes this unsuitable for the
        hot path.

        Args:
            episode: Episode dict containing states, actions, and
                metadata for the full episode.

        Returns:
            A dict with keys: reward, label, confidence, reasoning.
        """
        ...
