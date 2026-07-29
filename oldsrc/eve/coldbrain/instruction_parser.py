"""
Instruction parser — LLM-backed text intent extraction with rule-based fallback (cold path).

Uses an external LLM for instruction parsing when available, falls back to
rule-based keyword and pattern matching. Deterministic in pure rule mode;
LLM mode introduces non-determinism but handles complex instructions better.

Runs in cold path; never blocks the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InstructionParser:
    """LLM-assisted instruction parser with rule-based fallback.

    When an LLM callable is provided, complex instructions are routed to
    the LLM while simple patterns still use the rule-based registry.
    Without an LLM, all parsing is purely rule-based.

    Maintains a registry of (intent, regex pattern) pairs for the fallback path.

    Attributes:
        confidence_threshold: Minimum confidence to return a result (default 0.3).
        llm_fn: Optional external LLM callable for complex instruction parsing.
    """

    confidence_threshold: float = 0.3
    llm_fn: Callable | None = None

    _patterns: list[tuple[str, str]] = field(default_factory=list)
    _pending: list[str] = field(default_factory=list)

    def parse(self, text: str) -> dict[str, Any] | None:
        """Parse text into a structured intent.

        May invoke the LLM callable for complex instructions when available,
        falling back to rule-based keyword/pattern matching otherwise.

        Args:
            text: Raw user instruction string.

        Returns:
            {intent, params, confidence} dict or None.
        """
        ...

    def register_pattern(self, intent: str, pattern: str) -> None:
        """Register a new (intent, regex pattern) pair.

        Args:
            intent: Intent name (e.g. 'track', 'explore').
            pattern: Regex pattern string for matching.
        """
        ...

    def list_known_intents(self) -> list[str]:
        """Return all registered intent names.

        Returns:
            Sorted list of unique intent names.
        """
        ...

    def enqueue(self, text: str) -> None:
        """Queue text for later batch processing (cold-path pattern)."""
        ...

    def process_pending(self) -> list[dict[str, Any]]:
        """Process all queued texts and return results.

        Returns:
            List of parsed intent dicts.
        """
        ...

    @property
    def pending_count(self) -> int:
        """Number of queued texts awaiting processing."""
        ...
