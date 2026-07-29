"""
EVE Cold Brain — offline reasoning and learning (cold path).

Modules that run asynchronously and never block the hot path:
- InstructionParser: LLM-assisted text intent extraction with rule fallback
- IntentionFieldMapper: maps intent/world/self into intention field
- ReflectionWorker: LLM/VLM episode reflection and insight generation
- ConsolidationDecider: heuristic memory consolidation
"""

from .instruction_parser import InstructionParser
from .intention_field import IntentionFieldMapper
from .reflection_worker import ReflectionWorker
from .consolidation_decider import ConsolidationDecider

__all__ = [
    "InstructionParser",
    "IntentionFieldMapper",
    "ReflectionWorker",
    "ConsolidationDecider",
]
